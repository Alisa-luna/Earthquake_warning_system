"""
撞击过滤器 - 真正的多模型投票实现
基于SeisBench的PhaseNet多模型集成
"""

import numpy as np
# 添加兼容性：如果 trapz 不存在，用 trapezoid 替代
if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid
import torch
import logging
from collections import deque
from scipy import signal
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 导入SeisBench
try:
    import seisbench.models as sbm
    import seisbench.util as sbu
    SEISBENCH_AVAILABLE = True
except ImportError:
    SEISBENCH_AVAILABLE = False
    logger.error("SeisBench未安装，请运行: pip install seisbench")
    raise


class ImpactFilter:
    """
    PhaseNet多模型集成投票器
    加载多个预训练模型，对波形进行集成推理
    """

    def __init__(self, sampling_rate=100, device='cpu'):
        """
        初始化模型集成

        Args:
            sampling_rate: 采样率 (Hz)
            device: 运行设备 ('cpu' 或 'cuda')
        """
        self.sampling_rate = sampling_rate
        self.device = device
        self.models = {}
        self.model_weights = {}  # 各模型权重（基于验证集性能）

        # PhaseNet需要的采样率
        self.target_sampling_rate = 100  # PhaseNet默认100Hz

        # 模型配置
        self.model_configs = {
            'stead': {
                'name': 'PhaseNet (STEAD)',
                'weight': 1.2,  # 在STEAD数据集上训练，泛化性好
                'threshold': 0.00,  # P波概率阈值
                'description': '在STEAD数据集训练，适合一般场景'
            },
            'geofon': {
                'name': 'PhaseNet (GEOFON)',
                'weight': 1.1,  # 在GEOFON数据集训练，欧洲数据为主
                'threshold': 0.28,
                'description': '在GEOFON数据集训练，适合区域地震'
            },
            'original': {
                'name': 'PhaseNet (Original)',
                'weight': 1.0,  # 原始论文模型
                'threshold': 0.90,
                'description': '原始论文模型，精度高但较严格'
            },
            'ethz': {
                'name': 'PhaseNet (ETHZ)',
                'weight': 0.9,  # ETH Zurich数据集
                'threshold': 0.00,
                'description': 'ETH Zurich数据集训练，适合瑞士地区'
            },
            'instance': {
                'name': 'PhaseNet (Instance)',
                'weight': 1.05,  # Instance数据集，包含更多噪声
                'threshold': 0.00,
                'description': 'Instance数据集，抗噪性好'
            }
        }

        self._load_models()
        logger.info(f"✅ PhaseNet集成初始化完成，已加载 {len(self.models)} 个模型")

    def _load_models(self):
        """加载所有可用的PhaseNet模型"""
        for model_name in self.model_configs.keys():
            try:
                logger.info(f"  加载模型: {model_name}...")
                model = sbm.PhaseNet.from_pretrained(model_name)
                model.eval()
                model.to(self.device)
                self.models[model_name] = model
                logger.info(f"  ✅ {model_name} 加载成功")
            except Exception as e:
                logger.warning(f"  ⚠️ 加载模型 {model_name} 失败: {e}")

        if not self.models:
            raise RuntimeError("没有成功加载任何PhaseNet模型")

        # 归一化权重
        total_weight = sum([self.model_configs[name]['weight'] for name in self.models.keys()])
        for name in self.models.keys():
            self.model_weights[name] = self.model_configs[name]['weight'] / total_weight

    def preprocess_waveform(self, waveform):
        """
        预处理波形以适应PhaseNet输入

        Args:
            waveform: (3, N) 原始波形

        Returns:
            torch.Tensor: (1, 3, N') 处理后的波形
        """
        # 确保是3分量
        if waveform.shape[0] != 3:
            if waveform.shape[0] == 1:  # 单分量，复制到三分量
                waveform = np.repeat(waveform, 3, axis=0)
            else:
                raise ValueError(f"需要三分量输入，得到 {waveform.shape[0]} 分量")

        # 重采样到100Hz（如果需要）
        if self.sampling_rate != self.target_sampling_rate:
            from scipy import signal
            resample_ratio = self.target_sampling_rate / self.sampling_rate
            new_length = int(waveform.shape[1] * resample_ratio)
            waveform_resampled = np.zeros((3, new_length))
            for i in range(3):
                waveform_resampled[i] = signal.resample(waveform[i], new_length)
            waveform = waveform_resampled

        # 归一化（每个通道独立）
        for i in range(3):
            std = np.std(waveform[i])
            if std > 0:
                waveform[i] = (waveform[i] - np.mean(waveform[i])) / std

        if waveform.shape[1] < 3000:
            pad_width = 3000 - waveform.shape[1]
            waveform = np.pad(waveform, ((0, 0), (0, pad_width)), 'edge')

        # 转换为tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # (1, 3, N)
        waveform_tensor = waveform_tensor.to(self.device)

        return waveform_tensor

    def ensemble_predict(self, waveform, return_details=False):
        """
        集成预测

        Args:
            waveform: (3, N) 原始波形
            return_details: 是否返回详细信息

        Returns:
            dict: 预测结果
        """
        if not self.models:
            return {'p_probability': 0.5, 'is_earthquake': True, 'confidence': 0}

        try:
            # 预处理
            input_tensor = self.preprocess_waveform(waveform)

            all_p_probs = []
            all_picks = []
            model_results = {}

            # 每个模型独立推理
            with torch.no_grad():
                for name, model in self.models.items():
                    try:
                        # 模型推理
                        predictions = model(input_tensor)

                        # 获取P波概率
                        if isinstance(predictions, dict):
                            p_prob = predictions['P'].cpu().numpy()
                        else:
                            # 如果是tuple，通常是 (P_prob, S_prob)
                            p_prob = predictions[0].cpu().numpy()

                        # 平滑概率（移动平均）
                        window_size = min(50, len(p_prob[0]) // 10)
                        kernel = np.ones(window_size) / window_size
                        p_prob_smooth = np.convolve(p_prob[0], kernel, mode='same')

                        # 最大概率
                        max_p_prob = np.max(p_prob_smooth)

                        # 找到P波到时
                        threshold = self.model_configs[name]['threshold']
                        p_idx = np.where(p_prob_smooth > threshold)[0]
                        if len(p_idx) > 0:
                            p_pick = p_idx[0] / self.target_sampling_rate  # 转换为秒
                        else:
                            p_pick = None

                        # 记录结果
                        model_results[name] = {
                            'p_probability': float(max_p_prob),
                            'p_pick': p_pick,
                            'threshold': threshold,
                            'weight': self.model_weights[name]
                        }

                        all_p_probs.append(max_p_prob)

                        logger.info(f"    模型 {name}: P波概率={max_p_prob:.3f}, 到时={p_pick:.2f}s")
                        if p_pick is not None:
                            logger.debug(f"    模型 {name}: P波概率={max_p_prob:.3f}, 到时={p_pick:.2f}s")
                        else:
                            logger.debug(f"    模型 {name}: P波概率={max_p_prob:.3f}, 到时=未检测到")

                    except Exception as e:
                        logger.warning(f"模型 {name} 推理失败: {e}")
                        logger.debug(f"输入张量形状: {input_tensor.shape}")
                        logger.debug(f"输入张量统计: min={input_tensor.min():.3f}, max={input_tensor.max():.3f}, mean={input_tensor.mean():.3f}")
                        continue

            if not all_p_probs:
                return {'p_probability': 0.5, 'is_earthquake': True, 'confidence': 0}

            # 加权平均概率
            weighted_prob = 0
            total_weight = 0
            for name, result in model_results.items():
                weighted_prob += result['p_probability'] * result['weight']
                total_weight += result['weight']

            avg_p_prob = weighted_prob / total_weight if total_weight > 0 else np.mean(all_p_probs)

            # 投票：多少模型认为这是地震
            votes = []
            for name, result in model_results.items():
                threshold = self.model_configs[name]['threshold']
                votes.append(1 if result['p_probability'] > threshold else 0)

            vote_ratio = sum(votes) / len(votes) if votes else 0.5

            # 计算置信度（基于模型一致性）
            prob_std = np.std(all_p_probs) if len(all_p_probs) > 1 else 0.2
            confidence = 1 - min(0.5, prob_std)  # 标准差越小，置信度越高

            # 最终判断：地震还是撞击？
            # 地震：P波概率高且投票通过
            is_earthquake = avg_p_prob > 0.27 and vote_ratio > 0.6

            result={
                'p_probability': float(avg_p_prob) if avg_p_prob is not None else 0.5,
                'is_earthquake': is_earthquake if is_earthquake is not None else True,
                'confidence': float(confidence) if confidence is not None else 0.5,
                'vote_ratio': float(vote_ratio) if vote_ratio is not None else 0.5,
                'votes': votes if votes is not None else [0],
                'model_count': len(votes) if votes is not None else 1
            }

            if return_details:
                result['model_details'] = model_results
                result['all_probabilities'] = all_p_probs

            return result

        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            import traceback
            traceback.print_exc()
            return {'p_probability': 0.5, 'is_earthquake': True, 'confidence': 0}

class ImpactAwarePipeline:
    """
    撞击过滤器 - 三重检测机制（带真正的多模型投票）
    基于：频谱特征、能量增长速率、多模型投票
    """

    def __init__(self, denoiser=None, phase_model=None, eqt_model=None, sampling_rate=100):
        """
        初始化撞击过滤器

        Args:
            denoiser: 降噪器对象（可选）
            phase_model: PhaseNet模型（可选，现在用内部集成）
            eqt_model: EQTransformer模型（可选）
            sampling_rate: 采样率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.denoiser = denoiser  # 保存降噪器
        self.phase_model = phase_model  # 保存PhaseNet模型（虽然可能用内部集成）
        self.eqt_model = eqt_model  # 保存EQTransformer模型

        # ===== 1. 振幅相关阈值（宽松）=====
        self.min_amplitude_for_impact = 0.05  # 低于0.6g不判为撞击
        self.max_earthquake_amplitude = 0.03  # 低于0.03g可能是微弱地震，绝对不拦

        # ===== 2. 起振特征阈值（严格）=====
        self.onset_ratio_threshold = 1.0  # 前0.3秒振幅占比>85%才判为撞击
        self.onset_window = 0.3  # 起振检测窗口（秒）

        # ===== 3. 能量增长阈值（宽松）=====
        self.energy_growth_threshold = 0.5  # 增长<0.8倍才判为撞击
        self.energy_window = 0.5  # 能量计算窗口（秒）

        # ===== 4. 频谱特征阈值（核心）=====
        self.low_freq_range = (1, 8)  # 地震主要频段
        self.high_freq_range = (12, 30)  # 撞击主要频段
        self.spectral_ratio_threshold = 0.6  # 低/高频比<0.8才判为撞击

        # ===== 5. 持续时间阈值 =====
        self.min_duration = 0.8  # 最短地震持续时间（秒）
        self.max_impact_duration = 4.0  # 最长撞击持续时间（秒）

        # ===== 6. 多模型投票 =====
        self.ensemble = ImpactFilter(sampling_rate)  # 使用真正的模型投票器
        self.model_vote_threshold = 0.3  # 投票通过所需比例
        self.min_p_probability = 0.3  # 最小P波概率

        # 撞击模式状态
        self.impact_mode = False
        self.impact_duration = 0
        self.consecutive_impacts = 0

        # 历史记录
        self.impact_history = deque(maxlen=20)
        self.earthquake_buffer = deque(maxlen=30)
        self.recent_decisions = deque(maxlen=50)

        # 统计信息
        self.total_checks = 0
        self.impact_count = 0
        self.earthquake_count = 0
        self.spectral_rejects = 0
        self.energy_rejects = 0
        self.onset_rejects = 0
        self.vote_rejects = 0

        logger.info("✅ 撞击过滤器初始化完成（三重检测+真正模型投票）")



    def _spectral_ratio(self, waveform):
        """计算低频能量与高频能量的比值"""
        # 用振幅最大的通道
        channel_energy = np.sum(waveform ** 2, axis=1)
        main_channel = np.argmax(channel_energy)
        data = waveform[main_channel]

        # 计算功率谱密度
        freqs, psd = signal.periodogram(data, fs=self.sampling_rate)

        # 低频能量
        low_mask = (freqs >= self.low_freq_range[0]) & (freqs <= self.low_freq_range[1])
        low_energy = np.sum(psd[low_mask]) + 1e-10

        # 高频能量
        high_mask = (freqs >= self.high_freq_range[0]) & (freqs <= self.high_freq_range[1])
        high_energy = np.sum(psd[high_mask]) + 1e-10

        return low_energy / high_energy

    def _energy_growth_rate(self, waveform):
        """计算能量增长速率"""
        # 用振幅最大的通道
        channel_energy = np.sum(waveform ** 2, axis=1)
        main_channel = np.argmax(channel_energy)
        data = waveform[main_channel]

        # 计算滑动窗口RMS
        window_size = int(self.energy_window * self.sampling_rate)
        if len(data) < window_size * 3:
            return 1.0

        stride = window_size // 2
        rms_values = []

        for i in range(0, len(data) - window_size, stride):
            rms = np.sqrt(np.mean(data[i:i + window_size] ** 2))
            rms_values.append(rms)

        if len(rms_values) < 3:
            return 1.0

        # 前30%窗口的平均RMS
        early_count = max(1, len(rms_values) // 3)
        early_rms = np.mean(rms_values[:early_count])

        # 后30%窗口的平均RMS
        late_rms = np.mean(rms_values[-early_count:])

        if early_rms < 1e-6:
            return 10.0  # 早期能量接近0，增长速率极大（可能是地震）

        return late_rms / early_rms

    def _onset_ratio(self, waveform):
        """计算起振比例（前N秒振幅 / 最大振幅）"""
        early_samples = int(self.onset_window * self.sampling_rate)
        if waveform.shape[1] < early_samples:
            return 1.0

        early_amp = np.max(np.abs(waveform[:, :early_samples]))
        full_amp = np.max(np.abs(waveform))

        if full_amp < 1e-6:
            return 1.0

        return early_amp / full_amp

    def is_impact(self, waveform, event_data=None):
        """
        三重检测：判断是否是撞击（使用真正的模型投票）

        Args:
            waveform: (3, N) 波形数据
            event_data: 原始事件数据（可选，用于兼容）

        Returns:
            bool: True=撞击（应过滤），False=地震（放过）
            dict: 诊断信息
        """
        self.total_checks += 1

        npts = waveform.shape[1]
        duration = npts / self.sampling_rate
        max_amp = np.max(np.abs(waveform))

        results = {
            'is_impact': False,
            'reasons': [],
            'scores': {},
            'model_result': None
        }

        # ===== 0. 快速预检 =====
        if max_amp < self.max_earthquake_amplitude:
            # 振幅很小，可能是微弱地震，不拦截
            logger.debug(f"振幅小 ({max_amp:.3f}g)，可能是微弱地震，放过")
            self.earthquake_count += 1
            self.earthquake_buffer.append('earthquake')
            results['reasons'].append('low_amplitude')
            return False, results

        if max_amp < self.min_amplitude_for_impact:
            # 振幅不够大，不太可能是撞击
            logger.debug(f"振幅 ({max_amp:.3f}g) 小于撞击阈值，放过")
            self.earthquake_count += 1
            self.earthquake_buffer.append('earthquake')
            results['reasons'].append('amplitude_too_low')
            return False, results

        if duration < self.min_duration:
            # 持续时间太短，肯定是撞击
            logger.info(f"🚫 持续时间太短 ({duration:.2f}s)，判为撞击")
            self.impact_count += 1
            self.impact_history.append(1)
            self.earthquake_buffer.append('impact')
            results['is_impact'] = True
            results['reasons'].append('duration_too_short')
            return True, results

        # ===== 1. 频谱特征检测 =====
        spectral_ratio = self._spectral_ratio(waveform)
        results['scores']['spectral_ratio'] = spectral_ratio

        if spectral_ratio < self.spectral_ratio_threshold:
            logger.info(f"🚫 频谱特征: 高频主导 (比值={spectral_ratio:.2f})")
            self.impact_count += 1
            self.spectral_rejects += 1
            self.impact_history.append(1)
            self.earthquake_buffer.append('impact')
            results['is_impact'] = True
            results['reasons'].append('high_frequency_dominant')
            return True, results
        else:
            logger.debug(f"✅ 频谱特征: 低频主导 (比值={spectral_ratio:.2f})")

        # ===== 2. 能量增长检测 =====
        growth_rate = self._energy_growth_rate(waveform)
        results['scores']['growth_rate'] = growth_rate

        if growth_rate < self.energy_growth_threshold:
            logger.info(f"🚫 能量增长太慢 ({growth_rate:.2f})，判为撞击")
            self.impact_count += 1
            self.energy_rejects += 1
            self.impact_history.append(1)
            self.earthquake_buffer.append('impact')
            results['is_impact'] = True
            results['reasons'].append('slow_energy_growth')
            return True, results
        else:
            logger.debug(f"✅ 能量增长正常 ({growth_rate:.2f})")

        # ===== 3. 起振特征检测 =====
        onset_ratio = self._onset_ratio(waveform)
        results['scores']['onset_ratio'] = onset_ratio

        if onset_ratio > self.onset_ratio_threshold:
            logger.info(f"🚫 起振太突然 (占比={onset_ratio:.2f})")
            self.impact_count += 1
            self.onset_rejects += 1
            self.impact_history.append(1)
            self.earthquake_buffer.append('impact')
            results['is_impact'] = True
            results['reasons'].append('sudden_onset')
            return True, results
        else:
            logger.debug(f"✅ 起振正常 (占比={onset_ratio:.2f})")

        # ===== 4. 真正的多模型投票 =====
        logger.info(f"   🤖 调用PhaseNet集成投票...")
        model_result = self.ensemble.ensemble_predict(waveform, return_details=True)
        results['model_result'] = model_result

        # 记录得分
        results['scores']['p_probability'] = model_result.get('p_probability', 0.5)
        results['scores']['vote_ratio'] = model_result.get('p_probability', 0.5)
        results['scores']['model_confidence'] = model_result.get('confidence', 0.5)

        # 模型判断
        if not model_result['is_earthquake']:
            logger.info(f"🚫 模型投票: 非地震 (P概率={model_result['p_probability']:.2f}, "
                       f"投票率={model_result['vote_ratio']:.2f})")
            self.impact_count += 1
            self.vote_rejects += 1
            self.impact_history.append(1)
            self.earthquake_buffer.append('impact')
            results['is_impact'] = True
            results['reasons'].append('ensemble_vote')
            return True, results
        else:
            logger.info(f"✅ 模型投票: 地震 (P概率={model_result['p_probability']:.2f}, "
                       f"投票率={model_result['vote_ratio']:.2f})")

        # ===== 通过所有检测 =====
        logger.debug(f"✅ 通过所有检测，可能是地震")
        self.earthquake_count += 1
        self.impact_history.append(0)
        self.earthquake_buffer.append('earthquake')
        results['reasons'] = results.get('reasons', []) + ['likely_earthquake']
        return False, results

    def get_stats(self):
        """获取详细统计信息"""
        recent = list(self.earthquake_buffer)[-20:] if self.earthquake_buffer else []
        earthquake_ratio = recent.count('earthquake') / max(1, len(recent))

        return {
            'total_checks': self.total_checks,
            'impact_detected': self.impact_count,
            'earthquake_confirmed': self.earthquake_count,
            'impact_rate': self.impact_count / max(1, self.total_checks),
            'earthquake_rate': self.earthquake_count / max(1, self.total_checks),
            'spectral_rejects': self.spectral_rejects,
            'energy_rejects': self.energy_rejects,
            'onset_rejects': self.onset_rejects,
            'vote_rejects': self.vote_rejects,
            'recent_earthquake_ratio': earthquake_ratio,
            'buffer_size': len(self.earthquake_buffer)
        }


# 测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("测试 PhaseNet 多模型集成投票")
    print("=" * 60)

    # 初始化过滤器
    filter = ImpactAwarePipeline(sampling_rate=100)

    # 生成模拟信号
    t = np.linspace(0, 5, 500)  # 5秒数据

    # 模拟地震信号（低频、逐渐增强）
    earthquake_signal = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti < 1:
            earthquake_signal[i] = 0.01 * ti * np.sin(2 * np.pi * 3 * ti)
        elif ti < 3:
            earthquake_signal[i] = 0.05 * (ti - 0.5) * np.sin(2 * np.pi * 3 * ti)
        else:
            earthquake_signal[i] = 0.1 * np.sin(2 * np.pi * 3 * ti)

    earthquake_waveform = np.array([
        earthquake_signal,
        earthquake_signal * 0.6,
        earthquake_signal * 0.3
    ])

    # 模拟撞击信号（高频、突然起振）
    impact_signal = np.zeros_like(t)
    impact_signal[100:300] = 0.5 * np.exp(-t[100:300] * 5) * np.sin(2 * np.pi * 20 * t[100:300])

    impact_waveform = np.array([
        impact_signal,
        impact_signal * 0.8,
        impact_signal * 0.4
    ])

    # 测试地震信号
    print("\n🔍 测试地震信号:")
    is_impact, info = filter.is_impact(earthquake_waveform)
    print(f"  结果: 撞击={is_impact}")
    print(f"  原因: {info['reasons']}")
    print(f"  频谱比: {info['scores'].get('spectral_ratio', 0):.2f}")
    print(f"  增长速率: {info['scores'].get('growth_rate', 0):.2f}")
    print(f"  起振比例: {info['scores'].get('onset_ratio', 0):.2f}")

    if 'model_result' in info and info['model_result']:
        print(f"  模型投票: P概率={info['model_result']['p_probability']:.2f}")
        print(f"  投票率: {info['model_result']['vote_ratio']:.2f}")

    # 测试撞击信号
    print("\n🔍 测试撞击信号:")
    is_impact, info = filter.is_impact(impact_waveform)
    print(f"  结果: 撞击={is_impact}")
    print(f"  原因: {info['reasons']}")
    print(f"  频谱比: {info['scores'].get('spectral_ratio', 0):.2f}")
    print(f"  增长速率: {info['scores'].get('growth_rate', 0):.2f}")
    print(f"  起振比例: {info['scores'].get('onset_ratio', 0):.2f}")

    if 'model_result' in info and info['model_result']:
        print(f"  模型投票: P概率={info['model_result']['p_probability']:.2f}")
        print(f"  投票率: {info['model_result']['vote_ratio']:.2f}")

    # 打印统计信息
    print("\n📊 统计信息:")
    stats = filter.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")