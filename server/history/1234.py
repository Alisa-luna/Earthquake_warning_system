import json
import math
import logging
import threading
import tkinter as tk
from tkinter import messagebox
import paho.mqtt.client as mqtt
import smtplib
import time
from collections import deque
import numpy as np
import queue
import warnings

from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore', module='daspy')

# DASPy 降噪库
from daspy import Section

if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

import obspy
from obspy import Stream, Trace
import enum
from dataclasses import dataclass
from typing import Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import seisbench.models as sbm
import torch
from impact_filter import ImpactAwarePipeline, ImpactFilter

# 抑制警告
warnings.filterwarnings("ignore", module="seisbench")
warnings.filterwarnings("ignore", module="daspy")

# ==================== 配置信息 ====================
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 8883
MQTT_TOPIC = "483FDA58BA79-publish"
MQTT_USERNAME = "User"
MQTT_PASSWORD = "1234567890"

QQ_EMAIL_ENABLE = True
QQ_EMAIL_SENDER = "3809191404@qq.com"
QQ_EMAIL_PASSWORD = "xcqtffkzkhsgcdcb"
QQ_EMAIL_RECEIVER = "2028024910@qq.com"
QQ_EMAIL_SMTP_SERVER = "smtp.qq.com"
QQ_EMAIL_SMTP_PORT = 587

LOG_FILE = "sensor_trend_monitor.log"

# ==================== 物理单位阈值配置 ====================
ACCELERATION_THRESHOLD_G = 0.05 # 触发阈值
GRAVITY_THRESHOLD_LOW_G = 0.98
GRAVITY_THRESHOLD_HIGH_G = 1.00
ZERO_THRESHOLD = 100
SEISMIC_BUFFER_SIZE = 6000
ACTUAL_SENSITIVITY = 4096.0
TRIGGER_WINDOW = 15.0
MODEL_COOLDOWN_POINTS = 50
CONFIRMATION_COUNT = 3
CONFIRMATION_TIME_WINDOW = 0.5
MODEL_INPUT_LENGTH = 3000

MIN_PHASE_POINTS = 500
MIN_EQT_POINTS = 800
MIN_FINAL_POINTS = 1500

# 弹窗队列
alert_queue = queue.Queue()


# ==================== 预警级别定义 ====================
class AlertLevel(enum.Enum):
    NONE = 0
    YELLOW = 1
    ORANGE = 2
    RED = 3


class AlertState(enum.Enum):
    IDLE = "待机"
    P_ALERT = "P波警觉"
    EQT_CONFIRM = "EQT确认"
    FINAL = "最终确认"


@dataclass
class EventData:
    start_time: float
    last_trigger_time: float
    p_arrival: Optional[float] = None
    eqt_alert_time: Optional[float] = None
    eqt_confirm_time: Optional[float] = None
    final_time: Optional[float] = None
    max_ax: float = 0
    max_ay: float = 0
    max_az: float = 0
    confidence: float = 0
    magnitude: float = 0
    intensity: int = 0
    description: str = ""
    last_phase_analysis_points: int = 0
    last_eqt_analysis_points: int = 0


# ==================== 初始化日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("🚀 三级地震预警系统启动（DASPy降噪版）...")

# 抑制第三方库日志
logging.getLogger('seisbench').setLevel(logging.ERROR)
logging.getLogger('daspy').setLevel(logging.ERROR)


# ==================== DASPy降噪器 ====================
# ==================== DASPy降噪器（完整修复版） ====================
class DASPyDenoiser:
    def __init__(self, sampling_rate=100, method='wavelet'):
        """
        method: 'wavelet' (推荐, 不需要daspy), 'bandpass', 'curvelet' (需要daspy)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.method = method
        logger.info(f"✅ DASPy降噪器初始化完成，方法: {method}")

        # 检查daspy是否可用（仅curvelet方法需要）
        if method == 'curvelet':
            try:
                from daspy import Section
                self.daspy_available = True
                logger.info("   curvelet方法可用")
            except ImportError:
                self.daspy_available = False
                logger.warning("   ⚠️ daspy未安装，curvelet方法不可用，将回退到wavelet")
                self.method = 'wavelet'

    def __call__(self, data_3c):
        """使对象可以像函数一样被调用"""
        return self.denoise(data_3c)

    def _bandpass_filter(self, data, lowcut=1.0, highcut=20.0):
        """带通滤波器，沿时间轴滤波"""
        nyq = 0.5 * self.sampling_rate
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(4, [low, high], btype='band')
        # 对每个通道独立滤波
        filtered = np.zeros_like(data)
        for i in range(data.shape[0]):
            filtered[i] = filtfilt(b, a, data[i])
        return filtered

    def _wavelet_denoise_channel(self, channel_data):
        """对单个通道进行小波去噪（需安装 PyWavelets）"""
        try:
            import pywt
            # 使用软阈值小波去噪，自动估计阈值
            coeffs = pywt.wavedec(channel_data, 'db4', level=4)
            # 估计噪声标准差（使用最细尺度的小波系数）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            # 通用阈值
            threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))
            # 软阈值处理细节系数
            coeffs_thresh = [coeffs[0]]  # 保留近似系数
            for i in range(1, len(coeffs)):
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            # 重构信号
            denoised = pywt.waverec(coeffs_thresh, 'db4')
            # 确保长度一致
            if len(denoised) > len(channel_data):
                denoised = denoised[:len(channel_data)]
            elif len(denoised) < len(channel_data):
                denoised = np.pad(denoised, (0, len(channel_data) - len(denoised)), 'edge')
            return denoised
        except ImportError:
            logger.warning("⚠️ 未安装PyWavelets，回退到带通滤波")
            return channel_data
        except Exception as e:
            logger.warning(f"⚠️ 小波去噪失败: {e}，回退到原始数据")
            return channel_data

    def _wavelet_denoise(self, data_3c):
        """对三分量数据进行小波去噪"""
        denoised = np.zeros_like(data_3c)
        for ch in range(3):
            denoised[ch] = self._wavelet_denoise_channel(data_3c[ch])
        return denoised

    def _curvelet_denoise(self, data_3c):
        """曲线波去噪（使用daspy）"""
        try:
            from daspy import Section

            # 保存原始形状
            original_shape = data_3c.shape
            logger.debug(f"曲线波去噪前形状: {original_shape}")

            # DASPy的Section期望 (time, channel) 格式
            data_t = data_3c.T  # (N, 3)

            # 创建Section对象
            sec = Section(data=data_t, dt=self.dt, dx=1.0)

            # 应用曲线波去噪
            sec_denoised = sec.curvelet_denoising()

            # 获取降噪后的数据
            denoised_t = sec_denoised.data.copy()  # (N', 3)
            logger.debug(f"曲线波去噪后形状: {denoised_t.shape}")

            # 转回 (3, N) 格式
            denoised = denoised_t.T

            # ===== 关键修复：确保输出形状和输入一致 =====
            if denoised.shape != original_shape:
                logger.warning(f"形状不匹配: 输入 {original_shape}，输出 {denoised.shape}，进行调整")

                # 如果时间维度不同
                if denoised.shape[1] != original_shape[1]:
                    # 如果输出比输入长，裁剪
                    if denoised.shape[1] > original_shape[1]:
                        denoised = denoised[:, :original_shape[1]]
                        logger.debug(f"裁剪后形状: {denoised.shape}")

                    # 如果输出比输入短，用边缘值填充
                    elif denoised.shape[1] < original_shape[1]:
                        pad_width = original_shape[1] - denoised.shape[1]
                        # 用最后一个值填充
                        last_values = denoised[:, -1:]
                        padding = np.repeat(last_values, pad_width, axis=1)
                        denoised = np.hstack([denoised, padding])
                        logger.debug(f"填充后形状: {denoised.shape}")

            return denoised

        except Exception as e:
            logger.error(f"曲线波去噪失败: {e}，回退到小波去噪")
            return self._wavelet_denoise(data_3c)

    def denoise(self, data_3c):
        """
        输入: data_3c shape (3, N)
        返回: 降噪后的数据，同形状
        """
        # 输入验证
        if data_3c.ndim != 2 or data_3c.shape[0] != 3:
            logger.error(f"需要 [3, N] 格式，当前: {data_3c.shape}，返回原始数据")
            return data_3c

        # 确保数据足够长
        if data_3c.shape[1] < 50:
            logger.warning(f"数据太短 ({data_3c.shape[1]}点)，返回原始数据")
            return data_3c

        # 记录原始形状
        original_shape = data_3c.shape
        logger.debug(f"降噪开始，形状: {original_shape}，方法: {self.method}")

        # 方法分发
        try:
            if self.method == 'bandpass':
                result = self._bandpass_filter(data_3c)
            elif self.method == 'wavelet':
                result = self._wavelet_denoise(data_3c)
            elif self.method == 'curvelet':
                result = self._curvelet_denoise(data_3c)
            else:
                logger.warning(f"未知方法 {self.method}，使用带通滤波")
                result = self._bandpass_filter(data_3c)

            # 最终安全检查：确保形状一致
            if result.shape != original_shape:
                logger.error(f"降噪后形状 {result.shape} 与原始形状 {original_shape} 不一致，返回原始数据")
                return data_3c

            logger.debug(f"降噪完成，形状: {result.shape}")
            return result

        except Exception as e:
            logger.error(f"降噪失败: {e}，返回原始数据")
            return data_3c

    def denoise_batch(self, data_batch):
        """
        批量降噪

        Args:
            data_batch: list of (3, N) arrays

        Returns:
            list of 降噪后的数据
        """
        results = []
        for i, data in enumerate(data_batch):
            logger.debug(f"批量降噪第 {i + 1}/{len(data_batch)} 个")
            results.append(self.denoise(data))
        return results

    def denoise_safe(self, data_3c):
        """
        安全的降噪方法，确保返回的数据和输入形状一致
        如果降噪失败或形状不匹配，返回原始数据
        """
        try:
            result = self.denoise(data_3c)
            if result.shape == data_3c.shape:
                return result
            else:
                logger.error(f"降噪后形状 {result.shape} != 输入形状 {data_3c.shape}")
                return data_3c
        except Exception as e:
            logger.error(f"降噪失败: {e}")
            return data_3c

# ==================== 邮件通知 ====================
def send_email_alert(subject, content):
    if not QQ_EMAIL_ENABLE:
        return

    def _send():
        try:
            msg = MIMEMultipart()
            msg['From'] = QQ_EMAIL_SENDER
            msg['To'] = QQ_EMAIL_RECEIVER
            msg['Subject'] = Header(subject, 'utf-8')
            msg.attach(MIMEText(content, 'plain', 'utf-8'))
            server = smtplib.SMTP(QQ_EMAIL_SMTP_SERVER, QQ_EMAIL_SMTP_PORT)
            server.starttls()
            server.login(QQ_EMAIL_SENDER, QQ_EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"📧 邮件已发送: {subject}")
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")

    threading.Thread(target=_send, daemon=True).start()


# ==================== 弹窗 ====================
def show_alert(title, message, confidence, source):
    alert_queue.put((title, message, confidence, source))


# ==================== 本地规则判断 ====================
trigger_count = 0
last_trigger_time = 0


def check_for_seismic_trigger(sensor_data):
    global trigger_count, last_trigger_time
    try:
        ax_g = abs(sensor_data.get('AX', 0)) / ACTUAL_SENSITIVITY
        ay_g = abs(sensor_data.get('AY', 0)) / ACTUAL_SENSITIVITY
        az_g = abs(sensor_data.get('AZ', 0)) / ACTUAL_SENSITIVITY

        horizontal = math.sqrt(ax_g ** 2 + ay_g ** 2)
        vertical = abs(az_g - 1.0)

        current_time = time.time()

        if horizontal > ACCELERATION_THRESHOLD_G or vertical > 0.05:
            if current_time - last_trigger_time < CONFIRMATION_TIME_WINDOW:
                trigger_count += 1
            else:
                trigger_count = 1

            last_trigger_time = current_time

            if trigger_count >= CONFIRMATION_COUNT:
                reason = f"水平={horizontal:.3f}g, 垂直={vertical:.3f}g"
                trigger_count = 0
                return True, reason
            return False, f"确认中 ({trigger_count}/{CONFIRMATION_COUNT})"

        return False, "正常"
    except Exception as e:
        return False, f"检测错误: {e}"


def precise_local_check(sensor_data):
    try:
        required = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        missing = [f for f in required if f not in sensor_data]
        if missing:
            return "异常", f"缺失字段: {missing}"
        ax = int(sensor_data['AX'])
        ay = int(sensor_data['AY'])
        az = int(sensor_data['AZ'])
        if abs(ax) < ZERO_THRESHOLD and abs(ay) < ZERO_THRESHOLD and abs(az) < ZERO_THRESHOLD:
            return "异常", "所有读数接近零"
        return "正常", "自检通过"
    except Exception as e:
        return "异常", f"自检错误: {e}"


def quick_magnitude(ax_max, ay_max):
    max_g = max(ax_max, ay_max)
    if max_g < 0.01:
        return 0
    return round(math.log10(max_g * 1000) * 2, 1)


def calculate_intensity(ax_g, ay_g):
    max_g = max(ax_g, ay_g)
    max_cm = max_g * 980
    if max_cm < 0.0022:
        return 1, "无感"
    elif max_cm < 0.0063:
        return 2, "微感"
    elif max_cm < 0.018:
        return 3, "有感"
    elif max_cm < 0.045:
        return 4, "室内有感"
    elif max_cm < 0.089:
        return 5, "室外有感"
    elif max_cm < 0.177:
        return 6, "惊慌"
    elif max_cm < 0.353:
        return 7, "房屋损坏"
    elif max_cm < 0.707:
        return 8, "建筑物破坏"
    elif max_cm < 1.414:
        return 9, "建筑物倒塌"
    elif max_cm < 2.5:
        return 10, "毁灭性"
    else:
        return 11, "灾难性"


# ==================== PhaseNet模型 ====================
class PhaseNetReal:
    def __init__(self):
        logger.info("正在初始化PhaseNet模型...")
        try:
            self.model = sbm.PhaseNet.from_pretrained("geofon")
            self.model.eval()
            self.threshold = 50
            self.in_samples = 3000
            logger.info("✅ PhaseNet模型加载成功")
        except Exception as e:
            logger.error(f"❌ PhaseNet模型加载失败: {e}")
            self.model = None

    def analyze(self, waveform_data):
        if self.model is None:
            return 0
        if len(waveform_data) < 300:
            return 0
        try:
            scale = 9.8 / ACTUAL_SENSITIVITY
            z_vals, n_vals, e_vals = [], [], []
            for d in waveform_data:
                if not isinstance(d, dict):
                    continue
                z_vals.append(float(d.get('AZ', 0)) * scale)
                n_vals.append(float(d.get('AY', 0)) * scale)
                e_vals.append(float(d.get('AX', 0)) * scale)

            if len(z_vals) < 300:
                return 0

            z = np.array(z_vals, dtype=np.float32)
            n = np.array(n_vals, dtype=np.float32)
            e = np.array(e_vals, dtype=np.float32)

            z -= np.mean(z)
            n -= np.mean(n)
            e -= np.mean(e)

            target_length = self.in_samples
            if len(z) > target_length:
                z = z[-target_length:]
                n = n[-target_length:]
                e = e[-target_length:]
            elif len(z) < target_length:
                pad_width = target_length - len(z)
                z = np.pad(z, (0, pad_width), 'edge')
                n = np.pad(n, (0, pad_width), 'edge')
                e = np.pad(e, (0, pad_width), 'edge')

            z = z / (np.std(z) + 1e-10)
            n = n / (np.std(n) + 1e-10)
            e = e / (np.std(e) + 1e-10)

            waveform = np.stack([z, n, e], axis=0)[np.newaxis, :, :]

            device = next(self.model.parameters()).device
            waveform_tensor = torch.from_numpy(waveform).float().to(device)

            with torch.no_grad():
                predictions = self.model(waveform_tensor)

            if predictions is not None:
                if isinstance(predictions, torch.Tensor):
                    pred_np = predictions.cpu().numpy()
                else:
                    pred_np = predictions
                if pred_np.ndim >= 3:
                    p_prob = float(np.max(pred_np[0, 0, :])) * 100
                    logger.info(f"PhaseNet P波概率: {p_prob:.1f}%")
                    return p_prob if p_prob > self.threshold else 0
            return 0
        except Exception as e:
            logger.error(f"PhaseNet分析失败: {e}", exc_info=True)
            return 0


# ==================== EQTransformer模型 ====================
class EQTReal:
    def __init__(self):
        logger.info("正在初始化EQTransformer模型...")
        try:
            self.model = sbm.EQTransformer.from_pretrained("geofon")
            self.model.eval()
            logger.info("✅ EQTransformer模型加载成功")
            self.alert_threshold = 80
            self.confirm_threshold = 85
            self.in_samples = 6000
            logger.info(f"📐 EQT模型期望: in_samples={self.in_samples}")
        except Exception as e:
            logger.error(f"❌ EQTransformer模型加载失败: {e}")
            self.model = None

    def analyze(self, waveform_data):
        if self.model is None:
            return {'confidence': 0, 'magnitude': 0}
        if not waveform_data or len(waveform_data) < 500:
            return {'confidence': 0, 'magnitude': 0}
        try:
            scale = 9.8 / ACTUAL_SENSITIVITY
            z_vals, n_vals, e_vals = [], [], []
            for d in waveform_data:
                if not isinstance(d, dict):
                    continue
                z_vals.append(float(d.get('AZ', 0)) * scale)
                n_vals.append(float(d.get('AY', 0)) * scale)
                e_vals.append(float(d.get('AX', 0)) * scale)

            if len(z_vals) < 300:
                return {'confidence': 0, 'magnitude': 0}

            z = np.array(z_vals, dtype=np.float32)
            n = np.array(n_vals, dtype=np.float32)
            e = np.array(e_vals, dtype=np.float32)

            z -= np.mean(z)
            n -= np.mean(n)
            e -= np.mean(e)

            target_length = self.in_samples
            if len(z) > target_length:
                z = z[-target_length:]
                n = n[-target_length:]
                e = e[-target_length:]
            elif len(z) < target_length:
                pad_width = target_length - len(z)
                z = np.pad(z, (0, pad_width), 'edge')
                n = np.pad(n, (0, pad_width), 'edge')
                e = np.pad(e, (0, pad_width), 'edge')

            z = z / (np.std(z) + 1e-10)
            n = n / (np.std(n) + 1e-10)
            e = e / (np.std(e) + 1e-10)

            waveform = np.stack([z, n, e], axis=0)[np.newaxis, :, :]

            device = next(self.model.parameters()).device
            waveform_tensor = torch.from_numpy(waveform).float().to(device)

            with torch.no_grad():
                predictions = self.model(waveform_tensor)

            if isinstance(predictions, tuple):
                pred = predictions[0]
            else:
                pred = predictions

            if isinstance(pred, torch.Tensor):
                pred_np = pred.cpu().numpy()
            else:
                pred_np = pred

            if pred_np.ndim >= 2:
                if pred_np.ndim == 3:
                    detection_prob = float(np.max(pred_np[0, 0, :]))
                else:
                    detection_prob = float(np.max(pred_np))
                confidence = detection_prob * 100
                logger.info(f"EQT检测: 置信度={confidence:.1f}%")

                if confidence > 0:
                    recent = waveform_data[-100:] if len(waveform_data) >= 100 else waveform_data
                    max_ax = max(abs(d.get('AX', 0)) for d in recent) / ACTUAL_SENSITIVITY
                    max_ay = max(abs(d.get('AY', 0)) for d in recent) / ACTUAL_SENSITIVITY
                    magnitude = quick_magnitude(max_ax, max_ay)
                    return {'confidence': confidence, 'magnitude': magnitude}

            return {'confidence': 0, 'magnitude': 0}
        except Exception as e:
            logger.error(f"❌ EQT分析失败: {str(e)}", exc_info=True)
            return {'confidence': 0, 'magnitude': 0}


# ==================== 最终确认 ====================
class FinalValidator:
    def __init__(self, phase_model=None, eqt_model=None):
        logger.info("正在初始化最终确认模型...")
        self.models = {}
        if phase_model:
            self.models['phasenet'] = phase_model
            logger.info("✅ PhaseNet已共享")
        else:
            try:
                self.models['phasenet'] = PhaseNetReal()
                logger.info("✅ PhaseNet加载成功")
            except Exception as e:
                logger.error(f"❌ PhaseNet加载失败: {e}")
        if eqt_model:
            self.models['eqt'] = eqt_model
            logger.info("✅ EQTransformer已共享")
        else:
            try:
                self.models['eqt'] = EQTReal()
                logger.info("✅ EQTransformer加载成功")
            except Exception as e:
                logger.error(f"❌ EQTransformer加载失败: {e}")
        self.threshold = 0.7
        logger.info(f"最终确认模型加载完成: {len(self.models)}个")

    def analyze(self, data_points):
        if not self.models:
            return {'is_earthquake': False, 'confidence': 0, 'magnitude': 0}
        if len(data_points) < 1000:
            return {'is_earthquake': False, 'confidence': 0, 'magnitude': 0}
        votes, confidences = [], []
        if 'phasenet' in self.models:
            p_prob = self.models['phasenet'].analyze(data_points)
            votes.append(1 if p_prob >= 70 else 0)
            confidences.append(p_prob / 100.0)
        if 'eqt' in self.models:
            eqt_res = self.models['eqt'].analyze(data_points)
            confidence = eqt_res['confidence'] / 100.0
            votes.append(1 if confidence > self.threshold else 0)
            confidences.append(confidence)
        if not votes:
            return {'is_earthquake': False, 'confidence': 0, 'magnitude': 0}
        total = len(votes)
        yes = sum(votes)
        is_eq = yes >= (total /2)
        confidence = (yes / total) * 100 * (np.mean(confidences) if confidences else 0)
        max_ax = max(abs(d.get('AX', 0)) for d in data_points) / ACTUAL_SENSITIVITY
        max_ay = max(abs(d.get('AY', 0)) for d in data_points) / ACTUAL_SENSITIVITY
        magnitude = quick_magnitude(max_ax, max_ay)
        logger.info(f"最终确认: 地震={is_eq}, 置信度={confidence:.1f}%, 震级={magnitude}, 投票={yes}/{total}")
        return {'is_earthquake': is_eq, 'confidence': round(confidence, 1), 'magnitude': magnitude,
                'details': f"投票: {yes}/{total}"}


# ==================== 状态机 ====================
class EarthquakeStateMachine:
    def __init__(self):
        self.lock = threading.Lock()
        self.state = AlertState.IDLE
        self.current_event: Optional[EventData] = None
        self.alerts_sent = {level: False for level in AlertLevel}
        self.data_buffer = deque(maxlen=SEISMIC_BUFFER_SIZE)
        self.total_points_processed = 0
        self.total_triggers = 0

        # ===== 新增：撞击检测相关 =====
        self.last_impact_check = time.time()
        self.impact_check_interval = 1.0  # 每5秒检测一次
        # =============================

        logger.info("正在初始化降噪器...")
        try:
            # 使用 wavelet 方法，不需要 daspy
            self.denoiser = DASPyDenoiser(sampling_rate=100, method='wavelet')
            # 测试降噪器是否可用
            test_data = np.random.randn(3, 100).astype(np.float32)
            test_result = self.denoiser(test_data)
            logger.info(f"✅ 降噪器测试通过，输出形状: {test_result.shape}")
        except Exception as e:
            logger.error(f"❌ 降噪器初始化失败: {e}")
            self.denoiser = None

        logger.info("正在初始化所有模型...")
        self.phase = PhaseNetReal()
        self.eqt = EQTReal()
        self.final = FinalValidator(self.phase, self.eqt)

        # ===== 新增：初始化撞击感知流水线 =====
        logger.info("正在初始化撞击过滤器...")
        self.pipeline = ImpactAwarePipeline(
            denoiser=self.denoiser,
            phase_model=self.phase,
            eqt_model=self.eqt,
            sampling_rate=100
        )

        # ===== 新增：投票历史记录 =====
        self.vote_history = deque(maxlen=10)

        logger.info("✅ 撞击感知流水线初始化完成")
        # ===================================

        logger.info("✅ 所有模型初始化完成")

    def on_data(self, sensor_data):
        with self.lock:
            current_time = time.time()
            self.total_points_processed += 1

            # 存入缓存
            self.data_buffer.append({
                'time': current_time,
                'data': sensor_data.copy()
            })

            # 计算当前振幅
            ax_g = sensor_data.get('AX', 0) / ACTUAL_SENSITIVITY
            ay_g = sensor_data.get('AY', 0) / ACTUAL_SENSITIVITY
            az_g = sensor_data.get('AZ', 0) / ACTUAL_SENSITIVITY
            current_amp = max(ax_g, ay_g, az_g)

            # ===== 极度宽松的撞击检测 =====
            # 只在满足所有条件时才检测：
            # 1. 已经有事件在进行
            # 2. 事件点数超过200
            # 3. 距离上次检测超过5秒
            # 4. 当前振幅 > 0.5g（只有大信号才可能是撞击）

            # 在 on_data 方法中，找到撞击检测部分，修改为：

            if (self.current_event and
                    self._count_event_points() > 200 and
                    current_time - self.last_impact_check > 1.0 and
                    current_amp > 0.1):

                logger.info(f"🔍 执行撞击检测 (当前振幅={current_amp:.3f}g)")
                waveform = self._get_recent_waveform(seconds=3)
                if waveform is not None:
                    try:
                        # 通过流水线处理
                        result = self.pipeline.is_impact(waveform, event_data=self._get_event_data())

                        # ===== 更安全的访问方式 =====
                        # 方法1：直接打印整个 result 看看有什么字段
                        logger.debug(f"撞击检测结果: {result}")

                        # 方法2：使用 get 方法安全访问
                        #status = result.get('status', 'unknown')
                        is_impact, impact_info = self.pipeline.is_impact(waveform,event_data=self._get_event_data())
                        status = impact_info.get('status', 'unknown') if isinstance(impact_info, dict) else 'unknown'

                        result = {
                            'status': status,
                            'is_impact': is_impact,
                            'reasons': impact_info.get('reasons', []) if isinstance(impact_info, dict) else [],
                            'scores': impact_info.get('scores', {}) if isinstance(impact_info, dict) else {}
                        }

                        if status == 'impact_filtered' or is_impact:
                            # 获取原因（多种可能）
                            reason = result.get('reason', 'unknown')
                            if reason == 'unknown' and 'reasons' in result:
                                reasons = result['reasons']
                                reason = reasons[0] if reasons else 'unknown'
                            if reason == 'unknown' and 'impact_info' in result:
                                impact_info = result['impact_info']
                                if isinstance(impact_info, dict):
                                    reasons = impact_info.get('reasons', [])
                                    reason = reasons[0] if reasons else 'unknown'

                            logger.info(f"🚫 撞击被过滤: {reason}")
                            # 重置事件
                            self._reset()
                            self.last_impact_check = current_time
                            return

                    except Exception as e:
                        logger.error(f"撞击检测出错: {e}")
                        import traceback
                        traceback.print_exc()  # 打印完整错误堆栈

                self.last_impact_check = current_time
            # ==================================

            # 更新事件最大值（如果有）
            if self.current_event:
                self.current_event.max_ax = max(self.current_event.max_ax, abs(sensor_data.get('AX', 0)))
                self.current_event.max_ay = max(self.current_event.max_ay, abs(sensor_data.get('AY', 0)))
                self.current_event.max_az = max(self.current_event.max_az, abs(sensor_data.get('AZ', 0)))

            # 触发检测
            is_triggered, reason = check_for_seismic_trigger(sensor_data)

            # 打印每个点的加速度值（调试用）
            logger.debug(f"数据点: AX={ax_g:.3f}g, AY={ay_g:.3f}g, AZ={az_g:.3f}g, 触发={is_triggered} ({reason})")

            if is_triggered:
                self.total_triggers += 1
                logger.info(f"⚡ 触发! {reason}")

            if is_triggered:
                self._handle_trigger(current_time)

            # 状态处理
            if self.state == AlertState.IDLE:
                self._handle_idle()
            elif self.state == AlertState.P_ALERT:
                self._handle_p_alert()
            elif self.state == AlertState.EQT_CONFIRM:
                self._handle_eqt_confirm()

    def _send_alert(self, level, message, confidence, source):
        """发送预警（弹窗+邮件）"""
        if self.alerts_sent[level]:
            return

        self.alerts_sent[level] = True
        titles = {
            AlertLevel.YELLOW: "🟡 P波预警",
            AlertLevel.ORANGE: "🟠 地震预警",
            AlertLevel.RED: "🔴 地震最终确认"
        }

        # 弹窗
        show_alert(titles[level], message, confidence, source)

        # 橙色及以上发邮件
        if level.value >= AlertLevel.ORANGE.value:
            threading.Thread(
                target=send_email_alert,
                args=(titles[level], message)
            ).start()

    def _handle_trigger(self, current_time):
        if not self.current_event:
            self.current_event = EventData(start_time=current_time, last_trigger_time=current_time,
                                           p_arrival=current_time)
            logger.info(f"🌟 新事件开始 at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        else:
            if current_time - self.current_event.last_trigger_time < TRIGGER_WINDOW:
                self.current_event.last_trigger_time = current_time
                logger.info(f"事件延续 at {current_time:.3f}")
            else:
                logger.info(f"事件超时（间隔>{TRIGGER_WINDOW}s），开始新事件")
                self._reset()
                self.current_event = EventData(start_time=current_time, last_trigger_time=current_time,
                                               p_arrival=current_time)

    def _sensor_to_waveform(self, sensor_data):
        """
        把单点传感器数据转换成波形格式
        用于撞击检测
        """
        scale = 9.8 / ACTUAL_SENSITIVITY
        return np.array([
            [float(sensor_data.get('AZ', 0)) * scale],  # Z通道
            [float(sensor_data.get('AY', 0)) * scale],  # N通道
            [float(sensor_data.get('AX', 0)) * scale]  # E通道
        ], dtype=np.float32)

    def _get_recent_waveform(self, seconds=3):
        """
        从buffer中获取最近的一段波形

        Args:
            seconds: 获取最近多少秒的数据

        Returns:
            numpy array shape (3, N) 或 None
        """
        if len(self.data_buffer) < 100:  # 至少1秒数据
            return None

        # 取最近 N 秒的数据
        n_points = int(seconds * 100)  # 100Hz
        recent = list(self.data_buffer)[-n_points:]

        if not recent:
            return None

        scale = 9.8 / ACTUAL_SENSITIVITY
        z_vals, n_vals, e_vals = [], [], []

        for item in recent:
            d = item['data']
            z_vals.append(float(d.get('AZ', 0)) * scale)
            n_vals.append(float(d.get('AY', 0)) * scale)
            e_vals.append(float(d.get('AX', 0)) * scale)

        return np.array([z_vals, n_vals, e_vals], dtype=np.float32)

    def _get_event_data(self):
        if not self.current_event:
            return []
        start = self.current_event.start_time
        end = self.current_event.last_trigger_time
        if end - start > 60:
            start = end - 60
        return [item['data'] for item in self.data_buffer if start <= item['time'] <= end]

    def _count_event_points(self):
        if not self.current_event:
            return 0
        start = self.current_event.start_time
        end = self.current_event.last_trigger_time
        if end - start > 60:
            start = end - 60
        return sum(1 for item in self.data_buffer if start <= item['time'] <= end)

    def _preprocess_with_denoise(self, event_data):
        """将原始事件数据转换为三通道波形并降噪，失败时返回原始波形"""
        scale = 9.8 / ACTUAL_SENSITIVITY
        z_vals, n_vals, e_vals = [], [], []
        for d in event_data:
            z_vals.append(float(d.get('AZ', 0)) * scale)
            n_vals.append(float(d.get('AY', 0)) * scale)
            e_vals.append(float(d.get('AX', 0)) * scale)

        # 构建三通道数据 [3, time]
        waveform = np.array([z_vals, n_vals, e_vals], dtype=np.float32)

        # 去均值
        waveform -= np.mean(waveform, axis=1, keepdims=True)

        # DASPy降噪（使用安全的降噪方法）
        logger.info("🎯 正在应用DASPy智能降噪...")
        try:
            # 使用 denoise_safe 方法确保形状一致
            if hasattr(self.denoiser, 'denoise_safe'):
                denoised = self.denoiser.denoise_safe(waveform)
            else:
                denoised = self.denoiser.denoise(waveform)

            # 最终安全检查
            if denoised.shape != waveform.shape:
                logger.error(f"降噪后形状 {denoised.shape} 与原始 {waveform.shape} 不匹配，使用原始数据")
                return waveform

            logger.info("✅ 降噪成功")
            return denoised
        except Exception as e:
            logger.error(f"降噪失败: {e}，返回原始数据")
            return waveform

    def _three_way_vote(self, waveform, event_data=None):
        """三方投票机制（新增辅助函数）"""

        # 3. 撞击过滤器得分
        is_impact, impact_info = self.pipeline.is_impact(waveform, event_data)

        # 1. PhaseNet 得分
        phase_score = self.phase.analyze(event_data) if event_data else 0

        # 2. EQTransformer 得分
        eqt_result = self.eqt.analyze(event_data) if event_data else {'confidence': 0}
        eqt_score = eqt_result['confidence']



        # 从撞击信息中提取噪声得分
        if 'model_result' in impact_info and impact_info['model_result']:
            impact_noise_score = 1 - impact_info['model_result'].get('confidence', 0.5)
        else:
            scores = impact_info.get('scores', {})
            spectral = scores.get('spectral_ratio', 1.0)
            impact_noise_score = max(0, min(1, 1 - spectral / 2))

        # 归一化
        phase_norm = phase_score / 100
        eqt_norm = eqt_score / 100

        # 规则1：撞击过滤器一票否决
        if impact_noise_score > 0.8:
            return 'reject', 0, {'reason': '撞击过滤器一票否决'}

        # 规则2：三方投票
        votes = 0
        if phase_norm > 0.45: votes += 1
        if eqt_norm > 0.75: votes += 1
        if impact_noise_score < 0.4: votes += 1

        # 地震得分
        earthquake_score = (phase_norm * 0.3 + eqt_norm * 0.3 + (1 - impact_noise_score) * 0.4)

        # 记录投票
        self.vote_history.append({
            'time': time.time(),
            'votes': votes,
            'score': earthquake_score
        })

        # 决策
        if votes >= 2 and earthquake_score > 0.6:
            return 'earthquake', earthquake_score, {}
        elif votes >= 1 and earthquake_score > 0.3:
            return 'suspicious', earthquake_score, {}
        else:
            return 'noise', earthquake_score, {}

    def _handle_idle(self):
        if not self.current_event:
            return

        event_points = self._count_event_points()
        if event_points < MIN_PHASE_POINTS:
            return

        waveform = self._get_recent_waveform(seconds=3)
        event_data = self._get_event_data()

        if waveform is None or len(event_data) < MIN_PHASE_POINTS:
            return

        # 降噪（复用原有逻辑）
        denoised_waveform = self._preprocess_with_denoise(event_data)

        # 三方投票
        decision, score, _ = self._three_way_vote(denoised_waveform, event_data)

        if decision == 'earthquake':
            # 直接进EQT确认
            if not self.alerts_sent[AlertLevel.ORANGE]:
                self._send_alert(AlertLevel.ORANGE, f"三方投票确认地震", score * 100, "三方投票")
            self.state = AlertState.EQT_CONFIRM
            logger.info(f"🟠 IDLE → EQT_CONFIRM")

        elif decision == 'suspicious':
            # 进P_ALERT
            if not self.alerts_sent[AlertLevel.YELLOW]:
                self._send_alert(AlertLevel.YELLOW, f"检测到可疑信号", score * 100, "三方投票")
            self.state = AlertState.P_ALERT
            logger.info(f"🟡 IDLE → P_ALERT")

        elif decision == 'reject':
            logger.info(f"🚫 事件被否决")
            self._reset()

        else:  # noise
            logger.info(f"🌫️ 判定为噪声")
            self._reset()

    def _handle_p_alert(self):
        if not self.current_event:
            self._reset()
            return

        if time.time() - self.current_event.start_time > 60:
            logger.info("P_ALERT超时")
            self._reset()
            return

        event_points = self._count_event_points()
        if event_points < MIN_EQT_POINTS:
            return

        waveform = self._get_recent_waveform(seconds=5)
        event_data = self._get_event_data()

        if waveform is None or len(event_data) < MIN_EQT_POINTS:
            return

        denoised_waveform = self._preprocess_with_denoise(event_data)

        # 再次投票
        decision, score, _ = self._three_way_vote(denoised_waveform, event_data)

        if decision == 'earthquake':
            if not self.alerts_sent[AlertLevel.ORANGE]:
                self._send_alert(AlertLevel.ORANGE, f"二次投票确认地震", score * 100, "三方投票")
            self.state = AlertState.EQT_CONFIRM
            logger.info(f"🟠 P_ALERT → EQT_CONFIRM")

        elif decision == 'reject' or decision == 'noise':
            logger.info(f"🚫 事件被否决")
            self._reset()
    def _handle_eqt_confirm(self):
        if not self.current_event or self.current_event.final_time is not None:
            return

        event_points = self._count_event_points()
        if event_points < MIN_FINAL_POINTS:
            return

        event_data = self._get_event_data()
        if len(event_data) < MIN_FINAL_POINTS:
            return

        denoised_waveform = self._preprocess_with_denoise(event_data)

        # 先用FinalValidator分析
        scale = ACTUAL_SENSITIVITY / 9.8
        denoised_event_data = []
        for i in range(len(event_data)):
            denoised_event_data.append({
                'AX': denoised_waveform[2, i] * scale,
                'AY': denoised_waveform[1, i] * scale,
                'AZ': denoised_waveform[0, i] * scale,
            })

        result = self.final.analyze(denoised_event_data)

        if result and result['is_earthquake']:
            # 原有逻辑不变
            self.current_event.final_time = time.time()
            self.current_event.magnitude = result['magnitude']
            self.current_event.confidence = result['confidence']
            intensity, desc = calculate_intensity(
                self.current_event.max_ax / ACTUAL_SENSITIVITY,
                self.current_event.max_ay / ACTUAL_SENSITIVITY
            )
            self.current_event.intensity = intensity
            self.current_event.description = desc

            if not self.alerts_sent[AlertLevel.RED]:
                self._send_alert(AlertLevel.RED, f"最终确认：烈度{intensity}度", result['confidence'], "SeisBench")
                self._send_final_email()

            self.state = AlertState.FINAL
            logger.info(f"🔴 EQT_CONFIRM → FINAL")
        else:
            # 新增：最终分析否定时，用三方投票复核
            decision, score, _ = self._three_way_vote(denoised_waveform, event_data)

            if decision == 'earthquake':
                logger.info(f"⚠️ 最终分析否定，但三方投票仍确认地震，信任投票")
                # 可以发一个低级别警告
            else:
                logger.warning("最终分析否定地震")
                self._reset()

    def _send_final_email(self):
        if not self.current_event:
            return
        duration = self.current_event.final_time - self.current_event.start_time
        content = f"""
地震最终确认报告
{'=' * 50}

📅 时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.current_event.start_time))}
📊 震级：M{self.current_event.magnitude:.1f}
📈 烈度：{self.current_event.intensity}度（{self.current_event.description}）
📌 置信度：{self.current_event.confidence:.1f}%

⏱️ 时间线：
├─ P波到达：0.0秒
├─ EQT预警：{self.current_event.eqt_alert_time - self.current_event.start_time:.1f}秒
├─ EQT确认：{self.current_event.eqt_confirm_time - self.current_event.start_time:.1f}秒
└─ 最终确认：{duration:.1f}秒

📊 峰值加速度：
├─ X轴：{self.current_event.max_ax / ACTUAL_SENSITIVITY:.3f}g
├─ Y轴：{self.current_event.max_ay / ACTUAL_SENSITIVITY:.3f}g
└─ Z轴：{self.current_event.max_az / ACTUAL_SENSITIVITY:.3f}g

📈 统计信息：
├─ 处理数据点：{self.total_points_processed}
├─ 总触发次数：{self.total_triggers}
└─ 事件持续时间：{duration:.1f}秒
"""
        send_email_alert("🔴 地震最终确认报告", content)

    def _reset(self):
        self.state = AlertState.IDLE
        self.current_event = None
        self.alerts_sent = {level: False for level in AlertLevel}
        logger.info("🔄 状态机重置")


# ==================== MQTT ====================
state_machine = None


def on_message(client, userdata, msg):
    global state_machine
    try:
        payload_str = msg.payload.decode('utf-8')
        sensor_data = json.loads(payload_str)

        if 'status' in sensor_data:
            logger.info(f"💓 心跳包: {sensor_data}")
            return

        required = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        if all(field in sensor_data for field in required):
            status, reason = precise_local_check(sensor_data)
            if status == "异常":
                logger.warning(f"⚠️ 硬件异常: {reason}")
            if state_machine:
                state_machine.on_data(sensor_data)
            return

        lower_required = [f.lower() for f in required]
        if all(field in sensor_data for field in lower_required):
            converted = {k.upper(): sensor_data[k] for k in lower_required}
            logger.debug("转换小写字段为大写")
            if state_machine:
                state_machine.on_data(converted)
            return

        logger.warning(f"⚠️ 未知数据格式: {list(sensor_data.keys())}")
    except Exception as e:
        logger.error(f"处理消息错误: {e}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("✅ 成功连接到MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
    else:
        logger.error(f"❌ 连接失败: {rc}")


# ==================== 主程序 ====================
def main():
    global state_machine

    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid

    state_machine = EarthquakeStateMachine()

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    import ssl
    try:
        client.tls_set(ca_certs=r"D:\Mosquitto\ca.crt")
        client.tls_insecure_set(True)
        logger.info("✅ SSL证书配置成功")
    except Exception as e:
        logger.error(f"SSL配置失败: {e}")
        logger.warning("⚠️ 启用非安全SSL连接")

    try:
        logger.info(f"🔄 尝试连接至 {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()

        # 主线程循环处理弹窗
        while True:
            try:
                title, msg, conf, src = alert_queue.get(timeout=0.1)
                root = tk.Tk()
                root.attributes('-topmost', True)
                root.withdraw()
                alert_msg = f"{title}\n\n{msg}\n置信度: {conf}%\n来源: {src}"
                messagebox.showwarning("⚠️ 地震预警 ⚠️", alert_msg)
                root.destroy()
            except queue.Empty:
                pass
            time.sleep(0.01)

    except KeyboardInterrupt:
        logger.info("🛑 收到退出信号")
    except Exception as e:
        logger.error(f"连接错误: {e}")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()