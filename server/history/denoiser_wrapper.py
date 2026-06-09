"""
DASPy 降噪器封装
将 DASPy 的降噪功能封装成易用的接口
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from daspy import Section

    DASPY_AVAILABLE = True
except ImportError:
    DASPY_AVAILABLE = False
    logger.warning("DASPy 未安装，降噪功能将不可用")


class DASPyDenoiser:
    """
    基于 DASPy 的地震数据降噪器
    专门为三分量 [3, time] 数据设计
    """

    def __init__(self, sampling_rate=100):
        """
        初始化降噪器

        Args:
            sampling_rate: 采样率 (Hz)
        """
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.is_available = DASPY_AVAILABLE

        if self.is_available:
            logger.info("✅ DASPy 降噪器初始化成功")
        else:
            logger.warning("⚠️ DASPy 不可用，降噪器将返回原始数据")

    def __call__(self, data_3c):
        """
        使对象可调用，方便集成

        Args:
            data_3c: numpy array, shape (3, N) 三分量数据

        Returns:
            降噪后的数据，相同形状
        """
        return self.denoise(data_3c)

    def denoise(self, data_3c):
        """
        对三分量数据进行降噪

        Args:
            data_3c: numpy array, shape (3, N)  [Z, N, E] 顺序

        Returns:
            denoised: 降噪后的数据，相同形状
        """
        # 输入验证
        if data_3c.ndim != 2 or data_3c.shape[0] != 3:
            logger.error(f"输入格式错误: 需要 [3, N]，实际 {data_3c.shape}")
            return data_3c

        if not self.is_available:
            logger.debug("DASPy 不可用，跳过降噪")
            return data_3c

        try:
            # 转换为 DASPy 的 Section 格式
            # DASPy 期望 (channel, time) 格式，和我们的 data_3c 一致
            sec = Section(data=data_3c, dt=self.dt, dx=1.0)

            # 应用曲波变换降噪
            # 这是 DASPy 里效果最好的降噪算法
            sec_denoised = sec.curvelet_denoising()

            denoised_data = sec_denoised.data.copy()
            logger.debug(f"降噪完成，数据形状: {denoised_data.shape}")

            return denoised_data

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
        for data in data_batch:
            results.append(self.denoise(data))
        return results


# 简单的测试代码
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 测试降噪器
    denoiser = DASPyDenoiser(sampling_rate=100)

    # 生成模拟数据
    test_data = np.random.randn(3, 3000) * 0.1
    print(f"输入形状: {test_data.shape}")

    # 测试直接调用
    result1 = denoiser(test_data)
    print(f"直接调用结果形状: {result1.shape}")

    # 测试 denoise 方法
    result2 = denoiser.denoise(test_data)
    print(f"denoise方法结果形状: {result2.shape}")

    print("测试完成!")