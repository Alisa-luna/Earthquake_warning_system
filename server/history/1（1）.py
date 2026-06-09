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
import re
from openai import OpenAI
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import math
# ==================== 配置信息 ====================
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 8883
MQTT_TOPIC = "483FDA58BA79-publish"
MQTT_USERNAME = "User"
MQTT_PASSWORD = "1234567890"

# SiliconFlow配置
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_API_KEY = "sk-kyxwdnmgzmcmcyzxbhtibhjvsgjpiefexuyrjkjgmjubyspv"
SILICONFLOW_MODEL = "Qwen/Qwen3-8B"

# QQ邮箱配置（需要开启SMTP服务）
QQ_EMAIL_ENABLE = True  # 是否启用邮件通知
QQ_EMAIL_SENDER = "3809191404@qq.com"  # 你的QQ邮箱
QQ_EMAIL_PASSWORD = "xcqtffkzkhsgcdcb"  # QQ邮箱授权码（不是登录密码）
QQ_EMAIL_RECEIVER = "2028024910@qq.com"  # 接收通知的邮箱
QQ_EMAIL_SMTP_SERVER = "smtp.qq.com"
QQ_EMAIL_SMTP_PORT = 587  # 或 465 (SSL)

LOG_FILE = "sensor_trend_monitor.log"

# ==================== 物理单位阈值配置 ====================
ACCELERATION_THRESHOLD_G = 0.05
GRAVITY_THRESHOLD_LOW_G = 0.98
GRAVITY_THRESHOLD_HIGH_G = 1.03
GYROSCOPE_MIN_DPS = 0.05
ZERO_THRESHOLD = 100
OVERFLOW_THRESHOLD = 32000

CONFIRMATION_TIME_WINDOW = 20.0
SEISMIC_BUFFER_SIZE = 150

SENSITIVITY_ACCEL = 4096.0
SENSITIVITY_GYRO = 16.4

# ==================== 初始化日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("地震模拟监控系统启动...")

# ==================== 初始化SiliconFlow客户端 ====================
try:
    siliconflow_client = OpenAI(
        base_url=SILICONFLOW_API_URL,
        api_key=SILICONFLOW_API_KEY,
        timeout=30.0
    )
    logging.info("✅ SiliconFlow客户端初始化成功")
except Exception as e:
    logging.error(f"❌ SiliconFlow客户端初始化失败: {e}")
    siliconflow_client = None


# ==================== 邮件通知函数 ====================
def send_email_alert(subject, content):
    """发送QQ邮件通知"""
    if not QQ_EMAIL_ENABLE:
        return

    def _send():
        try:
            # 创建邮件对象
            msg = MIMEMultipart()
            msg['From'] = QQ_EMAIL_SENDER
            msg['To'] = QQ_EMAIL_RECEIVER
            msg['Subject'] = Header(subject, 'utf-8')

            # 邮件正文
            msg.attach(MIMEText(content, 'plain', 'utf-8'))

            # 连接QQ邮箱SMTP服务器
            server = smtplib.SMTP(QQ_EMAIL_SMTP_SERVER, QQ_EMAIL_SMTP_PORT)
            server.starttls()  # 启用TLS加密
            server.login(QQ_EMAIL_SENDER, QQ_EMAIL_PASSWORD)

            # 发送邮件
            server.send_message(msg)
            server.quit()

            logging.info(f"📧 邮件通知已发送: {subject}")
        except Exception as e:
            logging.error(f"❌ 邮件发送失败: {e}")

    # 在新线程中发送邮件，避免阻塞
    email_thread = threading.Thread(target=_send)
    email_thread.daemon = True
    email_thread.start()

#===================== 降采样函数 ======================
class SimpleDownsampler:
    """改进版：保留窗口内的最大值，不是简单抽样"""

    def __init__(self, factor=10):
        self.factor = factor
        self.counter = 0
        self.window_data = []  # 存窗口内所有数据
        self.max_values = {  # 存最大值
            'AX': 0, 'AY': 0, 'AZ': 0,
            'GX': 0, 'GY': 0, 'GZ': 0
        }

    def process(self, data):
        # 1. 存原始数据（用于平均值）
        self.window_data.append(data)

        # 2. 更新最大值
        for key in ['AX', 'AY', 'AZ']:
            self.max_values[key] = max(
                self.max_values[key],
                abs(data.get(key, 0))
            )

        self.counter += 1

        # 3. 窗口结束，输出
        if self.counter >= self.factor:
            self.counter = 0

            # 计算平均值（用于趋势判断）
            avg_data = {}
            for key in ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']:
                values = [d.get(key, 0) for d in self.window_data]
                avg_data[key] = sum(values) / len(values)

            # 准备结果
            result = {
                'avg': avg_data,  # 平均值 → 趋势判断
                'max': self.max_values.copy(),  # 最大值 → 触发检测
                'peak_count': self._count_peaks()  # 新增：峰值计数
            }

            # 清空窗口
            self.window_data = []
            for key in self.max_values:
                self.max_values[key] = 0

            return result

        return None

    def _count_peaks(self):
        """统计窗口内超过阈值的次数"""
        threshold = 0.05  # 触发阈值
        count = 0
        for data in self.window_data:
            if abs(data.get('AX', 0)) > threshold:
                count += 1
        return count
# ==================== 事件簇处理 ======================
class SimpleEventCluster:
    def __init__(self, time_window=3.0, min_confidence=60):
        self.time_window = time_window
        self.min_confidence = min_confidence
        self.current_event = None
        self.completed_events = []

    def add_trigger(self, trigger_time, data, peak_count=1):
        """添加触发，增加峰值计数参数"""

        # 计算这次触发的"强度"
        intensity = (
                abs(data.get('AX', 0)) +
                abs(data.get('AY', 0)) +
                abs(data.get('AZ', 0)) - 9.8  # 减去重力
        )

        # 【新增】计算旋转强度
        rotation_intensity = (
                                     abs(data.get('GX', 0)) +
                                     abs(data.get('GY', 0)) +
                                     abs(data.get('GZ', 0))
                             ) / 16.4  # 转换为 °/s

        if not self.current_event:
            # 新事件
            self.current_event = {
                'start': trigger_time,
                'last': trigger_time,
                'max_ax': abs(data.get('AX', 0)),
                'max_ay': abs(data.get('AY', 0)),
                'max_intensity': intensity,
                'max_rotation': rotation_intensity,
                'count': 1,
                'peak_total': peak_count,  # 总峰值次数
                'confidence': 30  # 初始置信度
            }
        else:
            time_diff = trigger_time - self.current_event['last']

            if time_diff < self.time_window:
                # 同一个事件
                self.current_event['last'] = trigger_time
                self.current_event['max_ax'] = max(
                    self.current_event['max_ax'],
                    abs(data.get('AX', 0))
                )
                self.current_event['max_intensity'] = max(
                    self.current_event['max_intensity'],
                    rotation_intensity
                )
                self.current_event['count'] += 1
                self.current_event['peak_total'] += peak_count

                # 动态计算置信度
                self._update_confidence()
            else:
                # 事件结束
                self._finalize_event()
                # 新事件开始
                self.current_event = {
                    'start': trigger_time,
                    'last': trigger_time,
                    'max_ax': abs(data.get('AX', 0)),
                    'max_ay': abs(data.get('AY', 0)),
                    'max_intensity': intensity,
                    'count': 1,
                    'peak_total': peak_count,
                    'confidence': 30
                }

    def _update_confidence(self):
        """动态更新置信度"""
        conf = 0

        # 1. 触发次数越多越可信
        conf += min(40, self.current_event['count'] * 10)

        # 2. 强度越大越可信
        if self.current_event['max_intensity'] > 0.5:
            conf += 30
        elif self.current_event['max_intensity'] > 0.2:
            conf += 15

        # 3. 峰值次数越多越可信
        conf += min(30, self.current_event['peak_total'] * 5)

        self.current_event['confidence'] = min(95, conf)

    def _finalize_event(self):
        """事件结束，最终判断"""
        if self.current_event and self.current_event['confidence'] >= self.min_confidence:
            # 计算持续时间
            self.current_event['duration'] = (
                    self.current_event['last'] - self.current_event['start']
            )
            self.completed_events.append(self.current_event)

            # 记录日志
            logging.info(f"✅ 事件确认: 置信度{self.current_event['confidence']}%, "
                         f"触发{self.current_event['count']}次, "
                         f"持续{self.current_event['duration']:.1f}秒")

    def get_confirmed_events(self):
        """只返回高置信度的事件"""
        return [e for e in self.completed_events if e['confidence'] >= self.min_confidence]

#===================== 烈度计算函数 ========================



class NationalStandardIntensity:
    """
    根据《中国地震烈度表 GB/T 17742-2020》计算仪器烈度
    参考：中国地震局官方说明[citation:2][citation:6]
    """

    def __init__(self):
        # 烈度与PGA对应关系（根据国家标准）
        # 来源：中国地震烈度表[citation:8]
        self.intensity_table = [
            (1, 0, 0.0022, "无感"),  # I度 < 0.22 cm/s²
            (2, 0.0022, 0.0063, "微感"),  # II度 0.22-0.63 cm/s²
            (3, 0.0063, 0.018, "有感"),  # III度 0.63-1.8 cm/s²
            (4, 0.018, 0.045, "室内有感"),  # IV度 1.8-4.5 cm/s²
            (5, 0.045, 0.089, "室外有感"),  # V度 4.5-8.9 cm/s²
            (6, 0.089, 0.177, "惊慌"),  # VI度 8.9-17.7 cm/s²
            (7, 0.177, 0.353, "房屋损坏"),  # VII度 17.7-35.3 cm/s²
            (8, 0.353, 0.707, "建筑物破坏"),  # VIII度 35.3-70.7 cm/s²
            (9, 0.707, 1.414, "建筑物倒塌"),  # IX度 70.7-141.4 cm/s²
            (10, 1.414, 2.5, "毁灭性"),  # X度 141.4-250 cm/s²
            (11, 2.5, 5.0, "灾难性"),  # XI度 250-500 cm/s²
            (12, 5.0, float('inf'), "山河改观")  # XII度 >500 cm/s²
        ]

    def pga_to_intensity(self, pga_cm):
        """
        根据峰值加速度(PGA)计算烈度
        pga_cm: 峰值加速度 (单位 cm/s²)
        返回: (烈度, 描述, 置信度)
        """
        for intensity, low, high, desc in self.intensity_table:
            if low <= pga_cm < high:
                # 计算置信度（越靠近区间中间越高）
                mid = (low + high) / 2 if high != float('inf') else low * 1.5
                confidence = 100 - min(30, abs(pga_cm - mid) / (mid) * 50)
                return intensity, desc, round(confidence)

        return 12, "山河改观", 100

    def from_acceleration(self, ax_g, ay_g, az_g=None):
        """
        从水平加速度计算烈度
        ax_g, ay_g: 水平加速度 (单位 g)
        """
        # 1. 取最大水平加速度
        max_horiz_g = max(abs(ax_g), abs(ay_g))
        max_horiz_cm = max_horiz_g * 980  # 转换为 cm/s²

        # 2. 如果有垂直分量，也可以加权（可选）
        if az_g:
            vert_cm = abs(az_g) * 980
            # 垂直分量一般影响较小，可做加权修正
            # 这里简单取水平为主
            pass

        # 3. 查表得烈度
        intensity, desc, conf = self.pga_to_intensity(max_horiz_cm)

        return {
            'intensity': intensity,
            'description': desc,
            'pga_cm': round(max_horiz_cm, 2),
            'pga_g': round(max_horiz_g, 4),
            'confidence': conf
        }

    def from_event(self, event_data):
        """
        从事件数据计算烈度
        event_data: 事件的所有数据点
        """
        if not event_data:
            return None

        # 找出最大水平加速度
        max_ax = max(abs(d.get('AX', 0)) for d in event_data)
        max_ay = max(abs(d.get('AY', 0)) for d in event_data)

        # 计算烈度
        result = self.from_acceleration(max_ax, max_ay)

        # 附加统计信息
        result['peak_count'] = len([d for d in event_data
                                    if max(abs(d.get('AX', 0)), abs(d.get('AY', 0))) > 0.05])
        result['duration'] = event_data[-1].get('time', 0) - event_data[0].get('time', 0)

        return result

    def get_warning_level(self, intensity):
        """根据烈度返回警告级别"""
        if intensity >= 9:
            return "🔴🔴 红色最高预警", "严重破坏，立即避险"
        elif intensity >= 7:
            return "🔴 红色预警", "房屋损坏，紧急避险"
        elif intensity >= 6:
            return "🟠 橙色预警", "惊慌，准备避险"
        elif intensity >= 4:
            return "🟡 黄色预警", "有感，注意安全"
        elif intensity >= 3:
            return "🟢 蓝色提示", "微感"
        else:
            return "⚪ 无感", "正常"
#==================== 创建实例 ==========================
downsampler = SimpleDownsampler(factor=10)  # 200Hz→20Hz
event_cluster = SimpleEventCluster(time_window=3.0)
intensity_calc = NationalStandardIntensity()

# ==================== 地震事件状态机 ====================
class SeismicMonitor:
    def __init__(self):
        self.potential_event_buffer = deque(maxlen=SEISMIC_BUFFER_SIZE)
        self.event_trigger_time = None
        self.event_triggered = False
        self.trigger_reason = ""

    def reset_event(self):
        self.event_triggered = False
        self.event_trigger_time = None
        self.trigger_reason = ""
        # 注意：不清空buffer，保留用于分析


seismic_monitor = SeismicMonitor()


# ==================== 单位换算函数 ====================
def convert_mpu6050_data(raw_data_dict):
    """将MPU6050原始数据字典转换为物理单位"""
    converted = {}
    try:


        ax_raw = int(raw_data_dict['AX'])
        ay_raw = int(raw_data_dict['AY'])
        az_raw = int(raw_data_dict['AZ'])
        gx_raw = int(raw_data_dict['GX'])
        gy_raw = int(raw_data_dict['GY'])
        gz_raw = int(raw_data_dict['GZ'])

        converted['AX_g'] = ax_raw / SENSITIVITY_ACCEL
        converted['AY_g'] = ay_raw / SENSITIVITY_ACCEL
        converted['AZ_g'] = az_raw / SENSITIVITY_ACCEL

        converted['GX_dps'] = gx_raw / SENSITIVITY_GYRO
        converted['GY_dps'] = gy_raw / SENSITIVITY_GYRO
        converted['GZ_dps'] = gz_raw / SENSITIVITY_GYRO

        converted['AX_raw'] = ax_raw
        converted['AY_raw'] = ay_raw
        converted['AZ_raw'] = az_raw
        converted['GX_raw'] = gx_raw
        converted['GY_raw'] = gy_raw
        converted['GZ_raw'] = gz_raw

    except (ValueError, KeyError, TypeError) as e:
        logging.error(f"数据转换错误: {e}")
        return {}

    return converted


# ==================== 本地规则判断 ====================
def check_for_seismic_trigger(sensor_data):
    """核心触发逻辑：检测是否可能发生地震事件"""
    try:
        converted_data = convert_mpu6050_data(sensor_data)
        if not converted_data:
            return False, "数据转换失败"

        gx_dps, gy_dps, gz_dps = converted_data['GX_dps'], converted_data['GY_dps'], converted_data['GZ_dps']
        gx_abs, gy_abs, gz_abs = abs(gx_dps), abs(gy_dps), abs(gz_dps)

        if all(gyro < GYROSCOPE_MIN_DPS for gyro in [gx_abs, gy_abs, gz_abs]):
            return False, f"陀螺仪无响应(＜{GYROSCOPE_MIN_DPS}°/s)"

        ax_g, ay_g, az_g = converted_data['AX_g'], converted_data['AY_g'], converted_data['AZ_g']
        ax_abs, ay_abs, az_abs = abs(ax_g), abs(ay_g), abs(az_g)

        if ax_abs > ACCELERATION_THRESHOLD_G or ay_abs > ACCELERATION_THRESHOLD_G:
            abnormal = []
            if ax_abs > ACCELERATION_THRESHOLD_G:
                abnormal.append(f"AX({ax_g:.2f}g)")
            if ay_abs > ACCELERATION_THRESHOLD_G:
                abnormal.append(f"AY({ay_g:.2f}g)")
            return True, f"水平加速度异常触发: {', '.join(abnormal)}"

        if az_abs < GRAVITY_THRESHOLD_LOW_G or az_abs > GRAVITY_THRESHOLD_HIGH_G:
            return True, f"垂直加速度异常触发: AZ={az_g:.2f}g"

        return False, "无触发"

    except Exception as e:
        return False, f"触发检查错误: {e}"



# ==================== 震级计算函数 ====================
def calculate_magnitude(ax_max_g, ay_max_g):
    """根据最大水平加速度估算里氏震级（使用物理单位g）"""
    max_horiz_g = max(abs(ax_max_g), abs(ay_max_g))
    max_horiz_gal = max_horiz_g * 980.665

    if max_horiz_gal > 0:
        magnitude = math.log10(max_horiz_gal) * 3.0 - 0.5
        return round(max(1.0, magnitude), 1)
    return 0.0

#======================快速震级计算========================
def quick_magnitude(ax_max, ay_max, rotation_max=None):
    """快速估算震级（加入旋转修正）"""
    max_g = max(ax_max, ay_max)
    if max_g < 0.01:
        return 0

    # 基础震级
    mag = round(math.log10(max_g * 1000) * 2, 1)

    # 【新增】如果有剧烈旋转，可能是近场地震，适当提高震级
    if rotation_max and rotation_max > 5:
        mag = min(9.0, mag + 0.5)  # 近场加0.5级

    return mag


# ==================== 弹窗函数（修复版）====================
# ==================== 弹窗函数（支持烈度）====================
def show_alert(title, value, confidence, trigger_reason, value_type="烈度"):
    """在独立线程中弹出告警窗口
    title: 标题
    value: 烈度或震级数值
    confidence: 置信度
    trigger_reason: 触发原因
    value_type: "烈度" 或 "震级"
    """

    def _show():
        root = tk.Tk()
        root.withdraw()
        root.lift()
        root.attributes('-topmost', True)

        if value_type == "烈度":
            alert_msg = (f"🚨 地震预警！\n\n"
                         f"{title}\n"
                         f"预估烈度: {value} 度\n"
                         f"置信度: {confidence}%\n"
                         f"触发原因: {trigger_reason}")
        else:  # 震级
            alert_msg = (f"🚨 地震预警！\n\n"
                         f"{title}\n"
                         f"预估震级: {value} 级\n"
                         f"置信度: {confidence}%\n"
                         f"触发原因: {trigger_reason}")

        messagebox.showwarning("⚠️ 地震预警 ⚠️", alert_msg)
        root.destroy()

    alert_thread = threading.Thread(target=_show)
    alert_thread.daemon = True
    alert_thread.start()
# ==================== 本地系统自检 ====================
def precise_local_check(sensor_data):
    """精确的本地规则判断（使用原始数据进行硬件检查）"""
    try:
        required_fields = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        missing_fields = [field for field in required_fields if field not in sensor_data]
        if missing_fields:
            return "异常", f"数据字段缺失: {missing_fields}"

        ax = int(sensor_data['AX'])
        ay = int(sensor_data['AY'])
        az = int(sensor_data['AZ'])
        gx = int(sensor_data['GX'])
        gy = int(sensor_data['GY'])
        gz = int(sensor_data['GZ'])

        if all(abs(val) < ZERO_THRESHOLD for val in [ax, ay, az, gx, gy, gz]):
            return "异常", "所有传感器读数接近零（可能断电）"

        if any(abs(val) >= OVERFLOW_THRESHOLD for val in [ax, ay, az, gx, gy, gz]):
            return "异常", "传感器读数达到极限值（可能溢出）"

        return "正常", "硬件自检通过"

    except Exception as e:
        return "异常", f"数据格式错误: {e}"

#=================== 陀螺仪数据处理 ======================
def extract_gyro_features(event_data):
    """
    从事件数据中提取陀螺仪特征
    event_data: 原始数据列表，每个元素有 'GX', 'GY', 'GZ' 字段
    """
    if not event_data:
        return {'max_rotation': 0, 'rotation_energy': 0, 'direction_changes': 0}

    gyro_x = []
    gyro_y = []
    gyro_z = []

    for data in event_data:
        # 兼容两种数据格式
        if isinstance(data, dict):
            if 'GX' in data:
                gyro_x.append(abs(data['GX']))
                gyro_y.append(abs(data['GY']))
                gyro_z.append(abs(data['GZ']))
            elif 'gx' in data:
                gyro_x.append(abs(data['gx']))
                gyro_y.append(abs(data['gy']))
                gyro_z.append(abs(data['gz']))

    if not gyro_x:
        return {'max_rotation': 0, 'rotation_energy': 0, 'direction_changes': 0}

    # 1. 最大旋转速度
    max_rotation = max(max(gyro_x), max(gyro_y), max(gyro_z)) / 16.4  # 转换为 °/s

    # 2. 旋转能量（均方根）
    all_gyro = gyro_x + gyro_y + gyro_z
    rms = math.sqrt(sum(g * g for g in all_gyro) / len(all_gyro)) / 16.4

    # 3. 方向变化次数（简化版）
    direction_changes = 0
    if len(gyro_x) > 3:
        # 粗略估计：看正负变化次数
        for i in range(1, len(gyro_x)):
            if gyro_x[i] * gyro_x[i - 1] < 0:  # 符号改变
                direction_changes += 1

    return {
        'max_rotation': round(max_rotation, 2),
        'rotation_energy': round(rms, 2),
        'direction_changes': direction_changes
    }

# ==================== AI地震分析函数 ====================
def analyze_waveform_with_qwen(event_data):
    """
    用Qwen3-8B分析地震波形
    返回: (is_earthquake, magnitude, reason, confidence)
    """
    if siliconflow_client is None:
        return False, 0, "AI客户端未初始化", 0

    if len(event_data) < 10:
        return False, 0, "数据点不足", 0

    formatted_data = []
    max_ax = max_ay = 0
    for data in event_data:
        converted = convert_mpu6050_data(data)
        if converted:
            formatted_data.append({
                'ax': converted['AX_g'],
                'ay': converted['AY_g'],
                'az': converted['AZ_g'],
                'gx': converted['GX_dps'],
                'gy': converted['GY_dps'],
                'gz': converted['GZ_dps']
            })
            max_ax = max(max_ax, abs(converted['AX_g']))
            max_ay = max(max_ay, abs(converted['AY_g']))

    if len(formatted_data) < 10:
        return False, 0, "有效数据点不足", 0

    n = len(formatted_data)
    p_wave = formatted_data[:n // 3]
    s_wave = formatted_data[-n // 3:]

    def get_stats(data):
        if not data:
            return {'az_mean': 0, 'az_max': 0, 'horiz_mean': 0}
        az_vals = [abs(d['az']) for d in data]
        horiz_vals = [math.sqrt(d['ax'] ** 2 + d['ay'] ** 2) for d in data]
        return {
            'az_mean': sum(az_vals) / len(az_vals),
            'az_max': max(az_vals),
            'horiz_mean': sum(horiz_vals) / len(horiz_vals),
        }

    p_stats = get_stats(p_wave)
    s_stats = get_stats(s_wave)
    gyro_features = extract_gyro_features(event_data)
    prompt = f"""
    你是一个地震波形分析专家。请根据以下多维度特征判断这段震动是否是地震。

    【加速度特征 - P波/S波分析】
    前段（可能是P波）：
    - 垂直加速度均值：{p_stats['az_mean']:.3f} g
    - 水平加速度均值：{p_stats['horiz_mean']:.3f} g

    后段（可能是S波）：
    - 垂直加速度均值：{s_stats['az_mean']:.3f} g
    - 水平加速度均值：{s_stats['horiz_mean']:.3f} g

    【旋转特征 - 陀螺仪分析】
    - 最大旋转速度：{gyro_features['max_rotation']} °/s
    - 旋转能量：{gyro_features['rotation_energy']}
    - 方向变化次数：{gyro_features['direction_changes']}

    【判断规则（多维度综合）】

    1️⃣ 远场地震（有明显P-S波分离）：
       - 前段垂直 > 前段水平（P波先到）
       - 后段水平 > 前段水平（S波增强）
       - 后段水平 > 后段垂直（S波主导）
       - 旋转较小（远场旋转不明显）

    2️⃣ 近场地震（震中附近）：
       - 前后段垂直和水平都较大（无明显分离）
       - 旋转明显（最大旋转 > 3°/s）
       - 旋转能量高
       - 方向变化多（地震波方向复杂）

    3️⃣ 单次冲击（如落物、敲击）：
       - 突然的峰值，很快衰减
       - 后段明显减弱
       - 旋转小或无
       - 持续时间短

    4️⃣ 人为干扰（如旋转设备）：
       - 加速度小，旋转大
       - 方向单一（方向变化少）
       - 可能持续较长时间

    5️⃣ 环境噪音：
       - 各轴数值小且稳定
       - 无明显特征

    请综合以上所有特征，只返回JSON格式：
    {{
        "is_earthquake": true/false,
        "earthquake_type": "远场/近场/未知",
        "confidence": 0-100,
        "reason": "判断依据（一句话说明）"
    }}
    """
    try:
        response = siliconflow_client.chat.completions.create(
            model=SILICONFLOW_MODEL,
            messages=[
                {"role": "system", "content": "你是一个严谨的地震波形分析专家，只返回JSON格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )

        result_text = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_eq = result.get('is_earthquake', False)
            confidence = result.get('confidence', 0)
            reason = result.get('reason', '无说明')

            est_magnitude = calculate_magnitude(max_ax, max_ay)

            return is_eq, est_magnitude, reason, confidence
        else:
            return False, 0, "AI返回格式异常", 0

    except Exception as e:
        logging.error(f"AI调用失败: {e}")
        return False, 0, f"API错误: {str(e)}", 0





# ==================== MQTT消息处理 ====================
def on_message(client, userdata, msg):
    try:
        payload_str = msg.payload.decode('utf-8')
        sensor_data = json.loads(payload_str)

        # 0. 基础硬件自检
        basic_status, basic_reason = precise_local_check(sensor_data)
        if basic_status == "异常":
            logging.error(f"🚨 硬件异常: {basic_reason}")
            show_alert(f"硬件异常: {basic_reason}", 0, 0, "硬件自检")
            send_email_alert("硬件异常告警", f"硬件异常: {basic_reason}\n数据: {payload_str[:100]}")
            return

        # 1. 持续存入缓冲区
        seismic_monitor.potential_event_buffer.append({
            'time': time.time(),  # 存时间戳
            'data': sensor_data.copy()  # 存数据
        })

        # 2. 检查触发条件
        is_triggered, trigger_reason = check_for_seismic_trigger(sensor_data)
        # 降采样处理
        downsampled = downsampler.process(sensor_data)
        if downsampled is not None:
            avg_data = downsampled['avg']
            max_data = downsampled['max']

            # 用平均值判断触发

            if is_triggered:
                event_cluster.add_trigger(time.time(), max_data)

                # ✅【关键】获取最新事件时也要判断
                confirmed_events = event_cluster.get_confirmed_events()
                if confirmed_events:  # 如果有确认的事件
                    latest = confirmed_events[-1]
                    if latest is not None:
                        # ===== 1. 烈度计算（新增，用于实时告警）=====
                      intensity_result = intensity_calc.from_acceleration(
                            latest['max_ax'],
                            latest['max_ay']
                        )

                      # 根据烈度级别决定是否立即告警
                      if intensity_result['intensity'] >= 4:  # IV度以上就有感
                            warning_level, warning_desc = intensity_calc.get_warning_level(
                                intensity_result['intensity']
                            )

                            # 立即弹窗告警（不等待AI）
                            show_alert(
                                f"{warning_level}",
                                intensity_result['intensity'],  # 烈度值
                                80,
                                f"PGA={intensity_result['pga_g']:.4f}g",
                                value_type="烈度"  # ← 新增参数
                            )
                      mag = quick_magnitude(latest['max_ax'], latest['max_ay'])
                      logging.info(f"事件中: {latest['count']}次触发, 当前震级{mag}")

            # 如果有触发，加入事件簇
        confirmed_events= event_cluster.get_confirmed_events()
        if confirmed_events:  # 如果有确认的事件
            latest_event = confirmed_events[-1]
            if latest_event:
            # 计算震级
              mag = quick_magnitude(
                latest_event['max_ax'],
                latest_event['max_ay']
              )
              logging.info(f"📊 事件完成: {latest_event['count']}次触发, 最大震级{mag}")

              if latest_event['count'] > 5:
                  # 从缓冲区提取原始数据
                  event_raw_data = []
                  event_start = latest_event['start']
                  event_end = latest_event['last']

                  for item in seismic_monitor.potential_event_buffer:
                      if event_start <= item['time'] <= event_end:
                          event_raw_data.append(item['data'])
                  if event_raw_data:
                             threading.Thread(target=analyze_waveform_with_qwen, args=(event_raw_data,)).start()
        # 3. 状态机处理
        if is_triggered and not seismic_monitor.event_triggered:
            # 首次触发
            seismic_monitor.event_triggered = True
            seismic_monitor.event_trigger_time = time.time()
            seismic_monitor.trigger_reason = trigger_reason
            logging.warning(f"⚠️ 地震事件触发: {trigger_reason}")
            logging.info("开始缓存事件数据，等待后续波形...")

        elif seismic_monitor.event_triggered:
            # 已经在事件监控中
            current_time = time.time()
            time_elapsed = current_time - seismic_monitor.event_trigger_time

            if time_elapsed >= CONFIRMATION_TIME_WINDOW:
                # 时间窗口结束，提交分析
                logging.info("⏰ 事件时间窗口结束，提交AI分析...")

                # 提取缓存数据用于分析
                buffer_data = list(seismic_monitor.potential_event_buffer)

                # AI分析
                is_earthquake, magnitude, reason, confidence = analyze_waveform_with_qwen([item['data'] for item in buffer_data])

                if is_earthquake and confidence > 60:
                    # 确认为地震
                    show_alert(
                        "地震确认",
                        magnitude,  # 震级值
                        confidence,
                        seismic_monitor.trigger_reason,
                        value_type="震级"  # ← 新增参数
                    )

                    # 弹出窗口（修复参数）
                    show_alert(reason, magnitude, confidence, seismic_monitor.trigger_reason)

                    # 发送邮件通知
                    email_content = f"""
                    地震预警详情：

                    时间：{time.strftime('%Y-%m-%d %H:%M:%S')}
                    预估震级：{magnitude} 级
                    置信度：{confidence}%
                    触发原因：{seismic_monitor.trigger_reason}
                    AI分析：{reason}

                    请保持警惕！
                    """
                    send_email_alert(f"⚠️ 地震预警 - {magnitude}级", email_content)
                else:
                    logging.info(f"📝 AI分析为干扰: {reason} (置信度:{confidence}%)")

                # 重置事件状态
                seismic_monitor.reset_event()

        # 4. 正常状态日志（降低频率）
        elif len(seismic_monitor.potential_event_buffer) % 10 == 0:
            converted_data = convert_mpu6050_data(sensor_data)
            if converted_data:
                ax_g, ay_g, az_g = converted_data['AX_g'], converted_data['AY_g'], converted_data['AZ_g']
                gx_dps, gy_dps, gz_dps = converted_data['GX_dps'], converted_data['GY_dps'], converted_data['GZ_dps']
                logging.info(f"📊 数据正常 - 加速度: AX={ax_g:.2f}g, AY={ay_g:.2f}g, AZ={az_g:.2f}g")

    except Exception as e:
        import traceback
        logging.error(f"处理消息错误: {e}")
        logging.error(f"处理消息错误: {e}\n{traceback.format_exc()}")


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logging.info("✅ 成功连接到MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        logging.info(f"📡 已订阅Topic: {MQTT_TOPIC}")
    else:
        logging.error(f"❌ 连接失败，错误代码: {rc}")


# ==================== 主程序 ====================
def main():
    # 测试邮件配置（启动时发送测试邮件）
    if QQ_EMAIL_ENABLE:
        send_email_alert("监控系统启动", f"地震监控系统已启动\n时间：{time.strftime('%Y-%m-%d %H:%M:%S')}")

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # SSL配置
    try:
        client.tls_set(ca_certs=r"D:\Mosquitto\ca.crt")
        client.tls_insecure_set(True)
        logging.info("✅ SSL证书配置成功")
    except Exception as e:
        logging.error(f"SSL配置失败: {e}")
        client.tls_insecure_set(True)
        logging.warning("⚠️ 启用非安全SSL连接")

    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        logging.info(f"🔄 尝试连接至 {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        logging.error(f"连接错误: {e}")


if __name__ == "__main__":
    main()