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

    prompt = f"""
    你是一个地震波形分析专家。请根据以下传感器数据特征，判断这段震动是否可能是地震。

    【数据特征】
    - 前段（可能是P波）：
        * 垂直加速度均值：{p_stats['az_mean']:.3f} g
        * 水平加速度均值：{p_stats['horiz_mean']:.3f} g

    - 后段（可能是S波）：
        * 垂直加速度均值：{s_stats['az_mean']:.3f} g
        * 水平加速度均值：{s_stats['horiz_mean']:.3f} g

    【判断规则】
    - 如果前段垂直 > 前段水平 且 后段水平 > 前段水平 且 后段水平 > 后段垂直，则高度疑似地震
    - 如果只有前段垂直大，但后段无明显水平增强，可能是单次冲击
    - 如果前后段无明显规律，可能是环境噪音

    只返回JSON格式：
    {{
        "is_earthquake": true/false,
        "confidence": 0-100,
        "reason": "判断依据"
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


# ==================== 震级计算函数 ====================
def calculate_magnitude(ax_max_g, ay_max_g):
    """根据最大水平加速度估算里氏震级（使用物理单位g）"""
    max_horiz_g = max(abs(ax_max_g), abs(ay_max_g))
    max_horiz_gal = max_horiz_g * 980.665

    if max_horiz_gal > 0:
        magnitude = math.log10(max_horiz_gal) * 3.0 - 0.5
        return round(max(1.0, magnitude), 1)
    return 0.0


# ==================== 弹窗函数（修复版）====================
def show_alert(reason, magnitude, confidence, trigger_reason):
    """在独立线程中弹出告警窗口"""

    def _show():
        # 创建新的Tk实例，确保线程安全
        root = tk.Tk()
        root.withdraw()  # 隐藏主窗口

        # 确保窗口在最前
        root.lift()
        root.attributes('-topmost', True)

        alert_msg = (f"🚨 地震预警！\n\n"
                     f"预估震级: {magnitude} 级\n"
                     f"置信度: {confidence}%\n"
                     f"触发原因: {trigger_reason}\n"
                     f"AI分析: {reason}")

        # 弹出警告框
        messagebox.showwarning("⚠️ 地震预警 ⚠️", alert_msg)

        # 销毁窗口
        root.destroy()

    # 创建并启动线程
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
        seismic_monitor.potential_event_buffer.append(sensor_data)

        # 2. 检查触发条件
        is_triggered, trigger_reason = check_for_seismic_trigger(sensor_data)

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
                is_earthquake, magnitude, reason, confidence = analyze_waveform_with_qwen(buffer_data)

                if is_earthquake and confidence > 60:
                    # 确认为地震
                    alert_msg = (f"检测到地震波！\n"
                                 f"预估震级: {magnitude}级\n"
                                 f"置信度: {confidence}%\n"
                                 f"触发原因: {seismic_monitor.trigger_reason}\n"
                                 f"AI分析: {reason}")
                    logging.critical(f"🚨 {alert_msg}")

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
        logging.error(f"处理消息错误: {e}")


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