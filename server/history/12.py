import json
import logging
import threading
import tkinter as tk
from tkinter import messagebox
import paho.mqtt.client as mqtt
import requests
import time
from collections import deque
import numpy as np

# ==================== 配置信息 ====================
MQTT_BROKER = "w8381dbc.ala.cn-hangzhou.emqxsl.cn"
MQTT_PORT = 8883
MQTT_TOPIC = "483FDA58BA79-publish"
MQTT_USERNAME = "User"
MQTT_PASSWORD = "1234567890"

# 使用SiliconFlow进行趋势预测
SILICONFLOW_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
SILICONFLOW_API_KEY = "sk-kyxwdnmgzmcmcyzxbhtibhjvsgjpiefexuyrjkjgmjubyspv"
SILICONFLOW_MODEL = "Qwen/Qwen3-8B"

LOG_FILE = "sensor_trend_monitor.log"

# ==================== 阈值配置 ====================
ACCELERATION_THRESHOLD = 5000  # X、Y轴加速度阈值
GRAVITY_THRESHOLD_LOW = 15000  # Z轴重力加速度下限
GRAVITY_THRESHOLD_HIGH = 17000  # Z轴重力加速度上限
GYROSCOPE_THRESHOLD = 300  # 陀螺仪阈值
ZERO_THRESHOLD = 100  # 零值检测阈值
OVERFLOW_THRESHOLD = 32000  # 溢出检测阈值

# ==================== 数据缓存配置 ====================
DATA_HISTORY_SIZE = 10  # 保存最近10条数据用于趋势分析
data_history = deque(maxlen=DATA_HISTORY_SIZE)

# ==================== 初始化日志 ====================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info("传感器趋势监控系统启动...")


# ==================== 弹窗函数 ====================
def show_alert(reason, data):
    """在独立线程中弹出告警窗口"""

    def _show():
        root = tk.Tk()
        root.withdraw()
        alert_msg = f"传感器预警！\n\n预警原因: {reason}\n\n当前数据: {data}"
        messagebox.showwarning("趋势预警", alert_msg)
        root.destroy()

    alert_thread = threading.Thread(target=_show)
    alert_thread.daemon = True
    alert_thread.start()


# ==================== 本地精确判断 ====================
def precise_local_check(sensor_data):
    """精确的本地规则判断"""
    try:
        required_fields = ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']
        missing_fields = [field for field in required_fields if field not in sensor_data]
        if missing_fields:
            return "异常", f"数据字段缺失: {missing_fields}"

        # 转换为数值
        ax = int(sensor_data['AX'])
        ay = int(sensor_data['AY'])
        az = int(sensor_data['AZ'])
        gx = int(sensor_data['GX'])
        gy = int(sensor_data['GY'])
        gz = int(sensor_data['GZ'])

        # 1. 检查零值（断电）
        if all(abs(val) < ZERO_THRESHOLD for val in [ax, ay, az, gx, gy, gz]):
            return "异常", "所有传感器读数接近零（可能断电）"

        # 2. 检查溢出
        if any(abs(val) >= OVERFLOW_THRESHOLD for val in [ax, ay, az, gx, gy, gz]):
            return "异常", "传感器读数达到极限值（可能溢出）"

        # 3. 检查X、Y轴加速度
        ax_abs, ay_abs = abs(ax), abs(ay)
        if ax_abs > ACCELERATION_THRESHOLD or ay_abs > ACCELERATION_THRESHOLD:
            abnormal = []
            if ax_abs > ACCELERATION_THRESHOLD: abnormal.append(f"AX({ax})")
            if ay_abs > ACCELERATION_THRESHOLD: abnormal.append(f"AY({ay})")
            return "异常", f"水平加速度异常: {', '.join(abnormal)}"

        # 4. 检查重力加速度（Z轴）
        az_abs = abs(az)
        if az_abs < GRAVITY_THRESHOLD_LOW or az_abs > GRAVITY_THRESHOLD_HIGH:
            return "异常", f"重力加速度异常: AZ={az}（正常范围15000-17000）"

        # 5. 检查陀螺仪
        gx_abs, gy_abs, gz_abs = abs(gx), abs(gy), abs(gz)
        if any(gyro > GYROSCOPE_THRESHOLD for gyro in [gx_abs, gy_abs, gz_abs]):
            abnormal = []
            if gx_abs > GYROSCOPE_THRESHOLD: abnormal.append(f"GX({gx})")
            if gy_abs > GYROSCOPE_THRESHOLD: abnormal.append(f"GY({gy})")
            if gz_abs > GYROSCOPE_THRESHOLD: abnormal.append(f"GZ({gz})")
            return "异常", f"角速度异常: {', '.join(abnormal)}"

        return "正常", "所有参数在正常范围内"

    except Exception as e:
        return "异常", f"数据格式错误: {e}"


# ==================== 趋势分析函数 ====================
def analyze_trends():
    """分析数据趋势，检测潜在问题"""
    if len(data_history) < 3:  # 至少需要3个数据点
        return "数据不足", "需要更多数据进行分析"

    # 提取历史数据
    history = list(data_history)

    # 计算各轴的变化趋势
    trends = {}
    for axis in ['AX', 'AY', 'AZ', 'GX', 'GY', 'GZ']:
        values = [data[axis] for data in history]
        # 计算斜率（简单线性趋势）
        if len(values) >= 2:
            x = np.arange(len(values))
            slope, _ = np.polyfit(x, values, 1)
            trends[axis] = slope

    # 检测异常趋势
    warnings = []

    # 加速度趋势警告
    for axis in ['AX', 'AY']:
        if abs(trends.get(axis, 0)) > 100:  # 斜率过大
            warnings.append(f"{axis}加速度正在快速变化")

    # 重力加速度趋势警告
    if abs(trends.get('AZ', 0)) > 50:  # 重力变化不应太快
        warnings.append("重力加速度异常变化")

    # 角速度趋势警告
    for axis in ['GX', 'GY', 'GZ']:
        if abs(trends.get(axis, 0)) > 20:  # 角速度变化过快
            warnings.append(f"{axis}角速度正在加速")

    if warnings:
        return "趋势预警", f"检测到潜在问题: {'; '.join(warnings)}"
    else:
        return "趋势正常", "数据变化平稳"


# ==================== AI趋势预测函数 ====================
def predict_with_ai(current_status, trend_analysis):
    """使用AI预测未来趋势"""

    # 准备历史数据摘要
    history_summary = "\n".join([f"时刻-{i}: {data}" for i, data in enumerate(data_history)])

    prompt = f"""
[系统指令]
你是一个工业传感器趋势预测专家。基于当前状态和历史数据，预测未来可能的问题。

## 当前数据状态:
{current_status}

## 趋势分析结果:
{trend_analysis}

## 最近{DATA_HISTORY_SIZE}条历史数据:
{history_summary}

## 你的任务:
1. 分析当前趋势是否可能导致未来问题
2. 预测未来30秒内可能发生的异常
3. 提供预防建议

## 输出格式:
请用以下JSON格式回答:
{{
  "risk_level": "低/中/高",
  "prediction": "未来可能发生的问题描述",
  "suggestion": "预防建议"
}}

请开始分析:
"""

    headers = {
        "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": SILICONFLOW_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 300
    }

    try:
        response = requests.post(SILICONFLOW_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        ai_response = result['choices'][0]['message']['content']

        # 提取JSON响应
        try:
            json_start = ai_response.find('{')
            json_end = ai_response.rfind('}') + 1
            json_str = ai_response[json_start:json_end]
            prediction = json.loads(json_str)
            return prediction
        except:
            return {"risk_level": "未知", "prediction": "AI响应格式错误", "suggestion": "请检查系统"}

    except Exception as e:
        logging.error(f"AI预测失败: {e}")
        return {"risk_level": "未知", "prediction": "预测服务暂不可用", "suggestion": "请依赖本地检测"}


# ==================== MQTT回调函数 ====================
def on_connect(client, userdata, flags, rc):
    """连接MQTT服务器时的回调"""
    if rc == 0:
        logging.info("成功连接到MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        logging.info(f"已订阅Topic: {MQTT_TOPIC}")
    else:
        logging.error(f"连接失败，错误代码: {rc}")


def on_message(client, userdata, msg):
    """收到MQTT消息时的回调"""
    try:
        payload_str = msg.payload.decode('utf-8')
        sensor_data = json.loads(payload_str)

        # 保存到历史数据
        data_history.append(sensor_data)

        logging.info(f"收到数据: {sensor_data}")

        # 第一步：本地精确判断
        status, reason = precise_local_check(sensor_data)

        if status == "异常":
            logging.error(f"🚨 立即异常: {reason}")
            show_alert(f"立即异常: {reason}", payload_str)
            return

        # 第二步：趋势分析
        trend_status, trend_reason = analyze_trends()
        logging.info(f"趋势分析: {trend_status} - {trend_reason}")

        if trend_status == "趋势预警":
            # 第三步：AI预测
            prediction = predict_with_ai(f"{status}: {reason}", f"{trend_status}: {trend_reason}")

            logging.warning(f"🔶 趋势预警: {trend_reason}")
            logging.info(f"AI预测: 风险{prediction['risk_level']} - {prediction['prediction']}")

            if prediction['risk_level'] in ["高", "中"]:
                show_alert(
                    f"趋势预警: {trend_reason}\nAI预测: {prediction['prediction']}\n建议: {prediction['suggestion']}",
                    payload_str
                )

        else:
            logging.info(f"✅ 状态正常: {reason}")

    except Exception as e:
        logging.error(f"处理消息错误: {e}")


# ==================== 主程序 ====================
def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    # SSL配置
    try:
        client.tls_set(ca_certs="C:/Users/L1370/Desktop/main/emqxsl-ca.crt")
        logging.info("SSL证书配置成功")
    except Exception as e:
        logging.error(f"SSL配置失败: {e}")
        client.tls_insecure_set(True)
        logging.warning("启用非安全SSL连接")

    # 设置认证
    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        logging.info(f"尝试连接至 {MQTT_BROKER}:{MQTT_PORT}...")
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        logging.error(f"连接错误: {e}")


if __name__ == "__main__":
    main()
