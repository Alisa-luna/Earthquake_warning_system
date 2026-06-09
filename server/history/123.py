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
import queue  # 新增

if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
    np.trapz = np.trapezoid

import obspy
from obspy import Stream, Trace
import enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import seisbench.models as sbm
import torch
import warnings

# 抑制SeisBench警告
warnings.filterwarnings("ignore", module="seisbench")
warnings.filterwarnings("ignore", message=".*fragments shorter.*")

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
ACCELERATION_THRESHOLD_G = 0.03
GRAVITY_THRESHOLD_LOW_G = 0.99
GRAVITY_THRESHOLD_HIGH_G = 1.01
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

# 弹窗队列（新增）
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
logger.info("🚀 三级地震预警系统启动（最终修复版）...")
logging.getLogger('seisbench').setLevel(logging.ERROR)

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

# ==================== 弹窗（放入队列，由主线程处理）====================
def show_alert(title, message, confidence, source):
    """将弹窗信息放入队列，由主线程处理"""
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
        horizontal = math.sqrt(ax_g**2 + ay_g**2)
        vertical = abs(az_g - 1.0)
        current_time = time.time()
        if horizontal > ACCELERATION_THRESHOLD_G or vertical > 0.2:
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
    if max_cm < 0.0022: return 1, "无感"
    elif max_cm < 0.0063: return 2, "微感"
    elif max_cm < 0.018: return 3, "有感"
    elif max_cm < 0.045: return 4, "室内有感"
    elif max_cm < 0.089: return 5, "室外有感"
    elif max_cm < 0.177: return 6, "惊慌"
    elif max_cm < 0.353: return 7, "房屋损坏"
    elif max_cm < 0.707: return 8, "建筑物破坏"
    elif max_cm < 1.414: return 9, "建筑物倒塌"
    elif max_cm < 2.5: return 10, "毁灭性"
    else: return 11, "灾难性"

# ==================== Stream转换 ====================
def to_standard_stream(data_points, target_length=MODEL_INPUT_LENGTH):
    if not data_points or len(data_points) < 100:
        return None
    npts = len(data_points)
    sampling_rate = 100
    scale = 9.8 / ACTUAL_SENSITIVITY
    data = np.zeros((3, npts), dtype=np.float32)
    for i, d in enumerate(data_points):
        data[0, i] = float(d.get('AZ', 0)) * scale
        data[1, i] = float(d.get('AY', 0)) * scale
        data[2, i] = float(d.get('AX', 0)) * scale
    data -= np.mean(data, axis=1, keepdims=True)
    if npts < target_length:
        pad_width = target_length - npts
        data = np.column_stack([data, np.tile(data[:, -1:], (1, pad_width))])
        npts = target_length
    elif npts > target_length:
        data = data[:, :target_length]
        npts = target_length
    current_time = time.time()
    starttime = obspy.UTCDateTime(current_time - npts / sampling_rate)
    traces = []
    for comp_idx, channel in enumerate(['Z', 'N', 'E']):
        trace = Trace(data=data[comp_idx])
        trace.stats.sampling_rate = sampling_rate
        trace.stats.delta = 1.0 / sampling_rate
        trace.stats.channel = f'BH{channel}'
        trace.stats.starttime = starttime
        trace.stats.npts = npts
        trace.stats.calib = 1.0
        trace.stats.network = 'XX'
        trace.stats.station = 'STA1'
        trace.stats.location = ''
        traces.append(trace)
    return Stream(traces=traces)

# ==================== PhaseNet（数组版）====================
class PhaseNetReal:
    def __init__(self):
        logger.info("正在初始化PhaseNet模型...")
        try:
            self.model = sbm.PhaseNet.from_pretrained("stead")
            self.model.eval()
            self.threshold = 70
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

# ==================== EQTransformer（数组版，6000点）====================
class EQTReal:
    def __init__(self):
        logger.info("正在初始化EQTransformer模型...")
        try:
            self.model = sbm.EQTransformer.from_pretrained("stead")
            self.model.eval()
            logger.info("✅ EQTransformer模型加载成功")
            self.alert_threshold = 80
            self.confirm_threshold = 90
            self.in_channels = getattr(self.model, 'in_channels', 3)
            self.in_samples = getattr(self.model, 'in_samples', 6000)
            logger.info(f"📐 EQT模型期望: in_channels={self.in_channels}, in_samples={self.in_samples}")
        except Exception as e:
            logger.error(f"❌ EQTransformer模型加载失败: {e}")
            self.model = None
            self.in_samples = 6000

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
                # 取能量最强的窗口（简化：取最后target_length点）
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

# ==================== 最终确认（共享模型）====================
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
        is_eq = yes >= (total * 2 / 3)
        confidence = (yes / total) * 100 * (np.mean(confidences) if confidences else 0)
        max_ax = max(abs(d.get('AX', 0)) for d in data_points) / ACTUAL_SENSITIVITY
        max_ay = max(abs(d.get('AY', 0)) for d in data_points) / ACTUAL_SENSITIVITY
        magnitude = quick_magnitude(max_ax, max_ay)
        logger.info(f"最终确认: 地震={is_eq}, 置信度={confidence:.1f}%, 震级={magnitude}, 投票={yes}/{total}")
        return {'is_earthquake': is_eq, 'confidence': round(confidence, 1), 'magnitude': magnitude, 'details': f"投票: {yes}/{total}"}

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
        logger.info("正在初始化所有模型...")
        self.phase = PhaseNetReal()
        self.eqt = EQTReal()
        self.final = FinalValidator(self.phase, self.eqt)
        logger.info("✅ 所有模型初始化完成")

    def on_data(self, sensor_data):
        with self.lock:
            current_time = time.time()
            self.total_points_processed += 1
            self.data_buffer.append({'time': current_time, 'data': sensor_data.copy()})
            if self.current_event:
                self.current_event.max_ax = max(self.current_event.max_ax, abs(sensor_data.get('AX', 0)))
                self.current_event.max_ay = max(self.current_event.max_ay, abs(sensor_data.get('AY', 0)))
                self.current_event.max_az = max(self.current_event.max_az, abs(sensor_data.get('AZ', 0)))
            is_triggered, reason = check_for_seismic_trigger(sensor_data)
            if is_triggered:
                self.total_triggers += 1
                logger.info(f"⚡ 触发! {reason}")
            if is_triggered:
                self._handle_trigger(current_time)
            if self.state == AlertState.IDLE:
                self._handle_idle()
            elif self.state == AlertState.P_ALERT:
                self._handle_p_alert()
            elif self.state == AlertState.EQT_CONFIRM:
                self._handle_eqt_confirm()

    def _handle_trigger(self, current_time):
        if not self.current_event:
            self.current_event = EventData(start_time=current_time, last_trigger_time=current_time, p_arrival=current_time)
            logger.info(f"🌟 新事件开始 at {time.strftime('%H:%M:%S', time.localtime(current_time))}")
        else:
            if current_time - self.current_event.last_trigger_time < TRIGGER_WINDOW:
                self.current_event.last_trigger_time = current_time
                logger.debug(f"事件延续 at {current_time:.3f}")
            else:
                logger.info(f"事件超时（间隔>{TRIGGER_WINDOW}s），开始新事件")
                self._reset()
                self.current_event = EventData(start_time=current_time, last_trigger_time=current_time, p_arrival=current_time)

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

    def _handle_idle(self):
        if not self.current_event:
            return
        event_points = self._count_event_points()
        if event_points < MIN_PHASE_POINTS:
            return
        if event_points - self.current_event.last_phase_analysis_points < MODEL_COOLDOWN_POINTS:
            return
        event_data = self._get_event_data()
        if len(event_data) < MIN_PHASE_POINTS:
            return
        p_prob = self.phase.analyze(event_data)
        self.current_event.last_phase_analysis_points = event_points
        if p_prob >= self.phase.threshold:
            if not self.alerts_sent[AlertLevel.YELLOW]:
                self._send_alert(AlertLevel.YELLOW, f"PhaseNet检测到P波，概率{p_prob:.1f}%", p_prob, "PhaseNet")
            self.state = AlertState.P_ALERT
            logger.info(f"🟡 状态转移: IDLE → P_ALERT (P波概率 {p_prob:.1f}%)")
        else:
            logger.info(f"PhaseNet未确认P波 (概率 {p_prob:.1f}%)，重置事件")
            self._reset()

    def _handle_p_alert(self):
        if not self.current_event:
            self._reset()
            return
        if time.time() - self.current_event.start_time > 60:
            logger.info("P_ALERT超时 (60秒)，重置")
            self._reset()
            return
        event_points = self._count_event_points()
        if event_points < MIN_EQT_POINTS:
            return
        if event_points - self.current_event.last_eqt_analysis_points < MODEL_COOLDOWN_POINTS:
            return
        event_data = self._get_event_data()
        if len(event_data) < MIN_EQT_POINTS:
            return
        eqt_result = self.eqt.analyze(event_data)
        self.current_event.last_eqt_analysis_points = event_points
        if eqt_result['confidence'] >= self.eqt.alert_threshold:
            if not self.alerts_sent[AlertLevel.ORANGE]:
                self._send_alert(AlertLevel.ORANGE, f"EQT检测到地震，置信度{eqt_result['confidence']:.1f}%", eqt_result['confidence'], "EQT")
                self.current_event.eqt_alert_time = time.time()
            if eqt_result['confidence'] >= self.eqt.confirm_threshold:
                self.current_event.eqt_confirm_time = time.time()
                self.current_event.magnitude = max(self.current_event.magnitude, eqt_result['magnitude'])
                self.state = AlertState.EQT_CONFIRM
                logger.info(f"🟠 状态转移: P_ALERT → EQT_CONFIRM (置信度 {eqt_result['confidence']:.1f}%)")

    def _handle_eqt_confirm(self):
        if not self.current_event:
            self._reset()
            return
        if self.current_event.final_time is not None:
            return
        event_points = self._count_event_points()
        if event_points < MIN_FINAL_POINTS:
            return
        event_data = self._get_event_data()
        if len(event_data) < MIN_FINAL_POINTS:
            return
        result = self.final.analyze(event_data)
        if result and result['is_earthquake']:
            self.current_event.final_time = time.time()
            self.current_event.magnitude = result['magnitude']
            self.current_event.confidence = result['confidence']
            intensity, desc = calculate_intensity(self.current_event.max_ax / ACTUAL_SENSITIVITY, self.current_event.max_ay / ACTUAL_SENSITIVITY)
            self.current_event.intensity = intensity
            self.current_event.description = desc
            if not self.alerts_sent[AlertLevel.RED]:
                self._send_alert(AlertLevel.RED, f"最终确认：烈度{intensity}度，震级{result['magnitude']}级", result['confidence'], "SeisBench集成")
                self._send_final_email()
            self.state = AlertState.FINAL
            logger.info(f"🔴 状态转移: EQT_CONFIRM → FINAL (震级 {result['magnitude']}, 烈度 {intensity})")
        else:
            logger.warning("最终分析否定地震，可能误报")
            self._reset()

    def _send_alert(self, level, message, confidence, source):
        if self.alerts_sent[level]:
            return
        self.alerts_sent[level] = True
        titles = {AlertLevel.YELLOW: "🟡 P波预警", AlertLevel.ORANGE: "🟠 地震预警", AlertLevel.RED: "🔴 地震最终确认"}
        show_alert(titles[level], message, confidence, source)
        if level.value >= AlertLevel.ORANGE.value:  # 修复点：比较value
            threading.Thread(target=send_email_alert, args=(titles[level], message)).start()

    def _send_final_email(self):
        if not self.current_event:
            return
        duration = self.current_event.final_time - self.current_event.start_time
        content = f"""
地震最终确认报告
{'='*50}

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

# ==================== 主程序（处理弹窗队列）====================
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
        client.loop_start()  # 非阻塞循环
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