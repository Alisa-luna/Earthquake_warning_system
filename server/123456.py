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
from pca_matrices import pca_inverse, PCA_OUT_PRED, PCA_OUT_HIST, QUANT_SCALE
# 抑制警告
warnings.filterwarnings("ignore", module="seisbench")
warnings.filterwarnings("ignore", module="daspy")

# ==================== PCA 矩阵数据（从网关端头文件复制）====================

# ==================== 空间插值与震中推算 ====================
def haversine(lat1, lng1, lat2, lng2):
    """计算两点间距离 (km)"""
    R = 6371
    dLat = math.radians(lat2 - lat1)
    dLng = math.radians(lng2 - lng1)
    a = math.sin(dLat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dLng / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def pga_to_intensity(pga_cm_s2):
    """峰值加速度 (cm/s²) → 烈度 (1-12)"""
    thresholds = [(0.31, 1), (0.63, 2), (1.25, 3), (2.50, 4), (5.00, 5),
                  (10.0, 6), (25.0, 7), (50.0, 8), (100.0, 9),
                  (250.0, 10), (500.0, 11), (float('inf'), 12)]
    for thresh, level in thresholds:
        if pga_cm_s2 < thresh:
            return level
    return 12


def waveform_to_intensity(waveform):
    """从三轴波形计算烈度"""
    horizontal = np.sqrt(waveform[0] ** 2 + waveform[1] ** 2)
    pga_cm_s2 = np.max(np.abs(horizontal)) * 100
    return pga_to_intensity(pga_cm_s2), pga_cm_s2


def get_node_intensities(pca_buffer, node_locations):
    node_data = []
    for node_id, buf in pca_buffer.items():
        if not buf or node_id not in node_locations:
            continue
        lat, lng = node_locations[node_id]
        recent_waveforms = [item['waveform'] for item in list(buf)[-3:]]  # 直接用，波形已是 g 值
        intensities, pgas, max_amps = [], [], []
        for wf in recent_waveforms:
            intensity, pga = waveform_to_intensity(wf)
            intensities.append(intensity)
            pgas.append(pga)
            max_amps.append(np.max(np.abs(wf)))
        node_data.append({...})
    return node_data


def estimate_epicenter(node_locations, pca_buffer):
    """多节点加权平均推算震中"""
    triggered = []
    for node_id, buf in pca_buffer.items():
        if buf and node_id in node_locations:
            latest = buf[-1]
            max_amp = np.max(np.abs(latest['waveform']))
            lat, lng = node_locations[node_id]
            triggered.append({'lat': lat, 'lng': lng, 'weight': max_amp})
    if not triggered:
        return None, None
    total_w = sum(n['weight'] for n in triggered)
    if total_w == 0:
        return None, None
    est_lat = sum(n['lat'] * n['weight'] for n in triggered) / total_w
    est_lng = sum(n['lng'] * n['weight'] for n in triggered) / total_w
    logger.info(f"📍 震中推算: {est_lat:.4f}, {est_lng:.4f} ({len(triggered)}个节点)")
    return est_lat, est_lng


def idw_interpolate(node_data, target_lat, target_lng, power=2.0):
    """反距离权重插值"""
    if not node_data:
        return None
    total_w, weighted_sum = 0, 0
    for n in node_data:
        dist = haversine(target_lat, target_lng, n['lat'], n['lng'])
        if dist < 0.001:
            return n['intensity']
        w = 1.0 / (dist ** power)
        weighted_sum += n['intensity'] * w
        total_w += w
    return weighted_sum / total_w if total_w else None


def generate_shakemap(node_data, radius_km=20):
    """生成烈度分布网格"""
    if len(node_data) < 2:
        return None
    lats = [n['lat'] for n in node_data]
    lngs = [n['lng'] for n in node_data]
    center_lat, center_lng = np.mean(lats), np.mean(lngs)
    lat_span = radius_km / 111.32
    lng_span = radius_km / (111.32 * math.cos(math.radians(center_lat)))
    n_points = 30
    grid_lats = np.linspace(center_lat - lat_span, center_lat + lat_span, n_points)
    grid_lngs = np.linspace(center_lng - lng_span, center_lng + lng_span, n_points)
    grid = np.zeros((n_points, n_points))
    for i, lat in enumerate(grid_lats):
        for j, lng in enumerate(grid_lngs):
            est = idw_interpolate(node_data, lat, lng)
            grid[i, j] = est if est else 0
    return grid_lats, grid_lngs, grid

def generate_shakemap_html(node_data, epicenter_lat, epicenter_lng, ev=None):
    if not node_data:
        return

    js_api_key = "e64f575fd50af118d93112f67c27523c"
    js_api_secret = "0705644930985c809183b3554a1f3cb0"

    if ev:
        event_time = time.strftime('%H:%M:%S', time.localtime(ev.start_time)) if ev.start_time else '--'
        distance = f"{ev.estimated_distance_km:.1f}" if ev.estimated_distance_km else '--'
        azimuth = f"{ev.p_wave_azimuth:.0f}" if ev.p_wave_azimuth is not None else '--'
        magnitude = f"{ev.magnitude:.1f}" if ev.magnitude else '--'
    else:
        event_time = distance = azimuth = '--'
        magnitude = '--'
    node_count = len(node_data)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>地震烈度分布</title>
    <style>
        body, html {{ margin: 0; height: 100%; }}
        #map {{ width: 100%; height: 100%; }}
        .info-card {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(0,0,0,0.75);
            color: white;
            padding: 15px;
            border-radius: 10px;
            font-family: Arial, sans-serif;
            font-size: 14px;
            z-index: 999;
            max-width: 250px;
        }}
        .info-card h3 {{ margin: 0 0 8px 0; color: #FF6B6B; }}
        .info-card p {{ margin: 3px 0; }}
    </style>
</head>
<body>
<div id="map"></div>
    <div class="info-card">
        <h3>⚠️ 地震预警信息</h3>
        <p>📅 时间: <span id="eventTime">{event_time}</span></p>
        <p>📏 震中距: <span id="distance">{distance}</span> km</p>
        <p>🧭 方位角: <span id="azimuth">{azimuth}</span>°</p>
        <p>📊 震级: <span id="magnitude">M{magnitude}</span></p>
        <p>📈 节点数: <span id="nodeCount">{node_count}</span></p>
    </div>
<script>
    window._AMapSecurityConfig = {{
        securityJsCode: '{js_api_secret}'
    }};
</script>
<script src="https://webapi.amap.com/maps?v=2.0&key={js_api_key}&plugin=AMap.HeatMap"></script>
<script>
(function() {{
    var epicenter = [{epicenter_lng}, {epicenter_lat}];
    var nodes = {json.dumps([{'lat': n['lat'], 'lng': n['lng'], 'int': n['intensity'], 'pga': round(float(n['pga']), 1)} for n in node_data], ensure_ascii=False)};

    // 初始化地图
    var map = new AMap.Map('map', {{
        zoom: 12,
        center: epicenter,
        mapStyle: 'amap://styles/darkblue'  // 深色风格，更有科技感
    }});

        var heatmapData = nodes.map(function(n) {{
        return {{ lng: n.lng, lat: n.lat, count: n.int * 10 }};
    }});
    heatmapData.max = 100;
    
    var heatmap = new AMap.HeatMap(map, {{
        radius: 40,
        opacity: [0, 0.8],
        gradient: {{
            0.2: '#4ecca3', 0.4: '#f5c518', 0.6: '#ff8c00', 0.8: '#ff4500', 1.0: '#e94560'
        }},
        dataSet: {{
            data: heatmapData,
            max: 100
        }}
    }});

    // ===== 震中标记（脉冲动画） =====
    var epicenterContent = '<div style="position:relative;width:40px;height:40px;">' +
        '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
        'width:40px;height:40px;background:rgba(233,69,96,0.4);border-radius:50%;' +
        'animation:pulse 1.5s infinite;"></div>' +
        '<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);' +
        'width:16px;height:16px;background:#e94560;border:3px solid white;border-radius:50%;"></div></div>';

    var epicenterMarker = new AMap.Marker({{
        position: epicenter,
        content: epicenterContent,
        offset: new AMap.Pixel(-20, -20),
        zIndex: 1000
    }});
    epicenterMarker.setMap(map);

    // ===== 节点烈度标记 =====
    nodes.forEach(function(node) {{
        var color = node.int >= 7 ? '#e94560' : node.int >= 5 ? '#ff8c00' : '#f5c518';
        var content = '<div style="background:' + color + ';color:white;padding:4px 10px;' +
            'border-radius:15px;font-size:13px;font-weight:bold;' +
            'box-shadow:0 2px 8px rgba(0,0,0,0.3);white-space:nowrap;">' +
            node.int + '度</div>';
        var marker = new AMap.Marker({{
            position: [node.lng, node.lat],
            content: content,
            offset: new AMap.Pixel(-20, -20)
        }});
        marker.setMap(map);

        // 点击弹出详细信息
        marker.on('click', function() {{
            var info = '<div style="padding:10px;font-size:14px;">' +
                '<h4>节点信息</h4>' +
                '<p>烈度: ' + node.int + '度</p>' +
                '<p>PGA: ' + node.pga + ' cm/s²</p>' +
                '<p>坐标: ' + node.lat.toFixed(4) + ', ' + node.lng.toFixed(4) + '</p>' +
                '</div>';
            var infoWindow = new AMap.InfoWindow({{
                content: info,
                offset: new AMap.Pixel(0, -30)
            }});
            infoWindow.open(map, [node.lng, node.lat]);
        }});
    }});

    // ===== 震中标签 =====
    var epicenterLabel = new AMap.Marker({{
        position: epicenter,
        label: {{
            content: '<div style="background:#e94560;color:white;padding:2px 8px;' +
                'border-radius:4px;font-size:12px;margin-top:-50px;">推定震中</div>',
            offset: new AMap.Pixel(-28, -50)
        }},
        icon: new AMap.Icon({{ size: new AMap.Size(1,1), image: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7' }})
    }});
    epicenterLabel.setMap(map);

        // ===== S波扩散动画（无限循环） =====
    
    var waveTimer = null;
    function startWaveAnimation() {{
        for (var i = 1; i <= 5; i++) {{
            (function(delay) {{
                setTimeout(function() {{
                    var circle = new AMap.Circle({{
                        center: epicenter,
                        radius: 5000,
                        strokeColor: '#e94560',
                        strokeWeight: 2,
                        strokeOpacity: 0.5,
                        fillColor: '#e94560',
                        fillOpacity: 0.05,
                        zIndex: 100
                    }});
                    circle.setMap(map);
                    
                    var radius = 5000;
                    var timer = setInterval(function() {{
                        radius += 3000;
                        circle.setRadius(radius);
                        circle.setOptions({{ strokeOpacity: Math.max(0, circle.getOptions().strokeOpacity - 0.08) }});
                        if (radius > 80000) {{
                            clearInterval(timer);
                            circle.setMap(null);
                        }}
                    }}, 100);
                }}, delay);
            }})(i * 600);
        }}
    }}
    startWaveAnimation();
    setInterval(startWaveAnimation, 5 * 600 + 8000);
    // 自适应视野
    map.setFitView();

    

}})();
</script>
</body>
</html>"""

    with open("shakemap.html", "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("✅ 烈度分布图已生成: shakemap.html")


def generate_shakemap_html_embed(node_data, epicenter_lat, epicenter_lng):
    """生成可嵌入邮件的轻量版地图"""
    js_api_key = "e64f575fd50af118d93112f67c27523c"
    js_api_secret = "0705644930985c809183b3554a1f3cb0"

    html = f"""
    <div id="map" style="width:100%;height:400px;border-radius:10px;"></div>
    <script>
        window._AMapSecurityConfig = {{
            securityJsCode: '{js_api_secret}'
        }};
    </script>
    <script src="https://webapi.amap.com/maps?v=2.0&key={js_api_key}&plugin=AMap.HeatMap"></script>
    <script>
    (function() {{
        var epicenter = [{epicenter_lng}, {epicenter_lat}];
        var nodes = {json.dumps([{'lat': n['lat'], 'lng': n['lng'], 'int': n['intensity']} for n in node_data], ensure_ascii=False)};

        var map = new AMap.Map('map', {{
            zoom: 12,
            center: epicenter
        }});

        // 震中标记
        new AMap.Marker({{
            position: epicenter,
            title: '推定震中',
            label: {{ content: '震中', offset: new AMap.Pixel(20, 0) }},
            icon: new AMap.Icon({{
                size: new AMap.Size(30, 30),
                image: 'https://webapi.amap.com/theme/v1.3/markers/n/mark_r.png'
            }})
        }}).setMap(map);

        // 节点标记
        nodes.forEach(function(node) {{
            var color = node.int >= 7 ? '#e94560' : node.int >= 5 ? '#ff8c00' : '#f5c518';
            new AMap.Marker({{
                position: [node.lng, node.lat],
                content: '<div style="background:' + color + ';color:white;padding:3px 8px;border-radius:12px;font-size:12px;">' + node.int + '度</div>',
                offset: new AMap.Pixel(-15, -15)
            }}).setMap(map);
        }});

        map.setFitView();
    }})();
    </script>
    """
    return html

def send_email_alert_html(subject, html_content):
        def _send():
            try:
                msg = MIMEMultipart('alternative')
                msg['From'] = QQ_EMAIL_SENDER
                msg['To'] = QQ_EMAIL_RECEIVER
                msg['Subject'] = Header(subject, 'utf-8')
                msg.attach(MIMEText(html_content, 'html', 'utf-8'))

                server = smtplib.SMTP(QQ_EMAIL_SMTP_SERVER, QQ_EMAIL_SMTP_PORT)
                server.starttls()
                server.login(QQ_EMAIL_SENDER, QQ_EMAIL_PASSWORD)
                server.send_message(msg)
                server.quit()
                logger.info(f"📧 HTML邮件已发送: {subject}")
            except Exception as e:
                logger.error(f"❌ 邮件发送失败: {e}")

        threading.Thread(target=_send, daemon=True).start()


def send_email_alert_with_attachment(subject, html_content, attachment_path):
    def _send():
        try:
            msg = MIMEMultipart()
            msg['From'] = QQ_EMAIL_SENDER
            msg['To'] = QQ_EMAIL_RECEIVER
            msg['Subject'] = Header(subject, 'utf-8')
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))

            # 添加 HTML 附件
            with open(attachment_path, 'rb') as f:
                att = MIMEText(f.read(), 'base64', 'utf-8')
                att["Content-Type"] = "text/html; charset=utf-8"
                att["Content-Disposition"] = f"attachment; filename=shakemap.html"
                msg.attach(att)

            server = smtplib.SMTP(QQ_EMAIL_SMTP_SERVER, QQ_EMAIL_SMTP_PORT)
            server.starttls()
            server.login(QQ_EMAIL_SENDER, QQ_EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"📧 邮件+附件已发送: {subject}")
        except Exception as e:
            logger.error(f"❌ 邮件发送失败: {e}")

    threading.Thread(target=_send, daemon=True).start()


def generate_shakemap_static(node_data, epicenter_lat, epicenter_lng):
    """生成高德静态地图，返回 HTML img 标签"""
    if not node_data or not epicenter_lat or not epicenter_lng:
        return '<p style="color:#888;">暂无烈度分布数据</p>'

    # 构建标记：震中红色 + 节点橙色
    markers = [f"mid,0xFF0000,A:{epicenter_lng},{epicenter_lat}"]
    for n in node_data:
        intensity = n['intensity']
        if intensity >= 7:
            color = "0xFF0000"  # 红色
        elif intensity >= 5:
            color = "0xFF8C00"  # 橙色
        else:
            color = "0x00BFFF"  # 深天蓝，不用 0x0000FF
        markers.append(f"mid,{color},C:{n['lng']},{n['lat']}")

    marker_str = "&markers=" + "|".join(markers)
    url = f"https://restapi.amap.com/v3/staticmap?size=800*500{marker_str}&key=a8c640d3f0ea6c654a66cc22e0ed6106"

    return f'<img src="{url}" style="width:100%;max-width:800px;border-radius:10px;"/>'

# ==================== 配置信息 ====================
RECEIVE_MQTT_BROKER = "w8381dbc.ala.cn-hangzhou.emqxsl.cn"
RECEIVE_MQTT_PORT = 8883
RECEIVE_MQTT_TOPIC = "earthquake/data"
RECEIVE_MQTT_USERNAME = "***"
RECEIVE_MQTT_PASSWORD = "***"
RECEIVE_MQTT_CA_CERT = "***"

SEND_MQTT_BROKER = "w8381dbc.ala.cn-hangzhou.emqxsl.cn"
SEND_MQTT_PORT = 8883
SEND_MQTT_TOPIC = "earthquake/alert"
SEND_MQTT_USERNAME = "***"
SEND_MQTT_PASSWORD = "***"
SEND_MQTT_CA_CERT = "***"

QQ_EMAIL_ENABLE = True
QQ_EMAIL_SENDER = "***"
QQ_EMAIL_PASSWORD = "***"
QQ_EMAIL_RECEIVER = "***"
QQ_EMAIL_SMTP_SERVER = "smtp.qq.com"
QQ_EMAIL_SMTP_PORT = 587

LOG_FILE = "sensor_trend_monitor.log"

ACCELERATION_THRESHOLD_G = 0.02
GRAVITY_THRESHOLD_LOW_G = 0.98
GRAVITY_THRESHOLD_HIGH_G = 1.00
ZERO_THRESHOLD = 100
SEISMIC_BUFFER_SIZE = 6000
ACTUAL_SENSITIVITY = 4096.0
TRIGGER_WINDOW = 15.0
CONFIRMATION_COUNT = 3
CONFIRMATION_TIME_WINDOW = 0.5
MIN_PHASE_POINTS = 500
MIN_EQT_POINTS = 800
MIN_FINAL_POINTS = 800

alert_queue = queue.Queue()
TEST_MODE = False
TEST_OUTPUT_FILE = "test_results.json"


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
    epicenter_lat: Optional[float] = None
    epicenter_lng: Optional[float] = None
    estimated_distance_km: Optional[float] = None
    p_wave_azimuth: Optional[float] = None  # ← 加这行
    epicenter_intensity: int = 0  # ← 加
    epicenter_magnitude: float = 0.0


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(LOG_FILE, encoding='utf-8'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger('seisbench').setLevel(logging.ERROR)
logging.getLogger('daspy').setLevel(logging.ERROR)


class EarthquakeMQTT:
    def __init__(self):
        self.client = None
        self.connected = False
        self.node_locations = {}
        self.connect()

    def connect(self):
        try:
            self.client = mqtt.Client(client_id="eq_sender_" + str(int(time.time())))
            self.client.username_pw_set(SEND_MQTT_USERNAME, SEND_MQTT_PASSWORD)
            if SEND_MQTT_CA_CERT:
                self.client.tls_set(ca_certs=SEND_MQTT_CA_CERT)
            self.client.on_connect = self.on_connect
            self.client.connect(SEND_MQTT_BROKER, SEND_MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"❌ 发送MQTT连接失败: {e}")
            self.connected = False

    def on_connect(self, client, userdata, flags, rc):
        self.connected = (rc == 0)
        logger.info("✅ 发送MQTT连接成功" if self.connected else f"❌ 失败: {rc}")

    def send_alert(self, level, confidence, intensity, node_id=None,
                   epicenter_lat=None, epicenter_lng=None):
        if not self.connected:
            self.connect()
            if not self.connected:
                return False
        try:
            payload = {
                "type": "trigger",
                "node": node_id or 0,
                "confidence": confidence,
                "intensity": intensity,
                "timestamp": int(time.time()),
            }
            if epicenter_lat and epicenter_lng:
                payload["gwLat"] = epicenter_lat
                payload["gwLng"] = epicenter_lng
                payload["isEpicenter"] = True
            for nid, (nlat, nlng) in self.node_locations.items():
                payload[f"node{nid}_lat"] = nlat
                payload[f"node{nid}_lng"] = nlng
            result = self.client.publish(SEND_MQTT_TOPIC, json.dumps(payload))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"✅ [MQTT] 预警已发送: 烈度{intensity}度, 震中={epicenter_lat},{epicenter_lng}")
                return True
        except Exception as e:
            logger.error(f"❌ [MQTT] 发送异常: {e}")
        return False

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


class MQTTReceiver:
    def __init__(self, callback):
        self.callback = callback
        self.client = None
        self.connected = False
        self.connect()

    def connect(self):
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
            self.client.on_connect = self.on_connect
            self.client.on_message = self.on_message
            self.client.username_pw_set(RECEIVE_MQTT_USERNAME, RECEIVE_MQTT_PASSWORD)
            try:

                self.client.tls_set(ca_certs=RECEIVE_MQTT_CA_CERT)
                logger.info("✅ 接收服务器SSL证书配置成功")

            except:
                self.client.tls_insecure_set(True)
                logger.info("⚠️ SSL配置失败，尝试无SSL连接")
            self.client.connect(RECEIVE_MQTT_BROKER, RECEIVE_MQTT_PORT, 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"❌ 接收MQTT连接失败: {e}")

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self.connected = True
            client.subscribe(RECEIVE_MQTT_TOPIC)
            client.subscribe("earthquake/data")
            client.subscribe("earthquake/alert")
            client.subscribe("earthquake/heartbeat")
            logger.info("📥 已订阅:  earthquake/data, earthquake/alert, earthquake/heartbeat")

    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            if topic == "earthquake/data":
                handle_pca_message(msg)
            elif topic == "earthquake/alert":
                handle_alert_message(msg)
            elif self.callback:
                self.callback(msg)
        except Exception as e:
            logger.error(f"处理消息错误: {e}")

    def disconnect(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


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


def show_alert(title, message, confidence, source):
    alert_queue.put((title, message, confidence, source))


import requests


def push_to_geohub(node_data, epicenter_lat, epicenter_lng):
    GEOHUB_API_URL = "https://restapi.amap.com/rest/lbs/geohub/geo/feature/bulkadd"
    GEOHUB_KEY = "a8c640d3f0ea6c654a66cc22e0ed6106"
    DATASET_ID = "25026909-91b3-483e-8598-2630045848d2"

    features = []
    for i, node in enumerate(node_data):
        features.append({
            "geometry": {
                "type": "Point",
                "coordinates": [node['lng'], node['lat']]  # 经度在前，纬度在后
            },
            "properties": {
                "node_index": i,
                "intensity": int(node['intensity']),
                "pga": round(float(node['pga']), 1),
                "is_epicenter": False,
                "timestamp": int(time.time())
            }
        })

    if epicenter_lat and epicenter_lng:
        features.append({
            "geometry": {
                "type": "Point",
                "coordinates": [float(epicenter_lng), float(epicenter_lat)]
            },
            "properties": {
                "node_index": -1,
                "intensity": int(max(n['intensity'] for n in node_data)),
                "pga": 0,
                "is_epicenter": True,
                "timestamp": int(time.time())
            }
        })

    payload = {
        "key": GEOHUB_KEY,
        "dataset_id": DATASET_ID,
        "features": features
    }

    try:
        resp = requests.post(GEOHUB_API_URL, json=payload, timeout=10)
        print(f"[GEOHUB] Response: {resp.status_code} {resp.text}")
    except Exception as e:
        logger.error(f"❌ GeoHUB推送失败: {e}")


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
        if horizontal > ACCELERATION_THRESHOLD_G or vertical > 0.02:
            if current_time - last_trigger_time < CONFIRMATION_TIME_WINDOW:
                trigger_count += 1
            else:
                trigger_count = 1
            last_trigger_time = current_time
            if trigger_count >= CONFIRMATION_COUNT:
                trigger_count = 0
                return True, f"水平={horizontal:.3f}g, 垂直={vertical:.3f}g"
            return False, f"确认中 ({trigger_count}/{CONFIRMATION_COUNT})"
        return False, "正常"
    except Exception as e:
        return False, f"检测错误: {e}"


def precise_local_check(sensor_data):
    try:
        required = ['AX', 'AY', 'AZ']
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
    thresholds = [
        (0.0022, 1), (0.0063, 2), (0.018, 3), (0.045, 4), (0.089, 5),
        (0.177, 6), (0.353, 7), (0.707, 8), (1.414, 9), (2.5, 10), (float('inf'), 11)
    ]
    descs = ["无感", "微感", "有感", "室内有感", "室外有感", "惊慌",
             "房屋损坏", "建筑物破坏", "建筑物倒塌", "毁灭性", "灾难性"]
    for i, (thresh, level) in enumerate(thresholds):
        if max_g < thresh:
            return level, descs[i]
    return 11, "灾难性"


class DASPyDenoiser:
    def __init__(self, sampling_rate=100, method='wavelet'):
        self.sampling_rate = sampling_rate
        self.dt = 1.0 / sampling_rate
        self.method = method

    def __call__(self, data_3c):
        return self.denoise(data_3c)

    def _wavelet_denoise_channel(self, channel_data):
        try:
            import pywt
            coeffs = pywt.wavedec(channel_data, 'db4', level=4)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))
            coeffs_thresh = [coeffs[0]]
            for i in range(1, len(coeffs)):
                coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            denoised = pywt.waverec(coeffs_thresh, 'db4')
            if len(denoised) > len(channel_data):
                denoised = denoised[:len(channel_data)]
            elif len(denoised) < len(channel_data):
                denoised = np.pad(denoised, (0, len(channel_data) - len(denoised)), 'edge')
            return denoised
        except ImportError:
            return channel_data

    def _wavelet_denoise(self, data_3c):
        denoised = np.zeros_like(data_3c)
        for ch in range(3):
            denoised[ch] = self._wavelet_denoise_channel(data_3c[ch])
        return denoised

    def denoise(self, data_3c):
        if data_3c.ndim != 2 or data_3c.shape[0] != 3:
            return data_3c
        if data_3c.shape[1] < 50:
            return data_3c
        return self._wavelet_denoise(data_3c)

    def denoise_safe(self, data_3c):
        try:
            result = self.denoise(data_3c)
            return result if result.shape == data_3c.shape else data_3c
        except:
            return data_3c


class PhaseNetReal:
    def __init__(self):
        try:
            self.model = sbm.PhaseNet.from_pretrained("geofon")
            self.model.eval()
            self.threshold = 0.5
            self.in_samples = 3000
            logger.info("✅ PhaseNet加载成功")
        except Exception as e:
            logger.error(f"❌ PhaseNet加载失败: {e}")
            self.model = None

    def analyze(self, waveform_data):
        if self.model is None or len(waveform_data) < 300:
            return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                    's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                    'detection_score': 0}

        try:
            scale = 9.8 / ACTUAL_SENSITIVITY
            z_vals, n_vals, e_vals = [], [], []
            for d in waveform_data:
                z_vals.append(float(d.get('AZ', 0)) * scale)
                n_vals.append(float(d.get('AY', 0)) * scale)
                e_vals.append(float(d.get('AX', 0)) * scale)

            z = np.array(z_vals, dtype=np.float32)
            n = np.array(n_vals, dtype=np.float32)
            e = np.array(e_vals, dtype=np.float32)
            z -= np.mean(z);
            n -= np.mean(n);
            e -= np.mean(e)

            target = self.in_samples
            if len(z) > target:
                z, n, e = z[-target:], n[-target:], e[-target:]
            elif len(z) < target:
                pad = target - len(z)
                z = np.pad(z, (0, pad), 'edge')
                n = np.pad(n, (0, pad), 'edge')
                e = np.pad(e, (0, pad), 'edge')

            z /= (np.std(z) + 1e-10)
            n /= (np.std(n) + 1e-10)
            e /= (np.std(e) + 1e-10)

            waveform = np.stack([z, n, e], axis=0)[np.newaxis, :, :]
            device = next(self.model.parameters()).device
            waveform_tensor = torch.from_numpy(waveform).float().to(device)

            with torch.no_grad():
                predictions = self.model(waveform_tensor)

            if predictions is not None and isinstance(predictions, torch.Tensor):
                pred_np = predictions.cpu().numpy()  # (1, 3, T)
                p_probs = pred_np[0, 0, :]  # P 波概率序列
                s_probs = pred_np[0, 1, :]  # S 波概率序列

                p_idx = int(np.argmax(p_probs))
                p_conf = float(p_probs[p_idx])

                # S 波必须在 P 波之后
                if p_idx < len(s_probs) - 1:
                    s_idx = p_idx + int(np.argmax(s_probs[p_idx:]))
                else:
                    s_idx = p_idx
                s_conf = float(s_probs[s_idx])

                # 计算走时差
                s_p_samples = s_idx - p_idx
                s_p_sec = s_p_samples / 100.0  # 假设 100Hz 采样
                dist_km = s_p_sec * 8.0  # 粗略震中距估算

                detection_score = float(np.max(pred_np[0, 0, :]))

                logger.info(f"📍 PhaseNet: P={p_idx / 100.0:.2f}s S={s_idx / 100.0:.2f}s "
                            f"ΔTs-p={s_p_sec:.2f}s 震中距≈{dist_km:.1f}km")

                return {
                    'p_arrival_sec': p_idx / 100.0,
                    's_arrival_sec': s_idx / 100.0,
                    'p_confidence': p_conf,
                    's_confidence': s_conf,
                    's_p_diff_sec': s_p_sec,
                    'estimated_distance_km': dist_km,
                    'detection_score': detection_score
                }

            return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                    's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                    'detection_score': 0}
        except Exception as e:
            logger.error(f"PhaseNet分析错误: {e}")
            return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                    's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                    'detection_score': 0}


class EQTReal:
    def __init__(self):
        try:
            self.model = sbm.EQTransformer.from_pretrained("geofon")
            self.model.eval()
            self.in_samples = 6000
            logger.info("✅ EQTransformer加载成功")
        except Exception as e:
            logger.error(f"❌ EQTransformer加载失败: {e}")
            self.model = None

    def analyze(self, waveform_data):
        if self.model is None or len(waveform_data) < 500:
            return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                    's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                    'detection_score': 0, 'magnitude': 0}

        try:
            scale = 9.8 / ACTUAL_SENSITIVITY
            z_vals, n_vals, e_vals = [], [], []
            for d in waveform_data:
                z_vals.append(float(d.get('AZ', 0)) * scale)
                n_vals.append(float(d.get('AY', 0)) * scale)
                e_vals.append(float(d.get('AX', 0)) * scale)

            z = np.array(z_vals, dtype=np.float32)
            n = np.array(n_vals, dtype=np.float32)
            e = np.array(e_vals, dtype=np.float32)
            z -= np.mean(z);
            n -= np.mean(n);
            e -= np.mean(e)

            target = self.in_samples
            if len(z) > target:
                z, n, e = z[-target:], n[-target:], e[-target:]
            elif len(z) < target:
                pad = target - len(z)
                z = np.pad(z, (0, pad), 'edge')
                n = np.pad(n, (0, pad), 'edge')
                e = np.pad(e, (0, pad), 'edge')

            z /= (np.std(z) + 1e-10)
            n /= (np.std(n) + 1e-10)
            e /= (np.std(e) + 1e-10)

            waveform = np.stack([z, n, e], axis=0)[np.newaxis, :, :]
            device = next(self.model.parameters()).device
            waveform_tensor = torch.from_numpy(waveform).float().to(device)

            with torch.no_grad():
                predictions = self.model(waveform_tensor)


                pred = predictions  # 元组，不用再取 [0]
                if isinstance(pred, tuple) and len(pred) == 3:
                    # EQT 输出: (detection, P, S) 三个 (1, T) 张量
                    detection_probs = pred[0].cpu().numpy().flatten()
                    p_probs = pred[1].cpu().numpy().flatten()
                    s_probs = pred[2].cpu().numpy().flatten()
                elif isinstance(pred, torch.Tensor):
                    pred_np = pred.cpu().numpy()
                    if pred_np.ndim == 3 and pred_np.shape[1] >= 3:
                        detection_probs = pred_np[0, 0, :]
                        p_probs = pred_np[0, 1, :]
                        s_probs = pred_np[0, 2, :]
                    elif pred_np.ndim == 2 and pred_np.shape[0] >= 3:
                        detection_probs = pred_np[0, :]
                        p_probs = pred_np[1, :]
                        s_probs = pred_np[2, :]
                    else:
                        logger.error(f"EQT输出维度异常: {pred_np.shape}")
                        return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                                's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                                'detection_score': 0, 'magnitude': 0}
                else:
                    return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                            's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                            'detection_score': 0, 'magnitude': 0}


                p_idx = int(np.argmax(p_probs))
                p_conf = float(p_probs[p_idx])

                if p_idx < len(s_probs) - 1:
                    s_idx = p_idx + int(np.argmax(s_probs[p_idx:]))
                else:
                    s_idx = p_idx
                s_conf = float(s_probs[s_idx])

                s_p_samples = s_idx - p_idx
                s_p_sec = s_p_samples / 100.0
                dist_km = s_p_sec * 8.0

                detection_score = float(np.max(detection_probs))

                recent = waveform_data[-100:]
                max_ax = max(abs(d.get('AX', 0)) for d in recent) / ACTUAL_SENSITIVITY
                max_ay = max(abs(d.get('AY', 0)) for d in recent) / ACTUAL_SENSITIVITY

                logger.info(f"📍 EQT: P={p_idx / 100.0:.2f}s S={s_idx / 100.0:.2f}s "
                            f"ΔTs-p={s_p_sec:.2f}s 震中距≈{dist_km:.1f}km")

                return {
                    'p_arrival_sec': p_idx / 100.0,
                    's_arrival_sec': s_idx / 100.0,
                    'p_confidence': p_conf,
                    's_confidence': s_conf,
                    's_p_diff_sec': s_p_sec,
                    'estimated_distance_km': dist_km,
                    'detection_score': detection_score,
                    'magnitude': quick_magnitude(max_ax, max_ay)
                }


        except Exception as e:
            logger.error(f"EQT分析错误: {e}")
            return {'p_arrival_sec': None, 's_arrival_sec': None, 'p_confidence': 0,
                    's_confidence': 0, 's_p_diff_sec': None, 'estimated_distance_km': None,
                    'detection_score': 0, 'magnitude': 0}


class FinalValidator:
    def __init__(self, phase_model=None, eqt_model=None):
        self.models = {}
        self.threshold = 0.5

        if phase_model:
            self.models['phasenet'] = phase_model
        else:
            try:
                self.models['phasenet'] = PhaseNetReal()
            except:
                pass

        if eqt_model:
            self.models['eqt'] = eqt_model
        else:
            try:
                self.models['eqt'] = EQTReal()
            except:
                pass

    def analyze(self, data_points):
        if not self.models or len(data_points) < 1000:
            return {'is_earthquake': False, 'confidence': 0, 'magnitude': 0}

        votes = []
        confidences = []

        # 1. PhaseNet 投票
        if 'phasenet' in self.models:
            result = self.models['phasenet'].analyze(data_points)
            if isinstance(result, dict):
                score = result.get('detection_score', 0)
                votes.append(1 if score > self.threshold else 0)
                confidences.append(score)

        # 2. EQT 投票
        if 'eqt' in self.models:
            result = self.models['eqt'].analyze(data_points)
            if isinstance(result, dict):
                score = result.get('detection_score', 0)
                votes.append(1 if score > self.threshold else 0)
                confidences.append(score)

        if not votes:
            return {'is_earthquake': False, 'confidence': 0, 'magnitude': 0}

        total = len(votes)
        yes = sum(votes)

        # 计算最终置信度（百分比）
        confidence = (yes / total) * 100 * np.mean(confidences) if confidences else 0

        # 计算震级
        max_ax = max(abs(d.get('AX', 0)) for d in data_points) / ACTUAL_SENSITIVITY
        max_ay = max(abs(d.get('AY', 0)) for d in data_points) / ACTUAL_SENSITIVITY

        return {
            'is_earthquake': yes >= total / 2,
            'confidence': round(confidence, 1),
            'magnitude': quick_magnitude(max_ax, max_ay),
            'details': f"投票: {yes}/{total}"
        }


class EarthquakeStateMachine:
    def __init__(self, mqtt_sender):
        self.lock = threading.Lock()
        self.state = AlertState.IDLE
        self.current_event = None
        self.alerts_sent = {level: False for level in AlertLevel}
        self.data_buffer = deque(maxlen=SEISMIC_BUFFER_SIZE)
        self.total_points_processed = 0
        self.total_triggers = 0
        self.mqtt_sender = mqtt_sender
        self.orange_alert_times = deque(maxlen=5)
        self.orange_alert_count = 0
        self.orange_window = 45
        self.fast_upgrade_triggered = False
        self.test_mode = True
        self.trigger_times = []
        self.alert_times = {level: [] for level in AlertLevel}
        self.pca_buffer = {}
        self.node_locations = {}
        self.denoiser = DASPyDenoiser(sampling_rate=100, method='wavelet')
        self.phase = PhaseNetReal()
        self.eqt = EQTReal()
        self.final = FinalValidator(self.phase, self.eqt)
        self.vote_history = deque(maxlen=10)
        self._suspicious_count = 0
        self._last_reset_time = 0
        self._orange_cooldown = 30
        logger.info("✅ 状态机初始化完成")

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
        elif current_time - self.current_event.last_trigger_time < TRIGGER_WINDOW:
            self.current_event.last_trigger_time = current_time
        else:
            self._reset()
            self.current_event = EventData(start_time=current_time, last_trigger_time=current_time, p_arrival=current_time)

    def _get_event_data(self):
        if not self.current_event:
            return []
        start = self.current_event.start_time
        end = self.current_event.last_trigger_time
        return [item['data'] for item in self.data_buffer if start <= item['time'] <= end]

    def _count_event_points(self):
        if not self.current_event:
            return 0
        start = self.current_event.start_time
        end = self.current_event.last_trigger_time
        return sum(1 for item in self.data_buffer if start <= item['time'] <= end)

    def _get_recent_waveform(self, seconds=3):
        if len(self.data_buffer) < 100:
            return None
        n_points = int(seconds * 100)
        recent = list(self.data_buffer)[-n_points:]
        scale = 9.8 / ACTUAL_SENSITIVITY
        z_vals, n_vals, e_vals = [], [], []
        for item in recent:
            d = item['data']
            z_vals.append(float(d.get('AZ', 0)) * scale)
            n_vals.append(float(d.get('AY', 0)) * scale)
            e_vals.append(float(d.get('AX', 0)) * scale)
        return np.array([z_vals, n_vals, e_vals], dtype=np.float32)


    def _preprocess_with_denoise(self, event_data):
        scale = 9.8 / ACTUAL_SENSITIVITY
        z_vals, n_vals, e_vals = [], [], []
        for d in event_data:
            z_vals.append(float(d.get('AZ', 0)) * scale)
            n_vals.append(float(d.get('AY', 0)) * scale)
            e_vals.append(float(d.get('AX', 0)) * scale)
        waveform = np.array([z_vals, n_vals, e_vals], dtype=np.float32)
        waveform -= np.mean(waveform, axis=1, keepdims=True)
        try:
            denoised = self.denoiser.denoise_safe(waveform)
            return denoised if denoised.shape == waveform.shape else waveform
        except:
            return waveform



    def _two_way_vote(self, waveform, event_data=None):
        phase_result = self.phase.analyze(event_data) if event_data else None
        eqt_result = self.eqt.analyze(event_data) if event_data else None

        if not phase_result and not eqt_result:
            return 'noise', 0, {}

        # ===== 融合 P/S 波到时 =====
        p_times, s_times, dists = [], [], []
        p_weight, s_weight = 0, 0

        if phase_result and phase_result['p_arrival_sec'] is not None:
            p_times.append(phase_result['p_arrival_sec'])
            s_times.append(phase_result['s_arrival_sec'])
            if phase_result['estimated_distance_km']:
                dists.append(phase_result['estimated_distance_km'])
            p_weight += phase_result['p_confidence']

        if eqt_result and eqt_result['p_arrival_sec'] is not None:
            p_times.append(eqt_result['p_arrival_sec'])
            s_times.append(eqt_result['s_arrival_sec'])
            if eqt_result['estimated_distance_km']:
                dists.append(eqt_result['estimated_distance_km'])
            p_weight += eqt_result['p_confidence']

        # 加权平均
        avg_p = np.average(p_times, weights=[phase_result['p_confidence'], eqt_result['p_confidence']]) if len(
            p_times) == 2 else p_times[0] if p_times else None
        avg_s = np.average(s_times, weights=[phase_result['s_confidence'], eqt_result['s_confidence']]) if len(
            s_times) == 2 else s_times[0] if s_times else None
        avg_dist = np.mean(dists) if dists else None

        # 地震判定
        phase_detect = phase_result['detection_score'] if phase_result else 0
        eqt_detect = eqt_result['detection_score'] if eqt_result else 0
        votes = (1 if phase_detect > 0.5 else 0) + (1 if eqt_detect > 0.75 else 0)
        score = phase_detect * 0.4 + eqt_detect * 0.6

        pick_info = {
            'p_arrival': avg_p,
            's_arrival': avg_s,
            'distance_km': avg_dist,
            'phase_p_conf': phase_result['p_confidence'] if phase_result else 0,
            'eqt_p_conf': eqt_result['p_confidence'] if eqt_result else 0
        }
        logger.info(f"📊 置信度: PhaseNet P={pick_info['phase_p_conf']:.2f} EQT P={pick_info['eqt_p_conf']:.2f}")

        if votes >= 2 and eqt_detect > 0.95:
            return 'earthquake', score, pick_info
        elif votes >= 1 and score > 0.8:
            return 'earthquake', score, pick_info
        elif votes >= 1 and score > 0.65:
            return 'suspicious', score, pick_info
        return 'noise', score, pick_info

    def _handle_idle(self):
        # 冷却期内不触发
        if time.time() - self._last_reset_time < self._orange_cooldown:
            return
        if not self.current_event or self._count_event_points() < MIN_PHASE_POINTS:
            return
        event_data = self._get_event_data()
        if len(event_data) < MIN_PHASE_POINTS:
            return
        denoised = self._preprocess_with_denoise(event_data)
        decision, score, pick_info = self._two_way_vote(denoised, event_data)
        # ===== 立刻记录已有信息 =====
        if pick_info.get('distance_km'):
            self.current_event.estimated_distance_km = pick_info['distance_km']
        if pick_info.get('p_arrival') and denoised.shape[1] > int(pick_info['p_arrival'] * 100):
            p_idx = int(pick_info['p_arrival'] * 100)
            ax_p = denoised[2, p_idx]
            ay_p = denoised[1, p_idx]
            azimuth = math.degrees(math.atan2(ay_p, ax_p))
            if azimuth < 0:
                azimuth += 360
            self.current_event.p_wave_azimuth = azimuth
        # ==============================
            # ===== 推算震中位置 =====
            if self.current_event.estimated_distance_km and self.node_locations:
                ref_node = list(self.node_locations.values())[0]
                ref_lat, ref_lng = ref_node
                az_rad = math.radians(azimuth)
                dist_km = self.current_event.estimated_distance_km
                d_lat = dist_km * math.cos(az_rad) / 111.32
                d_lng = dist_km * math.sin(az_rad) / (111.32 * math.cos(math.radians(ref_lat)))
                self.current_event.epicenter_lat = ref_lat + d_lat
                self.current_event.epicenter_lng = ref_lng + d_lng
        # ==============================

        # ===== 计算烈度 =====
        if self.current_event.max_ax > 0 or self.current_event.max_ay > 0:
            intensity, desc = calculate_intensity(
                self.current_event.max_ax / ACTUAL_SENSITIVITY,
                self.current_event.max_ay / ACTUAL_SENSITIVITY
            )
            self.current_event.intensity = intensity
            # ===== 推算震中烈度和震级 =====
            if self.current_event.estimated_distance_km and intensity > 0:
                dist = self.current_event.estimated_distance_km
                epic_int = intensity + 1.5 * math.log10(max(dist, 1) / 10.0)
                epic_int = max(1, min(12, int(round(epic_int))))
                epic_mag = 0.6 * epic_int + 1.5
                self.current_event.epicenter_intensity = epic_int
                self.current_event.epicenter_magnitude = round(epic_mag, 1)
                logger.info(f"📊 震中烈度: {intensity}→{epic_int}度, 震级: M{epic_mag}")
            self.current_event.description = desc
            self.current_event.magnitude = quick_magnitude(
                self.current_event.max_ax / ACTUAL_SENSITIVITY,
                self.current_event.max_ay / ACTUAL_SENSITIVITY
            )
        # ==================
        if decision == 'earthquake':
            self._send_alert(AlertLevel.ORANGE, f"双模型确认地震", score * 100, "双模型")
            self.state = AlertState.EQT_CONFIRM
        elif decision == 'suspicious':
            self._send_alert(AlertLevel.YELLOW, f"疑似地震信号", score * 100, "双模型")
            self.state = AlertState.P_ALERT
        else:
            self._reset()

    def _handle_p_alert(self):
        if not self.current_event or time.time() - self.current_event.start_time > 60:
            self._reset()
            return
        if self._count_event_points() < MIN_EQT_POINTS:
            return
        event_data = self._get_event_data()
        denoised = self._preprocess_with_denoise(event_data)

        decision, score, pick_info = self._two_way_vote(denoised, event_data)
        # ===== 立刻记录已有信息 =====
        if pick_info.get('distance_km'):
            self.current_event.estimated_distance_km = pick_info['distance_km']
        if pick_info.get('p_arrival') and denoised.shape[1] > int(pick_info['p_arrival'] * 100):
            p_idx = int(pick_info['p_arrival'] * 100)
            ax_p = denoised[2, p_idx]
            ay_p = denoised[1, p_idx]
            azimuth = math.degrees(math.atan2(ay_p, ax_p))
            if azimuth < 0:
                azimuth += 360
            self.current_event.p_wave_azimuth = azimuth
        # ==============================
        if decision == 'earthquake':
            self._send_alert(AlertLevel.ORANGE, f"二次确认地震", score * 100, "双模型")
            self.state = AlertState.EQT_CONFIRM
        elif decision == 'suspicious':
            # 连续 suspicious，累计计数
            self._suspicious_count += 1
            if self._suspicious_count >= 3:  # 连续 3 次 suspicious 自动升级
                self._send_alert(AlertLevel.ORANGE, f"持续可疑信号升级", score * 100, "双模型")
                self.state = AlertState.EQT_CONFIRM
        elif decision == 'noise':
            self._reset()

    def _handle_eqt_confirm(self):
        if not self.current_event or self.current_event.final_time:
            return
        if self._count_event_points() < MIN_FINAL_POINTS:
            return

        event_data = self._get_event_data()
        denoised = self._preprocess_with_denoise(event_data)
        scale = ACTUAL_SENSITIVITY / 9.8
        denoised_event_data = []
        for i in range(len(event_data)):
            denoised_event_data.append({
                'AX': denoised[2, i] * scale, 'AY': denoised[1, i] * scale, 'AZ': denoised[0, i] * scale
            })
            # ===== 震相拾取 + 震中参数估算 =====
        decision, score, pick_info = self._two_way_vote(denoised, denoised_event_data)

        # 1. 记录震中距
        if pick_info.get('distance_km'):
            self.current_event.estimated_distance_km = pick_info['distance_km']

        # 2. 计算方位角（P波初动方向）
        if pick_info.get('p_arrival') is not None:
            p_idx = int(pick_info['p_arrival'] * 100)
            if p_idx < denoised.shape[1]:
                ax_p = denoised[2, p_idx]  # 东向
                ay_p = denoised[1, p_idx]  # 北向
                azimuth = math.degrees(math.atan2(ay_p, ax_p))
                if azimuth < 0:
                    azimuth += 360
                self.current_event.p_wave_azimuth = azimuth
                logger.info(f"🧭 P波方位角: {azimuth:.1f}°")

        # 3. 如果有方位角 + 距离，推算震中
        if self.current_event.estimated_distance_km and self.current_event.p_wave_azimuth:
            # 从网关位置 + 距离 + 方位角推算震中
            if self.node_locations:
                # 取第一个节点的位置作为参考点
                ref_node = list(self.node_locations.values())[0]
                ref_lat, ref_lng = ref_node

                # 方位角转弧度（正北为0，顺时针）
                az_rad = math.radians(self.current_event.p_wave_azimuth)

                # 距离转经纬度偏移
                dist_km = self.current_event.estimated_distance_km
                d_lat = dist_km * math.cos(az_rad) / 111.32
                d_lng = dist_km * math.sin(az_rad) / (111.32 * math.cos(math.radians(ref_lat)))

                est_lat = ref_lat + d_lat
                est_lng = ref_lng + d_lng

                self.current_event.epicenter_lat = est_lat
                self.current_event.epicenter_lng = est_lng
                logger.info(f"📍 方位角+距离推算震中: {est_lat:.4f}, {est_lng:.4f}")
        # =========================================


        # =========================================

        result = self.final.analyze(denoised_event_data)
        if result and result['is_earthquake']:
            self.current_event.final_time = time.time()
            # 使用事件记录的最大加速度来计算震级
            self.current_event.magnitude = quick_magnitude(
                self.current_event.max_ax / ACTUAL_SENSITIVITY,
                self.current_event.max_ay / ACTUAL_SENSITIVITY
            )
            # 转换 float32 为 Python 原生 float
            if isinstance(self.current_event.magnitude, (np.float32, np.float64)):
                self.current_event.magnitude = float(self.current_event.magnitude)

            self.current_event.confidence = float(result['confidence'])
            intensity, desc = calculate_intensity(
                self.current_event.max_ax / ACTUAL_SENSITIVITY,
                self.current_event.max_ay / ACTUAL_SENSITIVITY
            )
            if self.current_event.estimated_distance_km and intensity > 0:
                dist = self.current_event.estimated_distance_km
                epic_int = intensity + 1.5 * math.log10(max(dist, 1) / 10.0)
                epic_int = max(1, min(12, int(round(epic_int))))
                epic_mag = 0.6 * epic_int + 1.5
                self.current_event.epicenter_intensity = epic_int
                self.current_event.epicenter_magnitude = round(epic_mag, 1)
            self.current_event.intensity = intensity
            self.current_event.description = desc
            # ===== 震中推算：方位角+距离优先，多节点加权次之 =====
            if self.current_event.estimated_distance_km and self.current_event.p_wave_azimuth:
                # 单台站：用距离 + 方位角推算震中
                # 取第一个已知节点作为参考点
                if self.node_locations:
                    ref_node = list(self.node_locations.values())[0]
                    ref_lat, ref_lng = ref_node

                    az_rad = math.radians(self.current_event.p_wave_azimuth)
                    dist_km = self.current_event.estimated_distance_km
                    d_lat = dist_km * math.cos(az_rad) / 111.32
                    d_lng = dist_km * math.sin(az_rad) / (111.32 * math.cos(math.radians(ref_lat)))

                    self.current_event.epicenter_lat = ref_lat + d_lat
                    self.current_event.epicenter_lng = ref_lng + d_lng
                    logger.info(
                        f"📍 方位角+距离推算震中: {self.current_event.epicenter_lat:.4f}, {self.current_event.epicenter_lng:.4f}")
                else:
                    self.current_event.epicenter_lat, self.current_event.epicenter_lng = estimate_epicenter(
                        self.node_locations, self.pca_buffer)
            elif len(self.pca_buffer) >= 2:
                # 多节点：加权平均
                self.current_event.epicenter_lat, self.current_event.epicenter_lng = estimate_epicenter(
                    self.node_locations, self.pca_buffer)
            else:
                self.current_event.epicenter_lat, self.current_event.epicenter_lng = None, None
            self._send_alert(AlertLevel.RED, f"最终确认：烈度{intensity}度", result['confidence'], "SeisBench")

            self.state = AlertState.FINAL
        else:
            self._reset()

    def _send_alert(self, level, message, confidence, source):
        if self.alerts_sent[level]:
            return
        self.alerts_sent[level] = True
        titles = {AlertLevel.YELLOW: "🟡 P波警觉", AlertLevel.ORANGE: "🟠 地震预警", AlertLevel.RED: "🔴 地震最终确认"}
        show_alert(titles[level], message, confidence, source)

        if level == AlertLevel.RED:
            # 只有 RED 级别发详细报告
            self._send_final_email()
        elif level == AlertLevel.ORANGE:
            ev = self.current_event
            if ev:
                dist = float(ev.estimated_distance_km) if ev.estimated_distance_km else 0
                az = float(ev.p_wave_azimuth) if ev.p_wave_azimuth else 0
                mag = ev.epicenter_magnitude if ev.epicenter_magnitude > 0 else ev.magnitude

                content = f"""🟠 地震预警 (ORANGE)
            {'=' * 20}
            📅 时间: {time.strftime('%H:%M:%S', time.localtime(ev.start_time))}
            📏 震中距: {dist:.1f} km
            🧭 方位角: {az:.0f}°
            📊 震级估算: M{mag:.1f}
            📈 烈度: {ev.intensity}度（{ev.description}）
            📌 置信度: {confidence:.1f}%
            📍 推算震中: {ev.epicenter_lat:.4f}, {ev.epicenter_lng:.4f}
            {'=' * 20}
            """
            else:
                content = f"🟠 地震预警\n置信度: {confidence:.1f}%"

            send_email_alert("🟠 地震预警", content)

            # MQTT 发送
            if self.mqtt_sender and ev:
                self.mqtt_sender.send_alert(
                    level=2,
                    confidence=confidence,
                    intensity=ev.intensity,
                    epicenter_lat=ev.epicenter_lat,
                    epicenter_lng=ev.epicenter_lng
                )
            print(f"[DEBUG ORANGE] 离开ORANGE分支")

    def _send_final_email(self):
            if not self.current_event:
                return
            ev = self.current_event

            # 获取节点烈度数据
            node_data = []
            if len(self.pca_buffer) >= 1:
                node_data = get_node_intensities(self.pca_buffer, self.node_locations)

            map_html = generate_shakemap_static(node_data, ev.epicenter_lat, ev.epicenter_lng)
            mag = ev.epicenter_magnitude if ev.epicenter_magnitude > 0 else ev.magnitude

            # 构建节点烈度表格
            node_table = ""
            if node_data:
                for n in node_data:
                    node_table += f"<tr><td>Node</td><td>({n['lat']:.4f},{n['lng']:.4f})</td><td>{n['intensity']}度</td><td>{n['pga']:.1f} cm/s²</td></tr>"

            # 构建 HTML 邮件内容
            content = f"""
    <html><body>
    <h2>🔴 地震最终确认报告</h2>
    <hr>
    <table>
    <tr><td>📅 时间：</td><td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ev.start_time))}</td></tr>
    <tr><td>📏 震中距：</td><td>{ev.estimated_distance_km:.1f} km (S-P走时差法)</td></tr>
    <tr><td>🧭 方位角：</td><td>{ev.p_wave_azimuth:.0f}°</td></tr>
    <tr><td>📊 震级：</td><td>M{mag:.1f}</td></tr>
    <tr><td>📈 烈度：</td><td>{ev.intensity}度（{ev.description}）</td></tr>
    <tr><td>📌 置信度：</td><td>{ev.confidence:.1f}%</td></tr>
    <tr><td>📍 推算震中：</td><td>{ev.epicenter_lat:.4f}, {ev.epicenter_lng:.4f}</td></tr>
    </table>
    <hr>
    <h3>📊 峰值加速度</h3>
    <table>
    <tr><td>├─ X轴：</td><td>{ev.max_ax / ACTUAL_SENSITIVITY:.3f}g</td></tr>
    <tr><td>├─ Y轴：</td><td>{ev.max_ay / ACTUAL_SENSITIVITY:.3f}g</td></tr>
    <tr><td>└─ Z轴：</td><td>{ev.max_az / ACTUAL_SENSITIVITY:.3f}g</td></tr>
    </table>
    <hr>
    <h3>🌋 烈度分布图</h3>
    {map_html}
    <hr>
    <h3>📊 节点烈度分布</h3>
    <table border="1" cellpadding="5" cellspacing="0" style="border-collapse:collapse;width:100%">
    <tr><th>节点</th><th>坐标</th><th>烈度</th><th>PGA</th></tr>
    {node_table}
    </table>
    <hr>
    <p><small>本邮件由地震预警系统自动发送</small></p>
    </body></html>
    """

            # 发送 HTML 邮件
            send_email_alert_with_attachment(
                "🔴 地震最终确认报告",
                content,
                "shakemap.html"
            )

            # MQTT 推送
            if self.mqtt_sender:
                self.mqtt_sender.send_alert(level=3, confidence=ev.confidence, intensity=ev.intensity,
                                            epicenter_lat=ev.epicenter_lat, epicenter_lng=ev.epicenter_lng)

            # 生成离线 HTML 地图文件（备份）
            if node_data:
                try:
                    generate_shakemap_html(node_data, ev.epicenter_lat, ev.epicenter_lng, ev)
                except Exception as e:
                    logger.error(f"❌ HTML地图生成失败: {e}")

    def _reset(self):
        self.state = AlertState.IDLE
        self.current_event = None
        self.alerts_sent = {level: False for level in AlertLevel}
        self.orange_alert_times.clear()
        self.orange_alert_count = 0
        self.fast_upgrade_triggered = False
        self._suspicious_count = 0
        self._last_reset_time = time.time()
        self.state = AlertState.IDLE


def handle_pca_message(msg):
    global state_machine

    try:
        data = json.loads(msg.payload.decode('utf-8'))
        print(f"[DEBUG] JSON解析成功: type={data.get('type')}, node={data.get('node')}")
        if data.get('type') != 'pca_forward':
            return
        node_id = data.get('node', 0)
        frame_type = data.get('frame', 'pred')
        pca_coeffs = data.get('pca', [])
        norm_factors = data.get('norm_factors', None)
        if norm_factors and len(norm_factors) == 3:
            norm_factors = [float(f) for f in norm_factors]
        else:
            norm_factors = None
        lat = data.get('lat', 0)
        lng = data.get('lng', 0)
        if not pca_coeffs:
            return
        waveform = pca_inverse(pca_coeffs, frame_type)
        # ===== 预测帧反归一化（在存入缓冲区和注入状态机之前） =====
        norm_factors = data.get('norm_factors', None)
        if frame_type == 'pred' and norm_factors and len(norm_factors) == 3:
            norm_factors = [float(f) for f in norm_factors]
            for c in range(3):
                if norm_factors[c] > 1e-10:  # 防止除零
                    waveform[c] *= norm_factors[c]
            logger.debug(f"📊 Node{node_id} pred 反归一化: factors={norm_factors}")
        # =========================================================
        if lat and lng:
            state_machine.node_locations[node_id] = (lat, lng)
            state_machine.mqtt_sender.node_locations[node_id] = (lat, lng)
        if node_id not in state_machine.pca_buffer:
            state_machine.pca_buffer[node_id] = deque(maxlen=50)
        state_machine.pca_buffer[node_id].append({
            'timestamp': data.get('timestamp', 0),
            'waveform': waveform, 'lat': lat, 'lng': lng, 'frame_type': frame_type,
            'norm_factors': norm_factors
        })
        logger.info(f"📊 PCA: Node{node_id} {frame_type} 位置={lat:.4f},{lng:.4f}")
        if waveform.shape[1] >= 300:  # pred=300点, hist=900点，至少300点才处理
            logger.info(f"📊 Node{node_id} {frame_type} 波形注入状态机 ({waveform.shape[1]}点)")
            for i in range(waveform.shape[1]):
                sensor_data = {
                    'AX': int(waveform[2, i] * 4096 / 9.8),
                    'AY': int(waveform[1, i] * 4096 / 9.8),
                    'AZ': int(waveform[0, i] * 4096 / 9.8),

                }
                state_machine.on_data(sensor_data)
    except Exception as e:
        logger.error(f"PCA处理错误: {e}")


def handle_alert_message(msg):
    pass


def handle_received_message(msg):
    global state_machine
    try:
        payload_str = msg.payload.decode('utf-8')
        sensor_data = json.loads(payload_str)
        if 'status' in sensor_data:
            return
        required = ['AX', 'AY', 'AZ']
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
            if state_machine:
                state_machine.on_data(converted)
            return
    except Exception as e:
        logger.error(f"处理消息错误: {e}")


def main():
    global state_machine
    if not hasattr(np, 'trapz') and hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
    logger.info("📤 初始化发送MQTT...")
    mqtt_sender = EarthquakeMQTT()
    state_machine = EarthquakeStateMachine(mqtt_sender)
    state_machine.test_mode = TEST_MODE
    logger.info("📥 初始化接收MQTT...")
    mqtt_receiver = MQTTReceiver(handle_received_message)
    try:
        while True:
            try:
                title, msg, conf, src = alert_queue.get(timeout=0.1)
                root = tk.Tk()
                root.attributes('-topmost', True)
                root.withdraw()
                messagebox.showwarning("⚠️ 地震预警 ⚠️", f"{title}\n\n{msg}\n置信度: {conf}%\n来源: {src}")
                root.destroy()
            except queue.Empty:
                pass
            time.sleep(0.01)
    except KeyboardInterrupt:
        logger.info("🛑 退出")
    finally:
        mqtt_receiver.disconnect()
        mqtt_sender.disconnect()


if __name__ == "__main__":
    main()
