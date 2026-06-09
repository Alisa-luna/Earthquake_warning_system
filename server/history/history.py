#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import obspy
import paho.mqtt.client as mqtt
import time
import json
import argparse
import sys
from pathlib import Path

# MPU6050 灵敏度常数
ACCEL_SENSITIVITY = 4096.0  # LSB/g
G_TO_LSB = ACCEL_SENSITIVITY
LSB_TO_G = 1.0 / ACCEL_SENSITIVITY

# 默认 MQTT 配置（可修改）
MQTT_BROKER = "192.168.1.100"
MQTT_PORT = 8883
MQTT_TOPIC = "483FDA58BA79-publish"
MQTT_USER = "User"
MQTT_PASSWORD = "1234567890"
USE_SSL = True
CA_CERT = r"D:\Mosquitto\ca.crt"


class HistoricalEarthquakePlayer:
    """历史地震回放器（动态缩放版）"""

    def __init__(self, file_z, file_n=None, file_e=None, sampling_rate=200, target_peak_g=0.05):
        """
        加载地震波形文件，支持动态缩放
        target_peak_g: 目标峰值加速度（单位g），默认0.05g
        """
        self.target_rate = sampling_rate
        self.interval = 1.0 / sampling_rate
        self.target_peak_g = target_peak_g  # 动态缩放目标
        self.data = None
        self.times = None
        self._load_data(file_z, file_n, file_e)

        # MQTT 客户端
        self.client = mqtt.Client()
        self._setup_mqtt()

    def _load_data(self, file_z, file_n, file_e):
        """加载并处理波形数据，动态缩放"""
        # 读取主文件
        st_z = obspy.read(file_z)

        if file_n is None and file_e is None:
            # 自动识别通道
            traces = {tr.stats.channel: tr for tr in st_z}
            channel_map = None

            if 'BHZ' in traces and 'BHN' in traces and 'BHE' in traces:
                channel_map = {'Z': 'BHZ', 'N': 'BHN', 'E': 'BHE'}
                print("使用标准 BHZ/BHN/BHE 通道")
            elif 'HHZ' in traces and 'HHN' in traces and 'HHE' in traces:
                channel_map = {'Z': 'HHZ', 'N': 'HHN', 'E': 'HHE'}
                print("使用 HHZ/HHN/HHE 通道")
            elif 'EHZ' in traces and 'EHN' in traces and 'EHE' in traces:
                channel_map = {'Z': 'EHZ', 'N': 'EHN', 'E': 'EHE'}
                print("使用 EHZ/EHN/EHE 通道")
            else:
                # 尝试用最后一个字符匹配
                last_char_map = {}
                for ch_name, tr in traces.items():
                    last_char = ch_name[-1]
                    if last_char not in last_char_map:
                        last_char_map[last_char] = tr

                if 'Z' in last_char_map and 'N' in last_char_map and 'E' in last_char_map:
                    channel_map = {
                        'Z': last_char_map['Z'].stats.channel,
                        'N': last_char_map['N'].stats.channel,
                        'E': last_char_map['E'].stats.channel
                    }
                    print(f"按末字符匹配: {channel_map}")

            if channel_map is None:
                raise ValueError(f"找不到合适的通道，现有通道: {list(traces.keys())}")

            tr_z = traces[channel_map['Z']]
            tr_n = traces[channel_map['N']]
            tr_e = traces[channel_map['E']]
        else:
            tr_z = st_z[0]
            tr_n = obspy.read(file_n)[0]
            tr_e = obspy.read(file_e)[0]

        # 统一采样率
        for tr in [tr_z, tr_n, tr_e]:
            if tr.stats.sampling_rate != self.target_rate:
                print(f"重采样: {tr.stats.channel} 从 {tr.stats.sampling_rate}Hz 到 {self.target_rate}Hz")
                tr.resample(self.target_rate)

        # 对齐时间
        start_time = max(tr_z.stats.starttime, tr_n.stats.starttime, tr_e.stats.starttime)
        end_time = min(tr_z.stats.endtime, tr_n.stats.endtime, tr_e.stats.endtime)
        tr_z.trim(start_time, end_time)
        tr_n.trim(start_time, end_time)
        tr_e.trim(start_time, end_time)

        # 获取数据
        z = tr_z.data.astype(np.float32)
        n = tr_n.data.astype(np.float32)
        e = tr_e.data.astype(np.float32)

        # 统一长度
        min_len = min(len(z), len(n), len(e))
        z = z[:min_len]
        n = n[:min_len]
        e = e[:min_len]

        print(f"\n原始数据统计 (单位: counts):")
        print(f"  Z: min={z.min():.2f}, max={z.max():.2f}, std={np.std(z):.2f}")
        print(f"  N: min={n.min():.2f}, max={n.max():.2f}, std={np.std(n):.2f}")
        print(f"  E: min={e.min():.2f}, max={e.max():.2f}, std={np.std(e):.2f}")

        # ===== 动态缩放核心代码 =====
        # 1. 去均值
        z = z - np.mean(z)
        n = n - np.mean(n)
        e = e - np.mean(e)

        # 2. 计算原始峰值（counts）
        peak_counts = max(
            np.max(np.abs(z)),
            np.max(np.abs(n)),
            np.max(np.abs(e))
        )

        # 3. 计算缩放因子
        # 目标：让峰值达到 target_peak_g
        # counts → g 的典型转换因子是 1e-6 到 1e-5
        # 我们先假设 1 count ≈ 1e-6 g，然后缩放
        BASE_SCALE = 1e-8  # 基础转换因子
        current_peak_g = peak_counts * BASE_SCALE
        scale_factor = self.target_peak_g / current_peak_g if current_peak_g > 0 else 1.0

        print(f"\n📊 动态缩放参数:")
        print(f"  原始峰值: {peak_counts:.0f} counts")
        print(f"  基础转换: {BASE_SCALE:.2e} g/count")
        print(f"  当前峰值: {current_peak_g:.4f} g")
        print(f"  目标峰值: {self.target_peak_g:.3f} g")
        print(f"  缩放因子: {scale_factor:.2f}")

        # 4. 应用缩放
        z_g = z * BASE_SCALE * scale_factor
        n_g = n * BASE_SCALE * scale_factor
        e_g = e * BASE_SCALE * scale_factor

        print(f"\n转换后统计 (单位: g):")
        print(f"  Z: min={z_g.min():.4f}g, max={z_g.max():.4f}g, std={np.std(z_g):.4f}g")
        print(f"  N: min={n_g.min():.4f}g, max={n_g.max():.4f}g, std={np.std(n_g):.4f}g")
        print(f"  E: min={e_g.min():.4f}g, max={e_g.max():.4f}g, std={np.std(e_g):.4f}g")

        # 5. 转换为LSB（ESP32格式）
        self.az_lsb = np.round((z_g + 1.0) * G_TO_LSB).astype(int)
        self.ay_lsb = np.round(n_g * G_TO_LSB).astype(int)
        self.ax_lsb = np.round(e_g * G_TO_LSB).astype(int)

        # 6. 生成时间轴
        self.times = np.arange(len(z_g)) / self.target_rate
        self.num_samples = len(z_g)

        print(f"\n✅ 数据加载成功")
        print(f"  总样本数: {self.num_samples}, 时长: {self.times[-1]:.2f} 秒")
        print(f"  LSB范围: AX=[{self.ax_lsb.min()}, {self.ax_lsb.max()}], "
              f"AY=[{self.ay_lsb.min()}, {self.ay_lsb.max()}], "
              f"AZ=[{self.az_lsb.min()}, {self.az_lsb.max()}]")

    def _setup_mqtt(self):
        """配置 MQTT 连接"""
        if USE_SSL:
            try:
                self.client.tls_set(ca_certs=CA_CERT)
                self.client.tls_insecure_set(True)
            except Exception as e:
                print(f"SSL 配置失败: {e}")
        if MQTT_USER and MQTT_PASSWORD:
            self.client.username_pw_set(MQTT_USER, MQTT_PASSWORD)
        self.client.on_connect = self._on_connect

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("✅ 已连接到 MQTT Broker")
        else:
            print(f"❌ 连接失败，错误码 {rc}")

    def connect(self):
        """连接到 MQTT 服务器"""
        try:
            self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
            self.client.loop_start()
            time.sleep(1)
            return True
        except Exception as e:
            print(f"连接错误: {e}")
            return False

    def play(self, loop=True, start_skip=0):
        """
        开始发送数据
        loop: 是否循环播放
        start_skip: 跳过开头多少秒
        """
        if not self.connect():
            return

        start_idx = int(start_skip * self.target_rate)
        sent = 0
        start_time = time.time()

        try:
            while True:
                for i in range(start_idx, self.num_samples):
                    data = {
                        "AX": int(self.ax_lsb[i]),
                        "AY": int(self.ay_lsb[i]),
                        "AZ": int(self.az_lsb[i]),
                        "GX": 0,
                        "GY": 0,
                        "GZ": 0,
                        "timestamp": time.time(),
                        "sample_idx": i,
                        "total": self.num_samples
                    }
                    self.client.publish(MQTT_TOPIC, json.dumps(data))
                    sent += 1

                    if sent % 200 == 0:
                        elapsed = time.time() - start_time
                        speed = sent / elapsed
                        remaining = (self.num_samples - i - 1) / self.target_rate
                        az_g = data['AZ'] / G_TO_LSB - 1
                        print(f"进度: {i / self.num_samples * 100:.1f}% | "
                              f"速度: {speed:.0f}点/秒 | 剩余: {remaining:.1f}秒 | "
                              f"当前AZ={az_g:.3f}g")

                    expected_time = (i - start_idx) * self.interval
                    actual_elapsed = time.time() - start_time
                    sleep_time = max(0, expected_time - actual_elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                if not loop:
                    break
                print("\n🔄 一轮发送完成，重新开始循环...")
                start_time = time.time()
                start_idx = 0

        except KeyboardInterrupt:
            print("\n🛑 用户中断")
        finally:
            self.client.loop_stop()
            self.client.disconnect()


def main():
    parser = argparse.ArgumentParser(description="历史地震事件回放器（动态缩放版）")
    parser.add_argument("file", help="地震波形文件")
    parser.add_argument("--n", help="N分量文件（可选）")
    parser.add_argument("--e", help="E分量文件（可选）")
    parser.add_argument("--rate", type=float, default=200, help="目标采样率，默认200Hz")
    parser.add_argument("--peak", type=float, default=0.05,
                        help="目标峰值加速度(g)，默认0.05g（对应5-6度烈度）")
    parser.add_argument("--skip", type=float, default=0,
                        help="跳过开头多少秒（用于远震数据）")
    parser.add_argument("--loop", action="store_true", help="循环播放")
    parser.add_argument("--no-loop", dest="loop", action="store_false")
    parser.set_defaults(loop=True)

    args = parser.parse_args()

    print(f"\n🎯 动态缩放参数: 目标峰值={args.peak}g")
    player = HistoricalEarthquakePlayer(
        args.file, args.n, args.e,
        args.rate,
        target_peak_g=args.peak
    )
    player.play(loop=args.loop, start_skip=args.skip)


if __name__ == "__main__":
    main()