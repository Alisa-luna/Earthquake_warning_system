#  地震预警系统 (Earthquake Early Warning System)

基于 ESP32 + LoRa + 4G DTU + Edge Impulse AI 的分布式地震预警系统。（仍在持续更新与优化）

##  系统架构

```

节点 (ESP32 + MPU6050) ──LoRa──→ 网关 (ESP32-S3 + DTU) ──MQTT──→ 云端 (Python)
│                                │                              │
├── 100Hz 加速度采集            ├── PCA 反压缩                  ├── PhaseNet + EQT
├── Edge Impulse 推理           ├── 状态机投票                  ├── 震中推算
├── 波形预测                    ├── 钉钉/邮件告警               ├── 烈度分布图
└── WiFi AP (Web 管理)          └── 4G 透传                     └── 手机 APP 通知

```

##  快速开始

### 硬件要求
- **节点**：ESP32 + MPU6050 + DX-LR22 (LoRa)
- **网关**：ESP32-S3 + Air780EPM (4G DTU) + DX-LR22 (LoRa)

### 烧录步骤
1. 安装 Arduino IDE 和 ESP32 开发板支持
2. 安装所需库：MPU6050, PubSubClient, Edge Impulse SDK 等
3. 将 `node/` 烧录到节点 ESP32
4. 将 `gateway/` 烧录到网关 ESP32-S3
5. 配置 `config.h` 中的 WiFi、MQTT、钉钉 Token 等参数

##  目录结构

```

├── node/               # 节点端 Arduino 代码
├── gateway/            # 网关端 Arduino 代码
├── server/             # 云端 Python 处理代码
├── models/             # TFLite 模型和 PCA 矩阵
├── app/                # React Native 手机 APP
├── docs/               # 文档
└── README.md

```

## 🔧 主要功能

-  100Hz 三轴加速度实时采集
-  Edge Impulse 地震检测（Mcu-quake开源检测模型）
-  Conv1d 全连接扩张卷积 地震波形预测模型
-  LoRa 远距离数据传输 (SF11, BW125)
-  PCA 数据压缩 (900→5, 300→10)
-  4G DTU 透传钉钉告警 + MQTT
-  双模型投票 (PhaseNet + EQTransformer)
-  烈度分布热力图
-  手机 APP 全屏预警弹窗
-  时间同步 (NTP + LoRa 广播)
-  局域网内参与点相互导航和通信、坐标发送

##  许可证

本项目采用 [MIT License](LICENSE)

##  免责声明

本系统仅供研究和实验目的，不构成正式的地震预警系统。请勿将本系统用于生命财产安全相关的决策。

##  致谢

### MCU-Quake 模型
- **论文**：[Real-time discrimination of earthquake signals by integrating AI technology into IoT devices](https://doi.org/10.1038/s43247-025-02003-y)
- **作者**：Zhi Geng, Yanfei Wang, Wenyong Pan, Caixia Yu, Zhijing Bai, Hongzhou Zhang
- **发表**：Communications Earth & Environment, 2025
- **许可证**：[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **模型代码**：[ScienceDB](https://doi.org/10.57760/sciencedb.10775)

- **PhaseNet / EQTransformer** - SeisBench 预训练模型
  - 来源：[SeisBench](https://github.com/seisbench/seisbench)
  - 许可证：MIT License
