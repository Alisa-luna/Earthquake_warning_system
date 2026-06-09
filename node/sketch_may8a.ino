/**
 * 地震预警节点 v4.1
 * 
 * Core 0: 波形预测 + 三轴分类 + PCA压缩 + LoRa收发
 * Core 1: 100Hz采集(中断) + Z轴流式推理 + Web + 邻居维护
 */

#include <Arduino.h>
#include <Wire.h>
#include "MPU6050_6Axis_MotionApps20.h"
#include <SPI.h>

#include <WiFi.h>
#include <WebServer.h>
#include <FFat.h>
#include <math.h>
#include "config.h"
#include "pca_matrix_900.h"
#include "pca_matrix_300.h"
#include "waveform_predictor.h"
#include "esp_wifi.h"
#include <USER01253-project-1_inferencing.h>
#include <sys/time.h>

// ==================== 硬件 ====================
// LoRa 模块 (DX-LR22)
#define DXLR_SERIAL Serial2
#define DXLR_BAUDRATE 9600
#define DXLR_RX 4   // ESP32 GPIO4 接 DX-LR22 TX
#define DXLR_TX 5   // ESP32 GPIO5 接 DX-LR22 RX
#define DXLR_M0 15  // ← 加
#define DXLR_M1 16  // ← 加
#define DXLR_AUX 13
// I2C 传感器 (MPU6050)
#define I2C_SDA 17  // 根据你的接线确定
#define I2C_SCL 18
#define I2C_FREQ 400000
#define MPU6050_ADDR 0x68
#define MPU6050_INT 3  // MPU6050 INT 引脚接 GPIO3（根据你的实际接线修改）
#define MPU6050_DMP_FIFO_RATE 9
// ==================== 帧协议 ====================
#define FRAME_HEADER 0xAA
#define FRAME_DATA_PRED 0x01
#define FRAME_DATA_HIST 0x02
#define FRAME_ALERT 0x03
#define FRAME_RELAY 0x04
#define FRAME_REGISTER 0x05
#define FRAME_PROBE 0x06
#define FRAME_PROBE_RESP 0x07
#define FRAME_NEIGHBOR_LIST 0x08
#define FRAME_POSITION 0x09
#define FRAME_DEPLOY_CONFIRM 0x10
#define FRAME_PRESENCE 0x20
#define RELAY_TTL 5
#define FRAME_TIME_SYNC 0x11
// =============== 消息区 =================
#define FRAME_MESSAGE 0x30
#define MAX_STORED_MSGS 50

struct StoredMessage {
  uint32_t timestamp;
  uint16_t fromNode;
  char text[200];
};
StoredMessage msgStore[MAX_STORED_MSGS];
int msgCount = 0;

// ==================== 缓冲区 ====================
#define HISTORY_SEC 10
#define HISTORY_SIZE (HISTORY_SEC * 100)
#define MAX_NEIGHBORS 20
#define TX_QUEUE_LEN 10
#define PCA_OUT_HIST 10
#define PCA_OUT_PRED 5
#define PAYLOAD_HIST 20
#define PAYLOAD_PRED 10
#define SLIDE_INTERVAL_MS 150
// 基线值（根据你的观察）
#define Z_BASELINE 0.5f
#define XY_BASELINE 0.0f


// ==================== 结构体 ====================
struct NeighborInfo {
  uint16_t nodeID;
  float lat, lng, alt;
  int8_t rssi;
  float distance;
  uint32_t lastSeen;
};

// ==================== 全局变量 ====================
String cachedNodePage = "";
uint32_t cachedNodePageTime = 0;
float gwDistance = 0;
// 硬件定时器
hw_timer_t* sampleTimer = NULL;
volatile int16_t rawAX, rawAY, rawAZ;
volatile bool sampleReady = false;

// ACK 异步等待
bool waitingForAck = false;
uint8_t ackFrameType = 0;
uint32_t ackSentTime = 0;
uint8_t* ackRetryBuf = nullptr;
int ackRetryLen = 0;

// 历史环形缓冲区
float historyBuffer[HISTORY_SIZE][3];
volatile int historyHead = 0;
volatile bool historyFull = false;
int historyCount = 0;

// 本节点
uint16_t myID = NODE_ID;
float myLat = PREASSIGNED_LAT, myLng = PREASSIGNED_LNG, myAlt = PREASSIGNED_ALT;
bool positionVerified = false;
float gwLat = 0, gwLng = 0, gwAlt = 0;
bool gwPositionKnown = false;

// 邻居
NeighborInfo neighbors[MAX_NEIGHBORS];
int neighborCount = 0;

// 信号量与队列
SemaphoreHandle_t windowReadySemaphore;
SemaphoreHandle_t historyAlertSemaphore;
SemaphoreHandle_t dataMutex;
QueueHandle_t txQueue;
SemaphoreHandle_t inferenceMutex;

// 共享数据
float sharedInputWindow[3][100];
float sharedPredWaveform[3][300];
float sharedPCAOut[PCA_OUT_PRED];
int16_t sharedQuantized[PCA_OUT_PRED];
float sharedPredPeak = 0;
bool predictionSent = false;
uint32_t predictionSentTime = 0;

// Web
WebServer* server = nullptr;
bool apMode = false;

// 统计
volatile uint32_t sampleCount = 0;
volatile uint32_t inferenceCount = 0;
volatile uint32_t alertCount = 0;
volatile uint32_t relayCount = 0;

// MPU6050
MPU6050_6Axis_MotionApps20 mpu;
float tempOffset = 0;

// Edge Impulse 推理缓冲区
float g_inference_data[700];

bool dmpReady = false;
uint8_t fifoBuffer[64];
Quaternion q;
VectorFloat gravity;
float ypr[3];  // yaw, pitch, roll
volatile bool mpuInterrupt = false;

#define PRED_STA_WINDOW 30   // 短时窗：0.3秒 (100Hz * 0.3)
#define PRED_LTA_WINDOW 200  // 长时窗：2.0秒 (预测波形总长300点，LTA取前2秒)
#define PRED_STA_LTA_THRESHOLD 2.5f

float predZ[300];  // 存储预测波形Z通道
float predStaBuffer[PRED_STA_WINDOW] = { 0 };
float predLtaBuffer[PRED_LTA_WINDOW] = { 0 };
int predStaIdx = 0, predLtaIdx = 0;
float predStaSum = 0, predLtaSum = 0;
bool predStaInitialized = false, predLtaInitialized = false;

float bg_peak[3] = { 0.5, 0.3, 0.3 };  // X, Y, Z 背景峰值初始估计
float bg_alpha = 0.005;                // 背景更新平滑系数（越小越稳定）

void dmpDataReady() {
  mpuInterrupt = true;
}

time_t g_epochTime = 0;     // 当前 Unix 时间戳
bool g_timeValid = false;   // 时间是否有效
uint32_t lastTimeSync = 0;  // 上次收到时间同步的时间


// 获取格式化的时间字符串
String getTimeString() {
  if (!g_timeValid) return String(millis() / 1000) + "s";

  time_t now = g_epochTime + (millis() - lastTimeSync) / 1000;
  struct tm* timeinfo = localtime(&now);

  char buf[32];
  snprintf(buf, sizeof(buf), "%04d-%02d-%02d %02d:%02d:%02d",
           timeinfo->tm_year + 1900, timeinfo->tm_mon + 1, timeinfo->tm_mday,
           timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec);
  return String(buf);
}

// 获取 Unix 时间戳（秒）
uint32_t getEpochTime() {
  if (!g_timeValid) return millis() / 1000;
  return g_epochTime + (millis() - lastTimeSync) / 1000;
}

// ==================== 中断采集 ====================
// 中断函数只设置标志
void IRAM_ATTR onSampleTimer() {
  sampleReady = true;
}

// 最大绝对值归一化（适用于任意长度的一维浮点数组）
void normalizeMaxAbs(float* data, int len) {
  float max_val = 0;
  for (int i = 0; i < len; i++) {
    if (fabs(data[i]) > max_val) max_val = fabs(data[i]);
  }
  if (max_val > 1e-10) {
    for (int i = 0; i < len; i++) data[i] /= max_val;
  }
}


// ==================== 历史缓冲区 ====================
void addToHistory(float ax, float ay, float az) {
  historyBuffer[historyHead][0] = ax;
  historyBuffer[historyHead][1] = ay;
  historyBuffer[historyHead][2] = az;
  historyHead = (historyHead + 1) % HISTORY_SIZE;
  if (historyCount < HISTORY_SIZE) historyCount++;
  if (historyCount >= HISTORY_SIZE) historyFull = true;
}

void getRecentAxis(float* out, int axis, int seconds) {
  int count = seconds * 100;
  if (count > historyCount) count = historyCount;
  int start = (historyHead - count + HISTORY_SIZE) % HISTORY_SIZE;
  for (int i = 0; i < count; i++) {
    int idx = (start + i) % HISTORY_SIZE;
    out[i] = historyBuffer[idx][axis];
  }
}

void getRecentWindow(float (*out)[3], int count) {
  if (count > historyCount) count = historyCount;
  int start = (historyHead - count + HISTORY_SIZE) % HISTORY_SIZE;
  for (int i = 0; i < count; i++) {
    int idx = (start + i) % HISTORY_SIZE;
    out[i][0] = historyBuffer[idx][0];
    out[i][1] = historyBuffer[idx][1];
    out[i][2] = historyBuffer[idx][2];
  }
}

// ==================== PCA ====================
void pcaProject900(const float* input, float* output) {
  for (int j = 0; j < PCA_OUT_PRED; j++) {
    output[j] = 0;
    for (int i = 0; i < 900; i++) output[j] += (input[i] - pca_mean_900[i]) * pca_matrix_900[j][i];
  }
}

void pcaProject300(const float* input, float* output) {
  for (int j = 0; j < PCA_OUT_HIST; j++) {
    output[j] = 0;
    for (int i = 0; i < 300; i++) output[j] += (input[i] - pca_mean_300[i]) * pca_matrix_300[j][i];
  }
}

void quantize(const float* input, int16_t* output, int dim) {
  const float scale = 0.001f;
  for (int i = 0; i < dim; i++) {
    float v = input[i] / scale;
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    output[i] = (int16_t)round(v);
  }
}

// ==================== MCU-Quake 推理 ====================
int raw_feature_get_data(size_t offset, size_t length, float* out_ptr) {
  memcpy(out_ptr, g_inference_data + offset, length * sizeof(float));
  return 0;
}

float runInference(float* data, int len) {
  // 锁定，保证同一时间只有一个核心执行推理
  xSemaphoreTake(inferenceMutex, portMAX_DELAY);

  memcpy(g_inference_data, data, len * sizeof(float));
  signal_t signal;
  signal.total_length = EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE;
  signal.get_data = &raw_feature_get_data;
  ei_impulse_result_t result = { 0 };
  EI_IMPULSE_ERROR res = run_classifier(&signal, &result, false);

  xSemaphoreGive(inferenceMutex);
  return (res == 0) ? result.classification[0].value : 0;
}

// ==================== LoRa 发送 ====================
void enqueueFrame(uint8_t frameType, uint8_t* payload, uint8_t len) {
  uint8_t* buf = (uint8_t*)malloc(len + 5);
  if (!buf) return;
  buf[0] = FRAME_HEADER;
  buf[1] = frameType;
  buf[2] = len;
  buf[3] = (myID >> 8) & 0xFF;
  buf[4] = myID & 0xFF;
  memcpy(buf + 5, payload, len);
  xQueueSend(txQueue, &buf, 0);
}

// 修改后的 PCA 帧发送：在量化系数后追加归一化因子（每个通道 2 字节，共 6 字节）
void enqueuePCAFrame(uint8_t frameType, int16_t* quantized, int dim, float* factors) {
  // 原始 PCA 系数占用 dim*2 字节，再加 6 字节归一化因子
  uint8_t payload[26];  // 20 + 6 = 26
  memcpy(payload, quantized, dim * 2);

  // 将三个 float 归一化因子转为 uint16_t（乘 1000 保留 3 位小数）
  for (int c = 0; c < 3; c++) {
    uint16_t f_quant = (uint16_t)(factors[c] * 1000.0f + 0.5f);  // 四舍五入
    payload[dim * 2 + c * 2] = (f_quant >> 8) & 0xFF;
    payload[dim * 2 + c * 2 + 1] = f_quant & 0xFF;
  }

  enqueueFrame(frameType, payload, dim * 2 + 6);
}
void enqueuePCAFrame(uint8_t frameType, int16_t* quantized, int dim) {
  uint8_t payload[20];
  memcpy(payload, quantized, dim * 2);
  enqueueFrame(frameType, payload, dim * 2);
}

void sendAlertFrame() {
  uint8_t payload[12];

  // 置信度
  // ===== 动态计算 conf =====
  float dev_peak[3] = { 0 };
  for (int c = 0; c < 3; c++) {
    float baseline = (c == 2) ? Z_BASELINE : XY_BASELINE;
    for (int t = 0; t < 300; t++) {
      float dev = fabs(sharedPredWaveform[c][t] - baseline);
      if (dev > dev_peak[c]) dev_peak[c] = dev;
    }
  }
  float conf = fmax(fmax(dev_peak[0], dev_peak[1]), dev_peak[2]);
  if (conf < 0.1f) conf = 0.1f;

  // ===== 动态计算 accel =====
  float maxZ = 0;
  for (int t = 0; t < 300; t++) {
    float val = fabs(sharedPredWaveform[2][t]);
    if (val > maxZ) maxZ = val;
  }
  float maxAccel = maxZ * 9.8f;

  // 简易烈度计算
  uint8_t intensity = 1;
  if (maxAccel >= 0.31f) intensity = 2;
  else if (maxAccel >= 0.63f) intensity = 3;
  else if (maxAccel >= 1.25f) intensity = 4;
  else if (maxAccel >= 2.50f) intensity = 5;
  else if (maxAccel >= 5.00f) intensity = 6;
  else if (maxAccel >= 10.0f) intensity = 7;
  else if (maxAccel >= 25.0f) intensity = 8;
  else if (maxAccel >= 50.0f) intensity = 9;
  else if (maxAccel >= 100.0f) intensity = 10;
  else if (maxAccel >= 250.0f) intensity = 11;
  else intensity = 12;  // 实际上限为 12

  uint8_t ttl = RELAY_TTL;  // 初始中继跳数
  uint8_t reserved[2] = { 0 };

  memcpy(payload, &conf, 4);
  memcpy(payload + 4, &maxAccel, 4);
  payload[8] = intensity;
  payload[9] = ttl;
  memcpy(payload + 10, reserved, 2);

  enqueueFrame(FRAME_ALERT, payload, 12);
  alertCount++;
}

void waitAuxHigh() {
  if (DXLR_AUX <= 0) {
    delay(20);
    return;
  }
  uint32_t t = millis();
  while (digitalRead(DXLR_AUX) == LOW) {
    if (millis() - t > 500) break;
    delay(1);
  }
}


// 修改 sendLoRaFromQueue()
void sendLoRaFromQueue() {
  uint8_t* buf = NULL;
  if (xQueueReceive(txQueue, &buf, 0) == pdTRUE) {
    if (buf) {
      waitAuxHigh();  // 等待模块空闲
      int len = buf[2] + 5;
      DXLR_SERIAL.write(buf, len);
      free(buf);
    }
  }
}

void relayAlertFrame(uint16_t source, uint8_t* data, int len, uint8_t ttl) {
  if (ttl == 0) return;

  // ---- 去重：滑动窗口记录 (源ID, conf×100) ----
  float conf;
  memcpy(&conf, data, 4);
  int16_t confKey = (int16_t)(conf * 100);

  static uint16_t lastSources[10] = { 0 };
  static int16_t lastConfs[10] = { 0 };
  static int lastIdx = 0;
  for (int i = 0; i < 10; i++) {
    if (lastSources[i] == source && lastConfs[i] == confKey) {
      return;  // 已转发过，丢弃
    }
  }
  lastSources[lastIdx] = source;
  lastConfs[lastIdx] = confKey;
  lastIdx = (lastIdx + 1) % 10;

  // ---- 构造中继帧 ----
  uint8_t* relay = (uint8_t*)malloc(len + 5);
  if (!relay) return;
  relay[0] = FRAME_HEADER;
  relay[1] = FRAME_RELAY;
  relay[2] = len;  // 荷载长度，12
  relay[3] = (source >> 8) & 0xFF;
  relay[4] = source & 0xFF;
  memcpy(relay + 5, data, len);  // 复制原报警荷载
  // TTL 在荷载偏移 9 处，即 relay 数组索引 5+9=14
  relay[14] = ttl - 1;

  xQueueSend(txQueue, &relay, 0);
  relayCount++;
}
// ==================== RSSI 测距 ====================
float rssiToDistance(int8_t rssi) {
  return pow(10.0f, (-30.0f - rssi) / (10.0f * 2.8f));
}
// ==================== LoRa 接收 ====================
void handleLoRaRX() {

  if (DXLR_SERIAL.available() < 5) return;  // 至少需要帧头+类型+长度+源地址

  uint8_t buf[256];
  int idx = 0;

  // 找帧头 0xAA
  while (DXLR_SERIAL.available() && idx < 256) {
    buf[idx] = DXLR_SERIAL.read();
    if (buf[0] != FRAME_HEADER) {
      idx = 0;  // 不是帧头，丢弃
      continue;
    }
    idx++;
    if (idx >= 5) break;  // 至少读够帧头+类型+长度+源地址
  }

  if (idx < 5) return;

  uint8_t len = buf[2];
  uint16_t srcID = (buf[3] << 8) | buf[4];

  // 读取负载
  int totalLen = 5 + len;
  while (idx < totalLen && DXLR_SERIAL.available()) {
    buf[idx++] = DXLR_SERIAL.read();
  }

  if (idx < totalLen) return;  // 数据不完整

  int8_t rssi = 0;  // DX-LR22 已开启 AT+DRSSI1，最后一个字节是 RSSI
  if (idx > 5) {
    rssi = -(0xFF - buf[idx - 1]);
    idx--;
  }

  uint8_t type = buf[1];

  Serial.printf("[LoRa RX] type=0x%02X, srcID=%d, len=%d\n", type, srcID, len);

  switch (type) {
    case 0x0A:
      {  // ACK 帧
        uint16_t ackTarget = (buf[3] << 8) | buf[4];
        if (ackTarget == myID && waitingForAck) {
          waitingForAck = false;
          if (ackRetryBuf) {
            free(ackRetryBuf);
            ackRetryBuf = nullptr;
          }
          Serial.println("✅ ACK received (async)");
        }
        break;
      }
    case FRAME_ALERT:
    case FRAME_RELAY:
      if (srcID != myID) {
        uint8_t ttl = buf[14];
        relayAlertFrame(srcID, buf + 5, buf[2], buf[19]);
      }
      break;
    case FRAME_PROBE:
      if (srcID != myID) {
        uint8_t payload[12];
        memcpy(payload, &myLat, 4);
        memcpy(payload + 4, &myLng, 4);
        memcpy(payload + 8, &myAlt, 4);
        enqueueFrame(FRAME_PROBE_RESP, payload, 12);
      }
      break;
    case FRAME_PROBE_RESP:
      if (srcID != myID && buf[2] >= 12) {
        float nlat, nlng, nalt;
        memcpy(&nlat, buf + 5, 4);
        memcpy(&nlng, buf + 9, 4);
        memcpy(&nalt, buf + 13, 4);
        bool found = false;
        for (int i = 0; i < neighborCount; i++) {
          if (neighbors[i].nodeID == srcID) {
            neighbors[i].lat = nlat;
            neighbors[i].lng = nlng;
            neighbors[i].rssi = rssi;
            neighbors[i].lastSeen = millis();
            found = true;
            break;
          }
        }
        if (!found && neighborCount < MAX_NEIGHBORS) {
          neighbors[neighborCount].nodeID = srcID;
          neighbors[neighborCount].lat = nlat;
          neighbors[neighborCount].lng = nlng;
          neighbors[neighborCount].alt = nalt;
          neighbors[neighborCount].rssi = rssi;
          neighbors[neighborCount].lastSeen = millis();
          neighbors[neighborCount].distance = rssiToDistance(rssi);
          neighborCount++;
        }
      }
      break;
    case FRAME_NEIGHBOR_LIST:
      {
        uint8_t count = buf[5];
        int off = 6;
        for (int i = 0; i < count && off + 14 <= idx; i++) {
          uint16_t nid = (buf[off] << 8) | buf[off + 1];
          float nlat, nlng, nalt;
          memcpy(&nlat, buf + off + 2, 4);
          memcpy(&nlng, buf + off + 6, 4);
          memcpy(&nalt, buf + off + 10, 4);
          if (nid != myID) {
            bool found = false;
            for (int j = 0; j < neighborCount; j++) {
              if (neighbors[j].nodeID == nid) {
                neighbors[j].lat = nlat;
                neighbors[j].lng = nlng;
                neighbors[j].lastSeen = millis();
                neighbors[j].distance = rssiToDistance(neighbors[j].rssi);
                found = true;
                break;
              }
            }
            if (!found && neighborCount < MAX_NEIGHBORS) {
              neighbors[neighborCount].nodeID = nid;
              neighbors[neighborCount].lat = nlat;
              neighbors[neighborCount].lng = nlng;
              neighbors[neighborCount].lastSeen = millis();
              neighbors[neighborCount].distance = 0;
              neighborCount++;
            }
          }
          off += 14;
        }
        break;
      }
    case FRAME_POSITION:
      memcpy(&gwLat, buf + 5, 4);
      memcpy(&gwLng, buf + 9, 4);
      memcpy(&gwAlt, buf + 13, 4);
      gwDistance = rssiToDistance(rssi);
      gwPositionKnown = true;
      Serial.printf("[POS] Received GW pos: %.6f, %.6f, dist=%.1f m\n",
                    gwLat, gwLng, gwDistance);
      break;
    case FRAME_DEPLOY_CONFIRM:
      if (buf[2] >= 14) {  // 负载长度至少14字节
        uint16_t targetID = (buf[5] << 8) | buf[6];
        if (targetID == myID) {
          memcpy(&myLat, buf + 7, 4);
          memcpy(&myLng, buf + 11, 4);
          memcpy(&myAlt, buf + 15, 4);
          positionVerified = true;
          Serial.printf("[REG] ✅ 收到部署确认！新坐标: %.6f, %.6f, %.1f\n", myLat, myLng, myAlt);
        }
      }
      break;
    case FRAME_MESSAGE:
      {
        uint16_t targetID = (buf[5] << 8) | buf[6];
        if (targetID == 0 || targetID == myID) {
          if (msgCount < MAX_STORED_MSGS) {
            msgStore[msgCount].timestamp = millis();
            msgStore[msgCount].fromNode = srcID;
            int len = min((int)(buf[2] - 3), 199);
            memcpy(msgStore[msgCount].text, &buf[8], len);
            msgStore[msgCount].text[len] = '\0';
            msgCount++;
          }
        }
        break;
      }
    case FRAME_TIME_SYNC:
      if (buf[2] == 4) {  // 4字节时间戳
        uint32_t epoch;
        memcpy(&epoch, buf + 5, 4);

        struct timeval tv = { epoch, 0 };
        settimeofday(&tv, NULL);

        g_epochTime = epoch;
        lastTimeSync = millis();
        g_timeValid = true;

        Serial.printf("[TIME] Synced: %u\n", epoch);
      }
      break;
  }
}
// ==================== 三边定位自校验 ====================
void verifyPositionByNeighbors() {
  if (neighborCount < 3) return;  // 至少3个邻居才能做三边定位

  // 取最近的3个邻居（距离已知且最近）
  struct {
    uint16_t id;
    float lat, lng, dist;
  } cand[3];
  int cnt = 0;

  for (int i = 0; i < neighborCount && cnt < 3; i++) {
    if (neighbors[i].distance > 0 && neighbors[i].lat != 0 && neighbors[i].lng != 0) {
      cand[cnt].id = neighbors[i].nodeID;
      cand[cnt].lat = neighbors[i].lat;
      cand[cnt].lng = neighbors[i].lng;
      cand[cnt].dist = neighbors[i].distance;
      cnt++;
    }
  }

  if (cnt < 3) return;

  // 简化三边定位：用三个邻居坐标和距离，加权平均推算自己位置
  float sumLat = 0, sumLng = 0, sumWeight = 0;
  for (int i = 0; i < 3; i++) {
    float weight = 1.0f / fmax(cand[i].dist, 1.0f);  // 距离越近权重越大
    sumLat += cand[i].lat * weight;
    sumLng += cand[i].lng * weight;
    sumWeight += weight;
  }
  float estLat = sumLat / sumWeight;
  float estLng = sumLng / sumWeight;

  // 和预分配坐标对比
  float dLat = (myLat - estLat) * 111320.0f;
  float dLng = (myLng - estLng) * 111320.0f * cos(estLat * M_PI / 180.0f);
  float dev = sqrt(dLat * dLat + dLng * dLng);

  if (dev > 200) {
    positionVerified = false;
    Serial.printf("[TRILATERAL] ⚠️ Position deviation: %.1fm (pre %.4f,%.4f est %.4f,%.4f)\n",
                  dev, myLat, myLng, estLat, estLng);
  } else {
    positionVerified = true;
    Serial.printf("[TRILATERAL] ✅ Position verified (dev=%.1fm)\n", dev);
  }
}
// ==================== MPU6050 自检 ====================
bool runSelfCheck() {
  Serial.println("[SELFCHECK] MPU6050 calibration...");
  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(I2C_FREQ);
  mpu.initialize();
  if (!mpu.testConnection()) return false;
  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
  mpu.setDLPFMode(MPU6050_DLPF_BW_42);
  mpu.setRate(9);
  delay(100);
  int16_t tempRaw = mpu.getTemperature();
  tempOffset = (tempRaw / 340.0f) + 36.53f;
  float sa = 0, sx = 0, sy = 0;
  for (int i = 0; i < 500; i++) {
    int16_t ax, ay, az, gx, gy, gz;
    mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
    sa += ax;
    sx += ay;
    sy += az;
    delay(10);
  }
  Serial.printf("[SELFCHECK] Offsets: %.1f, %.1f, %.1f, Temp: %.1f°C\n",
                sa / 500, sx / 500, sy / 500 - 4096, tempOffset);
  return true;
}

// ==================== WiFi 部署注册 ====================
bool registerViaWiFi()
{
  for (int attempt = 1; attempt <= 3; attempt++) {  // ← 循环开始
    Serial.printf("[REG] Attempt %d/3\n", attempt);

  
    // =========================
    // 1. 扫描部署AP
    // =========================
    WiFi.mode(WIFI_STA);

    int n = WiFi.scanNetworks();

    String apName = "";
    int deployChannel = 1;

    for (int i = 0; i < n; i++)
    {
        if (WiFi.SSID(i).indexOf("EQ_Deploy") >= 0)
        {
            apName = WiFi.SSID(i);
            deployChannel = WiFi.channel(i);

            Serial.printf(
                "[REG] Found %s Channel=%d\n",
                apName.c_str(),
                deployChannel
            );

            break;
        }
    }
    WiFi.scanDelete();

    if (apName.length() == 0) {
      Serial.println("[REG] 未找到部署 AP");
      if (attempt < 3) delay(5000);
      continue;  // ← 在循环内部
    }

    
  
  

    // =========================
    // 2. AP_STA模式
    // =========================
    WiFi.mode(WIFI_AP_STA);

    String apSSID = "EQ_Node_" + String(myID);

    WiFi.softAPConfig(
        IPAddress(192,168,50,1),
        IPAddress(192,168,50,1),
        IPAddress(255,255,255,0)
    );

    bool apOk = WiFi.softAP(
        apSSID.c_str(),
        "12345678",
        deployChannel
    );

    Serial.printf(
        "[REG] AP Started=%d Channel=%d\n",
        apOk,
        deployChannel
    );

    delay(1000);

    // =========================
    // 3. STA连接部署网关
    // =========================
    WiFi.begin(
        apName.c_str(),
        "12345678"
    );

    uint32_t start = millis();

    while (WiFi.status() != WL_CONNECTED &&
           millis() - start < 25000)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println();

    if (WiFi.status() != WL_CONNECTED)
    {
        Serial.println("[REG] WiFi连接失败");
        return false;
    }

    Serial.println("[REG] WiFi 已连接");

    Serial.printf(
        "[REG] STA Channel=%d\n",
        WiFi.channel()
    );

    Serial.printf(
        "[REG] STA IP=%s\n",
        WiFi.localIP().toString().c_str()
    );

    Serial.printf(
        "[REG] AP IP=%s\n",
        WiFi.softAPIP().toString().c_str()
    );

    // =========================
    // 4. TCP注册
    // =========================
    WiFiClient client;

    if (!client.connect("192.168.4.1", 5555))
    {
        Serial.println("[REG] TCP连接失败");
        return false;
    }

    Serial.println("[REG] 已连接网关 TCP");

    String reg =
        "{\"cmd\":\"register\",\"id\":" +
        String(myID) +
        ",\"lat\":" +
        String(myLat, 6) +
        ",\"lng\":" +
        String(myLng, 6) +
        ",\"alt\":" +
        String(myAlt, 1) +
        "}\n";

    client.print(reg);

    Serial.printf(
        "[REG] 已发送: %s",
        reg.c_str()
    );

    String resp = "";

    start = millis();

    while (client.connected() &&
           millis() - start < 5000)
    {
        if (client.available())
        {
            resp = client.readStringUntil('\n');
            break;
        }

        delay(10);
    }

    client.stop();

    Serial.printf(
        "[REG] 网关响应: %s\n",
        resp.c_str()
    );
  


    // =========================
    // 5. 解析坐标
    // =========================
    int gwLatIdx =
        resp.indexOf("\"gateway_lat\":");

    int gwLngIdx =
        resp.indexOf("\"gateway_lng\":");

    if (gwLatIdx < 0 || gwLngIdx < 0)
    {
        Serial.println(
            "[REG] 响应中未找到网关坐标"
        );
        return false;
    }

    gwLat =
        resp.substring(
            gwLatIdx + 14
        ).toFloat();

    gwLng =
        resp.substring(
            gwLngIdx + 14
        ).toFloat();

    gwPositionKnown = true;

    Serial.printf(
        "[REG] 注册成功！网关坐标: %.6f %.6f\n",
        gwLat,
        gwLng
    );

    saveConfig();

    return true;
  }

  Serial.println("[REG] All 3 attempts failed");
  return false;
  
}
// ==================== FFat 配置 ====================
void saveConfig() {
  File f = FFat.open("/node.json", FILE_WRITE);
  if (f) {
    f.printf("{\"lat\":%.6f,\"lng\":%.6f,\"alt\":%.1f,\"gw_lat\":%.6f,\"gw_lng\":%.6f}",
             myLat, myLng, myAlt, gwLat, gwLng);
    f.close();
  }
}

void loadConfig() {
  File f = FFat.open("/node.json", FILE_READ);
  if (f) {
    String json = "";
    while (f.available()) json += (char)f.read();
    f.close();

    auto findFloat = [&](String key) -> float {
      int s = json.indexOf("\"" + key + "\":");
      if (s < 0) return 0;
      s += key.length() + 3;
      int e = json.indexOf(",", s);
      if (e < 0) e = json.indexOf("}", s);
      return json.substring(s, e).toFloat();
    };

    float lat = findFloat("lat");
    float lng = findFloat("lng");
    float alt = findFloat("alt");
    float glat = findFloat("gw_lat");
    float glng = findFloat("gw_lng");

    if (lat != 0 && lng != 0) {
      myLat = lat;
      myLng = lng;
      myAlt = alt;
      positionVerified = true;
    }
    if (glat != 0 && glng != 0) {
      gwLat = glat;
      gwLng = glng;
      gwPositionKnown = true;
    }
    Serial.printf("[FFat] Loaded pos: %.4f,%.4f  GW: %.4f,%.4f\n", myLat, myLng, gwLat, gwLng);
  }
}

// ==================== Web ====================
String genWebPage() {
  String h = "<!DOCTYPE html><html><head><meta charset='UTF-8'><title>Node " + String(myID) + "</title>";
  h += "<meta name='viewport' content='width=device-width,initial-scale=1'>";
  h += "<style>body{font-family:Arial;background:#1a1a2e;color:#eee;padding:10px}";
  h += ".card{background:#16213e;padding:10px;margin:8px 0;border-radius:8px}";
  h += "h2{color:#e94560}h3{color:#f5c518}.ok{color:#4ecca3}.warn{color:#e94560}";
  h += "table{width:100%;border-collapse:collapse}th,td{padding:5px;border-bottom:1px solid #333}";
  h += "button{padding:12px;width:100%;background:#e94560;color:white;border:none;border-radius:6px;margin:5px 0;font-size:16px}";
  h += "</style></head><body>";
  h += "<h2>🌍 节点 #" + String(myID) + "</h2>";

  h += "<div class='card'><h3>📍 位置信息</h3>";
  h += "<p>本节点: " + String(myLat, 6) + ", " + String(myLng, 6) + " 海拔: " + String(myAlt, 1) + "m</p>";
  if (gwPositionKnown) {
    float bearing = atan2(gwLng - myLng, gwLat - myLat) * 180 / PI;
    h += "<p>网关: " + String(gwLat, 6) + ", " + String(gwLng, 6) + " (bearing: " + String(bearing, 0) + "°)</p>";
  }
  h += "</div>";

  h += "<div class='card'><h3>🔗 邻居节点 (" + String(neighborCount) + ")</h3>";
  h += "<table><tr><th>ID</th><th>距离</th><th>信号</th><th>坐标</th></tr>";
  h += "<tbody id='neighborTable'>";  // 添加 id，用于局部刷新
  for (int i = 0; i < neighborCount; i++) {
    h += "<tr><td>" + String(neighbors[i].nodeID) + "</td>"
         + "<td>" + String(neighbors[i].distance, 1) + "m</td>"
         + "<td>" + String(neighbors[i].rssi) + "dBm</td>"
         + "<td>" + String(neighbors[i].lat, 4) + "," + String(neighbors[i].lng, 4) + "</td></tr>";
  }
  h += "</tbody></table></div>";

  h += "<div class='card'><h3>📊 运行统计</h3>";
  h += "<p>采样次数: <span id='sampleCount'>" + String(sampleCount) + "</span> | 推理次数: <span id='inferCount'>" + String(inferenceCount) + "</span></p>";
  h += "<p>预警次数: <span id='alertCount'>" + String(alertCount) + "</span> | 中继次数: <span id='relayCount'>" + String(relayCount) + "</span></p>";
  h += "<p>State: " + String(predictionSent ? "<span class='warn'>TRIGGERED</span>" : "<span class='ok'>Normal</span>") + "</p></div>";

  h += "<div class='card'><h3>📍 操作</h3>";
  h += "<button onclick=\"fetch('/sendLocation')\">📡 上报我的位置</button>";
  h += "<button onclick=\"openNavigation()\" style='background:#4ecca3'>🗺️ 导航到网关</button>";
  if (neighborCount > 0) {
    h += "<button onclick=\"showNeighborNav()\" style='background:#f5c518;color:#000'>🔗 导航到最近邻居</button>";
  }
  h += "</div>";
  h += "<div class='card'><h3>💬 消息</h3>";
  h += "<div id='msgList' style='max-height:120px;overflow-y:auto;font-size:12px;margin-bottom:8px'></div>";
  h += "<input type='number' id='msgTarget' placeholder='目标节点ID (0=全部)' value='0' style='width:100%;padding:6px;margin:3px 0;background:#0a1628;color:#eee;border:1px solid #333;border-radius:4px'>";
  h += "<input type='text' id='msgText' placeholder='输入消息...' style='width:100%;padding:6px;margin:3px 0;background:#0a1628;color:#eee;border:1px solid #333;border-radius:4px'>";
  h += "<button onclick='sendMsg()'>📤 发送消息</button>";
  h += "</div>";
  h += "<div class='card'><h3>🕐 网络时间</h3>";
  h += "<p id='networkTime'>" + getTimeString() + "</p>";
  h += "</div>";

  h += "<div id='navToast' style='display:none;position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#333;color:#fff;padding:10px 20px;border-radius:8px;z-index:999'></div>";

  h += "<script>";

  // 核心导航函数：调起手机本地地图 APP
  h += "function openNavigation() {";
  h += "var mlat=" + String(myLat, 6) + ";";
  h += "var mlng=" + String(myLng, 6) + ";";
  if (gwPositionKnown) {
    h += "var dlat=" + String(gwLat, 6) + ";";
    h += "var dlng=" + String(gwLng, 6) + ";";
    h += "navigateTo(mlat, mlng, dlat, dlng, 'Node→Gateway');";
  } else {
    h += "navigateTo(null, null, mlat, mlng, 'Node #" + String(myID) + "');";
  }
  h += "}";

  // 导航到最近邻居
  h += "function showNeighborNav(){";
  h += "var best=null;";
  h += "var nodes=[";
  for (int i = 0; i < neighborCount; i++) {
    if (i > 0) h += ",";
    h += "{id:" + String(neighbors[i].nodeID) + ",lat:" + String(neighbors[i].lat, 6) + ",lng:" + String(neighbors[i].lng, 6) + ",dist:" + String(neighbors[i].distance, 1) + "}";
  }
  h += "];";
  h += "if(nodes.length>0){best=nodes.reduce((a,b)=>a.dist<b.dist?a:b);";
  h += "navigateTo(" + String(myLat, 6) + "," + String(myLng, 6) + ",best.lat,best.lng,'Node→Neighbor '+best.id);}}";

  h += "function navigateTo(fromLat,fromLng,toLat,toLng,label){";
  h += "var ua=navigator.userAgent.toLowerCase();";
  h += "var isIOS=/iphone|ipad|ipod/.test(ua);";
  h += "var isAndroid=/android/.test(ua);";
  h += "var fallback='https://uri.amap.com/navigation?';";
  h += "if(fromLat&&fromLng){fallback+='from='+fromLng+','+fromLat+',起点&';}";
  h += "fallback+='to='+toLng+','+toLat+','+encodeURIComponent(label)+'&mode=walk&callnative=1';";
  h += "if(isAndroid){";
  h += "var s='androidamap://navi?sourceApplication=EQNode&lat='+toLat+'&lon='+toLng+'&dev=0&style=2';";
  h += "var t=Date.now();window.location.href=s;";
  h += "setTimeout(function(){if(Date.now()-t<2500)window.location.href=fallback;},2000);";
  h += "}else if(isIOS){";
  h += "var s='iosamap://navi?sourceApplication=EQNode&lat='+toLat+'&lon='+toLng+'&dev=0&style=2';";
  h += "var t=Date.now();window.location.href=s;";
  h += "setTimeout(function(){if(Date.now()-t<2500)window.location.href=fallback;},2000);";
  h += "}else{window.open(fallback,'_blank');}";
  h += "}";
  h += "function sendMsg(){var t=document.getElementById('msgTarget').value;var m=document.getElementById('msgText').value;if(!m){alert('请输入消息内容');return;}var btn=event.target;btn.disabled=true;btn.textContent='发送中...';fetch('/sendMsg?target='+t+'&text='+encodeURIComponent(m)).then(r=>r.text()).then(a=>{document.getElementById('msgText').value='';btn.textContent='📤 发送消息';btn.disabled=false;alert(a);}).catch(function(e){btn.textContent='📤 发送消息';btn.disabled=false;alert('发送失败: '+e.message);});}";
  h += "function refreshMessages(){fetch('/api/messages').then(r=>r.json()).then(d=>{var h='';d.forEach(function(m){h+='<div style=\"font-size:12px;padding:2px 0;border-bottom:1px solid #333\"><span style=\"color:#f5c518\">[节点 '+m.from+']</span> '+m.text+'</div>';});document.getElementById('msgList').innerHTML=h||'暂无消息';});}";
  h += "setInterval(refreshMessages,3000);";
  h += "function refreshNodeData(){";
  h += "  fetch('/api/status').then(r=>r.json()).then(d=>{";
  h += "    document.getElementById('sampleCount').textContent = d.samples;";
  h += "    document.getElementById('inferCount').textContent = d.inferences;";
  h += "    document.getElementById('alertCount').textContent = d.alerts;";
  h += "    document.getElementById('relayCount').textContent = d.relays;";
  h += "  });";
  h += "  fetch('/api/neighbors').then(r=>r.json()).then(d=>{";
  h += "    var tb = document.getElementById('neighborTable');";
  h += "    if(d.length==0){";
  h += "      tb.innerHTML='<tr><td colspan=4>暂无邻居</td></tr>';";
  h += "    }else{";
  h += "      tb.innerHTML = d.map(function(n){";
  h += "        return '<tr><td>'+n.id+'</td><td>'+(n.dist?n.dist.toFixed(1):'?')+'m</td><td>'+n.rssi+'dBm</td><td>'+(n.lat?n.lat.toFixed(4):'?')+','+(n.lng?n.lng.toFixed(4):'?')+'</td></tr>';";
  h += "      }).join('');";
  h += "    }";
  h += "  });";
  h += "}";
  h += "setInterval(refreshNodeData, 5000);";  // 每5秒刷新一次
  h += "function refreshTime(){";
  h += "  fetch('/api/time').then(r=>r.text()).then(t=>{";
  h += "    document.getElementById('networkTime').textContent = t;";
  h += "  });";
  h += "}";
  h += "setInterval(refreshTime, 10000);";  // 每10秒刷新
  h += "</script></body></html>";
  return h;
}

// ==================== Core 0: 预测 + 三轴分类 + PCA + LoRa收发 ====================
void core0Task(void* pv) {
  Serial.println("[Core0] Started");
  float inputWindow[100][3];
  float predWaveform[3][300];
  float pcaOutPred[PCA_OUT_PRED];
  int16_t quantizedPred[PCA_OUT_PRED];

  for (;;) {
    if (xSemaphoreTake(windowReadySemaphore, portMAX_DELAY) == pdTRUE) {


      // 1. 取最近1秒窗口
      getRecentWindow(inputWindow, 100);
      Serial.printf("[数据检查] X通道前5点: %.3f, %.3f, %.3f, %.3f, %.3f\n",
                    inputWindow[0][0], inputWindow[1][0], inputWindow[2][0],
                    inputWindow[3][0], inputWindow[4][0]);



      float norm_factors[3] = { 1.0f, 1.0f, 1.0f };  // 默认值，防止除零

      for (int c = 0; c < 3; c++) {
        float ch_data[100];
        for (int t = 0; t < 100; t++) ch_data[t] = inputWindow[t][c];

        // 计算归一化因子（最大绝对值）
        float max_val = 0;
        for (int t = 0; t < 100; t++) {
          if (fabs(ch_data[t]) > max_val) max_val = fabs(ch_data[t]);
        }
        if (max_val < 1e-10f) max_val = 1.0f;
        norm_factors[c] = max_val;

        // 归一化
        for (int t = 0; t < 100; t++) ch_data[t] /= max_val;
        for (int t = 0; t < 100; t++) inputWindow[t][c] = ch_data[t];
      }

      // 2. 波形预测 → 未来3秒
      bool ok = runWaveformPredictor(inputWindow, predWaveform);
      if (ok) {

        Serial.printf("[Diag] PredWave X[0..2]: %.3f, %.3f, %.3f\n",
                      predWaveform[0][0], predWaveform[0][1], predWaveform[0][2]);
        Serial.printf("[Diag] PredWave Y[0..2]: %.3f, %.3f, %.3f\n",
                      predWaveform[1][0], predWaveform[1][1], predWaveform[1][2]);
        Serial.printf("[Diag] PredWave Z[0..2]: %.3f, %.3f, %.3f\n",
                      predWaveform[2][0], predWaveform[2][1], predWaveform[2][2]);
        // 3. PCA压缩预测波形 (900→5)
        float flat[900];
        int idx = 0;
        for (int ch = 0; ch < 3; ch++)
          for (int t = 0; t < 300; t++)
            flat[idx++] = predWaveform[ch][t];
        pcaProject900(flat, pcaOutPred);
        quantize(pcaOutPred, quantizedPred, PCA_OUT_PRED);

        // 计算每个通道当前帧的最大绝对值
        float peak[3] = { 0 };
        for (int c = 0; c < 3; c++) {
          for (int t = 0; t < 300; t++) {
            float abs_val = fabs(predWaveform[c][t]);
            if (abs_val > peak[c]) peak[c] = abs_val;
          }
        }



        // 计算每个通道的基线偏差峰值
        float dev_peak[3] = { 0 };
        for (int c = 0; c < 3; c++) {
          float baseline = (c == 2) ? Z_BASELINE : XY_BASELINE;
          for (int t = 0; t < 300; t++) {
            float dev = fabs(predWaveform[c][t] - baseline);
            if (dev > dev_peak[c]) dev_peak[c] = dev;
          }
        }

// 阈值（偏差超过此值视为异常）
#define Z_DEV_THRESH 0.35f   // Z轴偏离基线0.3以上（如0.2或0.8）
#define XY_DEV_THRESH 0.65f  // X/Y偏离基线0.35以上

        bool triggered = false;
        if (dev_peak[2] > Z_DEV_THRESH) {
          triggered = true;
          Serial.printf("🚨 [Z轴] 偏差峰值 %.3f > %.3f\n", dev_peak[2], Z_DEV_THRESH);
        }
        if (dev_peak[0] > XY_DEV_THRESH) {
          triggered = true;
          Serial.printf("🚨 [X轴] 偏差峰值 %.3f > %.3f\n", dev_peak[0], XY_DEV_THRESH);
        }
        if (dev_peak[1] > XY_DEV_THRESH) {
          triggered = true;
          Serial.printf("🚨 [Y轴] 偏差峰值 %.3f > %.3f\n", dev_peak[1], XY_DEV_THRESH);
        }

        // 打印偏差峰值（便于调参）
        Serial.printf("[偏差峰值] X:%.3f Y:%.3f Z:%.3f\n", dev_peak[0], dev_peak[1], dev_peak[2]);

        if (triggered) {
          enqueuePCAFrame(FRAME_DATA_PRED, quantizedPred, PCA_OUT_PRED, norm_factors);
          sendAlertFrame();
          Serial.printf("[Node] PRED PCA sent\n");
          predictionSent = true;
          predictionSentTime = millis();
          sharedPredPeak = fmax(fmax(dev_peak[0], dev_peak[1]), dev_peak[2]);
        } else {
          predictionSent = false;
          sharedPredPeak = 0;
        }

        // 6. 存共享数据
        xSemaphoreTake(dataMutex, portMAX_DELAY);
        memcpy(sharedPredWaveform, predWaveform, sizeof(sharedPredWaveform));
        memcpy(sharedPCAOut, pcaOutPred, sizeof(sharedPCAOut));
        memcpy(sharedQuantized, quantizedPred, sizeof(sharedQuantized));
        xSemaphoreGive(dataMutex);
      }

      // 7. 历史报警处理（Core1 Z轴检测到异常时触发）
      if (xSemaphoreTake(historyAlertSemaphore, 0) == pdTRUE) {
        if (!predictionSent || millis() - predictionSentTime > 10000) {
          // 取最近 300 个采样点（3 秒 × 100Hz）
          float histWindow[300][3];
          getRecentWindow(histWindow, 300);

          // 展平为 900 点：按通道排列，X0..X299, Y0..Y299, Z0..Z299
          float flat[900];
          int idx = 0;
          for (int ch = 0; ch < 3; ch++) {
            for (int t = 0; t < 300; t++) {
              flat[idx++] = histWindow[t][ch];
            }
          }

          float pcaOutHist[PCA_OUT_HIST];  // PCA_OUT_HIST = 10
          int16_t quantizedHist[PCA_OUT_HIST];
          pcaProject300(flat, pcaOutHist);
          quantize(pcaOutHist, quantizedHist, PCA_OUT_HIST);

          uint8_t payload[20];
          memcpy(payload, quantizedHist, PCA_OUT_HIST * 2);
          enqueueFrame(FRAME_DATA_HIST, payload, PCA_OUT_HIST * 2);
          sendAlertFrame();
          Serial.printf("[Node] HIST PCA sent\n");
        }
      }
    }

    handleLoRaRX();
    sendLoRaFromQueue();

    if (waitingForAck && millis() - ackSentTime > 2000) {
      if (ackRetryBuf) {
        Serial.println("⚠️ ACK timeout, retrying...");
        DXLR_SERIAL.write(ackRetryBuf, ackRetryLen);
        ackSentTime = millis();
        waitingForAck = false;
        free(ackRetryBuf);
        ackRetryBuf = nullptr;
      }
    }

    static uint32_t lastClean = 0;
    if (millis() - lastClean > 30000) {
      for (int i = 0; i < neighborCount; i++) {
        if (millis() - neighbors[i].lastSeen > 120000) {
          neighbors[i] = neighbors[--neighborCount];
          i--;
        }
      }
      lastClean = millis();
    }
    vTaskDelay(pdMS_TO_TICKS(1));
  }
}

// ==================== Core 1: 采集 + Z轴流式推理 + Web + 邻居探测 ====================
void core1Task(void* pv) {
  uint32_t lastHeartbeat = 0;
  Serial.println("[Core1] Started");
  uint32_t lastWindowTime = 0, lastSlideTime = 0, lastProbeTime = 0;

  for (;;) {
    // 1. 消费中断采集（主循环里读 I2C，中断只设标志）
    if (sampleReady) {
      sampleReady = false;
      Wire.beginTransmission(MPU6050_ADDR);
      Wire.write(0x3B);
      Wire.endTransmission(false);
      Wire.requestFrom(MPU6050_ADDR, 6);
      rawAX = (Wire.read() << 8) | Wire.read();
      rawAY = (Wire.read() << 8) | Wire.read();
      rawAZ = (Wire.read() << 8) | Wire.read();

      // ===== 旋转校正 =====
      float fax = (float)rawAX, fay = (float)rawAY, faz = (float)rawAZ;

      if (dmpReady && mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
        mpu.dmpGetQuaternion(&q, fifoBuffer);
        mpu.dmpGetGravity(&gravity, &q);
        mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);

        // 四元数转旋转矩阵
        float r11 = 1 - 2 * q.y * q.y - 2 * q.z * q.z;
        float r12 = 2 * (q.x * q.y - q.w * q.z);
        float r13 = 2 * (q.x * q.z + q.w * q.y);
        float r21 = 2 * (q.x * q.y + q.w * q.z);
        float r22 = 1 - 2 * q.x * q.x - 2 * q.z * q.z;
        float r23 = 2 * (q.y * q.z - q.w * q.x);
        float r31 = 2 * (q.x * q.z - q.w * q.y);
        float r32 = 2 * (q.y * q.z + q.w * q.x);
        float r33 = 1 - 2 * q.x * q.x - 2 * q.y * q.y;

        // 旋转加速度到地面坐标系
        fax = (r11 * rawAX + r12 * rawAY + r13 * rawAZ) / 4096.0f;
        fay = (r21 * rawAX + r22 * rawAY + r23 * rawAZ) / 4096.0f;
        faz = (r31 * rawAX + r32 * rawAY + r33 * rawAZ) / 4096.0f;



        if (isnan(fax) || isnan(fay) || isnan(faz)) {
          continue;  // 跳过此帧，不存入历史
        }
      } else {
        // DMP 未就绪，直接转换原始值为 g
        fax = rawAX / 4096.0f;
        fay = rawAY / 4096.0f;
        faz = rawAZ / 4096.0f;
      }

      // 毛刺过滤
      float max_abs_val = fmax(fmax(fabs(fax), fabs(fay)), fabs(faz));
      if (max_abs_val < 3.0f) {  // 合理的 g 值范围
        addToHistory(fax, fay, faz);
      } else {
        // 毛刺，丢弃
        static uint32_t glitch_count = 0;
        glitch_count++;
        if (glitch_count % 100 == 0) {
          Serial.printf("[过滤] 已丢弃 %d 个毛刺，最近值: %.3f, %.3f, %.3f\n",
                        glitch_count, fax, fay, faz);
        }
      }
      sampleCount++;
    }  // 结束 if (sampleReady)

    // 2. 每1秒发窗口信号给Core0
    if (historyFull && millis() - lastWindowTime >= 1000) {
      lastWindowTime = millis();
      xSemaphoreGive(windowReadySemaphore);
    }

    // 3. 滑动Z轴推理 (每150ms一次)
    if (historyCount >= 700 && millis() - lastSlideTime >= SLIDE_INTERVAL_MS) {
      lastSlideTime = millis();
      float histZ[700];
      getRecentAxis(histZ, 2, 7);           // 7 秒 = 700 点
      float hz = runInference(histZ, 700);  // 直接送入模型
      if (hz > 0.5f) {
        xSemaphoreTake(dataMutex, portMAX_DELAY);
        bool alreadySent = predictionSent;
        xSemaphoreGive(dataMutex);
        if (!alreadySent) {
          xSemaphoreGive(historyAlertSemaphore);
          Serial.printf("  [Core1] Z-axis alert: %.3f\n", hz);
        }
      }
    }

    // 4. Web服务器
    if (apMode && server) server->handleClient();

    // 5. 邻居探测 (每60秒)
    if (millis() - lastProbeTime >= 60000) {
      uint8_t dummy = 0;
      enqueueFrame(FRAME_PROBE, &dummy, 1);
      lastProbeTime = millis();
    }

    // 三边定位自校验
    static uint32_t lastTrilateral = 0;
    if (millis() - lastTrilateral >= 30000 && neighborCount >= 3) {
      verifyPositionByNeighbors();
      lastTrilateral = millis();
    }

    // 心跳帧
    if (millis() - lastHeartbeat >= 60000) {
      Serial.printf("[HB] gwDistance=%.1f m\n", gwDistance);
      uint8_t hbPayload[4];
      memcpy(hbPayload, &gwDistance, 4);
      enqueueFrame(0x0B, hbPayload, 4);
      lastHeartbeat = millis();
    }

    vTaskDelay(pdMS_TO_TICKS(5));
  }  // for (;;)
}  // core1Task

// ==================== 初始化 ====================
void setup() {



  Serial.begin(115200);
  delay(200);


  Serial.println("=== BOOT ===");
  Serial.printf("[PSRAM] Size: %d, Free: %d\n", ESP.getPsramSize(), ESP.getFreePsram());
  Serial.printf("[HEAP] Free: %d, Max block: %d\n", ESP.getFreeHeap(),
                heap_caps_get_largest_free_block(MALLOC_CAP_8BIT));

  windowReadySemaphore = xSemaphoreCreateBinary();
  historyAlertSemaphore = xSemaphoreCreateBinary();
  dataMutex = xSemaphoreCreateMutex();
  inferenceMutex = xSemaphoreCreateMutex();
  txQueue = xQueueCreate(TX_QUEUE_LEN, sizeof(uint8_t*));
  Serial.println("信号量/队列已创建");

  if (!FFat.begin(true)) {  // true = 挂载失败时自动格式化
    Serial.println("❌ FFat 挂载/格式化失败，使用硬编码配置");
    myLat = PREASSIGNED_LAT;
    myLng = PREASSIGNED_LNG;
    myAlt = PREASSIGNED_ALT;
    positionVerified = false;
  } else {
    Serial.println("✅ FFat 挂载成功");
    positionVerified = false;
    Serial.println("[5] Config loaded");
  }
  loadConfig();
  // DX-LR22 串口初始化
  DXLR_SERIAL.begin(DXLR_BAUDRATE, SERIAL_8N1, DXLR_RX, DXLR_TX);
  pinMode(DXLR_M0, OUTPUT);  // ← 加
  pinMode(DXLR_M1, OUTPUT);  // ← 加
  pinMode(DXLR_AUX, INPUT);  // ← 加
  Serial.println("[2] DX-LR22 begin");

  // ===== DX-LR22 初始化（进入 AT 模式 → 配置 → 退出） =====
  DXLR_SERIAL.println("+++");
  int cnt = 0;
  delay(2000);
  cnt = 0;
  while (DXLR_SERIAL.available() && cnt < 200) {
    uint8_t c = DXLR_SERIAL.read();
    Serial.printf("%02X ", c);  // 只输出 HEX，一行显示
    cnt++;
  }
  Serial.println();

  DXLR_SERIAL.println("AT+LEVEL0");  // SF11, BW125, CR4/8
  delay(2000);
  cnt = 0;
  while (DXLR_SERIAL.available() && cnt < 200) {
    uint8_t c = DXLR_SERIAL.read();
    Serial.printf("%02X ", c);  // 只输出 HEX，一行显示
    cnt++;
  }
  Serial.println();

  DXLR_SERIAL.println("AT+DRSSI1");  // 开启数据包 RSSI
  delay(2000);
  cnt = 0;
  while (DXLR_SERIAL.available() && cnt < 200) {
    uint8_t c = DXLR_SERIAL.read();
    Serial.printf("%02X ", c);  // 只输出 HEX，一行显示
    cnt++;
  }
  Serial.println();

  DXLR_SERIAL.println("AT+LBT1");  // 开启 LBT（先听后说）
  delay(2000);
  cnt = 0;
  while (DXLR_SERIAL.available() && cnt < 200) {
    uint8_t c = DXLR_SERIAL.read();
    Serial.printf("%02X ", c);  // 只输出 HEX，一行显示
    cnt++;
  }
  Serial.println();

  DXLR_SERIAL.println("+++");  // 退出 AT 模式
  delay(2000);
  cnt = 0;
  while (DXLR_SERIAL.available() && cnt < 200) {
    uint8_t c = DXLR_SERIAL.read();
    Serial.printf("%02X ", c);  // 只输出 HEX，一行显示
    cnt++;
  }
  Serial.println();

  digitalWrite(DXLR_M0, LOW);  // 透传模式
  digitalWrite(DXLR_M1, LOW);

  Serial.println("[3] DX-LR22 ready");









  Wire.begin(I2C_SDA, I2C_SCL);
  Wire.setClock(400000);  // 先用低速，提高稳定性
  Serial.println("[I2C] 总线已手动初始化");

  // ===== MPU6050 初始化（绕过 WHO_AM_I 检查） =====
  Serial.println("[6] 开始 MPU6050 初始化...");
  mpu.initialize();
  delay(50);



  // 通过 I2C 扫描确认传感器是否存在
  Wire.beginTransmission(MPU6050_ADDR);
  byte error = Wire.endTransmission();
  if (error == 0) {
    Serial.println("[6] MPU6050 在线（跳过 WHO_AM_I 检查）");

    uint8_t devStatus = mpu.dmpInitialize();
    if (devStatus == 0) {
      mpu.setDMPEnabled(true);
      dmpReady = true;
      Serial.println("[DMP] Initialized");
    } else {
      Serial.printf("[DMP] Init failed (code %d)\n", devStatus);
      dmpReady = false;
    }
  } else {
    Serial.printf("[6] MPU6050 不在线 (I2C error: %d)，跳过\n", error);
    dmpReady = false;
  }

  mpu.setFullScaleAccelRange(MPU6050_ACCEL_FS_8);
  Serial.println("[MPU] 已强制设置加速度量程为 ±8g");

  mpu.setDLPFMode(MPU6050_DLPF_BW_42);  // 你也可以用 MPU6050_DLPF_BW_42 等
  Serial.println("[MPU] 已开启低通滤波器");

  // 初始化波形预测模型
  if (!initWaveformPredictor()) {

    // 不卡死，Core0 里预测失败时也能正常运行其他逻辑
  } else {
    Serial.println("[7] initWaveformPredictor...");
  }

  registerViaWiFi();


  WiFi.mode(WIFI_AP);


  Serial.println("AP Started");
  


  server = new WebServer(80);
  server->on("/", []() {
    // 每 30 秒重新生成一次缓存
    if (cachedNodePage.length() == 0 || millis() - cachedNodePageTime > 30000) {
      cachedNodePage = genWebPage();
      cachedNodePageTime = millis();
      Serial.printf("[Web] Cached node page (%d bytes)\n", cachedNodePage.length());
    }
    server->send(200, "text/html", cachedNodePage);
  });
  server->on("/sendLocation", []() {
    uint8_t payload[12];
    memcpy(payload, &myLat, 4);
    memcpy(payload + 4, &myLng, 4);
    memcpy(payload + 8, &myAlt, 4);
    enqueueFrame(FRAME_PRESENCE, payload, 12);
    server->send(200, "text/plain", "📍 Location sent to gateway!");
  });
  server->on("/sendMsg", []() {
    if (server->hasArg("target") && server->hasArg("text")) {
      uint16_t target = server->arg("target").toInt();
      String text = server->arg("text");
      uint8_t payload[200];
      payload[0] = (target >> 8) & 0xFF;
      payload[1] = target & 0xFF;
      payload[2] = 0;
      int len = text.length() > 196 ? 196 : text.length();
      memcpy(payload + 3, text.c_str(), len);
      enqueueFrame(FRAME_MESSAGE, payload, len + 3);
      server->send(200, "text/plain", "✅ Sent");
    }else {
        server->send(400, "text/plain", "❌ 缺少参数");
    }
  });

  server->on("/api/messages", []() {
    String json = "[";
    for (int i = 0; i < msgCount; i++) {
      if (i > 0) json += ",";
      json += "{\"from\":" + String(msgStore[i].fromNode) + ",\"time\":" + String(msgStore[i].timestamp / 1000) + ",\"text\":\"" + String(msgStore[i].text) + "\"}";
    }
    json += "]";
    server->send(200, "application/json", json);
  });
  server->begin();
  apMode = true;

  server->on("/api/time", []() {
    server->send(200, "text/plain", getTimeString());
  });
  server->on("/api/status", []() {
    String json = "{";
    json += "\"samples\":" + String(sampleCount) + ",";
    json += "\"inferences\":" + String(inferenceCount) + ",";
    json += "\"alerts\":" + String(alertCount) + ",";
    json += "\"relays\":" + String(relayCount);
    json += "}";
    server->send(200, "application/json", json);
  });

  server->on("/api/neighbors", []() {
    String json = "[";
    for (int i = 0; i < neighborCount; i++) {
      if (i > 0) json += ",";
      json += "{";
      json += "\"id\":" + String(neighbors[i].nodeID) + ",";
      json += "\"dist\":" + String(neighbors[i].distance, 1) + ",";
      json += "\"rssi\":" + String(neighbors[i].rssi) + ",";
      json += "\"lat\":" + String(neighbors[i].lat, 6) + ",";
      json += "\"lng\":" + String(neighbors[i].lng, 6);
      json += "}";
    }
    json += "]";
    server->send(200, "application/json", json);
  });



  sampleTimer = timerBegin(0, 80, true);                    // 定时器0, 80分频(1μs), 向上计数
  timerAttachInterrupt(sampleTimer, &onSampleTimer, true);  // 上升沿触发
  timerAlarmWrite(sampleTimer, 10000, true);                // 10000μs = 100Hz
  timerAlarmEnable(sampleTimer);


  xTaskCreatePinnedToCore(core0Task, "Core0", 20480, NULL, 3, NULL, 0);
  xTaskCreatePinnedToCore(core1Task, "Core1", 49152, NULL, 2, NULL, 1);

  vTaskDelete(NULL);
}

void loop() {
  vTaskDelete(NULL);
}