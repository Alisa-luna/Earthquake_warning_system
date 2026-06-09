#ifndef CONFIG_H
#define CONFIG_H

// ==================== 节点身份 ====================
#define NODE_ID          0x0001
#define GATEWAY_ID       0xFFFF

// ==================== 预分配坐标 ====================
#define PREASSIGNED_LAT  31.230416
#define PREASSIGNED_LNG  121.473701
#define PREASSIGNED_ALT  15.0

// ==================== LoRa 参数（DX-LR22 已预设，仅保留记录）====================
#define LORA_FREQ        433E6

// ==================== AI 推理 ====================
#define WINDOW_SIZE      300
#define AI_THRESHOLD     0.5f
#define YELLOW_THRESHOLD 0.5f
#define ORANGE_THRESHOLD 0.7f
#define VOTE_COUNT       2
#define PRE_TRIGGER_SEC  5
#define POST_TRIGGER_SEC 3

// ==================== PCA 参数 ====================
#define PCA_INPUT_DIM    300
#define PCA_OUTPUT_DIM   20

// ==================== WiFi 重连 ====================
#define WIFI_RETRY_DELAY  10000
#define WIFI_RETRY_COUNT  5

// ==================== 调试 ====================
#define SERIAL_BAUD      115200

// ==================== 网络状态枚举 ====================
enum NetworkState {
    NET_DISCONNECTED,
    NET_WIFI,
    NET_4G_FALLBACK
};

#endif