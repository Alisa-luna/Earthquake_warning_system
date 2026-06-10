// ============================================
// tcn_inference.c — 精简FP32推理
// ============================================
#include "tcn_inference.h"
#include "tcn_weights.h"
#include <string.h>

#define MID_CH 16
#define HEAD_DIM 256
#define INPUT_LEN 100
#define OUTPUT_LEN 300
#define NUM_BLOCKS 8

// 静态缓冲区
static float feat[MID_CH * INPUT_LEN];
static float temp1[MID_CH * INPUT_LEN];
static float temp2[MID_CH * INPUT_LEN];
static float pooled[MID_CH];

// FP32 因果卷积
static void causal_conv1d(const float* input, const float* weight, const float* bias,
                          int in_ch, int out_ch, int len, int kernel, int dilation,
                          float* output) {
    int pad = (kernel - 1) * dilation;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int t = 0; t < len; t++) {
            float sum = bias[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int pos = t - k * dilation;
                    if (pos >= 0) {
                        int w_idx = ((oc * in_ch) + ic) * kernel + k;
                        sum += input[ic * len + pos] * weight[w_idx];
                    }
                }
            }
            output[oc * len + t] = sum;
        }
    }
}

// ReLU
static void relu(float* x, int n) {
    for (int i = 0; i < n; i++)
        if (x[i] < 0) x[i] = 0;
}

// 逐元素加法
static void add(float* a, const float* b, int n) {
    for (int i = 0; i < n; i++) a[i] += b[i];
}

// 全局平均池化
static void global_avg_pool(const float* x, int ch, int len, float* out) {
    for (int c = 0; c < ch; c++) {
        float sum = 0;
        for (int t = 0; t < len; t++) sum += x[c * len + t];
        out[c] = sum / len;
    }
}

// 全连接
static void linear(const float* x, const float* w, const float* b,
                   int in_dim, int out_dim, int use_relu, float* out) {
    for (int o = 0; o < out_dim; o++) {
        float sum = b[o];
        for (int i = 0; i < in_dim; i++)
            sum += x[i] * w[o * in_dim + i];
        out[o] = use_relu ? (sum > 0 ? sum : 0) : sum;
    }
}

void tcn_predict(const float input[100][3], float output[300]) {
    // 1. 转置输入 (100,3) → (3,100)
    float in[3 * 100];
    for (int c = 0; c < 3; c++)
        for (int t = 0; t < 100; t++)
            in[c * 100 + t] = input[t][c];

    // 2. Stem: 因果卷积 + ReLU
    causal_conv1d(in, stem_conv_weight, stem_conv_bias, 3, MID_CH, 100, 7, 1, feat);
    relu(feat, MID_CH * 100);

    // 3. TCN Blocks
    for (int b = 0; b < NUM_BLOCKS; b++) {
        const float* w1 = (const float*[]){
            tcn_blk0_conv1_weight, tcn_blk1_conv1_weight, tcn_blk2_conv1_weight,
            tcn_blk3_conv1_weight, tcn_blk4_conv1_weight, tcn_blk5_conv1_weight,
            tcn_blk6_conv1_weight, tcn_blk7_conv1_weight
        }[b];
        const float* b1 = (const float*[]){
            tcn_blk0_conv1_bias, tcn_blk1_conv1_bias, tcn_blk2_conv1_bias,
            tcn_blk3_conv1_bias, tcn_blk4_conv1_bias, tcn_blk5_conv1_bias,
            tcn_blk6_conv1_bias, tcn_blk7_conv1_bias
        }[b];
        const float* w2 = (const float*[]){
            tcn_blk0_conv2_weight, tcn_blk1_conv2_weight, tcn_blk2_conv2_weight,
            tcn_blk3_conv2_weight, tcn_blk4_conv2_weight, tcn_blk5_conv2_weight,
            tcn_blk6_conv2_weight, tcn_blk7_conv2_weight
        }[b];
        const float* b2 = (const float*[]){
            tcn_blk0_conv2_bias, tcn_blk1_conv2_bias, tcn_blk2_conv2_bias,
            tcn_blk3_conv2_bias, tcn_blk4_conv2_bias, tcn_blk5_conv2_bias,
            tcn_blk6_conv2_bias, tcn_blk7_conv2_bias
        }[b];
        
        int dilation = 1 << b;
        
        // 保存残差
        memcpy(temp2, feat, MID_CH * 100 * sizeof(float));
        
        // Conv1 + ReLU
        causal_conv1d(feat, w1, b1, MID_CH, MID_CH, 100, 3, dilation, temp1);
        relu(temp1, MID_CH * 100);
        
        // Conv2 + ReLU
        causal_conv1d(temp1, w2, b2, MID_CH, MID_CH, 100, 3, dilation, feat);
        relu(feat, MID_CH * 100);
        
        // 残差连接
        add(feat, temp2, MID_CH * 100);
    }

    // 4. 全局池化
    global_avg_pool(feat, MID_CH, 100, pooled);

    // 5. 全连接头
    float h0[HEAD_DIM], h1[HEAD_DIM];
    linear(pooled, head_dense0_weight, head_dense0_bias, MID_CH, HEAD_DIM, 1, h0);
    linear(h0, head_dense1_weight, head_dense1_bias, HEAD_DIM, HEAD_DIM, 1, h1);
    linear(h1, head_dense2_weight, head_dense2_bias, HEAD_DIM, OUTPUT_LEN, 0, output);
}