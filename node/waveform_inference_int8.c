#include "waveform_inference_int8.h"
#include "weights_int8.h"
#include <string.h>
#include <math.h>

// 扩张卷积块权重指针数组（7个block）
const int8_t* dil_conv1_w[7] = {
    dil_block_0_conv1_weight_q, dil_block_1_conv1_weight_q, dil_block_2_conv1_weight_q,
    dil_block_3_conv1_weight_q, dil_block_4_conv1_weight_q, dil_block_5_conv1_weight_q,
    dil_block_6_conv1_weight_q
};
const float dil_conv1_ws[7] = {
    dil_block_0_conv1_weight_scale, dil_block_1_conv1_weight_scale, dil_block_2_conv1_weight_scale,
    dil_block_3_conv1_weight_scale, dil_block_4_conv1_weight_scale, dil_block_5_conv1_weight_scale,
    dil_block_6_conv1_weight_scale
};
const float* dil_conv1_b[7] = {
    dil_block_0_conv1_bias, dil_block_1_conv1_bias, dil_block_2_conv1_bias,
    dil_block_3_conv1_bias, dil_block_4_conv1_bias, dil_block_5_conv1_bias,
    dil_block_6_conv1_bias
};

const int8_t* dil_conv2_w[7] = {
    dil_block_0_conv2_weight_q, dil_block_1_conv2_weight_q, dil_block_2_conv2_weight_q,
    dil_block_3_conv2_weight_q, dil_block_4_conv2_weight_q, dil_block_5_conv2_weight_q,
    dil_block_6_conv2_weight_q
};
const float dil_conv2_ws[7] = {
    dil_block_0_conv2_weight_scale, dil_block_1_conv2_weight_scale, dil_block_2_conv2_weight_scale,
    dil_block_3_conv2_weight_scale, dil_block_4_conv2_weight_scale, dil_block_5_conv2_weight_scale,
    dil_block_6_conv2_weight_scale
};
const float* dil_conv2_b[7] = {
    dil_block_0_conv2_bias, dil_block_1_conv2_bias, dil_block_2_conv2_bias,
    dil_block_3_conv2_bias, dil_block_4_conv2_bias, dil_block_5_conv2_bias,
    dil_block_6_conv2_bias
};

// 静态缓冲区（FP32，用于激活值）
static float stem_out[16 * 100];
static float block_in[16 * 100];
static float block_temp[16 * 100];
static float block_out[16 * 100];
static float pooled[16 * 4];
static float branch0[8 * 4], branch1[8 * 4], branch2[8 * 4];
static float concat[24 * 4];
static float flat[96];
static float dense0_out[768], dense1_out[768], final_out[900];


// INT8 卷积（带反量化，bias 直接 FP32）
static void conv1d_int8(const float* input, int in_ch, int in_len,
                        const int8_t* weight_q, float weight_scale,
                        const float* bias,
                        int out_ch, int kernel, int stride, int padding, int dilation,
                        float* output) {
    int out_len = (in_len + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    for (int oc = 0; oc < out_ch; oc++) {
        for (int ol = 0; ol < out_len; ol++) {
            float sum = bias[oc];  // 直接 FP32，无需反量化
            for (int ic = 0; ic < in_ch; ic++) {
                for (int k = 0; k < kernel; k++) {
                    int il = ol * stride + k * dilation - padding;
                    if (il >= 0 && il < in_len) {
                        int w_idx = oc * (in_ch * kernel) + ic * kernel + k;
                        sum += input[ic * in_len + il] * (weight_q[w_idx] * weight_scale);
                    }
                }
            }
            output[oc * out_len + ol] = sum;
        }
    }
}

// ReLU
static void relu(float* data, int size) {
    for (int i = 0; i < size; i++)
        if (data[i] < 0) data[i] = 0;
}

// Average Pool 1D
static void avgpool1d(const float* input, int channels, int in_len,
                      int pool_size, int stride, float* output) {
    int out_len = (in_len - pool_size) / stride + 1;
    for (int c = 0; c < channels; c++) {
        for (int ol = 0; ol < out_len; ol++) {
            float sum = 0;
            for (int k = 0; k < pool_size; k++) {
                int il = ol * stride + k;
                sum += input[c * in_len + il];
            }
            output[c * out_len + ol] = sum / pool_size;
        }
    }
}

// INT8 全连接（带反量化，bias 直接 FP32）
static void linear_int8(const float* input, int in_dim,
                        const int8_t* weight_q, float weight_scale,
                        const float* bias,
                        int out_dim, float* output) {
    for (int o = 0; o < out_dim; o++) {
        float sum = bias[o];
        for (int i = 0; i < in_dim; i++) {
            sum += input[i] * (weight_q[o * in_dim + i] * weight_scale);
        }
        output[o] = sum;
    }
}

void waveform_predict(float input[100][3], float output[3][300]) {

    memset(stem_out, 0, sizeof(stem_out));
    memset(block_in, 0, sizeof(block_in));
    memset(block_temp, 0, sizeof(block_temp));
    memset(block_out, 0, sizeof(block_out));
    memset(pooled, 0, sizeof(pooled));
    memset(branch0, 0, sizeof(branch0));
    memset(branch1, 0, sizeof(branch1));
    memset(branch2, 0, sizeof(branch2));
    memset(concat, 0, sizeof(concat));
    memset(flat, 0, sizeof(flat));
    memset(dense0_out, 0, sizeof(dense0_out));
    memset(dense1_out, 0, sizeof(dense1_out));
    memset(final_out, 0, sizeof(final_out));

    // 1. 输入转置
    float in_buf[3 * 100];
    for (int c = 0; c < 3; c++)
        for (int t = 0; t < 100; t++)
            in_buf[c * 100 + t] = input[t][c];

    // 2. Stem
    conv1d_int8(in_buf, 3, 100, stem_conv_weight_q, stem_conv_weight_scale,
                stem_conv_bias, 16, 7, 1, 3, 1, stem_out);
    relu(stem_out, 16 * 100);

    // 3. Dilated blocks
    memcpy(block_in, stem_out, sizeof(block_in));
    int dilations[] = {1, 2, 4, 8, 16, 32, 64};
    for (int b = 0; b < 7; b++) {
        conv1d_int8(block_in, 16, 100, dil_conv1_w[b], dil_conv1_ws[b],
                    dil_conv1_b[b], 16, 3, 1, dilations[b], dilations[b], block_temp);
        relu(block_temp, 16 * 100);
        conv1d_int8(block_temp, 16, 100, dil_conv2_w[b], dil_conv2_ws[b],
                    dil_conv2_b[b], 16, 1, 1, 0, 1, block_out);
        relu(block_out, 16 * 100);
        for (int i = 0; i < 16 * 100; i++) block_out[i] += block_in[i];
        memcpy(block_in, block_out, sizeof(block_in));
    }

    // 4. Time Segment
    avgpool1d(block_out, 16, 100, 25, 25, pooled);

    // 5. Branches
    conv1d_int8(pooled, 16, 4, branch_0_weight_q, branch_0_weight_scale,
                branch_0_bias, 8, 3, 1, 1, 1, branch0);
    relu(branch0, 8 * 4);
    conv1d_int8(pooled, 16, 4, branch_1_weight_q, branch_1_weight_scale,
                branch_1_bias, 8, 3, 1, 2, 2, branch1);
    relu(branch1, 8 * 4);
    conv1d_int8(pooled, 16, 4, branch_2_weight_q, branch_2_weight_scale,
                branch_2_bias, 8, 5, 1, 4, 2, branch2);
    relu(branch2, 8 * 4);

    // 6. Concat + Flatten
    memcpy(concat, branch0, 32 * sizeof(float));
    memcpy(concat + 32, branch1, 32 * sizeof(float));
    memcpy(concat + 64, branch2, 32 * sizeof(float));
    for (int i = 0; i < 96; i++) flat[i] = concat[i];

    // 7. Dense layers
    linear_int8(flat, 96, dense0_weight_q, dense0_weight_scale,
                dense0_bias, 768, dense0_out);
    relu(dense0_out, 768);
    linear_int8(dense0_out, 768, dense1_weight_q, dense1_weight_scale,
                dense1_bias, 768, dense1_out);
    relu(dense1_out, 768);
    linear_int8(dense1_out, 768, dense_out_weight_q, dense_out_weight_scale,
                dense_out_bias, 900, final_out);

    // 8. Reshape to (3, 300)
    for (int c = 0; c < 3; c++)
        for (int t = 0; t < 300; t++)
            output[c][t] = final_out[c * 300 + t];
}