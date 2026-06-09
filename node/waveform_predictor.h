#ifndef WAVEFORM_PREDICTOR_H
#define WAVEFORM_PREDICTOR_H

#include <Arduino.h>

// 延迟初始化：在 USER01253-project-1_inferencing.h 之后调用
bool initWaveformPredictor();
bool runWaveformPredictor(float input[100][3], float output[3][300]);

#endif