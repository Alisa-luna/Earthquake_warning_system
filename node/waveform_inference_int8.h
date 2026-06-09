#ifndef WAVEFORM_INFERENCE_INT8_H
#define WAVEFORM_INFERENCE_INT8_H

#ifdef __cplusplus
extern "C" {
#endif

void waveform_predict(float input[100][3], float output[3][300]);

#ifdef __cplusplus
}
#endif

#endif