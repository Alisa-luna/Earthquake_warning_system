#include "waveform_inference_int8.h"

bool initWaveformPredictor() { return true; }
bool runWaveformPredictor(float input[100][3], float output[3][300]) {
    waveform_predict(input, output);
    return true;
}