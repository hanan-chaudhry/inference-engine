#include"../../inc/kernals/sigmoid.h"
#include <math.h>
void sigmoid(const float* __restrict x, float* __restrict y,const uint32_t len){
    for (int i = 0;i < len;i++) {
        y[i] = 1.0f / (1.0f + expf(-1 * x[i]));
    }
}