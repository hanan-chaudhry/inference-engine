#include"../../inc/kernals/silu.h"
#include <math.h>

void SILU(const float* __restrict x, float* __restrict y, const uint32_t len){
    for (int i = 0;i < len;i++) {
        y[i] = x[i] / (1.0f + expf(-1 * x[i]));
    }
}