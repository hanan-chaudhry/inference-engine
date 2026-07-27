#include"../../inc/kernals/relu.h"
#include <string.h>
void RELU(const float* __restrict x, float* __restrict y, const uint32_t len){
    uint64_t mask = 0x7FFFFFFFFFFFFFFF;
    for (int i = 0;i < len;i++) {
        uint64_t bits;
        memcpy(&bits, (x + i), sizeof(bits));
        memcpy((y + i), (x + i), sizeof(float));
        uint64_t val = bits;
        val &= mask;
        float value;
        memcpy(&value, &val, sizeof(bits));
        y[i] += value;
        y[i] /= 2.0f;
    }
}

// void RELU(float* x, uint32_t len) {
//     for (int i = 0;i < len;i++) {
//         fmaxf(0.0f, x[i]);
//     }
//     return;
// }
