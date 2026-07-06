#include"blis.h"
#include"../../inc/kernals/silu.h"

// explicitely call sigmoid ??
void SILU(double* x, double* y, uint32_t len){
    for(int i = 0;i < len;i++){
        y[i] = x[i] / (1 + exp(-1 * x[i]));
    }
}