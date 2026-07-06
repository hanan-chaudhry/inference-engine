#include"blis.h"
#include"../../inc/kernals/sigmoid.h"
void sigmoid(double* x, double* y, uint32_t len){
    for(int i = 0;i < len;i++){
        y[i] = 1 / (1 + exp(-1 * x[i]));
    }
}