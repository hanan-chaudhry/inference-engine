#include "../../inc/kernals/softmax.h"
#include"blis.h"
void softmax(const float* __restrict x, float* __restrict y, const uint32_t n){
  float normalizer = 0.0f;
  for (int i = 0;i < n;i++) {
    y[i] = expf(x[i]);
    normalizer = normalizer + y[i];
  }
  bli_sinvscalv(BLIS_NO_CONJUGATE, n, &normalizer, y, 1);
}
