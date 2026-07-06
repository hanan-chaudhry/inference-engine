#include "../../inc/kernals/softmax.h"
#include"blis.h"
void softmax(double* x, double* y, uint32_t n){
  double normalizer = 0;
  for(int i = 0;i < n;i++){
    y[i] = exp(x[i]);
    normalizer = normalizer + y[i];
  }
  bli_dinvscalv(BLIS_NO_CONJUGATE, n, &normalizer, y, 1);
}
