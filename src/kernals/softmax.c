#include "../../inc/kernals/softmax.h"
#include"blis.h"
void softmax(double* x, double* y, uint32_t n){ // aoun bhai se poochna maximum vocabulary of any LLM
  double normalizer = 0;
  for(int i = 0;i < n;i++){
    y[i] = exp(x[i]);
    normalizer = normalizer + y[i];
  }
  bli_dinvscalv(BLIS_NO_CONJUGATE, n, &normalizer, y, 1);
}


// int main(){
//   double x[3] = {1.2, 0.5, -0.3};
//   double y[3];
//   uint16_t n = 3;
//   softmax(x, y, &n);
//   for(int i = 0;i < n;i++){
//     printf("%f", y[i]);
//     printf(" ");
//   }
//   return 0;
// }
