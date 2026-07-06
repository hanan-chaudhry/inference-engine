#ifndef __MULTIATTENTION_H__
#define __MULTIATTENTION_H__

#include<stdint.h>
void multiAttention(double* W_Q, double* W_K, double* W_V, double* X,
                            uint32_t n, uint32_t d, uint32_t q_h, uint32_t kv_h, 
                            double* W_O, double* out, bool masked);
#endif