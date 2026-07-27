#ifndef __MULTIATTENTION_H__
#define __MULTIATTENTION_H__

#include<stdint.h>
void multiAttention(const float* __restrict W_Q,const float* __restrict W_K, 
    const float* __restrict W_V,const float* __restrict X, const uint32_t n, 
    const uint32_t d, const uint32_t q_h, const uint32_t kv_h, 
    const float* __restrict W_O, float* __restrict out, const bool masked);
#endif