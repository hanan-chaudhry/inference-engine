#ifndef __ATTENTION_H__
#define __ATTENTION_H__
#include<stdint.h>
#include<stdbool.h>
void attention(const float* __restrict Q, const float* __restrict K, 
    const float* __restrict V,const uint32_t n, const uint32_t d, float*__restrict out, 
    const bool masked);
#endif