#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__
#include<stdint.h>
void softmax(const float* __restrict x, float* __restrict y, const uint32_t n);
#endif