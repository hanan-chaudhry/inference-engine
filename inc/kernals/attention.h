#ifndef __ATTENTION_H__
#define __ATTENTION_H__
#include<stdint.h>
#include<stdbool.h>
void attention(double* Q, double* K, double* V, uint32_t n, uint32_t d, double* out, 
                 bool masked);
#endif