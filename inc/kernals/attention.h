#ifndef __ATTENTION_H__
#define __ATTENTION_H__
#include<stdio.h>
#include<stdint.h>
#include<stdbool.h>
void attention(double* Q, double* K, double* V, uint16_t n, uint16_t d, double* out, 
                 bool masked);
#endif