#ifndef ADD_H
#define ADD_H
#include<stdio.h>
#include<stdint.h>

void kernel_add_cpu_f32_forward(
    const float* a, const float* b,
    float* out,
    const size_t length
);


#endif