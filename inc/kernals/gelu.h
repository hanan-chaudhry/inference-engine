#ifndef GELU_H
#define GELU_H
#include<stdio.h>
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_gelu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const size_t length);

void kernel_gelu_vulkan_f32_forward(
    VkContext* ctx,
    const float* x,
    float* y,
    const size_t len
);

#endif