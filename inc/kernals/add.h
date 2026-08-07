#ifndef ADD_H
#define ADD_H
#include<stdio.h>
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_add_cpu_f32_forward(
    const float* a, const float* b,
    float* out,
    const size_t len
);

void kernel_add_vulkan_f32_forward(
    VkContext* ctx,
    const float* a, const float* b,
    float* out,
    const size_t len
);

#endif