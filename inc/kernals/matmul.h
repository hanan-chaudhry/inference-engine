#ifndef MATMUL_H
#define MATMUL_H
#include<stdio.h>
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_matmul_cpu_f32_forward(
    const float* a, const float* b, float* out,
    const uint32_t M, const uint32_t N, const uint32_t K
);

void kernel_matmul_vulkan_f32_forward(
    VkContext* ctx,
    const float* a, const float* b, float* out,
    const uint32_t M, const uint32_t N, const uint32_t K
);

#endif