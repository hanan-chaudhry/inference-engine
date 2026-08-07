#ifndef __SILU_H__
#define __SILU_H__
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_silu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const uint32_t len);

void kernel_silu_vulkan_f32_forward(
    VkContext* ctx,
    const float* x,
    float* y,
    const size_t n
);

#endif