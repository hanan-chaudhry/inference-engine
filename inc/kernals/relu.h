#ifndef __RELU_H__
#define __RELU_H__
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_relu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const uint32_t len);

void kernel_relu_vulkan_f32_forward(
    VkContext* ctx,
    const float* x,
    float* y,
    const size_t n
);

#endif