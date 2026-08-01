#ifndef SCALE_H
#define SCALE_H

#include <stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_scale_cpu_f32(
    float* tensor,
    float alpha,
    uint32_t n
);

void kernel_scale_vulkan_f32(
    VkContext* ctx,
    float* tensor,
    float alpha,
    uint32_t n
);

#endif // SCALE_H