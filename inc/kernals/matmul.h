#ifndef MATMUL_H
#define MATMUL_H
#include<stdio.h>
#include<stdint.h>
#include "../backend/vk_core.h"
#include "../backend/vk_mem.h"
#include "../backend/vk_ops.h"

void kernel_matmul_cpu_f32_forward(
);

void kernel_matmul_vulkan_f32_forward(
    VkContext* ctx
);

#endif