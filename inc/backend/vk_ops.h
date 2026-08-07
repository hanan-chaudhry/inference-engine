#pragma once
#include <stdint.h>
#include "vk_core.h"
#include "vk_mem.h"

typedef struct {
    uint32_t batch;     // Batch size
    uint32_t M;         // Rows (or total elements for 1D arrays)
    uint32_t N;         // Columns
    uint32_t K;         // Inner dimension for MatMul
    float eps;          // Epsilon for normalization
    uint32_t flags;     // Additional flags for future use
} PushParams;

// The Universal Dispatcher
void vk_dispatch(
    VkContext* ctx,
    VkPipeline pipeline,
    VkPipelineLayout layout,
    VkTensorBuffer** buffers,
    uint32_t buffer_count,
    PushParams* params,
    uint32_t group_x, uint32_t group_y, uint32_t group_z
);