#include "../../inc/kernals/scale.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

void kernel_scale_cpu_f32(float* a, float alpha, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        a[i] = a[i] * alpha;
    }
}

void kernel_scale_vulkan_f32(
    VkContext* ctx,
    float* a,
    float alpha,
    uint32_t n
) {
    size_t byte_size = n * sizeof(float);

    VkTensorBuffer vk_a = { 0 };
    vk_allocate_tensor(ctx, &vk_a, byte_size);
    memcpy(vk_a.mapped_ptr, a, byte_size);

    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    vk_create_universal_layout(ctx, &desc_layout, &pipe_layout);

    VkPipeline pipeline = vk_create_compute_pipeline(ctx, "src/shaders/scale.spv", pipe_layout);
    if (pipeline == VK_NULL_HANDLE) {
        printf("Error: Check scale shader path.\n");
        return;
    }

    // Dispatch 
    VkTensorBuffer* buffers[] = { &vk_a, &vk_a, &vk_a };

    PushParams params = { 0 };
    params.N = n;
    params.eps = alpha;

    uint32_t group_x = (n + 15) / 16;

    vk_dispatch(ctx, pipeline, pipe_layout, buffers, 3, &params, group_x, 1, 1);

    VkResult wait_res = vkQueueWaitIdle(ctx->compute_queue);
    if (wait_res != VK_SUCCESS) {
        printf("CRITICAL: GPU crashed or OS kill! Code: %d\n", wait_res);
    }

    memcpy(a, vk_a.mapped_ptr, byte_size);

    vk_free_tensor(ctx, &vk_a);
    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
}