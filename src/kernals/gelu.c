#include"../../inc/kernals/gelu.h"
#include <math.h>

#define WORKGROUP_SIZE 256

void kernel_gelu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const size_t length) {
    const float half = 0.5f;
    const float root_two_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < length; i++) {
        float x1 = x[i];
        float x3 = x1 * x1 * x1;
        float inner = root_two_over_pi * (x1 + coeff * x3);
        y[i] = half * x1 * (1.0f + tanhf(inner));
    }
}

void kernel_gelu_vulkan_f32_forward(
    VkContext* ctx,
    const float* x,
    float* y,
    const size_t n
) {
    size_t byte_size = n * sizeof(float);

    VkTensorBuffer vk_x = { 0 };
    vk_allocate_tensor(ctx, &vk_x, byte_size);

    memcpy(vk_x.mapped_ptr, x, byte_size);

    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    vk_create_universal_layout(ctx, &desc_layout, &pipe_layout);

    VkPipeline pipeline = vk_create_compute_pipeline(ctx, "src/shaders/gelu.spv", pipe_layout);

    VkTensorBuffer* buffers[] = { &vk_x, &vk_x, &vk_x };
    PushParams params = { 0 };
    params.N = n;

    uint32_t group_x = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    vk_dispatch(ctx, pipeline, pipe_layout, buffers, 3, &params, group_x, 1, 1);

    vkQueueWaitIdle(ctx->compute_queue);

    memcpy(y, vk_x.mapped_ptr, byte_size);

    vk_free_tensor(ctx, &vk_x);

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
}