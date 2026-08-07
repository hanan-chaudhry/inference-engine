#include"../../inc/kernals/relu.h"
#include <string.h>

#define WORKGROUP_SIZE 256

void kernel_relu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const uint32_t len) {
    uint64_t mask = 0x7FFFFFFFFFFFFFFF;
    for (int i = 0;i < len;i++) {
        uint64_t bits;
        memcpy(&bits, (x + i), sizeof(bits));
        memcpy((y + i), (x + i), sizeof(float));
        uint64_t val = bits;
        val &= mask;
        float value;
        memcpy(&value, &val, sizeof(bits));
        y[i] += value;
        y[i] /= 2.0f;
    }
}

// void kernel_relu_cpu_f32_forward(float* x, uint32_t len) {
//     for (int i = 0;i < len;i++) {
//         fmaxf(0.0f, x[i]);
//     }
//     return;
// }

void kernel_relu_vulkan_f32_forward(
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

    VkPipeline pipeline = vk_create_compute_pipeline(ctx, "src/shaders/relu.spv", pipe_layout);

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
