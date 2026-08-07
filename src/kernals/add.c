#include"../../inc/kernals/add.h"
#include <stddef.h>
#include <string.h>

#define WORKGROUP_SIZE 16

void kernel_add_cpu_f32_forward(
    const float* a,
    const float* b,
    float* out,
    const size_t len
) {
    for (size_t i = 0; i < len; i++) {
        out[i] = a[i] + b[i];
    }
}


void kernel_add_vulkan_f32_forward(
    VkContext* ctx,
    const float* a, const float* b,
    float* out,
    const size_t len
) {
    size_t byte_size = len * sizeof(float);

    // 1. Allocate temporary VRAM tensors
    VkTensorBuffer vk_a, vk_b, vk_out;
    vk_allocate_tensor(ctx, &vk_a, byte_size);
    vk_allocate_tensor(ctx, &vk_b, byte_size);
    vk_allocate_tensor(ctx, &vk_out, byte_size);

    // 2. Copy CPU data to GPU mapped pointers
    memcpy(vk_a.mapped_ptr, a, byte_size);
    memcpy(vk_b.mapped_ptr, b, byte_size);

    memset(vk_out.mapped_ptr, 0, byte_size);

    // 3. Load the pipeline (later we will do this once globally only)
    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    vk_create_universal_layout(ctx, &desc_layout, &pipe_layout);

    VkPipeline pipeline = vk_create_compute_pipeline(ctx, "src/shaders/add.spv", pipe_layout);

    // 4. Setup Dispatch parameters
    VkTensorBuffer* buffers[] = { &vk_a, &vk_b, &vk_out };
    PushParams params = { 0 };
    params.N = len;

    uint32_t group_x = (len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    // 5. dispatch to the GPU
    vk_dispatch(ctx, pipeline, pipe_layout, buffers, 3, &params, group_x, 1, 1);

    // 6. wait for gpu ot finish working before reading the results
    vkQueueWaitIdle(ctx->compute_queue);

    // 7. Copy the GPU result back to the user's CPU array
    memcpy(out, vk_out.mapped_ptr, byte_size);

    // 8. Cleanup temporary VRAM and pipelines
    vk_free_tensor(ctx, &vk_a);
    vk_free_tensor(ctx, &vk_b);
    vk_free_tensor(ctx, &vk_out);

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
}
