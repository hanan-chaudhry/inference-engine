#define _GNU_SOURCE

#include "../../inc/kernals/matmul.h"
#include <string.h>
#include <time.h>
#include <blis/blis.h>

void kernel_matmul_cpu_f32_forward(
    const float* a, const float* b, float* out,
    const uint32_t M, const uint32_t N, const uint32_t K
) {
    float alpha = 1.0f;
    float beta = 0.0f;

    dim_t rs_a = K, cs_a = 1;
    dim_t rs_b = N, cs_b = 1;
    dim_t rs_c = N, cs_c = 1;

    bli_sgemm(
        BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE,
        M, N, K,
        &alpha,
        (float*)a, rs_a, cs_a,
        (float*)b, rs_b, cs_b,
        &beta,
        out, rs_c, cs_c
    );
}




void kernel_matmul_vulkan_f32_forward(
    VkContext* ctx,
    const float* a, const float* b, float* out,
    const uint32_t M, const uint32_t N, const uint32_t K,
    double* time
) {
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    VkTensorBuffer vk_a = { 0 }, vk_b = { 0 }, vk_out = { 0 };
    vk_allocate_tensor(ctx, &vk_a, bytes_A);
    vk_allocate_tensor(ctx, &vk_b, bytes_B);
    vk_allocate_tensor(ctx, &vk_out, bytes_C);

    memcpy(vk_a.mapped_ptr, a, bytes_A);
    memcpy(vk_b.mapped_ptr, b, bytes_B);
    memset(vk_out.mapped_ptr, 0, bytes_C);

    VkDescriptorSetLayout desc_layout;
    VkPipelineLayout pipe_layout;
    vk_create_universal_layout(ctx, &desc_layout, &pipe_layout);

    VkPipeline pipeline = vk_create_compute_pipeline(ctx, "src/shaders/matmul.spv", pipe_layout);

    if (pipeline == VK_NULL_HANDLE) {
        printf("Error: Check shader path.\n");
        return;
    }

    VkTensorBuffer* buffers[] = { &vk_a, &vk_b, &vk_out };
    PushParams params = { 0 };
    params.M = M;
    params.N = N;
    params.K = K;

    uint32_t group_x = (N + 15) / 16;
    uint32_t group_y = (M + 15) / 16;

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    vk_dispatch(ctx, pipeline, pipe_layout, buffers, 3, &params, group_x, group_y, 1);
    // vkQueueWaitIdle(ctx->compute_queue);
    VkResult wait_res = vkQueueWaitIdle(ctx->compute_queue);
    if (wait_res != VK_SUCCESS) {
        printf("CRITICAL: GPU crashed or was killed by the OS! Error code: %d\n", wait_res);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    if (time != NULL) {
        *time = (end.tv_sec - start.tv_sec) +
            (end.tv_nsec - start.tv_nsec) / 1e9;
    }

    memcpy(out, vk_out.mapped_ptr, bytes_C);

    vk_free_tensor(ctx, &vk_a);
    vk_free_tensor(ctx, &vk_b);
    vk_free_tensor(ctx, &vk_out);

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);
}