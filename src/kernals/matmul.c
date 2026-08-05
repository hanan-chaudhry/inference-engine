#define _GNU_SOURCE

#include "../../inc/kernals/matmul.h"
#include <string.h>
#include <time.h>
#include <blis/blis.h>

#define CALC_MS(start, end) (((end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9) * 1000.0)

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
    struct timespec t0, t1, t2, t3, t4, t5, t6;

    // --- PHASE 0: Start ---
    clock_gettime(CLOCK_MONOTONIC, &t0);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    VkTensorBuffer vk_a = { 0 }, vk_b = { 0 }, vk_out = { 0 };
    vk_allocate_tensor(ctx, &vk_a, bytes_A);
    vk_allocate_tensor(ctx, &vk_b, bytes_B);
    vk_allocate_tensor(ctx, &vk_out, bytes_C);

    // --- PHASE 1: Allocation complete ---
    clock_gettime(CLOCK_MONOTONIC, &t1);

    memcpy(vk_a.mapped_ptr, a, bytes_A);
    memcpy(vk_b.mapped_ptr, b, bytes_B);
    memset(vk_out.mapped_ptr, 0, bytes_C);

    // --- PHASE 2: Copy CPU -> GPU complete ---
    clock_gettime(CLOCK_MONOTONIC, &t2);

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

    // --- PHASE 3: Pipeline & Setup complete ---
    clock_gettime(CLOCK_MONOTONIC, &t3);

    vk_dispatch(ctx, pipeline, pipe_layout, buffers, 3, &params, group_x, group_y, 1);

    VkResult wait_res = vkQueueWaitIdle(ctx->compute_queue);
    if (wait_res != VK_SUCCESS) {
        printf("CRITICAL: GPU crashed or was killed by the OS! Error code: %d\n", wait_res);
    }

    // --- PHASE 4: GPU Dispatch complete ---
    clock_gettime(CLOCK_MONOTONIC, &t4);

    if (time != NULL) {
        *time = (t4.tv_sec - t3.tv_sec) + (t4.tv_nsec - t3.tv_nsec) / 1e9;
    }

    memcpy(out, vk_out.mapped_ptr, bytes_C);

    // --- PHASE 5: Copy GPU -> CPU complete ---
    clock_gettime(CLOCK_MONOTONIC, &t5);

    vk_free_tensor(ctx, &vk_a);
    vk_free_tensor(ctx, &vk_b);
    vk_free_tensor(ctx, &vk_out);

    vkDestroyPipeline(ctx->device, pipeline, NULL);
    vkDestroyPipelineLayout(ctx->device, pipe_layout, NULL);
    vkDestroyDescriptorSetLayout(ctx->device, desc_layout, NULL);

    // --- PHASE 6: Cleanup complete ---
    clock_gettime(CLOCK_MONOTONIC, &t6);

    printf("\n--- VULKAN FUNCTION TIME DISTRIBUTION ---\n");
    printf("1. VRAM Allocation:       %8.3f ms\n", CALC_MS(t0, t1));
    printf("2. Memcpy (CPU to GPU):   %8.3f ms\n", CALC_MS(t1, t2));
    printf("3. Pipeline Setup (I/O):  %8.3f ms\n", CALC_MS(t2, t3));
    printf("4. GPU Compute Dispatch:  %8.3f ms\n", CALC_MS(t3, t4));
    printf("5. Memcpy (GPU to CPU):   %8.3f ms\n", CALC_MS(t4, t5));
    printf("6. VRAM & Layout Cleanup: %8.3f ms\n", CALC_MS(t5, t6));
    printf("-----------------------------------------\n");
    printf("TOTAL FUNCTION TIME:      %8.3f ms\n\n", CALC_MS(t0, t6));
}