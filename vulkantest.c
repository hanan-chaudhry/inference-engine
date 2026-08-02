#include <stdio.h>
#include <stdlib.h>
#include "inc/backend/vk_core.h"
#include "inc/kernals/add.h"
#include "inc/kernals/matmul.h"

void test_add(VkContext* ctx) {
    size_t N = 2048;
    size_t bytes = N * sizeof(float);

    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C_cpu = (float*)malloc(bytes);
    float* C_gpu = (float*)malloc(bytes);

    for (size_t i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    kernel_add_cpu_f32_forward(A, B, C_cpu, N);
    kernel_add_vulkan_f32_forward(ctx, A, B, C_gpu, N);

    printf("\n[ADD TEST]\n");
    printf("CPU Result[0]: %f\n", C_cpu[0]);
    printf("GPU Result[0]: %f\n", C_gpu[0]);

    int mismatch = 0;
    for (size_t i = 0; i < N; i++) {
        if (C_cpu[i] != C_gpu[i]) {
            printf("MISMATCH at index %zu: CPU=%f, GPU=%f\n", i, C_cpu[i], C_gpu[i]);
            mismatch = 1;
            break;
        }
    }

    if (!mismatch) {
        printf("Status: SUCCESS (All %zu elements match!)\n", N);
    }
    printf("\n\n");

    free(A); free(B); free(C_cpu); free(C_gpu);
}

void test_matmul(VkContext* ctx) {
    uint32_t M = 2048;
    uint32_t K = 2048;
    uint32_t N = 2048;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float* A = (float*)malloc(bytes_A);
    float* B = (float*)malloc(bytes_B);
    float* C_cpu = (float*)malloc(bytes_C);
    float* C_gpu = (float*)malloc(bytes_C);

    for (size_t i = 0; i < M * K; i++) A[i] = 1.0f;
    for (size_t i = 0; i < K * N; i++) B[i] = 2.0f;

    kernel_matmul_cpu_f32_forward(A, B, C_cpu, M, N, K);
    double time_gpu = 0.0;
    kernel_matmul_vulkan_f32_forward(ctx, A, B, C_gpu, M, N, K, &time_gpu);

    double total_flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (total_flops / time_gpu) / 1e9;

    double theoretical_peak_gflops = 1792.0;

    double efficiency = (gflops / theoretical_peak_gflops) * 100.0;

    printf("GPU Compute Execution Time: %.3f ms\n", time_gpu * 1000.0);
    printf("Achieved Performance:       %.2f GFLOPs\n", gflops);
    printf("Hardware Efficiency:        %.2f%%\n", efficiency);


    printf("\n--- MATMUL TEST (%dx%d * %dx%d) ---\n", M, K, K, N);
    printf("CPU Result[0]: %f\n", C_cpu[0]);
    printf("GPU Result[0]: %f\n", C_gpu[0]);

    int mismatch = 0;
    for (size_t i = 0; i < M * N; i++) {
        float diff = C_cpu[i] - C_gpu[i];
        if (diff < -0.001f || diff > 0.001f) {
            printf("MISMATCH at index %zu: CPU=%f, GPU=%f\n", i, C_cpu[i], C_gpu[i]);
            mismatch = 1;
            break;
        }
    }

    if (!mismatch) {
        printf("Status: SUCCESS (All %u elements match!)\n", M * N);
    }
    printf("\n\n");

    free(A); free(B); free(C_cpu); free(C_gpu);
}

void test_scale(VkContext* ctx) {
    printf("pass");
}

int main() {
    VkContext ctx;
    if (!vk_init(&ctx)) {
        return -1;
    }

    int choice = -1;
    while (1) {
        printf("==============|Vulkan Inference Engine|==============\n");
        printf("[1] Test Add\n");
        printf("[2] Test MatMul\n");
        printf("[3] Test Scale\n");
        printf("[0] Exit\n");
        printf("Select Test: ");

        if (scanf("%d", &choice) != 1) {
            while (getchar() != '\n');
            continue;
        }

        if (choice == 0) break;

        switch (choice) {
        case 1:
            test_add(&ctx);
            break;
        case 2:
            test_matmul(&ctx);
            break;
        case 3:
            test_scale(&ctx);
            break;
        default:
            printf("\n[ Invalid choice. Try again. ]\n\n");
        }
    }

    vk_cleanup(&ctx);
    return 0;
}
