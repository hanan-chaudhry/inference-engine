#include <stdio.h>
#include <stdlib.h>
#include "inc/kernals/add.h"

int main() {
    size_t N = 1024;
    size_t bytes = N * sizeof(float);

    // Standard C allocations
    float* A = (float*)malloc(bytes);
    float* B = (float*)malloc(bytes);
    float* C_cpu = (float*)malloc(bytes);
    float* C_gpu = (float*)malloc(bytes);

    for (size_t i = 0; i < N; i++) {
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    // Run on CPU
    kernel_add_cpu_f32_forward(A, B, C_cpu, N);

    // Run on GPU
    VkContext ctx;
    vk_init(&ctx);

    // Look how simple this call is!
    kernel_add_vulkan_f32_forward(&ctx, A, B, C_gpu, N);

    // Verify
    printf("CPU Result[0]: %f\n", C_cpu[0]);
    printf("GPU Result[0]: %f\n", C_gpu[0]);

    vk_cleanup(&ctx);
    free(A); free(B); free(C_cpu); free(C_gpu);
    return 0;
}