#include "../../inc/kernals/matmul.h"
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