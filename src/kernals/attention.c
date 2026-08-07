#include"blis.h"
#include<stdio.h>
#include<stdint.h>
#include<float.h>
#include "../../inc/kernals/softmax.h"
#include"../../inc/kernals/attention.h"
#include<stdbool.h>
void attention(const float* __restrict Q, const float* __restrict K, 
    const float* __restrict V,const uint32_t n, const uint32_t d, float*__restrict out, 
    const bool masked){
                    
    uint32_t score_dim = n * n; 
    // uint32_t out_dim = n * d;
    float score[score_dim];
    float probs[score_dim];
    float mask[score_dim];
    float alpha = 1.0f;
    float beta = 0.0f;
    float min = -FLT_MAX * masked;
    float sqrt_d = sqrtf(d);
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, n, n, d, &alpha, Q, 1,
         n, K, 1, n, &beta, score, n, 1);
    bli_sinvscalv(BLIS_NO_CONJUGATE, score_dim, &sqrt_d, score, 1);
    bli_ssetm(BLIS_NO_CONJUGATE, 1, BLIS_NONUNIT_DIAG, BLIS_UPPER, n, n, &min, mask, n, 1);
    bli_saddm(1, BLIS_NONUNIT_DIAG, BLIS_UPPER, BLIS_NO_TRANSPOSE, n, n, mask, n, 1, score, n, 1);
    
    for(int i = 0;i < n;i++){
        softmax((score + (i * n)), (probs + (i * n)), n);
    }
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, n, &alpha, probs, n,
         1, V, 1, n, &beta, out, 1, n);
}