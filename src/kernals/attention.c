#include"blis.h"
#include<stdio.h>
#include<stdint.h>
#include "../../inc/kernals/softmax.h"
#include"../../inc/kernals/attention.h"
#include<stdbool.h>
void attention(double* Q, double* K, double* V, uint32_t n, uint32_t d, double* out, 
                 bool masked){
                    
    uint32_t score_dim = n * n; 
    uint32_t out_dim = n * d;
    double score[score_dim];
    double probs[score_dim];
    double mask[score_dim];
    double alpha = 1.0;
    double beta = 0.0;
    double min = -DBL_MAX * masked;
    double sqrt_d = sqrt(d);
    memset(mask, 0, sizeof(double) * score_dim);
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_TRANSPOSE, n, n, d, &alpha, Q, 1,
         n, K, 1, n, &beta, score, n, 1);
    bli_dinvscalv(BLIS_NO_CONJUGATE, score_dim, &sqrt_d, score, 1);
    bli_dsetm(BLIS_NO_CONJUGATE, 1, BLIS_NONUNIT_DIAG, BLIS_UPPER, n, n, &min, mask, n, 1);
    bli_daddm(1, BLIS_NONUNIT_DIAG, BLIS_UPPER, BLIS_NO_TRANSPOSE, n, n, mask, n, 1, score, n, 1);
    for(int i = 0;i < n;i++){
        softmax((score + (i * n)), (probs + (i * n)), n);
    }
    
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, n, &alpha, probs, n,
         1, V, 1, n, &beta, out, 1, n);
}