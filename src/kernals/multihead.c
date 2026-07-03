#include"blis.h"
#include<stdio.h>
void multi_head_attention(double* W_Q, double* W_K, double* W_V, double* X, uint16_t* w_dim,
                            uint16_t* n, uint16_t* d, uint8_t* h, double* sqrt_d){
    const uint16_t Q_dim = (*n) * (*d);
    double Q[Q_dim];
    double K[Q_dim];
    double V[Q_dim];
    double alpha = 1.0;
    double beta = 0.0;
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, *n, *d, *n, &alpha, W_Q, *n, 1, X, 
         *d,1, &beta, Q, 1, *n);
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, *n, *d, *n, &alpha, W_K, *n, 1, X, 
         *d,1, &beta, K, 1, *n);
    bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, *n, *d, *n, &alpha, W_V, *n, 1, X, 
         *d,1, &beta, V, 1, *n);
    for(int i = 0;i < h;i++){

    }
}