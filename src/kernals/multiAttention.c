#include"blis.h"
#include"../../inc/kernals/multiAttention.h"
#include"../../inc/kernals/attention.h"
#include<stdio.h>
// expecting in row major but self attention works in column major
// remeber to use assert for exceptions
// dimensions of Ws are being calculated
void multiAttention(const float* __restrict W_Q,const float* __restrict W_K, 
     const float* __restrict W_V,const float* __restrict X, const uint32_t n, 
     const uint32_t d, const uint32_t q_h, const uint32_t kv_h, 
     const float* __restrict W_O, float* __restrict out, const bool masked){
     uint32_t Q_dim = n * d;
     uint32_t q_cols = d / q_h;
     uint32_t dims = q_cols * kv_h;
     uint32_t group_size = q_h / kv_h;
     uint32_t KV_dim = n * dims;
     float Q[Q_dim];
     float K[KV_dim];
     float V[KV_dim];
     float res[Q_dim];
     float alpha = 1.0f;
     float beta = 0.0f;

     bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, d, &alpha, X, d, 1, W_Q, 
          d,1, &beta, Q, 1, n);
     bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, d, &alpha, X, d, 1, W_K, 
          dims,1, &beta, K, 1, n);
     bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, d, &alpha, X, d, 1, W_V, 
          dims,1, &beta, V, 1, n);
     for(int i = 0;i < q_h;i++){
          uint32_t group_idx = i / group_size;
          attention((Q + (i * n * q_cols)), (K + (group_idx * n * q_cols)), 
          (V + (group_idx * n * q_cols)), n, q_cols, (res + (i * n * q_cols)), masked);
     }
     for(int i = 0; i< n;i++){
          for(int j = 0;j < d;j++){
               printf("%f ", res[i * d + j]);
          }
          printf("\n");
     }
     printf("\n");
     bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, d, &alpha, res, 1, n, W_O, 
     d, 1, &beta, out, d, 1);
}