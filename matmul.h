#pragma once

#include <cuda.h>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MM_params {
    // The A & B matrices.
    void *__restrict__ A_ptr; // m x n
    void *__restrict__ B_ptr; // n x k

    // Shape
    int M, N, K;
};
struct MM_kernel_params : public MM_params {
    // The C matrix (output). m x k
    void * __restrict__ C_ptr;
    void * __restrict__ DB_ptr;

    int arch;
    int num_sm;
    int kernel_id;
};

void run_matmul(MM_kernel_params &params, cudaStream_t stream);