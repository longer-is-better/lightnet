#pragma once
#include <stdio.h>

#include <iostream>

enum class REDUCE_OP {
    SUM = 0,
    AVG
};

std::ostream& operator<<(std::ostream& os, REDUCE_OP op);

/// @brief calculate reduce result.
///        if n <= 2048: reduce<<<1, ceil_32(n / 2), ceil_32(n / 2) * sizeof(T)>>>.
///        else if 2048 < n < 2048^2:
///            in two steps, 0. grid = ceil(n / (1024 * 2))
///                          1. reduce<<<grid, 1024, 1024 * sizeof(T)>>>(n, I, O(in grid length), op)
///                          2. reduce<<<1, grid, grid * sizeof(T)>>>()
/// @param n element_count
/// @param I pointer to input data
/// @param O pointer to output data
/// @param op REDUCE_OP
/// @return O
template <typename T>
__global__ void kreduce(size_t n, T *I, T *O, REDUCE_OP op) {
    extern __shared__ T partial[];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    T back = 0;
    if (tid + gridDim.x * blockDim.x < n) back = I[tid + gridDim.x * blockDim.x];
    switch (op) {
        case REDUCE_OP::SUM:
            partial[threadIdx.x] = I[tid] + back;
            break;
        case REDUCE_OP::AVG:
            partial[threadIdx.x] = (I[tid] + back) / 2;
            break;
        
        default:
            printf("not support REDUCE_OP %d", op);
            return;
    }
    __syncthreads();

    for (int half_threads = blockDim.x / 2; half_threads >= 1; half_threads /= 2) {
        if (threadIdx.x < half_threads) {
            switch (op) {
                case REDUCE_OP::SUM:
                    partial[threadIdx.x] += partial[threadIdx.x + half_threads];
                    break;
                case REDUCE_OP::AVG:
                    partial[threadIdx.x] += partial[threadIdx.x + half_threads];
                    partial[threadIdx.x] /= 2;
                    break;
                
                default:
                    printf("not support REDUCE_OP %d", op);
                    return;
            }
        }
        __syncthreads();
    }

    O[blockIdx.x] = partial[0];
}