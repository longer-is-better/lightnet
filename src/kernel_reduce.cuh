#pragma once
#include <stdio.h>

#include <iostream>

enum class REDUCE_OP {
    SUM = 0,
    AVG
};

std::ostream& operator<<(std::ostream& os, const REDUCE_OP &op);

template <typename T>
__device__ void warpReduceSum(volatile T* shmem_ptr, int t) {
    shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

template <typename T>
__global__ void kreduce(size_t total_n, size_t current_n, T *I, T *O, REDUCE_OP op) {
    extern __shared__ T partial[];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T front = 0, back = 0;
    if (i < current_n) front = I[i];
    if (i + blockDim.x < current_n) back = I[i + blockDim.x];
    partial[threadIdx.x] = front + back;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        warpReduceSum(partial, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        if (gridDim.x == 1) {
            if (op == REDUCE_OP::AVG) {
                partial[0] /= total_n;
            }
        }
        O[blockIdx.x] = partial[0];
    }
}