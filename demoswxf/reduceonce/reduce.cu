#include "reduce.h"

__global__ void sum_gpu(int *in, int count, int *out) {
    __shared__ int in_s[BLOCKSIZE];

    // grid loop
    int shared_tmp = 0;
    for (int idx = blockDim.x * blockIdx.x + threadIdx.x; idx < count; idx += gridDim.x * blockDim.x) {
        shared_tmp += in[idx];
    }
    in_s[threadIdx.x] = shared_tmp;
    __syncthreads();


    int tmp = 0;
    for (int total_threads = BLOCKSIZE / 2; total_threads >= 1; total_threads /= 2) {
        if (threadIdx.x < total_threads) {
            tmp = in_s[threadIdx.x] + in_s[threadIdx.x + total_threads];
        }
        __syncthreads();
        if (threadIdx.x < total_threads) {
            in_s[threadIdx.x] = tmp;
        }
        // if (threadIdx.x < total_threads) {
        //     in_s[threadIdx.x] += in_s[threadIdx.x + total_threads];
        //     // printf("in_s: %d %d %d\n", in_s[0], in_s[1], in_s[2]);
        // }
        __syncthreads();
    }

    // block_sum -> shared memory[0]
    if (blockIdx.x * blockDim.x < count) {
        if (threadIdx.x == 0) {
            atomicAdd(out, in_s[0]);
        }
    }
}