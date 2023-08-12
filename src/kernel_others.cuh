#pragma once
#include <curand_kernel.h>



template <typename DATA_TYPE>
__global__ void kinitializeRandom(DATA_TYPE* data, int size, DATA_TYPE lower_bound, DATA_TYPE upper_bound) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        curandState state;
        curand_init(clock64(), tid, 0, &state);
        data[tid] = curand_uniform(&state) * (upper_bound - lower_bound) - lower_bound;
    }
}

template <typename T>
__global__ void kmemset(size_t N, T *I, T val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) I[tid] = val;
}

template <typename T>
__global__ void kmemset_d(size_t N, T *I, T alpha, T *val) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) I[tid] = alpha * (*val);
}
