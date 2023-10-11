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

template <typename T>
__global__ void find_greater(T *input, T *output, T target, int *index) {
    int tid_in_kernel = blockIdx.x * blockDim.x + threadIdx.x;
    int val = input[tid_in_kernel];
    if (val > target) output[atomicAdd(index, 1)] = val;
}

