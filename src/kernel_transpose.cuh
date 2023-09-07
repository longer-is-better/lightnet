#pragma once


template <typename T>
__global__ void ktranspose(size_t m, size_t n, T *I, T *O) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < m) o[x * m + y] = I[y * n + x];
}