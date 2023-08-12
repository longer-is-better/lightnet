#pragma once


/// @brief 
/// @param trans_W 
/// @param trans_X 
/// @param m after trans if need
/// @param k after trans if need
/// @param n after trans if need
/// @param W 
/// @param X 
/// @param Y 
/// @return 
template <typename T>
__global__ void kmatmul(bool trans_W, bool trans_X, size_t m, size_t k, size_t n, T *W, T *X, T *Y) {
    extern __shared__ float shared_mem[];

    float *tile_W = shared_mem;
    float *tile_X = shared_mem + blockDim.y * blockDim.x;

    for (int b = 0; b < (k + blockDim.x - 1) / blockDim.x; b++) {
        int W_y = blockIdx.y * blockDim.y + threadIdx.y;
        int W_x = b * blockDim.x + threadIdx.x;
        int X_y = b * blockDim.y + threadIdx.y;
        int X_x = blockIdx.x * blockDim.x + threadIdx.x;

        if (trans_W) {
            if (W_x < k || W_y < m) {
                tile_W[threadIdx.y * blockDim.x + threadIdx.x] = W[W_x * m + W_y];
            } else {
                tile_W[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        } else {
            if (W_x < k || W_y < m) {
                tile_W[threadIdx.y * blockDim.x + threadIdx.x] = W[W_y * k + W_x];
            } else {
                tile_W[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        }

        if (trans_X) {
            if (X_x < n || X_y < k) {
                tile_X[threadIdx.y * blockDim.x + threadIdx.x] = X[X_x + X_y * k];
            } else {
                tile_X[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        } else {
            if (X_x < n || X_y < k) {
                tile_X[threadIdx.y * blockDim.x + threadIdx.x] = X[X_y * n + X_x];
            } else {
                tile_X[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        }

        __syncthreads();

        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += tile_W[threadIdx.y * blockDim.x + i] * tile_X[i * blockDim.x + threadIdx.x];
        }
        Y[W_y * n + X_x] += sum;
        __syncthreads();
    }
}