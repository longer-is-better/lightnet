#pragma oncetrans_L

template <typename T>
__global__ void kmatmul_naive(bool trans_L, bool trans_R, size_t m, size_t k, size_t n, T *L, T *R, T *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < m && y < n) {
        for (size_t i = 0; i < k; i++) {
            O[x * n + y] += L[x * k + i] * R[i * n + y];
        }
    }
}

template <typename T>
__global__ void kmatmul_coalescing(bool trans_L, bool trans_R, size_t m, size_t k, size_t n, T *L, T *R, T *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < m) {
        for (size_t i = 0; i < k; i++) {
            O[y * n + x] += L[y * k + i] * R[i * n + x];
        }
    }
}

/// @brief 
/// @param trans_L 
/// @param trans_R 
/// @param m after trans if need
/// @param k after trans if need
/// @param n after trans if need
/// @param L 
/// @param R 
/// @param O 
/// @return 
template <typename T>
__global__ void kmatmul(bool trans_L, bool trans_R, size_t m, size_t k, size_t n, T *L, T *R, T *O) {
    float ans = 0.f;
    extern __shared__ float shared_mem[];

    float *tile_L = shared_mem;
    float *tile_R = shared_mem + blockDim.y * blockDim.x;
    
    int L_y = blockIdx.y * blockDim.y + threadIdx.y;
    int R_x = blockIdx.x * blockDim.x + threadIdx.x;
    for (int b = 0; b < (k + blockDim.x - 1) / blockDim.x; b++) {
        int L_x = b * blockDim.x + threadIdx.x;
        int R_y = b * blockDim.y + threadIdx.y;

        if (trans_L) {
            if (L_x < k && L_y < m) {
                tile_L[threadIdx.y * blockDim.x + threadIdx.x] = L[L_x * m + L_y];
            } else {
                tile_L[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        } else {
            if (L_x < k && L_y < m) {
                tile_L[threadIdx.y * blockDim.x + threadIdx.x] = L[L_y * k + L_x];
            } else {
                tile_L[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        }

        if (trans_R) {
            if (R_x < n && R_y < k) {
                tile_R[threadIdx.y * blockDim.x + threadIdx.x] = R[R_x * k + R_y];
            } else {
                tile_R[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        } else {
            if (R_x < n && R_y < k) {
                tile_R[threadIdx.y * blockDim.x + threadIdx.x] = R[R_y * n + R_x];
            } else {
                tile_R[threadIdx.y * blockDim.x + threadIdx.x] = 0;
            }
        }
        __syncthreads();

        float sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            sum += tile_L[threadIdx.y * blockDim.x + i] * tile_R[i * blockDim.x + threadIdx.x];
        }
        if (L_y < m && R_x < n) ans += sum;
        __syncthreads();
    }
    if (L_y < m && R_x < n) O[L_y * n + R_x] = ans;
}