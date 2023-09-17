#pragma once


template <typename T>
__global__ void ktranspose(size_t m, size_t n, T *I, T *O) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < n && y < m) O[x * m + y] = I[y * n + x];
}


template <typename T>
__global__ void ktranspose_smem(size_t m, size_t n, T *I, T *O) {
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    extern __shared__ T tile[];
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;

    tile[threadIdx.x * blockDim.y + threadIdx.y] = (x < n && y < m) ? I[y * n + x] : 0;
    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;

    if (x < m && y < n)
        O[y * m + x] = tile[threadIdx.y * blockDim.x + threadIdx.x];
}

// group_size = TILE_DIM
// template <typename T>
__device__ int ktranspose_nbkcft_index_fp32(int index, int group_size) {
    int row_index = index / 32;
    int sride = 32 / group_size;
    int group_index = index / group_size;
    int index_in_group = index % group_size;
    int index_in_group_nbkcft = (index_in_group + sride * row_index) % group_size;
    int nbkcft_index =  group_index * group_size + index_in_group_nbkcft;
    return nbkcft_index;
}


template <typename T>
__device__ int ktranspose_minbkcft_index(int index, int group_size) {
    int row_index = index * (sizeof(T) / 4) / 32;
    int sride = warpSize / group_size;
    int group_index = index / group_size;
    int index_in_group = index % group_size;
    int index_in_group_minbkcft = (index_in_group + sride * row_index) % group_size;
    int minbkcft_index =  group_index * group_size + index_in_group_minbkcft;
    return minbkcft_index;
}

// block only (8, 8)(16, 16)(32, 32)
// https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric
template <typename DATA_TYPE>
__global__ void ktranspose_smem_minbkcft(size_t m, size_t n, DATA_TYPE *I, DATA_TYPE *O) {
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    extern __shared__ DATA_TYPE ktranspose_smem_minbkcft_tile[];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    ktranspose_smem_minbkcft_tile[
        ktranspose_minbkcft_index<DATA_TYPE>(
            threadIdx.x * blockDim.y + threadIdx.y,
            blockDim.x
        )
    ] = (x < n && y < m) ? I[y * n + x] : 0;

    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < m && y < n) {
        O[y * m + x] = ktranspose_smem_minbkcft_tile[
            ktranspose_minbkcft_index<DATA_TYPE>(
                threadIdx.y * blockDim.x + threadIdx.x,
                blockDim.x
            )
        ];
    }
}

// template <typename T, typename T4>
// __global__ void ktranspose_vec4(size_t m, size_t n, T *I, T *O) {
//     uint y_in_grid = blockIdx.y * blockDim.y + threadIdx.y;
//     uint x_in_grid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     for (uint y = y_in_grid; y < m; y += gridDim.y * blockDim.y) {
//         for (uint x = x_in_grid; x < n; x += gridDim.x * blockDim.x) {
//             uint start = 4 * x, end = start + 4;
//             if (end <= n) {
//                 T4 frag = reinterpret_cast<T4*>(I + y * n)[x];
//                 O[(start + 0) * m + y] = frag.x;
//                 O[(start + 1) * m + y] = frag.y;
//                 O[(start + 2) * m + y] = frag.z;
//                 O[(start + 3) * m + y] = frag.w;
//             }
//         }
//         uint remainder = n % 4;
//         uint remainder_start = n / 4 * 4;
//         uint remainder_x = remainder_start + x_in_grid;
//         if (x_in_grid < remainder)
//             O[remainder_x * m + y] = I[y * n + remainder_x];
//     }    
// }