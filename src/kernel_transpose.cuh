#pragma once
#include <assert.h>

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
    extern __shared__ DATA_TYPE tile[];

    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    int minbkcft_index = ktranspose_minbkcft_index<DATA_TYPE>(
        threadIdx.x * blockDim.y + threadIdx.y,
        blockDim.x
    );
    tile[minbkcft_index] = (x < n && y < m) ? I[y * n + x] : 0;

    __syncthreads();

    x = blockIdx.y * blockDim.y + threadIdx.x;
    y = blockIdx.x * blockDim.x + threadIdx.y;
    if (x < m && y < n) {
        minbkcft_index = ktranspose_minbkcft_index<DATA_TYPE>(
            threadIdx.y * blockDim.x + threadIdx.x,
            blockDim.x
        );
        O[y * m + x] = tile[minbkcft_index];
    }
}


// frag & vec
template <typename DATA_TYPE, typename DATA_TYPE_4>
__global__ void ktranspose_smem_4xvec4(size_t m, size_t n, DATA_TYPE *I, DATA_TYPE *O) {
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    int TILE_DIM = blockDim.x;
    extern __shared__ DATA_TYPE tile[]; // 16 * blocksize

    size_t x_in_block = threadIdx.x;
    size_t y_in_block = threadIdx.y;

    size_t x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
    size_t y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;


    DATA_TYPE_4 row0 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 0) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row1 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 1) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row2 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 2) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row3 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 3) * (n / 4) + x_in_kernel];

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.w;

    __syncthreads();

    x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
    y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;

    reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + 0) * (m / 4) + x_in_kernel] = reinterpret_cast<DATA_TYPE_4*>(tile)[(4 * y_in_block + 0) * TILE_DIM + x_in_block];
    reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + 1) * (m / 4) + x_in_kernel] = reinterpret_cast<DATA_TYPE_4*>(tile)[(4 * y_in_block + 1) * TILE_DIM + x_in_block];
    reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + 2) * (m / 4) + x_in_kernel] = reinterpret_cast<DATA_TYPE_4*>(tile)[(4 * y_in_block + 2) * TILE_DIM + x_in_block];
    reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + 3) * (m / 4) + x_in_kernel] = reinterpret_cast<DATA_TYPE_4*>(tile)[(4 * y_in_block + 3) * TILE_DIM + x_in_block];
}


// frag & vec
template <typename DATA_TYPE, typename DATA_TYPE_4>
__global__ void ktranspose_smem_4xvec4_store_no_vec(size_t m, size_t n, DATA_TYPE *I, DATA_TYPE *O) {
    assert(blockDim.x == blockDim.y && blockDim.z == 1);
    int TILE_DIM = blockDim.x;
    extern __shared__ DATA_TYPE tile[]; // 16 * blocksize

    size_t x_in_block = threadIdx.x;
    size_t y_in_block = threadIdx.y;

    size_t x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
    size_t y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;


    DATA_TYPE_4 row0 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 0) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row1 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 1) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row2 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 2) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row3 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 3) * (n / 4) + x_in_kernel];

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0] = row0.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1] = row1.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2] = row2.w;

    tile[(4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.x;
    tile[(4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.y;
    tile[(4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.z;
    tile[(4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3] = row3.w;

    __syncthreads();

    x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
    y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;

    for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
            int oy = 4 * y_in_kernel + r;
            int ox = 4 * x_in_kernel + c;
            int ty = 4 * y_in_block + r;
            int tx = 4 * x_in_block + c;
            O[oy * m + ox] = tile[ty * 4 * TILE_DIM + tx];
        }
    }
}

template <typename DATA_TYPE = int>
__device__ int transpose_smem_4xvec4_minbkcft_index(int index) {
    int bank_row = index / 32;
    int bank = index % 32;
    int frag_in_smem_Idx_y = bank_row / 4;
    int frag_in_smem_Idx_x = bank / 4;
    int ele_in_frag_Idx_x = bank % 4;
    int minbkcft_index =\
        bank_row * 32 +
        frag_in_smem_Idx_x * 4 +
        (ele_in_frag_Idx_x + frag_in_smem_Idx_y) % 4;
    return minbkcft_index;
}


template <typename DATA_TYPE = int>
__device__ int transpose_smem_4xvec4_minbkcft_index_plus(int index) {
    int bank_row = index / 32;
    int bank = index % 32;
    int frag_in_smem_Idx_y = bank_row / 4;
    int frag_in_smem_Idx_x = bank / 4;
    int frag_group_in_smem_Idx_y = (frag_in_smem_Idx_y / 4) * 4;
    int ele_in_frag_Idx_x = bank % 4;
    int frag_in_smem_minbkcft_Idx_x = (frag_in_smem_Idx_x + frag_group_in_smem_Idx_y) % 8;
    int minbkcft_index =\
        bank_row * 32 +
        frag_in_smem_minbkcft_Idx_x * 4 +
        (ele_in_frag_Idx_x + frag_in_smem_Idx_y) % 4;
    return minbkcft_index;
}


// frag & vec & minbkcft & STG coalesced, float only
template <typename DATA_TYPE, typename DATA_TYPE_4>
__global__ void ktranspose_smem_4xvec4_minbkcft(size_t m, size_t n, DATA_TYPE *I, DATA_TYPE *O) {
    assert(m % 32 == 0 && n % 32 == 0);
    assert(blockDim.x == 8 && blockDim.y == 8 && blockDim.z == 1);

    extern __shared__ DATA_TYPE tile[]; // 16 * blocksize

    int TILE_DIM = blockDim.x;
    size_t x_in_block = threadIdx.x;
    size_t y_in_block = threadIdx.y;
    size_t x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
    size_t y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;

    DATA_TYPE_4 row0 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 0) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row1 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 1) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row2 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 2) * (n / 4) + x_in_kernel];
    DATA_TYPE_4 row3 = reinterpret_cast<DATA_TYPE_4*>(I)[(4 * y_in_kernel + 3) * (n / 4) + x_in_kernel];

    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.x;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.y;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.z;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.w;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.x;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.y;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.z;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.w;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.x;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.y;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.z;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.w;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.x;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.y;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.z;
    tile[transpose_smem_4xvec4_minbkcft_index_plus((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.w;
    __syncthreads();

    x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
    y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;

    for (int r = 0; r < 4; r++) {
        DATA_TYPE_4 vecOut = make_float4(
            tile[
                transpose_smem_4xvec4_minbkcft_index_plus(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 0)
                )
            ],
            tile[
                transpose_smem_4xvec4_minbkcft_index_plus(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 1)
                )
            ],
            tile[
                transpose_smem_4xvec4_minbkcft_index_plus(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 2)
                )
            ],
            tile[
                transpose_smem_4xvec4_minbkcft_index_plus(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 3)
                )
            ]
        );
        reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + r) * (m / 4) + x_in_kernel] = vecOut;
    }
}