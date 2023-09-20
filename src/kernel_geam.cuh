#pragma once

template <typename DATA_TYPE = int>
__device__ int geam_smem_4xvec4_minbkcft_index(int index) {
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
__global__ void kgeam_smem_4xvec4_minbkcft(size_t m, size_t n, bool transI1, DATA_TYPE *I1, bool transI2, DATA_TYPE *I2, DATA_TYPE *O) {
    assert(m % 32 == 0 && n % 32 == 0);
    assert(blockDim.x == 8 && blockDim.y == 8 && blockDim.z == 1);

    extern __shared__ DATA_TYPE tile[]; // 16 * blocksize

    int TILE_DIM = blockDim.x;
    size_t x_in_block = threadIdx.x;
    size_t y_in_block = threadIdx.y;
    size_t x_in_kernel;
    size_t y_in_kernel;

    if (transI1) {
        x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
        y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;
    } else {
        x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
        y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;
    }

    DATA_TYPE_4 row0 = reinterpret_cast<DATA_TYPE_4*>(I1)[(4 * y_in_kernel + 0) * ((transI1 ? m : n) / 4) + x_in_kernel];
    DATA_TYPE_4 row1 = reinterpret_cast<DATA_TYPE_4*>(I1)[(4 * y_in_kernel + 1) * ((transI1 ? m : n) / 4) + x_in_kernel];
    DATA_TYPE_4 row2 = reinterpret_cast<DATA_TYPE_4*>(I1)[(4 * y_in_kernel + 2) * ((transI1 ? m : n) / 4) + x_in_kernel];
    DATA_TYPE_4 row3 = reinterpret_cast<DATA_TYPE_4*>(I1)[(4 * y_in_kernel + 3) * ((transI1 ? m : n) / 4) + x_in_kernel];

    if (transI1){
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0)] = row0.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1)] = row1.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2)] = row2.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3)] = row3.w;
    } else {
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 0)] = row0.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 1)] = row0.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 2)] = row0.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 3)] = row0.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 0)] = row1.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 1)] = row1.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 2)] = row1.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 3)] = row1.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 0)] = row2.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 1)] = row2.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 2)] = row2.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 3)] = row2.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 0)] = row3.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 1)] = row3.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 2)] = row3.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 3)] = row3.w;
    }

    if (transI2) {
        x_in_kernel = blockIdx.y * TILE_DIM + x_in_block;
        y_in_kernel = blockIdx.x * TILE_DIM + y_in_block;
    } else {
        x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
        y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;
    }

    row0 = reinterpret_cast<DATA_TYPE_4*>(I2)[(4 * y_in_kernel + 0) * ((transI2 ? m : n) / 4) + x_in_kernel];
    row1 = reinterpret_cast<DATA_TYPE_4*>(I2)[(4 * y_in_kernel + 1) * ((transI2 ? m : n) / 4) + x_in_kernel];
    row2 = reinterpret_cast<DATA_TYPE_4*>(I2)[(4 * y_in_kernel + 2) * ((transI2 ? m : n) / 4) + x_in_kernel];
    row3 = reinterpret_cast<DATA_TYPE_4*>(I2)[(4 * y_in_kernel + 3) * ((transI2 ? m : n) / 4) + x_in_kernel];
    if (transI2){
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 0)] += row0.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 0)] += row0.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 0)] += row0.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 0)] += row0.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 1)] += row1.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 1)] += row1.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 1)] += row1.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 1)] += row1.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 2)] += row2.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 2)] += row2.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 2)] += row2.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 2)] += row2.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 0) * TILE_DIM * 4 + 4 * y_in_block + 3)] += row3.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 1) * TILE_DIM * 4 + 4 * y_in_block + 3)] += row3.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 2) * TILE_DIM * 4 + 4 * y_in_block + 3)] += row3.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * x_in_block + 3) * TILE_DIM * 4 + 4 * y_in_block + 3)] += row3.w;
    } else {
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 0)] += row0.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 1)] += row0.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 2)] += row0.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 0) * TILE_DIM * 4 + 4 * x_in_block + 3)] += row0.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 0)] += row1.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 1)] += row1.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 2)] += row1.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 1) * TILE_DIM * 4 + 4 * x_in_block + 3)] += row1.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 0)] += row2.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 1)] += row2.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 2)] += row2.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 2) * TILE_DIM * 4 + 4 * x_in_block + 3)] += row2.w;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 0)] += row3.x;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 1)] += row3.y;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 2)] += row3.z;
        tile[geam_smem_4xvec4_minbkcft_index((4 * y_in_block + 3) * TILE_DIM * 4 + 4 * x_in_block + 3)] += row3.w;
    }
    __syncthreads();

    x_in_kernel = blockIdx.x * TILE_DIM + x_in_block;
    y_in_kernel = blockIdx.y * TILE_DIM + y_in_block;

    for (int r = 0; r < 4; r++) {
        DATA_TYPE_4 vecOut = make_float4(
            tile[
                geam_smem_4xvec4_minbkcft_index(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 0)
                )
            ],
            tile[
                geam_smem_4xvec4_minbkcft_index(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 1)
                )
            ],
            tile[
                geam_smem_4xvec4_minbkcft_index(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 2)
                )
            ],
            tile[
                geam_smem_4xvec4_minbkcft_index(
                    (4 * y_in_block + r) * (4 * TILE_DIM) + (4 * x_in_block + 3)
                )
            ]
        );
        reinterpret_cast<DATA_TYPE_4*>(O)[(4 * y_in_kernel + r) * gridDim.x * TILE_DIM + x_in_kernel] = vecOut;
    }
}