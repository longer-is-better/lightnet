#include "transpose.h"

// matrix transpose
/*
                                        t57
    in  b00 b01 b02 | b03 b04 b05 | b06 b07 b08
        b10 b11 b12 | b13 b14 b15 | b16 b17 b18
        b20 b21 b22 | b23 b24 b25 | b26 b27 b28
        ------------+-------------+------------
        b30 b31 b32 | b33 b34 b35 | b36 b37 b38
        b40 b41 b42 | b43 b44 b45 | b46 b47 b48
        b50 b51 b52 | b53 b54 b55 | b56 b57 b58

    out b00 b10 b20 | b30 b40 b50 
        b01 b11 b21 | b31 b41 b51
        b02 b12 b22 | b32 b42 b52
        ------------+-------------
        b03 b13 b23 | b33 b43 b53 
        b04 b14 b24 | b34 b44 b54
        b05 b15 b25 | b35 b45 b55
        ------------+-------------
        b06 b16 b26 | b36 b46 b56 
        b07 b17 b27 | b37 b47 b57
        b08 b18 b28 | b38 b48 b58

    shared memory
    t57 从全局内存读 b57 到共享内存
    t57 从共享内存读 b48
    t57 将 b48 写入全局内存    
*/

__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N]) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if (x < M && y < N) {
        out[x][y] = in[y][x];
    }
}

__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N]) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ int in_s[BLOCKSIZE][BLOCKSIZE];
    if (x < M && y < N) {
        in_s[threadIdx.y][threadIdx.x] = in[y][x];
    }
    __syncthreads();

    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if (x1 < N && y1 < M) {
        out[y1][x1] = in_s[threadIdx.x][threadIdx.y];
    }
}

__global__ void gpu_shared_bank_matrix_transpose(int in[N][M], int out[M][N]) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ int in_s[BLOCKSIZE][BLOCKSIZE+1];
    if (x < M && y < N) {
        in_s[threadIdx.y][threadIdx.x] = in[y][x];
    }
    __syncthreads();

    int x1 = threadIdx.x + blockDim.y * blockIdx.y;
    int y1 = threadIdx.y + blockDim.x * blockIdx.x;
    if (x1 < N && y1 < M) {
        out[y1][x1] = in_s[threadIdx.x][threadIdx.y];
    }
}

void cpu_matrix_transpose(int in[N][M], int out[M][N]) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            out[x][y] = in[y][x];
        }
    }
}

bool checkResult(int gpu_result[M][N], int cpu_result[M][N]) {
    bool errors = false;

    for (int x = 0; x < M; x++) {
        for (int y = 0; y < N; ++y) {
            if (fabs(gpu_result[x][y] - cpu_result[x][y]) > 1e-6) {
                errors = true;
                return errors;
            }
        }
    }
    return errors;
}
