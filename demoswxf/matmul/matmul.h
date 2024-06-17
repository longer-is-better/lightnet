#pragma once

#include <math.h>

#define BLOCKSZ 16
// #define BLOCKSZ 2
#define SHARED_MEM_SIZE 48 * 1024\

#define tile_size 8
#define TM tile_size

#define BM  (BLOCKSZ * tile_size)
#define BN  BLOCKSZ
// #define BK (SHARED_MEM_SIZE / (4 * (BM + BN)))
#define BK BN



static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}

template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

template<typename T>
bool isConsistent(T* mat1, T* mat2, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        if (abs(mat1[i] - mat2[i]) > 1e-6) {
            printf("mat1[%d] = %.2f, mat2[%d] = %.2f\n", i, mat1[i], i, mat2[i]);
            return false;
        }
    }
    return true;
}

template<typename T>
void prtmat(T* mat, int R, int C) {
    printf("\n");
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            printf("%.2f\t", mat[i * C + j]);
        }
        printf("\n");
    }
}

template<typename T>
void matmulCPU(T* A, T* B, T* C, int M, int K, int N) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int p = 0; p < K; ++p) {
                C[i * N + j] += A[i * K + p] * B[p * N + j];
            }
        }
    }
}

template<typename T>
__global__ void matmulGPU(T* A, T* B, T* C, int M, int K, int N) {
    /*
        一个线程负责计算一个 C 中的结果
    */
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    T tmp;
    if (r < M && c < N) {
        tmp = 0.;
        for (int p = 0; p < K; ++p) {
            tmp += A[r * K + p] * B[p * N + c];
        }
        C[r * N + c] = tmp;
    }
}


template<typename T>
__global__ void matmulGPUSharedMem(T* A, T* B, T* C, int M, int K, int N) {
    /*
        一个线程负责计算一个 C 中的结果
    */
    int num_shared_block = (K + BK - 1) / BK;
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];

    T tmp = 0.;
    for (int i = 0; i < num_shared_block; ++i) {
        // copy data to shared mem
        int A_row = threadIdx.y;  
        int A_col = threadIdx.x;
        if (blockIdx.y * BM + A_row < M && i * BK + A_col < K) {
            As[A_row][A_col] = A[A_row * K + A_col];
        } else {
            As[A_row][A_col] = 0;
        }

        int B_row = threadIdx.y;  
        int B_col = threadIdx.x;
        if (i * BK + B_row < K && blockIdx.x * BN + B_col < N) {
            Bs[B_row][B_col] = B[B_row * N + B_col];
        } else {
            Bs[B_row][B_col] = 0;
        }
        __syncthreads();
        A += BK;
        B += BK * N;
        
        for (int k = 0; k < BK; ++k) {
            tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    int C_row = threadIdx.y;
    int C_col = threadIdx.x;
    if (blockIdx.y * BM + C_row < M && blockIdx.x * BN + C_col < N) {
        C[C_row * N + C_col] = tmp;
    }
}

template<typename T>
__global__ void matmulGPUSMOneTill(T* A, T* B, T* C, int M, int K, int N) {
    /*
        一个线程负责计算 TM 个 C 中的结果
    */
    int num_shared_block = (K + BK - 1) / BK;
    __shared__ T As[BM][BK];
    __shared__ T Bs[BK][BN];

    A = &A[blockIdx.y * BM * K];
    B = &B[blockIdx.x * BN];
    C = &C[blockIdx.y * BM * N + blockIdx.x * BN];

    T tmp[TM] = {0.};
    for (int i = 0; i < num_shared_block; ++i) {
        // copy data to shared mem
        for (int m = 0; m < TM; ++m) {
            int A_row = threadIdx.y * TM + m;  
            int A_col = threadIdx.x;
            if (blockIdx.y * BM + A_row < M && i * BK + A_col < K) {
                As[A_row][A_col] = A[A_row * K + A_col];
            // } else{
            } else if (A_row < BM && A_col < BK) {
                As[A_row][A_col] = 0.;
            }
        }

        int B_row = threadIdx.y;  
        int B_col = threadIdx.x;
        if (i * BK + B_row < K && blockIdx.x * BN + B_col < N) {
            Bs[B_row][B_col] = B[B_row * N + B_col];
        } else if (B_row < BK && B_col < BN) {
            Bs[B_row][B_col] = 0.;
        }
        __syncthreads();
        A += BK;
        B += BK * N;

        // if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        //     printf("As:\n");
        //     for (int i = 0; i < BM; ++i) {
        //         for (int j = 0; j < BK; ++j) {
        //             printf("%.2f\t", As[i][j]);
        //         }
        //         printf("\n");
        //     }

        //     printf("Bs:\n");
        //     for (int i = 0; i < BK; ++i) {
        //         for (int j = 0; j < BN; ++j) {
        //             printf("%.2f\t", Bs[i][j]);
        //         }
        //         printf("\n");
        //     }
        // }
        
        for (int k = 0; k < BK; ++k) {
            for (int m = 0; m < TM; ++m) {  
                int As_row = threadIdx.y * TM + m;  
                int Bs_col = threadIdx.x;
                if (As_row < BM && Bs_col < BN) {
                    tmp[m] += As[As_row][k] * Bs[k][Bs_col];
                }
            }
        }
        __syncthreads();
    }
    
    for (int m = 0; m < TM; ++m) {
        int C_row = threadIdx.y * TM + m;
        int C_col = threadIdx.x;
        if (blockIdx.y * BM + C_row < M && blockIdx.x * BN + C_col < N) {
            C[C_row * N + C_col] = tmp[m];
        }
    }
}