#pragma once



#include <stdio.h>
#include <math.h>

#define BLOCKSIZE 32
#define M 3000
#define N 1000
__global__ void gpu_matrix_transpose(int in[N][M], int out[M][N]);
__global__ void gpu_shared_matrix_transpose(int in[N][M], int out[M][N]);
__global__ void gpu_shared_bank_matrix_transpose(int in[N][M], int out[M][N]);
void cpu_matrix_transpose(int in[N][M], int out[M][N]);
bool checkResult(int gpu_result[M][N], int cpu_result[M][N]);
