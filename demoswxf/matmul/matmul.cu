/*
    shared memory: 48KB
    nvcc matmul.cu -o matmul.o -lcublas && ./matmul.o
    /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -o matmul_profile_report ./matmul.o 

    nvcc matmul.cu -o matmul.o -lcublas -G -g
    cuda-gdb --args ./matmul.o 2 3 4

*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include "matmul.h"

int main(int argc, char* argv[]) {
    int M = 1024, K = 512, N = 1024;
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) K = atoi(argv[2]);
    if (argc > 3) N = atoi(argv[3]);

    auto sz_a = sizeof(float) * M * K;
    auto sz_b = sizeof(float) * K * N;
    auto sz_c = sizeof(float) * M * N;

    // 分配空间
    float* h_a, *h_b, *h_c, *res_from_d;
    h_a = (float*)malloc(sz_a);
    h_b = (float*)malloc(sz_b);
    h_c = (float*)malloc(sz_c);
    res_from_d = (float*)malloc(sz_c);

    float* d_a, *d_b, *d_c;
    cudaMalloc(&d_a, sz_a);
    cudaMalloc(&d_b, sz_b);
    cudaMalloc(&d_c, sz_c);

    // 初始化数据
    for (int i = 0; i < M * K; ++i) h_a[i] = static_cast<float>(1);
    for (int i = 0; i < K * N; ++i) h_b[i] = static_cast<float>(1);
    // for (int i = 0; i < M * K; ++i) h_a[i] = static_cast<float>(i % 100);
    // for (int i = 0; i < K * N; ++i) h_b[i] = static_cast<float>(i % 100);
    // for (int i = 0; i < M * K; ++i) h_a[i] = (rand() % 10) * 1.0f;
    // for (int i = 0; i < K * N; ++i) h_b[i] = (rand() % 10) * 1.0f;

    // 拷贝计算数据到设备
    cudaMemcpy(d_a, h_a, sz_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, sz_b, cudaMemcpyHostToDevice);

    // 计算
    // cpu
    clock_t cpu_start = clock();
    matmulCPU(h_a, h_b, h_c, M, K, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // cublass
    float* res_from_sgemm;
    res_from_sgemm = (float*)malloc(sz_c * sizeof(float));
    float alpha = 1.0f, beta = 0.0f;
    cublasHandle_t handle;
    cublasCreate(&handle);
    clock_t sgemm_start = clock();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c, N);
    clock_t sgemm_end = clock();
    double sgemm_time = (double)(sgemm_end - sgemm_start) / CLOCKS_PER_SEC;    
    cudaMemcpy(res_from_sgemm, d_c, sz_c, cudaMemcpyDeviceToHost);

    dim3 threadsPerBlock(BLOCKSZ, BLOCKSZ);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // gpu
    clock_t gpu_start = clock();
    matmulGPU<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
    clock_t gpu_end = clock();
    double gpu_time = (double)(gpu_end - gpu_start) / CLOCKS_PER_SEC;
    cudaMemcpy(res_from_d, d_c, sz_c, cudaMemcpyDeviceToHost);

    // gpu_sm
    float* res_gpu_sm = (float*)malloc(sz_c * sizeof(float));
    clock_t gpu_sm_start = clock();
    matmulGPUSharedMem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, M, K, N);
    // checkCudaErrors(cudaGetLastError());
    clock_t gpu_sm_end = clock();
    double gpu_sm_time = (double)(gpu_sm_end - gpu_sm_start) / CLOCKS_PER_SEC;
    cudaMemcpy(res_gpu_sm, d_c, sz_c, cudaMemcpyDeviceToHost);


    // gpu_sm_onetill
    dim3 BlockSize(BLOCKSZ, BLOCKSZ);
    dim3 GridSize((N + BN - 1) / BN, (M + BM - 1) / BM);
    float* res_gpu_sm_ot = (float*)malloc(sz_c * sizeof(float));
    clock_t gpu_sm_ot_start = clock();
    matmulGPUSMOneTill<<<GridSize, BlockSize>>>(d_a, d_b, d_c, M, K, N);
    clock_t gpu_sm_ot_end = clock();
    double gpu_sm_ot_time = (double)(gpu_sm_ot_end - gpu_sm_ot_start) / CLOCKS_PER_SEC;
    cudaMemcpy(res_gpu_sm_ot, d_c, sz_c, cudaMemcpyDeviceToHost);

    // // 打印 
    // prtmat(h_a, M, K);
    // prtmat(h_b, K, N);
    // prtmat(h_c, M, N);
    // prtmat(res_from_sgemm, M, N);
    // prtmat(res_from_d, M, N);
    // prtmat(res_gpu_sm, M, N);
    // prtmat(res_gpu_sm_ot, M, N);

    printf("-----------------------------------------------------------------------------\n");
    printf("CPU Time: %f s\n", cpu_time);
    printf("cuBLAS is %s \n", isConsistent(h_c, res_from_sgemm, M, N) ? "true" : "false");
    printf("cuBLAS sgemm speed up: %.2f \n", cpu_time / sgemm_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config: grig (%d, %d), block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    printf("Normal GPU is %s \n", isConsistent(h_c, res_from_d, M, N) ? "true" : "false");
    printf("GPU Time: %f s\n", gpu_time);
    printf("GPU speed up: %.2f \n", cpu_time / gpu_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config: grig (%d, %d), block(%d, %d)\n", numBlocks.x, numBlocks.y, threadsPerBlock.x, threadsPerBlock.y);
    printf("Shared_MEM is %s \n", isConsistent(h_c, res_gpu_sm, M, N) ? "true" : "false");
    printf("Shared_MEM Time: %f s\n", gpu_sm_time);
    printf("Shared_MEM speed up: %.2f \n", cpu_time / gpu_sm_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config: grig (%d, %d), block(%d, %d)\n", GridSize.x, GridSize.y, BlockSize.x, BlockSize.y);
    printf("Shared_MEM_OneTill is %s \n", isConsistent(h_c, res_gpu_sm_ot, M, N) ? "true" : "false");
    printf("Shared_MEM_OneTill Time: %f s\n", gpu_sm_ot_time);
    printf("Shared_MEM_OneTill speed up: %.2f \n", cpu_time / gpu_sm_ot_time);

    free(h_a);
    free(h_b);
    free(h_c);
    free(res_from_sgemm);
    free(res_from_d);
    free(res_gpu_sm);
    free(res_gpu_sm_ot);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

