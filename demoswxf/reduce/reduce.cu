#include <stdio.h>
#include "cuda_runtime.h"
#include "reduce.h"

int main() {
    int N = 1024 * 1024;    // 4M 数据

    // 分配空间
    float* h_idata, *h_odata;
    int sz_i = sizeof(float) * N;
    int sz_o = sizeof(float);
    h_idata = (float*)malloc(sz_i);
    h_odata = (float*)malloc(sz_o);

    float* d_idata, *d_odata;
    cudaMalloc(&d_idata, sz_i);
    cudaMalloc(&d_odata, sz_i);

    // 初始化数据
    for (int i = 0; i < N; ++i) h_idata[i] = static_cast<float>(1);

    // 拷贝计算数据到设备
    cudaMemcpy(d_idata, h_idata, sz_i, cudaMemcpyHostToDevice);

    // 计算
    // cpu
    clock_t cpu_start = clock();
    reduceCPU(h_idata, h_odata, N);
    clock_t cpu_end = clock();
    double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // gpu base
    float* gpubl_odata = (float*)malloc(sz_o);
    dim3 block(THREADS_PER_BLOCK);
    dim3 grid(ceil(N / block.x));

    clock_t gpuBL_start = clock();
    reduceGpuBL<<<grid, block>>>(d_idata, d_odata, N);
    reduceGpuBL<<<grid.x / block.x, block>>>(d_odata, d_odata, grid.x);
    clock_t gpuBL_end = clock();
    double gpuBL_time = (double)(gpuBL_end - gpuBL_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpubl_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    // gpu wd
    float* gpuWD_odata = (float*)malloc(sz_o);
    dim3 block_WD(THREADS_PER_BLOCK);
    dim3 grid_WD(ceil(N / block_WD.x));

    clock_t gpuWD_start = clock();
    reduceGpuWD<<<grid_WD, block_WD>>>(d_idata, d_odata, N);
    reduceGpuWD<<<grid_WD.x / block_WD.x, block_WD>>>(d_odata, d_odata, grid_WD.x);
    clock_t gpuWD_end = clock();
    double gpuWD_time = (double)(gpuWD_end - gpuWD_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpuWD_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    // gpu BC
    float* gpuBC_odata = (float*)malloc(sz_o);
    dim3 block_BC(THREADS_PER_BLOCK);
    dim3 grid_BC(ceil(N / block_BC.x));

    clock_t gpuBC_start = clock();
    reduceGpuBC<<<grid_BC, block_BC>>>(d_idata, d_odata, N);
    reduceGpuBC<<<grid_BC.x / block_BC.x, block_BC>>>(d_odata, d_odata, grid_BC.x);
    clock_t gpuBC_end = clock();
    double gpuBC_time = (double)(gpuBC_end - gpuBC_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpuBC_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    // gpu ID
    float* gpuID_odata = (float*)malloc(sz_o);
    dim3 block_ID(THREADS_PER_BLOCK / 2);
    dim3 grid_ID(ceil(N / block_ID.x));

    clock_t gpuID_start = clock();
    reduceGpuID<<<grid_ID, block_ID>>>(d_idata, d_odata, N);
    reduceGpuID<<<grid_ID.x / 2 / block_ID.x, block_ID>>>(d_odata, d_odata, grid_ID.x / 2);
    clock_t gpuID_end = clock();
    double gpuID_time = (double)(gpuID_end - gpuID_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpuID_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    // gpu UR
    float* gpuUR_odata = (float*)malloc(sz_o);
    dim3 block_UR(THREADS_PER_BLOCK / 2);
    dim3 grid_UR(ceil(N / block_UR.x));

    clock_t gpuUR_start = clock();
    reduceGpuUR<<<grid_UR, block_UR>>>(d_idata, d_odata, N);
    reduceGpuUR<<<grid_UR.x / 2 / block_UR.x, block_UR>>>(d_odata, d_odata, grid_UR.x / 2);
    clock_t gpuUR_end = clock();
    double gpuUR_time = (double)(gpuUR_end - gpuUR_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpuUR_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    // gpu CUR
    float* gpuCUR_odata = (float*)malloc(sz_o);
    dim3 block_CUR(THREADS_PER_BLOCK / 2);
    dim3 grid_CUR(ceil(N / block_CUR.x));

    clock_t gpuCUR_start = clock();
    reduceGpuCUR<<<grid_CUR, block_CUR>>>(d_idata, d_odata, N);
    reduceGpuCUR<<<grid_CUR.x / 2 / block_CUR.x, block_CUR>>>(d_odata, d_odata, grid_CUR.x / 2);
    clock_t gpuCUR_end = clock();
    double gpuCUR_time = (double)(gpuCUR_end - gpuCUR_start) / CLOCKS_PER_SEC;
    cudaMemcpy(gpuCUR_odata, d_odata, sz_o, cudaMemcpyDeviceToHost);

    printf("-----------------------------------------------------------------------------\n");
    printf("CPU Result: %.2f\n", *h_odata);
    printf("CPU Time: %f s\n", cpu_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid.x, block.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid.x / block.x, block.x);
    printf("GpuBL is %s \n", *h_odata - *gpubl_odata < 1e-6 ? "true" : "false");
    printf("GpuBL Result: %.2f\n", *gpubl_odata);
    printf("GpuBL Time: %f s\n", gpuBL_time);
    printf("GpuBL speed up: %.2f \n", cpu_time / gpuBL_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid_WD.x, block_WD.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid_WD.x / block_WD.x, block_WD.x);
    printf("GpuWD is %s \n", *h_odata - *gpuWD_odata < 1e-6 ? "true" : "false");
    printf("GpuWD Result: %.2f\n", *gpuWD_odata);
    printf("GpuWD Time: %f s\n", gpuWD_time);
    printf("GpuWD speed up: %.2f \n", cpu_time / gpuWD_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid_BC.x, block_BC.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid_BC.x / block_BC.x, block_BC.x);
    printf("GpuBC is %s \n", *h_odata - *gpuBC_odata < 1e-6 ? "true" : "false");
    printf("GpuBC Result: %.2f\n", *gpuBC_odata);
    printf("GpuBC Time: %f s\n", gpuBC_time);
    printf("GpuBC speed up: %.2f \n", cpu_time / gpuBC_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid_ID.x, block_ID.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid_ID.x / 2 / block_ID.x, block_ID.x);
    printf("GpuID is %s \n", *h_odata - *gpuID_odata < 1e-6 ? "true" : "false");
    printf("GpuID Result: %.2f\n", *gpuID_odata);
    printf("GpuID Time: %f s\n", gpuID_time);
    printf("GpuID speed up: %.2f \n", cpu_time / gpuID_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid_UR.x, block_UR.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid_UR.x / 2 / block_UR.x, block_UR.x);
    printf("GpuUR is %s \n", *h_odata - *gpuUR_odata < 1e-6 ? "true" : "false");
    printf("GpuUR Result: %.2f\n", *gpuUR_odata);
    printf("GpuUR Time: %f s\n", gpuUR_time);
    printf("GpuUR speed up: %.2f \n", cpu_time / gpuUR_time);

    printf("-----------------------------------------------------------------------------\n");
    printf("GPU config firstly:\t grig (%d), block(%d)\n", grid_CUR.x, block_CUR.x);
    printf("GPU config secondly:\t grig (%d), block(%d)\n", grid_CUR.x / 2 / block_CUR.x, block_CUR.x);
    printf("GpuCUR is %s \n", *h_odata - *gpuCUR_odata < 1e-6 ? "true" : "false");
    printf("GpuCUR Result: %.2f\n", *gpuCUR_odata);
    printf("GpuCUR Time: %f s\n", gpuCUR_time);
    printf("GpuCUR speed up: %.2f \n", cpu_time / gpuCUR_time);

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    free(gpubl_odata);
    free(gpuWD_odata);
    free(gpuBC_odata);
    free(gpuID_odata);
}

/*

    nvcc reduce.cu -o reduce.o && ./reduce.o
    nvcc reduce.cu -o reduce.o && /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -o reduce_profile_report ./reduce.o 

    nvcc reduce.cu -o reduce.o -lcublas -G -g
    cuda-gdb --args ./reduce.o 2 3 4

*/