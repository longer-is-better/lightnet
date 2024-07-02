#include "transpose.h"

__managed__ int matrix[N][M];
__managed__ int gpu_result[M][N];
__managed__ int cpu_result[M][N];


int main() {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < M; ++x) {
            matrix[y][x] = rand() % 1024;
        }
    }

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    dim3 block(BLOCKSIZE, BLOCKSIZE);
    dim3 grid((M + BLOCKSIZE - 1) / BLOCKSIZE, (N + BLOCKSIZE - 1) / BLOCKSIZE);

    int times = 1;

    for (int i = 0; i < times; ++i) {
        gpu_matrix_transpose<<<grid, block>>>(matrix, gpu_result);
        // gpu_shared_matrix_transpose<<<grid, block>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    cpu_matrix_transpose(matrix, cpu_result);
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_gpu, time_cpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    bool err_naive_gpu = checkResult(gpu_result, cpu_result);

    cudaEvent_t start_s, stop_s;
    cudaEventCreate(&start_s);
    cudaEventCreate(&stop_s);
    cudaEventRecord(start_s);
    for (int i = 0; i < times; ++i) {
        gpu_shared_matrix_transpose<<<grid, block>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_s);
    bool err_shared_gpu = checkResult(gpu_result, cpu_result);

    cudaEvent_t start_b, stop_b;
    cudaEventCreate(&start_b);
    cudaEventCreate(&stop_b);
    cudaEventRecord(start_b);
    for (int i = 0; i < times; ++i) {
        gpu_shared_bank_matrix_transpose<<<grid, block>>>(matrix, gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_b);
    bool err_shared_bank_gpu = checkResult(gpu_result, cpu_result);

    float time_shared, time_shared_bank;
    cudaEventElapsedTime(&time_shared, start_s, stop_s);
    cudaEventElapsedTime(&time_shared_bank, start_b, stop_b);

    

    printf("Naive GPU:\n");
    printf("Result: %s\n", err_naive_gpu ? "Error" : "Pass");
    printf("CPU time: %.2f\t GPU time: %.2f\n", time_cpu, time_gpu/times);

    printf("Shared Memory:\n");
    printf("Result: %s\n", err_shared_gpu ? "Error" : "Pass");
    printf("CPU time: %.2f\t GPU time: %.2f\n", time_cpu, time_shared/times);

    printf("Shared Bank:\n");
    printf("Result: %s\n", err_shared_bank_gpu ? "Error" : "Pass");
    printf("CPU time: %.2f\t GPU time: %.2f\n", time_cpu, time_shared_bank/times);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_gpu);
    cudaEventDestroy(stop_cpu);
    cudaEventDestroy(start_s);
    cudaEventDestroy(start_b);
    cudaEventDestroy(stop_s);
    cudaEventDestroy(stop_b);

}

/*
 	nvcc main.cu transpose.cu -o main && ./main 
	nvcc main.cu transpose.cu -o main && /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -o transpose_profile_report ./main
*/