#include "reduce.h"

__managed__ int source[N];
__managed__ int gpu_result[1] = {0};

int main() {
    int cpu_result = 0;
    printf("Init input source...\n");
    for (int i = 0; i < N; ++i) {
        source[i] = rand() % 10;
    }
    printf("source: %d %d %d %d %d %d %d %d %d %d...\n", source[0], source[1], source[2], source[3], source[4], source[5],source[6], source[7], source[8], source[9]);

    cudaEvent_t start, stop_gpu, stop_cpu;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_gpu);
    cudaEventCreate(&stop_cpu);

    cudaEventRecord(start);
    cudaEventSynchronize(start);

    int times = 1;

    for (int i = 0; i < times; ++i) {
        gpu_result[0] = 0;
        sum_gpu<<<GRIDSIZE, BLOCKSIZE>>>(source, N, gpu_result);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    for (int i = 0; i < N; ++i) {
        cpu_result += source[i];
    }
    cudaEventRecord(stop_cpu);
    cudaEventSynchronize(stop_cpu);

    float time_gpu, time_cpu;
    cudaEventElapsedTime(&time_gpu, start, stop_gpu);
    cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
    printf("CPU time: %.2f\t\tGPU time: %.2f\n", time_cpu, time_gpu/times);

    printf("CPU Result: %d\tGPU Result: %d\n", cpu_result, gpu_result[0]);
    printf("Result: %s\n", cpu_result == gpu_result[0] ? "Pass" : "Error");

    return 0;
}

/*

    nvcc main.cu reduce.cu -o main && ./main
    nvcc main.cu reduce.cu -o main && /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -o reduce_profile_report ./main

*/