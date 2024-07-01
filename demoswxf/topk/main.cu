#include "topk.h"

__managed__ int source[N];   //原数组 
//__managed__  cuda关键字，用于声明所谓的托管内存，允许内存在CPU和GPU之间自动共享。
//用 __managed__ 声明的变量可以同时被 CPU 和 GPU 访问，无需手动在主机（CPU）和设备（GPU）之间复制数据。
//使用托管内存简化了内存管理，因为它允许 CPU 和 GPU 在无需显式数据传输命令的情况下访问相同的内存。
 
__managed__ int gpu_result[topk];  //topk最终结果
__managed__ int _1_pass_result[topk * GRID_SIZE];//每个block的前 topk 个，即中间结果
 
//理论，求一个大数组的前20个最大值，先将数组放入GPU内，每个block中求出最大的前20个值，放入_1_passresult
//然后每个block前20个值放一块在求前20个值得到最中结果


int main(){
 
	//为原数组赋初值
	printf("初始化源数据.....\n");
	for (int i = 0; i < N; i++) {
		// source[i] = rand();
		source[i] = (i + 1) % 100;
	}
	printf("完成初始化源数据.....\n");
 
	//cuda事件-计时
	cudaEvent_t start, stop_gpu, stop_cpu;
	cudaEventCreate(&start);
	cudaEventCreate(&stop_gpu);
	cudaEventCreate(&stop_cpu);
	cudaEventRecord(start);
	cudaEventSynchronize(start);//事件同步
	//这个函数用于等待一个 CUDA 事件完成。
	// 当你在 CUDA 程序中设置一个事件时，比如 cudaEventRecord(event, stream)，
	// 它会在特定的流（stream）中标记一个点。cudaEventSynchronize(event) 会阻塞调用线程，
	// 直到该事件发生，即直到 GPU 上的相关操作完成。
	printf("GPU Run *************\n");
	int times = 1;
	//计算
	for (int i = 0; i < times; i++) {
		gpu_topk <<<GRID_SIZE, BLOCK_SIZE >>> (source, _1_pass_result, N, topk);
		gpu_topk <<<1, BLOCK_SIZE >>> (_1_pass_result, gpu_result, topk * GRID_SIZE, topk);
		cudaDeviceSynchronize();
		//cudaDeviceSynchronize() 函数会阻塞调用线程，直到 GPU 完成所有队列中的操作。
		// 这包括所有 CUDA 核心、内存复制和其他相关的 GPU 操作。
	}
	printf("GPU Run Complete %d 次*************\n",times);
	cudaEventRecord(stop_gpu);
	cudaEventSynchronize(stop_gpu);
 
 
	//cpu结果初始化
	int cpu_result[topk] = { 0 }; //cpu结果存储
	printf("CPU Run *************\n");
	//计算
	cpu_topk(source, cpu_result, N, topk);
	printf("GPU Run Complete *************\n");
	cudaEventRecord(stop_cpu);
	cudaEventSynchronize(stop_cpu);
 
	//计算两次时间
	float time_cpu, time_gpu;
	cudaEventElapsedTime(&time_gpu, start, stop_gpu);
	cudaEventElapsedTime(&time_cpu, stop_gpu, stop_cpu);
 
	//判断GPU计算是否有误
	bool error = false;
	for (int i = 0; i < topk; i++) {
		printf(" CPU top%d:\t%d;\tGputop%d:\t%d;\n", i + 1, cpu_result[i], i + 1, gpu_result[i]);
		if (fabs(gpu_result[i] - cpu_result[i]) > 0) {
			error = true;
		}
	}
	printf("Result:%s\n", (error ? "Error" : "pass"));
	printf("CPU time: %.2f; GPU time: %.2f\n", time_cpu, time_gpu);
 
	return 0;
}
 
 
 
/*
 	nvcc main.cu topk.cu -o main && ./main 
	nvcc main.cu topk.cu -o main && /usr/local/NVIDIA-Nsight-Compute/ncu --set full -f -o topk_profile_report ./main
*/
 