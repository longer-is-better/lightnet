#include "topk.h"

__device__ __host__ void insert_sort(int*array,int k,int data) {
	//由__device__ __host__修饰符用于一个函数，表示该函数可以在GPU（设备）和CPU（主机）上执行
	//__device__ 声明一个函数为设备函数，该函数只能在GPU上执行，一般是被其他设备函数或者kernel调用
	// __host__ 声明一个函数为主机函数，该函数只能在CPU上执行，通常用于管理设备内存、启动核心等操作。
	// for (int i = 0; i < k; i++) {
	// 	//如果数据重复，就不参与排序了，选的是前20个最大的。
	// 	if (array[i] == data) {
	// 		return;
	// 	}
	// }
	//如果要插入的元素小于数组最后一个元素，那就不参与排序，因为就不是前20个最大的了
	if (data < array[k - 1]) {
		return;
	}
	//从倒数第二个开始，向前比较，如果当前数据比数据元素大，那么数组元素就向后移位，如果小于则将该位置的后一位作为插入位置。位置
	for (int i = k - 2; i >= 0; i--) {
		if (data > array[i]) {
			array[i + 1] = array[i];
		}
		else {
			array[i + 1] = data;
			return;
		}
	}
 
	//如果data比所有数据都大，那么将这个数据插入到改数组的第一个位置
	array[0] = data;
}
 
__global__ void gpu_topk(int* input, int* output, int length, int k) {
	/*
	参数说明：
	input 输入数组
	output 输出数组
	length 输入数组的长度
	*/
	//申请共享内存数据，用与保存每个块的计算结果
	__shared__ int ken[BLOCK_SIZE * topk];
	
 
	int top_array[topk];
	//top_array初始化，给最小值
	for (int i = 0; i < topk; i++) {
		top_array[i] = INT_MIN;
	}
 
	//插入排序
	//对数组中的所有数据进行插入排序
	for (int idx = threadIdx.x + blockDim.x * blockIdx.x; idx < length; idx += gridDim.x * blockDim.x) {
		insert_sort(top_array, topk, input[idx]);
	}
 
	//维护好的top array放进共享内存数组
	for (int i = 0; i < topk; i++) {
		ken[topk * threadIdx.x + i] = top_array[i];
	}
	__syncthreads();
 
	//共像内存中的数据合并，并行归约。
	// 每一步都将当前活动的线程数减半，这些线程合并相邻的 top_array。
	// 这个过程在每个线程块内部进行，最终得到该块的局部前 k 个最大值。
	for (int i = BLOCK_SIZE/2; i >= 1; i /= 2) {
		if (threadIdx.x < i) {
			for (int m = 0; m < topk; m++) {
				insert_sort(top_array, topk, ken[topk * (threadIdx.x + i) + m]);
			}
		}
		__syncthreads();
		if (threadIdx.x < i) {
			for (int m = 0; m < topk; m++) {				
				ken[topk * threadIdx.x + m] = top_array[m];
			}
		}
		__syncthreads();
	}
 
	//将最终结果写入输出数组，只使用每个线程块一个线程，可以是0，也可以使其他，
	// 用于将前topk个最大值写入输出数组相应的位置
	if (blockIdx.x * blockDim.x < length) {
		if (threadIdx.x == 0) {
			for (int i = 0; i < topk; i++) {
				output[topk * blockIdx.x + i] = ken[i];
			}
		}
	}
 
}
 
void cpu_topk(int* input, int* output, int length, int k) {
	for (int i = 0; i < length; i++) {
		insert_sort(output, k, input[i]);
	}
}
