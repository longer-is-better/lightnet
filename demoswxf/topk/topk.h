#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include "device_functions.h"
#include <stdio.h> //c标准输出输出库
#include <math.h>
#include <stdlib.h> //c标准库包含rand，产生随机数
 
//topk 问题 数组前k个大的元素
//归约，累加求和
 
#define N 100000000  //数据大小
#define BLOCK_SIZE 256  //一个块中有256个线程
#define GRID_SIZE 64 //32 每个网格中有32个块
#define topk 20 
 

//理论，求一个大数组的前20个最大值，先将数组放入GPU内，每个block中求出最大的前20个值，放入_1_passresult
//然后每个block前20个值放一块在求前20个值得到最中结果
 
 
__device__ __host__ void insert_sort(int*array,int k,int data);
	//由__device__ __host__修饰符用于一个函数，表示该函数可以在GPU（设备）和CPU（主机）上执行
	//__device__ 声明一个函数为设备函数，该函数只能在GPU上执行，一般是被其他设备函数或者kernel调用
	// __host__ 声明一个函数为主机函数，该函数只能在CPU上执行，通常用于管理设备内存、启动核心等操作。

__global__ void gpu_topk(int* input, int* output, int length, int k);
	/*
	参数说明：
	input 输入数组
	output 输出数组
	length 输入数组的长度
	*/
 
void cpu_topk(int* input, int* output, int length, int k);
