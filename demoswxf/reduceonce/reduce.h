#pragma once

#include <stdio.h>
#include "cuda_runtime_api.h"

#define N 100000000
#define BLOCKSIZE 256
#define GRIDSIZE 32

__global__ void sum_gpu(int *in, int count, int *out);

