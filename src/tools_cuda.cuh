#pragma once
#include <cstdio>
#include <map>
#include <string>

#include <cuda_runtime.h>

#include <glog/logging.h>



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

void check_device_data(float* p_data, size_t ele);

std::ostream& operator<<(std::ostream& os, const dim3 &dm);

struct GPU_TICKTOCK
{
    cudaEvent_t tick, tock;
    float interval;
};

extern std::map<
    std::string,
    GPU_TICKTOCK
> GPU_TICKTOCKS;


void GPU_TICK(std::string task, cudaStream_t stm);
void GPU_TOCK(std::string task, cudaStream_t stm);