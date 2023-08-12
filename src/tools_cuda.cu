#include "tools_cuda.cuh"


void check_device_data(float* p_data, size_t ele) {
  VLOG(8) << "check_device_data";
  float t[ele];
  checkCudaErrors(cudaMemcpy(t, p_data, ele * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < ele; i++) {
    VLOG(8) << t[i];
  }
}