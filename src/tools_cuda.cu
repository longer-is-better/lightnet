#include "tools_cuda.cuh"


std::map<
    std::string,
    GPU_TICKTOCK
> GPU_TICKTOCKS;

void check_device_data(float* p_data, size_t ele) {
  VLOG(8) << "check_device_data";
  float t[ele];
  checkCudaErrors(cudaMemcpy(t, p_data, ele * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < ele; i++) {
    VLOG(8) << t[i];
  }
}

std::ostream& operator<<(std::ostream& os, const dim3& dm) {
    os << "dim3(" << dm.x << ", " << dm.y << ", " << dm.z << ")";
    return os;
}



void GPU_TICK(std::string task, cudaStream_t stm){
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    GPU_TICKTOCKS[task] = GPU_TICKTOCK();
    GPU_TICKTOCKS[task].tick = start;
    GPU_TICKTOCKS[task].tock = stop;
    checkCudaErrors(cudaEventRecord(start, stm));
}

void GPU_TOCK(std::string task, cudaStream_t stm){
    checkCudaErrors(cudaEventRecord(GPU_TICKTOCKS[task].tock, stm));
    checkCudaErrors(cudaEventSynchronize(GPU_TICKTOCKS[task].tock));
    checkCudaErrors(
        cudaEventElapsedTime(
            &GPU_TICKTOCKS[task].interval,
            GPU_TICKTOCKS[task].tick,
            GPU_TICKTOCKS[task].tock
        )
    );
}