#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor.cuh"
#include "tools_cuda.cuh"
#include "tools_common.cuh"
#include "kernel_transpose.cuh"

template<class DATA_TYPE>
class test_transpose_T:
    public testing::TestWithParam<
        std::tuple<
            uint,  // BLOCK
            size_t,  // m
            size_t  // n
        >
    >
{
public:
    uint TILE_DIM;
    size_t m, n;

    size_t ele_count, sz;
    DATA_TYPE   *X_host, *X_device,\
            *Y_ground_truth_host, *Y_ground_truth_device,\
            *Y_predict_host, *Y_predict_device;

    cublasHandle_t handle;

    test_transpose_T();
    ~test_transpose_T();
};

template<class DATA_TYPE>
test_transpose_T<DATA_TYPE>::test_transpose_T() {
    std::tie(TILE_DIM, m, n) = GetParam();

    ele_count = m * n ;
    sz = ele_count * sizeof(DATA_TYPE);

    CHECK_NOTNULL(X_host = (DATA_TYPE*)malloc(sz));
    checkCudaErrors(cudaMalloc(&X_device, sz));

    CHECK_NOTNULL(Y_ground_truth_host = (DATA_TYPE*)malloc(sz));
    checkCudaErrors(cudaMalloc(&Y_ground_truth_device, sz));
    CHECK_NOTNULL(Y_predict_host = (DATA_TYPE*)malloc(sz));
    checkCudaErrors(cudaMalloc(&Y_predict_device, sz));

    auto X_gen = get_rand_data_gen<DATA_TYPE, std::uniform_real_distribution>(1.f, 1.5f);
    // auto X_gen = [](std::vector<int> i){return 10 * i[0] + i[1] + 1;};
#pragma omp parallel for
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) 
            X_host[r * n + c] = X_gen({r, c});

    checkCudaErrors(cudaMemcpy(X_device, X_host, sz, cudaMemcpyHostToDevice));

    cublasCreate(&handle);
}

template<class DATA_TYPE>
test_transpose_T<DATA_TYPE>::~test_transpose_T() {
    free(X_host);
    checkCudaErrors(cudaFree(X_device));
    free(Y_ground_truth_host);
    checkCudaErrors(cudaFree(Y_ground_truth_device));
    free(Y_predict_host);
    checkCudaErrors(cudaFree(Y_predict_device));

    cublasDestroy(handle);
}