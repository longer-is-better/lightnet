#include <random>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor.cuh"
#include "tools_cuda.cuh"
#include "tools_common.cuh"
#include "kernel_reduce.cuh"

class test_reduce:
    public testing::TestWithParam<
        std::tuple<
            dim3,  // BLOCK
            size_t,  // n < 2048 * 2048
            REDUCE_OP
        >
    >
{
public:
    dim3 BLOCK;
    size_t n;
    REDUCE_OP op;
    
    size_t X_size;
    float *X_host, *X_device, Y_ground_truth_host = 0, Y_predict_host;

    test_reduce();
    ~test_reduce();
};

test_reduce::test_reduce() {
    std::tie(BLOCK, n, op) = GetParam();

    X_size = n * sizeof(float);

    CHECK_NOTNULL(X_host = (float*)malloc(X_size));
    checkCudaErrors(cudaMalloc(&X_device, X_size));

    auto X_gen = get_rand_data_gen<float, std::uniform_real_distribution>(-2.f, 1.f);
    for (int i = 0; i < n; i++) X_host[i] = X_gen({i});
    checkCudaErrors(cudaMemcpy(X_device, X_host, X_size, cudaMemcpyHostToDevice));
}

test_reduce::~test_reduce() {
    free(X_host);
    checkCudaErrors(cudaFree(X_device));
}


INSTANTIATE_TEST_SUITE_P(
    pow_n_n,
    test_reduce,
    testing::Combine(
        testing::Values(
            32,
            64,
            128,
            256,
            512,
            1024
        ),
        testing::Values(
            1,
            4,
            6,
            8,
            16,
            17,
            27,
            256,
            512,
            1024,
            1025,
            3125,
            46656
        ),
        testing::Values(
            REDUCE_OP::SUM,
            REDUCE_OP::AVG
        )
    )
);


TEST_P(test_reduce, positive){
    size_t shared_mem = BLOCK.x * sizeof(float);

    size_t wn = n;
    while (wn != 1){
        dim3 GRID = ceil(wn, BLOCK.x * 2) / (BLOCK.x * 2);
        kreduce<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
            n,
            wn,
            X_device,
            X_device,
            op
        );
        wn = GRID.x;
    }

    checkCudaErrors(cudaMemcpy(&Y_predict_host, X_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];
    if (op == REDUCE_OP::AVG) Y_ground_truth_host /= n;    

    EXPECT_NEAR(Y_ground_truth_host, Y_predict_host, 0.1);
}