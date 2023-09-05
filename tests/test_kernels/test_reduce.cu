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
            size_t  // n < 2048 * 2048
        >
    >
{
public:
    dim3 BLOCK;
    size_t n;
    
    size_t X_size;
    float *X_host, *X_device, *Y_device, Y_ground_truth_host = 0, Y_predict_host;

    test_reduce();
    ~test_reduce();
};

test_reduce::test_reduce() {
    std::tie(BLOCK, n) = GetParam();

    X_size = n * sizeof(float);

    CHECK_NOTNULL(X_host = (float*)malloc(X_size));
    checkCudaErrors(cudaMalloc(&X_device, X_size));

    auto X_gen = get_rand_data_gen<float, std::uniform_real_distribution>(1.f, 1.f);
    for (int i = 0; i < n; i++) X_host[i] = X_gen({i});
    checkCudaErrors(cudaMemcpy(X_device, X_host, X_size, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&Y_device, sizeof(float)));
}

test_reduce::~test_reduce() {
    free(X_host);
    checkCudaErrors(cudaFree(X_device));
    checkCudaErrors(cudaFree(Y_device));
}


INSTANTIATE_TEST_SUITE_P(
    design,
    test_reduce,
    testing::Combine(
        testing::Values(
            512
        ),
        testing::Values(
            1024 * 1024 * 64
        )
    )
);

INSTANTIATE_TEST_SUITE_P(
    exhaustive_combine,
    test_reduce,
    testing::Combine(
        testing::Values(
            3,
            7,
            19,
            32,
            64,  // at lest 64
            128,
            256,
            512,
            1024
        ),
        testing::Values(
            1,
            4,
            6,
            7,
            8,
            16,
            17,
            27,
            33,
            256,
            512,
            777,
            1024,
            2240,
            2241,
            46656,
            1024 * 1024,
            1024 * 1024 * 8
        )
    )
);


TEST_P(test_reduce, kreduce_sum){
    size_t shared_mem = BLOCK.x * sizeof(float);

    size_t wn = n;
    while (wn != 1){
        dim3 GRID = ceil(wn, BLOCK.x * 2) / (BLOCK.x * 2);
        kreduce_sum<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
            n,
            wn,
            X_device,
            X_device
        );
        wn = GRID.x;
    }

    checkCudaErrors(cudaMemcpy(&Y_predict_host, X_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];
    // if (op == REDUCE_OP::AVG) Y_ground_truth_host /= n;    

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}

TEST_P(test_reduce, kreduce_sum_dbbf){
    size_t shared_mem = (2 * BLOCK.x + 1) * sizeof(float);

    dim3 GRID = min(ceil(n, BLOCK.x) / BLOCK.x, (size_t)1024);
    // dim3 GRID = 1;
    size_t n_pad = ceil(n, GRID.x * BLOCK.x);
    float *X_device_pad = nullptr;
    checkCudaErrors(cudaMalloc(&X_device_pad, n_pad * sizeof(float)));
    checkCudaErrors(cudaMemset(X_device_pad, 0, n_pad * sizeof(float)));
    checkCudaErrors(cudaMemcpy(X_device_pad, X_device, n * sizeof(float), cudaMemcpyDeviceToDevice));

    kreduce_sum_dbbf<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        n_pad,
        X_device_pad,
        Y_device
    );

    checkCudaErrors(cudaMemcpy(&Y_predict_host, Y_device, sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(X_device_pad));
    

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}

TEST_P(test_reduce, kreduce_sum_sfl){

    dim3 GRID = min(ceil(n, BLOCK.x) / BLOCK.x, (size_t)1024);
    kreduce_sum_sfl<<<GRID, BLOCK, 0, cudaStreamDefault>>>(
        X_device,
        X_device,
        n
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    kreduce_sum_sfl<<<1, GRID, 0, cudaStreamDefault>>>(
        X_device,
        X_device,
        GRID.x
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

    checkCudaErrors(cudaMemcpy(&Y_predict_host, X_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}

TEST_P(test_reduce, kreduce_sum_warpsfl_atom){

    dim3 GRID = min(ceil(n, BLOCK.x) / BLOCK.x, (size_t)1024);
    kreduce_sum_warpsfl_atom<<<GRID, BLOCK, 0, cudaStreamDefault>>>(
        X_device,
        Y_device,
        n
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

    checkCudaErrors(cudaMemcpy(&Y_predict_host, Y_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}

TEST_P(test_reduce, kreduce_sum_blksfl_atom){

    dim3 GRID = min(ceil(n, BLOCK.x) / BLOCK.x, (size_t)1024);
    kreduce_sum_blksfl_atom<<<GRID, BLOCK, 0, cudaStreamDefault>>>(
        X_device,
        Y_device,
        n
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

    checkCudaErrors(cudaMemcpy(&Y_predict_host, Y_device, sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) Y_ground_truth_host += X_host[i];

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}

TEST_P(test_reduce, kreduce_sum_vec4_blksfl_atom){

    dim3 GRID = min(ceil(n, BLOCK.x) / BLOCK.x, (size_t)1024);
    // dim3 GRID = 1;
    kreduce_sum_vec4_blksfl_atom<float, float4><<<GRID, BLOCK, 0, cudaStreamDefault>>>(
        X_device,
        Y_device,
        n
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));

    checkCudaErrors(cudaMemcpy(&Y_predict_host, Y_device, sizeof(float), cudaMemcpyDeviceToHost));

    float bf;
    for (int i = 0; i < n; i++) {
        bf = Y_ground_truth_host;
        Y_ground_truth_host += X_host[i];
        if (i % 100000 == 0) {
            LOG(INFO) << X_host[i] << "             " << Y_ground_truth_host<< "     diff    " << Y_ground_truth_host - bf;
        }
    }

    EXPECT_LE(abs((Y_ground_truth_host - Y_predict_host) / Y_ground_truth_host), 0.0002)
        << "\nY_ground_truth_host: " <<  Y_ground_truth_host
        << "\nY_predict_host: " <<  Y_predict_host;
}