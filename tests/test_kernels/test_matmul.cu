#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor.cuh"
#include "tools_cuda.cuh"
#include "tools_common.cuh"
#include "kernel_matmul.cuh"

class test_matmul:
    public testing::TestWithParam<
        std::tuple<
            bool,  // trans W
            bool,  // trans X
            int,  // m
            int,  // n
            int,  // k
            std::function<float(const std::vector<int>&)>,  // W gen
            std::function<float(const std::vector<int>&)>,  // X gen
            dim3  // block
        >
    >
{
public:
    bool trans_W, trans_X;
    int m ,n, k;
    std::function<float(const std::vector<int>&)> W_gen, X_gen;
    dim3 BLOCK;
    
    float alpha = 1.f, beta = 0.f;
    size_t W_size, X_size, Y_size;
    float *W_host, *X_host, *Y_ground_truth_host, *Y_predict_host, *W_device, *X_device, *Y_ground_truth_device, *Y_predict_device;
    dim3 GRID;
    size_t shared_mem;

    cublasHandle_t handle = nullptr;

    test_matmul();
    ~test_matmul();
};

test_matmul::test_matmul() {
    std::tie(
        trans_W,
        trans_X,
        m,
        n,
        k,
        W_gen,
        X_gen,
        BLOCK
    ) = GetParam();

    cublasCreate(&handle);
    W_size = m * k * sizeof(float);
    X_size = k * n * sizeof(float);
    Y_size = m * n * sizeof(float);

    CHECK_NOTNULL(W_host = (float*)malloc(W_size));
    CHECK_NOTNULL(X_host = (float*)malloc(X_size));
    CHECK_NOTNULL(Y_ground_truth_host = (float*)malloc(Y_size));
    CHECK_NOTNULL(Y_predict_host = (float*)malloc(Y_size));
    checkCudaErrors(cudaMalloc(&W_device, W_size));
    checkCudaErrors(cudaMalloc(&X_device, X_size));
    checkCudaErrors(cudaMalloc(&Y_ground_truth_device, Y_size));
    checkCudaErrors(cudaMalloc(&Y_predict_device, Y_size));

    for (int r = 0; r < m; r++) for (int c = 0; c < k; c++) W_host[r * k + c] = W_gen({r, c});
    for (int r = 0; r < k; r++) for (int c = 0; c < n; c++) X_host[r * n + c] = W_gen({r, c});
    checkCudaErrors(cudaMemcpy(W_device, W_host, W_size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(X_device, X_host, X_size, cudaMemcpyHostToDevice));


    GRID = dim3(ceil(n, BLOCK.x)/BLOCK.x, ceil(m, BLOCK.y)/BLOCK.y);
    shared_mem = BLOCK.x * BLOCK.y * sizeof(float) * 2;

}

test_matmul::~test_matmul() {
    free(W_host);
    free(X_host);
    free(Y_ground_truth_host);
    free(Y_predict_host);
    checkCudaErrors(cudaFree(W_device));
    checkCudaErrors(cudaFree(X_device));
    checkCudaErrors(cudaFree(Y_ground_truth_device));
    checkCudaErrors(cudaFree(Y_predict_device));

    cublasDestroy(handle);
}


INSTANTIATE_TEST_SUITE_P(
    design,
    test_matmul,
    testing::Values(
        std::make_tuple(
            false,
            false,
            8,
            4,
            6,
            [](const std::vector<int>& i){return i[0] + i[1];},
            [](const std::vector<int>& i){return i[1] - i[0];},
            dim3(4, 4)
        )
    )
);


// INSTANTIATE_TEST_SUITE_P(
//     exhaustive,
//     test_matmul,
//     testing::Values(
//         std::make_tuple(
//             false,
//             false,

//         )
//     )
// );

TEST_P(test_matmul, positive){
    Tensor show_W({1, 1, size_t(m), size_t(k)}, cudaMemoryTypeDevice, W_device);
    VLOG(8) << "show W \n" << show_W;
    Tensor show_X({size_t(k), size_t(n)}, cudaMemoryTypeDevice, X_device);
    VLOG(8) << "show X \n" << show_X;
    cublasSgemm(
        handle,
        trans_X ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_W ? CUBLAS_OP_T : CUBLAS_OP_N,
        n,
        m,
        k,
        &alpha,
        X_device,
        n,
        W_device,
        k,
        &beta,
        Y_ground_truth_device,
        n
    );
    cudaMemcpy(Y_ground_truth_host, Y_ground_truth_device, Y_size, cudaMemcpyDeviceToHost);

    kmatmul<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        trans_W,
        trans_X,
        m,
        k,
        n,
        W_device,
        X_device,
        Y_predict_device
    );
    checkCudaErrors(cudaStreamSynchronize(cudaStreamDefault));
    cudaMemcpy(Y_predict_host, Y_predict_device, Y_size, cudaMemcpyDeviceToHost);

    Tensor gt({size_t(m), size_t(n)}, cudaMemoryTypeHost, Y_ground_truth_host);
    VLOG(8) << "show gt \n" << gt;
    Tensor pd({size_t(m), size_t(n)}, cudaMemoryTypeHost, Y_predict_host);
    VLOG(8) << "show pd \n" << pd;

    for (int r = 0; r < m; r++) {
        for (int c = 0; c < n; c++) {
            CHECK_DOUBLE_EQ(
                Y_predict_host[r * n + c],
                Y_ground_truth_host[r * n + c]
            );
        }
    }
}