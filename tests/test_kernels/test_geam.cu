#include <random>
#include <chrono>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "tensor.cuh"
#include "tools_cuda.cuh"
#include "tools_common.cuh"
#include "kernel_geam.cuh"

class test_geam:
    public testing::TestWithParam<
        std::tuple<
            uint,  // BLOCK
            bool,  //trans1
            bool,  //trans2
            size_t,  // m
            size_t  // n
        >
    >
{
public:
    uint TILE_DIM;
    size_t m, n;
    bool trans1, trans2;

    size_t ele_count, sz;
    float   *X1_host, *X1_device,\
            *X2_host, *X2_device,\
            *Y_ground_truth_host, *Y_ground_truth_device,\
            *Y_predict_host, *Y_predict_device;

    cublasHandle_t handle;

    test_geam();
    ~test_geam();
};

test_geam::test_geam() {
    std::tie(TILE_DIM, trans1, trans2, m, n) = GetParam();

    ele_count = m * n ;
    sz = ele_count * sizeof(float);

    CHECK_NOTNULL(X1_host = (float*)malloc(sz));
    CHECK_NOTNULL(X2_host = (float*)malloc(sz));
    checkCudaErrors(cudaMalloc(&X1_device, sz));
    checkCudaErrors(cudaMalloc(&X2_device, sz));

    CHECK_NOTNULL(Y_ground_truth_host = (float*)malloc(sz));
    checkCudaErrors(cudaMalloc(&Y_ground_truth_device, sz));
    CHECK_NOTNULL(Y_predict_host = (float*)malloc(sz));
    checkCudaErrors(cudaMalloc(&Y_predict_device, sz));

    // auto gen = get_rand_data_gen<float, std::uniform_real_distribution>(1.f, 1.5f);
    // auto gen1 = [](std::vector<int> i){return (1000 * (i[0] + 1) + (i[1] + 1)) * 10000;};
    auto gen1 = [](std::vector<int> i){return 0;};
    // auto gen2 = [](std::vector<int> i){return i[0] + i[1];};
    auto gen2 = [](std::vector<int> i){return (1000 * (i[0] + 1) + (i[1] + 1));};
    #pragma omp parallel for
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) 
            X1_host[r * n + c] = gen1({r, c});
    #pragma omp parallel for
    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++) 
            X2_host[r * n + c] = gen2({r, c});

    // Tensor X1(trans1 ? std::vector<size_t>{n, m} : std::vector<size_t>{m, n}, cudaMemoryTypeHost, X1_host);
    // std::cout << X1;
    // Tensor X2(trans2 ? std::vector<size_t>{n, m} : std::vector<size_t>{m, n}, cudaMemoryTypeHost, X2_host);
    // std::cout << X2;

    checkCudaErrors(cudaMemcpy(X1_device, X1_host, sz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(X2_device, X2_host, sz, cudaMemcpyHostToDevice));

    cublasCreate(&handle);
}

test_geam::~test_geam() {
    free(X1_host);
    free(X2_host);
    checkCudaErrors(cudaFree(X1_device));
    checkCudaErrors(cudaFree(X2_device));
    free(Y_ground_truth_host);
    checkCudaErrors(cudaFree(Y_ground_truth_device));
    free(Y_predict_host);
    checkCudaErrors(cudaFree(Y_predict_device));

    cublasDestroy(handle);
}


INSTANTIATE_TEST_SUITE_P(
    exhaustive_combine,
    test_geam,
    testing::Combine(
        testing::Values(  // TILE.x == TILE.y
            8
        ),
        testing::Values(
            false,
            true
        ),
        testing::Values(
            false,
            true
        ),
        testing::Values(
            32,
            64,
            128,
            256,
            512,
            1024,
            2 * 1024,
            4 * 1024,
            8 * 1024,
            16 * 1024
        ),
        testing::Values(
            32,
            64,
            128,
            256,
            512,
            1024,
            2 * 1024,
            4 * 1024,
            8 * 1024,
            16 * 1024
        )
    )
);


INSTANTIATE_TEST_SUITE_P(
    design,
    test_geam,
    testing::Combine(
        testing::Values(
            // 1
            // 4
            8
            // 16
            // 32
        ),
        testing::Values(
            true
        ),
        testing::Values(
            true
        ),
        testing::Values(
            // 64
            16 * 1024
        ),
        testing::Values(
            // 32
            16 * 1024
        )
    )
);




TEST_P(test_geam, kgeam_smem_4xvec4_minbkcft){
    int BLOCK_TILE_DIM = TILE_DIM;
    int TILE_DIM = 4 * BLOCK_TILE_DIM;
    dim3 BLOCK = dim3(BLOCK_TILE_DIM, BLOCK_TILE_DIM);
    dim3 GRID = dim3(
        ceil(n, TILE_DIM) / TILE_DIM,
        ceil(m, TILE_DIM) / TILE_DIM
    );
    size_t shared_mem = TILE_DIM * TILE_DIM * sizeof(float);
    kgeam_smem_4xvec4_minbkcft<float, float4><<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
        m,
        n,
        trans1,
        X1_device,
        trans2,
        X2_device,
        Y_predict_device
    );
    checkCudaErrors(
        cudaMemcpy(
            Y_predict_host,
            Y_predict_device,
            sz,
            cudaMemcpyDeviceToHost
        )
    );

    // Tensor Y_P({m, n}, cudaMemoryTypeHost, Y_predict_host);
    // std::cout << Y_P;

    float alpha = 1.f, beta = 1.f;
    cublasSgeam(
        handle,
        trans1 ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans2 ? CUBLAS_OP_T : CUBLAS_OP_N,
        n, m,
        &alpha, X1_device, trans1 ? m : n,
        &beta, X2_device, trans2 ? m : n,
        Y_ground_truth_device, n
    );
    checkCudaErrors(
        cudaMemcpy(
            Y_ground_truth_host,
            Y_ground_truth_device,
            sz,
            cudaMemcpyDeviceToHost
        )
    );

    // Tensor Y_G({m, n}, cudaMemoryTypeHost, Y_ground_truth_host);
    // std::cout << Y_G;

    for (int r = 0; r < m; r++)
        for (int c = 0; c < n; c++)
            ASSERT_LE(
                abs((Y_ground_truth_host[r * n + c] - Y_predict_host[r * n + c]) / Y_ground_truth_host[r * n + c]),
                0.0002
            )   << "\npos[" << r << ", " << c << "]:"
                << "\nY_ground_truth_host: " <<  Y_ground_truth_host[r * n + c]
                << "\nY_predict_host: " <<  Y_predict_host[r * n + c];
}