#include <random>
#include <chrono>
#include <string>
#include <cublas_v2.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "cnpy.cuh"
#include "tensor.cuh"
#include "tools_cuda.cuh"
#include "tools_common.cuh"
#include "kernel_bev_pool_v2.cuh"


struct DIM
{
    int x = 1;
    int y = 1;
    int z = 1;
    int w = 1;
    DIM(int inx, int iny, int inz, int inw) : x(inx), y(iny), z(inz), w(inw){};
    DIM(int inx, int iny, int inz) : x(inx), y(iny), z(inz){};
    DIM(int inx, int iny) : x(inx), y(iny){};
    DIM(int inx) : x(inx){};
    DIM(){};
    template <typename T>
    size_t size()
    {
        return x * y * z * w * sizeof(T);
    }
    size_t nums()
    {
        return x * y * z * w;
    }
};


template<
    typename TENSORTYPE,
    typename ACCTYPE,
    const int TC,
    const int TN,
    const int BC,
    const int BN
>
class test_bev_pool_v2_fma_tnc:
    public testing::TestWithParam<
        std::tuple<
            INPUT_FILE_TYPE,
            std::string,  // input files dir
            DIM,  // depth_shape
            DIM,  // feat_shape
            DIM  // out_shape
        >
    >
{
public:
    INPUT_FILE_TYPE inputfiletype;
    std::string input_files_dir;
    DIM depth_shape;
    DIM feat_shape;
    DIM out_shape;


    cnpy::NpyArray  ranks_bev,\
                    ranks_depth,\
                    ranks_feat,\
                    interval_lengths,\
                    interval_starts;

    int c, n_intervals,\
        *ranks_depth_host, *ranks_depth_device,\
        *ranks_feat_host, *ranks_feat_device,\
        *ranks_bev_host, *ranks_bev_device,\
        *ranks_bev_mask_host, *ranks_bev_mask_device,\
        *interval_starts_host, *interval_starts_device,\
        *interval_lengths_host, *interval_lengths_device,\
        *interval_starts_e_host, *interval_starts_e_device,\
        *interval_lengths_e_host, *interval_lengths_e_device;
    TENSORTYPE  *depth_host, *depth_device,\
                *feat_host, *feat_device,\
                *out_gt_host, *out_gt_device,\
                *out_test_host, *out_test_device;

    // cudaStream_t stream;
    test_bev_pool_v2_fma_tnc();
    ~test_bev_pool_v2_fma_tnc();
};

template<
    typename TENSORTYPE,
    typename ACCTYPE,
    const int TC,
    const int TN,
    const int BC,
    const int BN
>
test_bev_pool_v2_fma_tnc<
    TENSORTYPE,
    ACCTYPE,
    TC,
    TN,
    BC,
    BN
>::test_bev_pool_v2_fma_tnc() {
    std::tie(
        inputfiletype,
        input_files_dir,
        depth_shape,
        feat_shape,
        out_shape
    ) = GetParam();

    if (inputfiletype == INPUT_FILE_TYPE::npy) {
        ranks_bev = cnpy::npy_load(input_files_dir + "/ranks_bev.npz.npy");
        ranks_depth = cnpy::npy_load(input_files_dir + "/ranks_depth.npz.npy");
        ranks_feat = cnpy::npy_load(input_files_dir + "/ranks_feat.npz.npy");
        interval_lengths = cnpy::npy_load(input_files_dir + "/interval_lengths.npz.npy");
        interval_starts = cnpy::npy_load(input_files_dir + "/interval_starts.npz.npy");
    }

    // checkCudaErrors(cudaStreamCreate(&stream));

    depth_host = (TENSORTYPE*)malloc(depth_shape.size<TENSORTYPE>());
    feat_host = (TENSORTYPE*)malloc(feat_shape.size<TENSORTYPE>());
    ranks_depth_host = (int*)malloc(ranks_depth.num_bytes());
    ranks_feat_host = (int*)malloc(ranks_feat.num_bytes());
    ranks_bev_host = (int*)malloc(ranks_bev.num_bytes());
    ranks_bev_mask_host = (int*)malloc(interval_starts.num_bytes()); memset(ranks_bev_mask_host, 0, interval_starts.num_bytes());
    interval_starts_host = (int*)malloc(interval_starts.num_bytes());
    interval_lengths_host = (int*)malloc(interval_lengths.num_bytes());
    interval_starts_e_host = (int*)malloc(interval_starts.num_bytes());
    interval_lengths_e_host = (int*)malloc(interval_lengths.num_bytes());
    out_gt_host = (TENSORTYPE*)malloc(out_shape.size<TENSORTYPE>());
    out_test_host = (TENSORTYPE*)malloc(out_shape.size<TENSORTYPE>());


    c = feat_shape.w;
    n_intervals = 192 * 256;
    for(int i = 0; i < ranks_depth.num_vals; i++) ranks_depth_host[i] = (int)ranks_depth.data<float>()[i];
    for(int i = 0; i < ranks_feat.num_vals; i++) ranks_feat_host[i] = (int)ranks_feat.data<float>()[i];
    for(int i = 0; i < ranks_bev.num_vals; i++) ranks_bev_host[i] = (int)ranks_bev.data<float>()[i];
    for(int i = 0; i < interval_starts.num_vals; i++) interval_starts_host[i] = (int)interval_starts.data<float>()[i];
    for(int i = 0; i < interval_lengths.num_vals; i++) interval_lengths_host[i] = (int)interval_lengths.data<float>()[i];

    for (int i = 0; i < ranks_bev.num_vals; i++) {
        int idx = ranks_bev_host[i];
        if (idx != -1 && ranks_bev_mask_host[idx] == 0)
        ranks_bev_mask_host[idx] = 1;
    }
    int j = 0;
    for (int i = 0; i < interval_starts.num_vals; i++) {
        if (ranks_bev_mask_host[i] == 0) {
            interval_starts_e_host[i] = 0;
            interval_lengths_e_host[i] = 0;
        } else {
            interval_starts_e_host[i] = interval_starts_host[j];
            interval_lengths_e_host[i] = interval_lengths_host[j];
            j++;
        }
    }


    checkCudaErrors(cudaMalloc(&depth_device, depth_shape.size<TENSORTYPE>()));
    checkCudaErrors(cudaMalloc(&feat_device, feat_shape.size<TENSORTYPE>()));
    checkCudaErrors(cudaMalloc(&ranks_depth_device, ranks_depth.num_bytes()));
    checkCudaErrors(cudaMalloc(&ranks_feat_device, ranks_feat.num_bytes()));
    checkCudaErrors(cudaMalloc(&ranks_bev_device, ranks_bev.num_bytes()));
    checkCudaErrors(cudaMalloc(&ranks_bev_mask_device, interval_starts.num_bytes()));
    checkCudaErrors(cudaMalloc(&interval_starts_device, interval_starts.num_bytes()));
    checkCudaErrors(cudaMalloc(&interval_lengths_device, interval_lengths.num_bytes()));
    checkCudaErrors(cudaMalloc(&interval_starts_e_device, interval_starts.num_bytes()));
    checkCudaErrors(cudaMalloc(&interval_lengths_e_device, interval_lengths.num_bytes()));
    checkCudaErrors(cudaMalloc(&out_gt_device, out_shape.size<TENSORTYPE>()));
    checkCudaErrors(cudaMalloc(&out_test_device, out_shape.size<TENSORTYPE>()));

    // init data
    for (int i = 0; i < depth_shape.nums(); i++)
    {
        depth_host[i] = TENSORTYPE((i % 100) * 0.01);
    }
    for (int i = 0; i < feat_shape.nums(); i++)
    {
        feat_host[i] = TENSORTYPE((100 - (i % 100)) * 0.01);
    }


    checkCudaErrors(cudaMemcpy(depth_device, depth_host, depth_shape.size<TENSORTYPE>(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(feat_device, feat_host, feat_shape.size<TENSORTYPE>(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ranks_depth_device, ranks_depth_host, ranks_depth.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ranks_feat_device, ranks_feat_host, ranks_feat.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ranks_bev_device, ranks_bev_host, ranks_bev.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(ranks_bev_mask_device, ranks_bev_mask_host, interval_starts.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(interval_starts_device, interval_starts_host, interval_starts.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(interval_lengths_device, interval_lengths_host, interval_lengths.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(interval_starts_e_device, interval_starts_e_host, interval_starts.num_bytes(), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(interval_lengths_e_device, interval_lengths_e_host, interval_lengths.num_bytes(), cudaMemcpyHostToDevice));





}

template<
    typename TENSORTYPE,
    typename ACCTYPE,
    const int TC,
    const int TN,
    const int BC,
    const int BN
>
test_bev_pool_v2_fma_tnc<
    TENSORTYPE,
    ACCTYPE,
    TC,
    TN,
    BC,
    BN
>::~test_bev_pool_v2_fma_tnc() {
    free(ranks_depth_host);
    free(ranks_feat_host);
    free(ranks_bev_host);
    free(ranks_bev_mask_host);
    free(interval_starts_host);
    free(interval_lengths_host);
    free(interval_starts_e_host);
    free(interval_lengths_e_host);
    free(depth_host);
    free(feat_host);
    free(out_gt_host);
    free(out_test_host);
    checkCudaErrors(cudaFree(ranks_depth_device));
    checkCudaErrors(cudaFree(ranks_feat_device));
    checkCudaErrors(cudaFree(ranks_bev_device));
    checkCudaErrors(cudaFree(ranks_bev_mask_device));
    checkCudaErrors(cudaFree(interval_starts_device));
    checkCudaErrors(cudaFree(interval_lengths_device));
    checkCudaErrors(cudaFree(interval_starts_e_device));
    checkCudaErrors(cudaFree(interval_lengths_e_device));
    checkCudaErrors(cudaFree(depth_device));
    checkCudaErrors(cudaFree(feat_device));
    checkCudaErrors(cudaFree(out_gt_device));
    checkCudaErrors(cudaFree(out_test_device));

    // checkCudaErrors(cudaStreamDestroy(stream));
}

using test_bev_pool_v2_fma_tnc_ff_1_1_32_8 = \
    test_bev_pool_v2_fma_tnc<float, float, 1, 1, 32, 8>;

INSTANTIATE_TEST_SUITE_P(
    design,
    test_bev_pool_v2_fma_tnc_ff_1_1_32_8,
    testing::Combine(
        testing::Values(
            INPUT_FILE_TYPE::npy
        ),
        testing::Values(
            "/home/jovyan/lightnet/tests/test_kernels/test_bev_pool_v2_inputs/npy"
            // "/home/dongwei/Workspace/lightnet/tests/test_kernels/test_bev_pool_v2_inputs/npy"
        ),
        testing::Values(
            DIM(7, 120, 64, 120)
        ),
        testing::Values(
            DIM(7, 64, 120, 128)
        ),
        testing::Values(
            DIM(1, 192, 256, 128)
        )
    )
);

TEST_P(test_bev_pool_v2_fma_tnc_ff_1_1_32_8, 0){
    constexpr int TC = 2;
    constexpr int TN = 1;
    constexpr int BC = 64;
    constexpr int BN = 8;
    using TENSORTYPE = float;
    using ACCTYPE = float;




    GPU_TICK("bev_pool_v2_b256", cudaStreamDefault);
    bev_pool_v2_b256(
        c,
        n_intervals,
        depth_device,
        feat_device,
        ranks_depth_device,
        ranks_feat_device,
        ranks_bev_device,
        interval_starts_device,
        interval_lengths_device,
        out_gt_device
    );
    GPU_TOCK("bev_pool_v2_b256", cudaStreamDefault);
    std::cout << "bev_pool_v2_b256 cost: " << GPU_TICKTOCKS["bev_pool_v2_b256"].interval << " ms." << std::endl;
    checkCudaErrors(cudaMemcpy(out_gt_host, out_gt_device, out_shape.size<float>(), cudaMemcpyDeviceToHost));


    // GPU_TICK("bev_pool_v2_mz", cudaStreamDefault);
    // // for (int i = 0; i < 100; i++) {
    //     bev_pool_v2_mz(
    //         interval_starts_host[n_intervals - 1] + interval_lengths_host[n_intervals - 1],
    //         c,
    //         n_intervals,
    //         depth_device,
    //         feat_device,
    //         ranks_depth_device,
    //         ranks_feat_device,
    //         ranks_bev_device,
    //         interval_starts_device,
    //         interval_lengths_device,
    //         out_test_device
    //     );
    // // }
    // GPU_TOCK("bev_pool_v2_mz", cudaStreamDefault);
    // std::cout << "bev_pool_v2_mz cost: " << GPU_TICKTOCKS["bev_pool_v2_mz"].interval << " ms." << std::endl;
    // checkCudaErrors(cudaMemcpy(out_test_host, out_test_device, out_shape.size<float>(), cudaMemcpyDeviceToHost));



    GPU_TICK("bev_pool_v2_morethread", cudaStreamDefault);
    // for (int i = 0; i < 100; i++) {
        bev_pool_v2_morethread(
            interval_starts_host[n_intervals - 1] + interval_lengths_host[n_intervals - 1],
            c,
            n_intervals,
            depth_device,
            feat_device,
            ranks_depth_device,
            ranks_feat_device,
            ranks_bev_device,
            interval_starts_device,
            interval_lengths_device,
            out_test_device
        );
    // }
    GPU_TOCK("bev_pool_v2_morethread", cudaStreamDefault);
    std::cout << "bev_pool_v2_morethread cost: " << GPU_TICKTOCKS["bev_pool_v2_morethread"].interval << " ms." << std::endl;
    checkCudaErrors(cudaMemcpy(out_test_host, out_test_device, out_shape.size<float>(), cudaMemcpyDeviceToHost));






    // dim3 gridSize(
    //     (c + TC * BC - 1)/(TC * BC),
    //     (n_intervals + TN * BN - 1)/(TN * BN)
    // );
    // dim3 blockSize(BC, BN);
    // GPU_TICK("bev_pool_kernel", cudaStreamDefault);
    // bev_pool_kernel<TENSORTYPE, ACCTYPE, TC, TN><<<gridSize, blockSize>>>(
    //     c, n_intervals,
    //     const_cast<const TENSORTYPE*>(depth_device),
    //     const_cast<const TENSORTYPE*>(feat_device),
    //     const_cast<const int*>(ranks_depth_device),
    //     const_cast<const int*>(ranks_feat_device),
    //     const_cast<const int*>(interval_starts_e_device),
    //     const_cast<const int*>(interval_lengths_e_device),
    //     out_test_device
    // );
    // GPU_TOCK("bev_pool_kernel", cudaStreamDefault);
    // std::cout << "bev_pool_kernel cost: " << GPU_TICKTOCKS["bev_pool_kernel"].interval << " ms." << std::endl;
    // checkCudaErrors(cudaMemcpy(out_test_host, out_test_device, out_shape.size<float>(), cudaMemcpyDeviceToHost));


    for (int h = 0; h < out_shape.y; h++) {  // 192
        for (int w = 0; w < out_shape.z; w++) {  // 256
            for (int c = 0; c < out_shape.w; c++) {  // 128
                float out_gt_host_i = out_gt_host[
                    h * out_shape.z * out_shape.w + \
                    w * out_shape.w + \
                    c
                ] + 1;
                float out_test_host_i = out_test_host[
                    h * out_shape.z * out_shape.w + \
                    w * out_shape.w + \
                    c
                ] + 1;

                std::cout << "check pos[" << h << ", " << w << ", " << c << "]" << std::endl;
                std::cout << "out_gt_host_i: " <<  out_gt_host_i << "\nout_test_host_i: " <<  out_test_host_i << std::endl;

                ASSERT_LE(
                    abs(
                        (out_gt_host_i - out_test_host_i) / out_gt_host_i
                    ),
                    0.00001
                )   << "\npos[" << h << ", " << w << ", " << c << "]:"
                    << "\nout_gt_host_i: " <<  out_gt_host_i
                    << "\nout_test_host_i: " <<  out_test_host_i
                    << std::endl << std::endl;
            }
            break;
        }
    }




}