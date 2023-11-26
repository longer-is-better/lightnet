#pragma once
#include <stdio.h>
#include <kernel_reduce.cuh>


/*
  https://github.com/HuangJunJie2017/BEVDet/blob/dev2.1/mmdet3d/ops/bev_pool_v2/src/bev_pool_cuda.cu
  Function: pillar pooling
*/
__global__ void kbev_pool_v2(
    int c,                                          // : number of channels 128                                          fixed
    int n_intervals,                                // : number of unique points   48043                                 variable
    const float *__restrict__ depth,                // : input depth, FloatTensor[b,n,d,h,w] [1, 7, 120, 64, 120]        fixed
    const float *__restrict__ feat,                 // : input feat, FloatTensor[b,n,h,w,c] [1, 7, 64, 120, 128]         fixed
    const int *__restrict__ ranks_depth,            // : input index of depth, IntTensor[n] [2500543]                    variable
    const int *__restrict__ ranks_feat,             // : input index of feat, IntTensor[n] [2500543] = interval_starts[n_intervals - 1] + interval_lengths[n_intervals - 1]                     variable
    const int *__restrict__ ranks_bev,              // : output index, IntTensor[n] [2500543]                            variable
    const int *__restrict__ interval_starts,        // : starting position for pooled point, IntTensor[n_intervals]      variable
    const int *__restrict__ interval_lengths,       // : how many points in each pooled point, IntTensor[n_intervals]    variable
    float* __restrict__ out                         // : output features, FloatTensor[b, d, h, w, c] [1, 1, 192, 256, 128]  fixed
                                                    //                                      h * w = 192 * 256 = 49152
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= n_intervals) return;
    int interval_start = interval_starts[index];
    int interval_length = interval_lengths[index];
    float psum = 0;
    const float* cur_depth;
    const float* cur_feat;
    for(int i = 0; i < interval_length; i++){
      cur_depth = depth + ranks_depth[interval_start+i];
      cur_feat = feat + ranks_feat[interval_start+i] * c + cur_c;
      psum += *cur_feat * *cur_depth;
    }

    const int* cur_rank = ranks_bev + interval_start;
    float* cur_out = out + *cur_rank * c + cur_c;
    *cur_out = psum;
}

void bev_pool_v2_b256(
  int c,
  int n_intervals,
  const float* depth,
  const float* feat,
  const int* ranks_depth,
  const int* ranks_feat,
  const int* ranks_bev,
  const int* interval_starts,
  const int* interval_lengths,
  float* out
) {
  kbev_pool_v2<<<(int)ceil(((double)n_intervals * c / 256)), 256>>>(
    c,
    n_intervals,
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    interval_lengths,
    out
  );
}

void bev_pool_v2_b1024(
  int c,
  int n_intervals,
  const float* depth,
  const float* feat,
  const int* ranks_depth,
  const int* ranks_feat,
  const int* ranks_bev,
  const int* interval_starts,
  const int* interval_lengths,
  float* out
) {
  kbev_pool_v2<<<(int)ceil(((double)n_intervals * c / 1024)), 1024>>>(
    c,
    n_intervals,
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    interval_lengths,
    out
  );
}


// https://gitlabee.chehejia.com/lpai/algorithm/bev_pool
template<typename TensorType, typename AccType, const int TC, const int TN>
__global__ void kbev_pool_fma_tnc(
    int c, int n_intervals,
    const TensorType *__restrict__ depth,
    const TensorType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ ranks_bev,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    TensorType *__restrict__ out
) {

  int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
  for (int tn = 0; tn < TN; tn++) {
    AccType psum[TC];
    int n_idx = tn_idx * TN + tn;
    if (n_idx >= n_intervals) return;

    int interval_start = interval_starts[n_idx];
    int interval_length = interval_lengths[n_idx];

    for (int tc = 0; tc < TC; tc++) {
      psum[tc] = 0;
    }

    for (int i = 0; i < interval_length; i++) {
      TensorType d = depth[ranks_depth[interval_start + i]];
#pragma unroll
      for (int tc = 0; tc < TC; tc++) { 
        int c_idx = tc_idx * TC + tc;
        if (c_idx >= c) continue;

        TensorType f = feat[ranks_feat[interval_start + i] * c + c_idx];
        if (std::is_same<TensorType, __half>::value && std::is_same<AccType, __half>::value)
          psum[tc] = __hfma((__half)d, (__half)f, (__half)psum[tc]);
        else if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(__half2float(d), __half2float(f), psum[tc]);
        else // (std::is_same<TensorType, float>::value && std::is_same<AccType, float>::value)
          psum[tc] = __fmaf_rn(d, f, psum[tc]);
      }
    }

#pragma unroll
    for (int tc = 0; tc < TC; tc++) {
      int c_idx = tc_idx * TC + tc;
      if (c_idx >= c) continue;
      if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
        out[ranks_bev[interval_start] * c + c_idx] = __float2half(psum[tc]);
      else
        out[ranks_bev[interval_start] * c + c_idx] = psum[tc];
    }

  }
}

template <typename TensorType, typename AccType, const int TC, const int TN>
__global__ void bev_pool_kernel(
    int c, int n_intervals,
    const TensorType *__restrict__ depth,
    const TensorType *__restrict__ feat,
    const int *__restrict__ ranks_depth,
    const int *__restrict__ ranks_feat,
    const int *__restrict__ interval_starts,
    const int *__restrict__ interval_lengths,
    TensorType *__restrict__ out)
{

    int tc_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tn_idx = blockIdx.y * blockDim.y + threadIdx.y;

#pragma unroll
    for (int tn = 0; tn < TN; tn++)
    {
        AccType psum[TC];
        int n_idx = tn_idx * TN + tn;
        if (n_idx >= n_intervals)
            return;

        int interval_start = __ldg(&interval_starts[n_idx]);
        int interval_length = __ldg(&interval_lengths[n_idx]);

        if (interval_start == -1)
            return;

        for (int tc = 0; tc < TC; tc++)
        {
            psum[tc] = 0;
        }

        for (int i = 0; i < interval_length; i++)
        {
            TensorType d = __ldg(&depth[ranks_depth[interval_start + i]]);
#pragma unroll
            for (int tc = 0; tc < TC; tc++)
            {
                int c_idx = tc_idx * TC + tc;
                if (c_idx >= c)
                    continue;

                TensorType f = __ldg(&feat[ranks_feat[interval_start + i] * c + c_idx]);
                if (std::is_same<TensorType, __half>::value && std::is_same<AccType, __half>::value)
                    psum[tc] = __hfma((__half)d, (__half)f, (__half)psum[tc]);
                else if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
                    psum[tc] = __fmaf_rn(__half2float(d), __half2float(f), psum[tc]);
                else // (std::is_same<TensorType, float>::value && std::is_same<AccType, float>::value)
                    psum[tc] = __fmaf_rn(d, f, psum[tc]);
            }
        }

#pragma unroll
        for (int tc = 0; tc < TC; tc++)
        {
            int c_idx = tc_idx * TC + tc;
            if (c_idx >= c)
                continue;
            int tid = n_idx * c + c_idx;
            if (std::is_same<TensorType, __half>::value && std::is_same<AccType, float>::value)
                out[tid] = __float2half(psum[tc]);
            else
                out[tid] = psum[tc];
        }
    }
}













__global__ void kbev_pool_v2_mz(
    int N,                                          // : [2500543] = interval_starts[n_intervals - 1] + interval_lengths[n_intervals - 1]
    int c,                                          // : number of channels 128                                          fixed
    int n_intervals,                                // : number of unique points   48043                                 variable
    const float *__restrict__ depth,                // : input depth, FloatTensor[b,n,d,h,w] [1, 7, 120, 64, 120]        fixed
    const float *__restrict__ feat,                 // : input feat, FloatTensor[b,n,h,w,c] [1, 7, 64, 120, 128]         fixed
    const int *__restrict__ ranks_depth,            // : input index of depth, IntTensor[n] [2500543]                    variable
    const int *__restrict__ ranks_feat,             // : input index of feat, IntTensor[n] [2500543] = interval_starts[n_intervals - 1] + interval_lengths[n_intervals - 1]                     variable
    const int *__restrict__ ranks_bev,              // : output index, IntTensor[n] [2500543]                            variable
    const int *__restrict__ interval_starts,        // : starting position for pooled point, IntTensor[n_intervals]      variable
    float* __restrict__ out                         // : output features, FloatTensor[b, d, h, w, c] [1, 1, 192, 256, 128]  fixed
                                                    //                                      h * w = 192 * 256 = 94152
) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index = idx / c;
    int cur_c = idx % c;
    if (index >= N) return;

    float cur_df = depth[ranks_depth[index]] * feat[ranks_feat[index] * c + cur_c];
    // atomicAdd(
    //     out + ranks_bev[index] * c + cur_c,
    //     cur_df
    // );
}

void bev_pool_v2_mz(
    int N,
    int c,
    int n_intervals,
    const float* depth,
    const float* feat,
    const int* ranks_depth,
    const int* ranks_feat,
    const int* ranks_bev,
    const int* interval_starts,
    const int* interval_lengths,
    float* out
) {
  kbev_pool_v2_mz<<<dim3((N * 128 + 1023) / 1024), 1024>>>(
    N,
    c,
    n_intervals,
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    out
  );
}







__global__ void kbev_pool_v2_morethread(
    int N,                                          // : [2500543] = interval_starts[n_intervals - 1] + interval_lengths[n_intervals - 1]
    int c,                                          // : number of channels 128                                          fixed
    int n_intervals,                                // : number of unique points   48043                                 variable
    const float *__restrict__ depth,                // : input depth, FloatTensor[b,n,d,h,w] [1, 7, 120, 64, 120]        fixed
    const float *__restrict__ feat,                 // : input feat, FloatTensor[b,n,h,w,c] [1, 7, 64, 120, 128]         fixed
    const int *__restrict__ ranks_depth,            // : input index of depth, IntTensor[n] [2500543]                    variable
    const int *__restrict__ ranks_feat,             // : input index of feat, IntTensor[n] [2500543] = interval_starts[n_intervals - 1] + interval_lengths[n_intervals - 1]                     variable
    const int *__restrict__ ranks_bev,              // : output index, IntTensor[n] [2500543]                            variable
    const int *__restrict__ interval_starts,        // : starting position for pooled point, IntTensor[n_intervals]      variable
    float* __restrict__ out                         // : output features, FloatTensor[b, d, h, w, c] [1, 1, 192, 256, 128]  fixed
                                                    //                                      h * w = 192 * 256 = 94152
) {

    unsigned n_index = blockDim.x * blockIdx.x + threadIdx.x,
             thread_lane = threadIdx.x & 0x1f,
             warp_lane = threadIdx.x & 0xffffffe0,
             cur_c = blockIdx.y,
             interval_n = 0;
    if (n_index >= N) return;
    interval_n = ranks_bev[n_index];  // opt
    float cur_depth, cur_feat, sum_df, down_sum_df;
    cur_depth = depth[ranks_depth[n_index]];
    cur_feat = feat[ranks_feat[n_index] * c + cur_c];  // opt
    sum_df = cur_depth * cur_feat;


    // write to ans!!!!!!

    // 1.
    // atomicAdd(out + interval_n * c + cur_c, sum_df);  // long lenght conflict


    // 2.
    for (unsigned int step = 1; step <=16; step = step << 1) {
        down_sum_df = __shfl_down_sync(0xffffffff, sum_df, step);
        if (interval_n == __shfl_down_sync(0xffffffff, interval_n, step) && thread_lane + step < warpSize)
            sum_df += down_sum_df;
    }

    // 2.1
    if (interval_n != __shfl_up_sync(0xffffffff, interval_n, 1) || thread_lane == 0)
        atomicAdd(out + interval_n * c + cur_c, sum_df);  // long lenght conflict
        // out[interval_n * c + cur_c] += sum_df;


    // 2.2
    // bool tail_head = \
    //     interval_n == __reduce_max_sync(0xffffffff, interval_n) && \
    //     (
    //         thread_lane == 0 || \
    //         __shfl_up_sync(0xffffffff, interval_n, 1) < interval_n
    //     );
    // __shared__ float s_head_dfs[16];  // hardcode 16 = (blockDim.x 512 / warpsize 32
    // if (tail_head) s_head_dfs[warp_lane] = sum_df;
    // __shared__ int s_interval_n[16];  // hardcode 16 = (blockDim.x 512 / warpsize 32)
    // s_interval_n[warp_lane] = interval_n;
    // __syncthreads();
    // for (int step = 1; step <= 8; step = step << 1) {
    //     if (tail_head && warp_lane + 1 < 16 && interval_n == s_interval_n[warp_lane + 1]) {  // hardcode 16 = (blockDim.x 512 / warpsize 32)
            
    //         // s_head_dfs[2 * warp_lane + 1] += s_head_dfs[2 * (warp_lane + 1)];
    //         // s_head_dfs[2 * (warp_lane + 1)] = 0;
    //         // if (thread_lane == 0) {sum_df  = s_head_dfs[2 * warp_lane];}
    //         // if (tail_head == 0) sum_df = s_head_dfs[2 * warp_lane + 1];
    //         // if (thread_lane == 0) s_head_dfs[2 * warp_lane] = sum_df;
    //         // if (tail_head == 0) s_head_dfs[2 * warp_lane + 1] = sum_df;
    //     }
    // }


    // 4.
    // bool tail_head = \
    //     interval_n == __reduce_max_sync(0xffffffff, interval_n) && \
    //     (
    //         thread_lane == 0 || \
    //         __shfl_up_sync(0xffffffff, interval_n, 1) < interval_n
    //     );

    // __shared__ float block_partial_res[1024];
    // if (interval_n != __shfl_up_sync(0xffffffff, interval_n, 1) || thread_lane == 0){
    //     block_partial_res[threadIdx.x] = sum_df;
    // }
    // __shared__ unsigned block_interval_n[1024];
    // block_interval_n[threadIdx.x] = interval_n;

    // // __syncthreads();
    // float res_down;
    // bool sf;
    // for (unsigned int step = 1; step <=16; step = step << 1) {
    //     sf = tail_head && interval_n == block_interval_n[(warp_lane + step) * warpSize];
    //     if (sf) res_down = block_partial_res[(warp_lane + step) * warpSize];
    //     // __syncthreads();
    //     if (sf) block_partial_res[threadIdx.x] += res_down;
    // }

    // if (
    //     thread_lane == 0 && interval_n != block_interval_n[threadIdx.x - 1] ||
    //     interval_n != __shfl_up_sync(0xffffffff, interval_n, 1) ||
    //     warp_lane == 0
    // ) {
    //     atomicAdd(out + interval_n * c + cur_c, block_partial_res[threadIdx.x]);
    // }
}

void bev_pool_v2_morethread(
    int N,
    int c,
    int n_intervals,
    const float* depth,
    const float* feat,
    const int* ranks_depth,
    const int* ranks_feat,
    const int* ranks_bev,
    const int* interval_starts,
    const int* interval_lengths,
    float* out
) {
  kbev_pool_v2_morethread<<<dim3((N + 127) / 128, 128), 128>>>(
  // kbev_pool_v2_morethread<<<dim3((N + 255) / 256, 128), 256>>>(
  // kbev_pool_v2_morethread<<<dim3((N + 511) / 512, 128), 512>>>(
  // kbev_pool_v2_morethread<<<dim3((N + 1023) / 1024, 128), 1024>>>(
    N,
    c,
    n_intervals,
    depth,
    feat,
    ranks_depth,
    ranks_feat,
    ranks_bev,
    interval_starts,
    out
  );
}