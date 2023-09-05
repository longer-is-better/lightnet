#pragma once
#include <stdio.h>
#include <iostream>
#include <cuda/pipeline>
#include <cooperative_groups/memcpy_async.h>

// enum class REDUCE_OP {
//     SUM = 0
// };

// std::ostream& operator<<(std::ostream& os, const REDUCE_OP &op);

template <typename T>
__device__ void warpReduceSum(volatile T* shmem_ptr, int t) {
    // shmem_ptr[t] += shmem_ptr[t + 32];
    shmem_ptr[t] += shmem_ptr[t + 16];
    shmem_ptr[t] += shmem_ptr[t + 8];
    shmem_ptr[t] += shmem_ptr[t + 4];
    shmem_ptr[t] += shmem_ptr[t + 2];
    shmem_ptr[t] += shmem_ptr[t + 1];
}

template <typename T>
__global__ void kreduce_sum(size_t total_n, size_t current_n, T *I, T *O) {
    extern __shared__ T partial[];

    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    T front = 0, back = 0;
    if (i < current_n) front = I[i];
    if (i + blockDim.x < current_n) back = I[i + blockDim.x];
    partial[threadIdx.x] = front + back;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 16; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 16) {
        warpReduceSum(partial, threadIdx.x);
    }

    if (threadIdx.x == 0) {
        O[blockIdx.x] = partial[0];
    }
}

template <typename T>
__global__ void kreduce_sum_dbbf(size_t total_n, T *I, T *O) {
    assert(total_n % gridDim.x == 0); // Assume input size fits batch_sz * grid_size

    constexpr size_t stages_count = 2; // Pipeline with two stages
    // Two batches must fit in shared memory:
    extern __shared__ T double_partial[];
    double_partial[2 * blockDim.x] = 0.f;
    size_t offset[stages_count] = { 0, blockDim.x }; // Offsets to each batch

    auto block = cooperative_groups::this_thread_block();
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
    auto pipeline = cuda::make_pipeline(block, &shared_state);
    auto block_batch = [&](size_t batch) -> int {
        return gridDim.x * blockDim.x * batch + blockIdx.x * blockDim.x;
    };

    size_t batch = 0;
    size_t stage_index = 0;

    if (block_batch(batch) < total_n) {
        pipeline.producer_acquire();
        cuda::memcpy_async(
            block,
            double_partial + offset[stage_index],
            I + block_batch(batch),
            sizeof(T) * blockDim.x,
            pipeline
        );
        pipeline.producer_commit();
    }

    while (batch * gridDim.x * blockDim.x < total_n) {
        // preload
        size_t next_batch = batch + 1;
        size_t next_stage_index = next_batch % stages_count;
        if (block_batch(next_batch) < total_n) {
            pipeline.producer_acquire();
            cuda::memcpy_async(
                block,
                double_partial + offset[next_stage_index],
                I + block_batch(next_batch),
                sizeof(T) * blockDim.x,
                pipeline
            );
            pipeline.producer_commit();
        }

        // reduce current batch
        pipeline.consumer_wait();
        T *partial = double_partial + offset[stage_index];
        for (int s = blockDim.x / 2; s > 16; s >>= 1) {
            if (threadIdx.x < s) {
                partial[threadIdx.x] += partial[threadIdx.x + s];
            }
            __syncthreads();
        }

        if (threadIdx.x < 32) {
            if (blockDim.x > 16) partial[threadIdx.x] += partial[threadIdx.x + 16];
            if (blockDim.x > 8) partial[threadIdx.x] += partial[threadIdx.x + 8];
            if (blockDim.x > 4) partial[threadIdx.x] += partial[threadIdx.x + 4];
            if (blockDim.x > 2) partial[threadIdx.x] += partial[threadIdx.x + 2];
            if (blockDim.x > 1) partial[threadIdx.x] += partial[threadIdx.x + 1];
        }

        if (threadIdx.x == 0) {
            double_partial[2 * blockDim.x] += partial[0];
        }
        pipeline.consumer_release();
        batch = next_batch;
        stage_index = next_stage_index;
    }
    if (threadIdx.x == 0) {
        atomicAdd(O, double_partial[2 * blockDim.x]);
    }
}


template<class DATA_TYPE>
__inline__ __device__
DATA_TYPE warp_reduce_sum(DATA_TYPE val) {
    for (unsigned int step = warpSize/2; step > 0; step = step >> 1) {
        val += __shfl_down_sync(0xffffffff, val, step);
    }
    return val;
}

template<class DATA_TYPE>
__inline__ __device__
DATA_TYPE block_reduce_sum(DATA_TYPE val) {
    val = warp_reduce_sum(val);
    __shared__ DATA_TYPE warp_ans[32];
    int wid = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    if (lane == 0) warp_ans[wid] = val;
    __syncthreads();
    if (wid == 0) {
        val = (lane < (blockDim.x + warpSize - 1) / warpSize) ? warp_ans[lane] : 0;
        val = warp_reduce_sum(val);
    }
    return val;
}

template<class DATA_TYPE>
__global__ void kreduce_sum_sfl(DATA_TYPE *I, DATA_TYPE *O, size_t N) {
    DATA_TYPE val = 0;
    for (
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        i < N;
        i += gridDim.x * blockDim.x
    ) {
        val += (i < N) ? I[i] : 0;
    }
    val = block_reduce_sum(val);
    if (threadIdx.x == 0) O[blockIdx.x] = val;
}


template<class DATA_TYPE>
__global__ void kreduce_sum_warpsfl_atom(DATA_TYPE *I, DATA_TYPE *O, size_t N) {
    DATA_TYPE val = 0;
    for (
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        i < N;
        i += gridDim.x * blockDim.x
    ) {
        val += (i < N) ? I[i] : 0;
    }
    val = warp_reduce_sum(val);
    if ((threadIdx.x & (warpSize - 1)) == 0)
        atomicAdd(O, val);
}


template<class DATA_TYPE>
__global__ void kreduce_sum_blksfl_atom(DATA_TYPE *I, DATA_TYPE *O, size_t N) {
    DATA_TYPE val = 0;
    for (
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        i < N;
        i += gridDim.x * blockDim.x
    ) {
        val += (i < N) ? I[i] : 0;
    }
    val = block_reduce_sum(val);
    if (threadIdx.x == 0) atomicAdd(O, val);
}





template<class DATA_TYPE, class DATA_TYPE4>
__global__ void kreduce_sum_vec4_blksfl_atom(DATA_TYPE *I, DATA_TYPE *O, size_t N) {
    DATA_TYPE val = 0;
    DATA_TYPE4 frag;
    for (
        size_t i = blockIdx.x * blockDim.x + threadIdx.x;
        i < N / 4;
        i += gridDim.x * blockDim.x
    ) {
        frag = reinterpret_cast<DATA_TYPE4*>(I)[i];
        val += frag.w + frag.x + frag.y + frag.z;
    }
    // in only one thread, process final elements (if there are any)

    if (threadIdx.x < N % 4 && blockIdx.x == 0) {
        val += I[N - threadIdx.x - 1];
    }

    val = block_reduce_sum(val);
    if (threadIdx.x == 0)
        atomicAdd(O, val);
}