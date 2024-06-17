

#define THREADS_PER_BLOCK 1024
#define blockSize 1024

template<typename T>
void reduceCPU(T* idata, T* odata, int N) {
    T sum = 0;
    // #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        sum += idata[i];
    }
    *odata = sum;
}

template<typename T>
__global__ void reduceGpuBL(T* g_idata, T* g_odata, int N) {
    __shared__ T s_data[THREADS_PER_BLOCK];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //在原始数组中的索引号

    s_data[tid] = g_idata[idx];
    __syncthreads();

    // do reduction in shared mem
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[tid];
        // printf("g_odata[%d] = %.2f\n", blockIdx.x, g_odata[blockIdx.x]);
    }
}

template<typename T>
__global__ void reduceGpuWD(T* g_idata, T* g_odata, int N) {
    /*
        mark: 两阶段存在一个问题就是要满足，grid的size <= block的size，不然就要增加计算阶段才能算完
        解决 warp divergence 问题 线程束分散 if (2*s*tid < blockDim.x)
        对于一个block而言，所有thread都是执行同一条命令，如果存在if-else这样的分支情况的话，thread会执行所有的分支。
        只是不满足条件的分支，所产生的结果不会被记录下来。每一轮迭代都会产生两个分支，严重影响效率

        解决方式: 尽可能地让所有线程走到同一个分支里面

        假定block中存在256个thread，即拥有256/32=8个warp。
        当进行第1次迭代时，0-3号warp的index<blockDim.x，4-7号warp的index>=blockDim.x。对于每个warp而言，都只是进入到一个分支内，不存在warp divergence的情况。
        当进行第2次迭代时，0、1号两个warp进入计算分支。
        当进行第3次迭代时，只有0号warp进入计算分支。
        当进行第4次迭代时，只有0号warp的前16个线程进入分支。此时开始产生warp divergence。
        通过这种方式，消除了前3次迭代的warp。
    */
    __shared__ T s_data[THREADS_PER_BLOCK];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //在原始数组中的索引号

    s_data[tid] = g_idata[idx];
    __syncthreads();

    // do reduction in shared mem
    for (int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            s_data[index] += s_data[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[tid];
    }
}

template<typename T>
__global__ void reduceGpuBC(T* g_idata, T* g_odata, int N) {
    /*
        解决bank冲突 反循环 int s = blockDim.x/2
        
        聚焦for循环的0号warp。
        第一次迭代，0号线程需要去load shared memory中的0号地址以及1号地址的数，然后写回到0号地址。
        此时，这个warp中的16号线程，需要去load shared memory中的32号地址和33号地址。
        可以发现，0号地址跟32号地址产生了两路bank冲突。
        第二次迭代，4路
        第三次迭代，8路
        ...

        解决方案：Sequential Addressing
    */
    __shared__ T s_data[THREADS_PER_BLOCK];

    // each thread loads one element from global to shared mem
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;    //在原始数组中的索引号

    s_data[tid] = g_idata[idx];
    __syncthreads();

    // do reduction in shared mem
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[tid];
    }
}

template<typename T>
__global__ void reduceGpuID(T* g_idata, T* g_odata, int N) {
    /*
        解决 idle 线程
    */
    __shared__ T s_data[THREADS_PER_BLOCK];

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    s_data[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = s_data[tid];
    }
}

__device__ void warpReduce(volatile float* cache,int tid){
    cache[tid]+=cache[tid+32];
    cache[tid]+=cache[tid+16];
    cache[tid]+=cache[tid+8];
    cache[tid]+=cache[tid+4];
    cache[tid]+=cache[tid+2];
    cache[tid]+=cache[tid+1];
}

template<typename T>
__global__ void reduceGpuUR(T* g_idata, T* g_odata, int N) {
    /*
        现有问题
        当进行到最后几轮迭代时，此时的block中只有warp0在干活时，线程还在进行同步操作。这一条语句造成了极大浪费。

        解决方式
        由于一个warp中的32个线程，其实是在一个SIMD单元上，这32个线程每次都是执行同一条命令，保持了同步状态。
        当s=32时，即只有一个SIMD单元在工作时，完全可以将__syncthreads()这条同步代码去掉。所以我们将最后一维展开以减少同步。伪代码如

    */
    __shared__ float s_data[THREADS_PER_BLOCK];
    
    //ench thread loads one element from global memory to shared mem
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;//当前的block和thread的索引，每个block有两个warp
    unsigned int tid = threadIdx.x;
    s_data[tid] = g_idata[i] + g_idata[i+blockDim.x];
    __syncthreads();
 
    //do reduction in shared mem
    for(unsigned int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s){
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    
    if(tid < 32) warpReduce(s_data, tid);
    //write result for this block to global mem
    if(tid == 0) g_odata[blockIdx.x] = s_data[tid];
}


// template <unsigned int blockSize>
__device__ void warpReduceCUR(volatile float* cache, unsigned int tid){
    if (blockSize >= 64) cache[tid]+=cache[tid+32];
    if (blockSize >= 32) cache[tid]+=cache[tid+16];
    if (blockSize >= 16) cache[tid]+=cache[tid+8];
    if (blockSize >= 8) cache[tid]+=cache[tid+4];
    if (blockSize >= 4) cache[tid]+=cache[tid+2];
    if (blockSize >= 2) cache[tid]+=cache[tid+1];
};
 
// template <unsigned int blockSize>
template<typename T>
__global__ void reduceGpuCUR(T* g_idata, T* g_odata, int N) {
    __shared__ float sdata[THREADS_PER_BLOCK];
 
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();
 
    // do reduction in shared mem
    if (blockSize >= 512) {
        if (tid < 256) { 
            sdata[tid] += sdata[tid + 256]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 256) {
        if (tid < 128) { 
            sdata[tid] += sdata[tid + 128]; 
        } 
        __syncthreads(); 
    }
    if (blockSize >= 128) {
        if (tid < 64) { 
            sdata[tid] += sdata[tid + 64]; 
        } 
        __syncthreads(); 
    }
    if (tid < 32) {
        warpReduceCUR(sdata, tid);
    }
 
    // write result for this block to global mem
    if (tid == 0) {
        g_odata[blockIdx.x] = sdata[0];
    }
}