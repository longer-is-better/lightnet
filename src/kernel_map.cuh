#pragma once

enum class MAP_OP {
    ADD = 0,
    MULTIPLY,
    POW,
    LOG,
    ABS,
    SIGN
};

std::ostream& operator<<(std::ostream& os, MAP_OP op);

template <typename T>
__global__ void kmap(size_t N, T *I, T *O, MAP_OP op, T operand) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op) {
            case MAP_OP::ADD:
                O[tid] = I[tid] + operand;
                break;
            case MAP_OP::MULTIPLY:
                O[tid] = I[tid] * operand;
                break;
            case MAP_OP::POW:
                O[tid] = powf(I[tid], operand);
                break;
            case MAP_OP::LOG:
                O[tid] = logf(I[tid]);
                break;
            case MAP_OP::ABS:
                O[tid] = fabsf(I[tid]);
                break;
            case MAP_OP::SIGN:
                O[tid] = I[tid] > 0 ? 1 : -1;
                break;

            default:
                break;
        }
    }
}

template <typename T>
__global__ void kmap_inplace(size_t N, T *I, MAP_OP op, T operand) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op) {
            case MAP_OP::ADD:
                I[tid] += operand;
                break;
            case MAP_OP::MULTIPLY:
                I[tid] *= operand;
                break;
            case MAP_OP::POW:
                I[tid] = powf(I[tid], operand);
                break;
            case MAP_OP::LOG:
                I[tid] = logf(I[tid]);
                break;
            case MAP_OP::ABS:
                I[tid] = fabsf(I[tid]);
                break;
            case MAP_OP::SIGN:
                I[tid] = I[tid] > 0 ? 1 : -1;
                break;
            
            
            default:
                break;
        }
    }
}