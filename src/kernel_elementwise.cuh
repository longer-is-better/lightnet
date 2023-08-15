#pragma once
#include <iostream>

enum class ELE_OP {
    ADD = 0,
    SUB,
    MULTIPLY,
    DIVIDE
};


std::ostream& operator<<(std::ostream& os, ELE_OP op);


template <typename T>
__global__ void kelementwise(size_t N, T *I, T alpha, T *operand, T *O, ELE_OP op) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op) {
            case ELE_OP::ADD:
                O[tid] = I[tid] + (alpha * operand[tid]);
                break;
            case ELE_OP::SUB:
                O[tid] = I[tid] - (alpha * operand[tid]);
                break;
            case ELE_OP::MULTIPLY:
                O[tid] = I[tid] * (alpha * operand[tid]);
                break;
            case ELE_OP::DIVIDE:
                O[tid] = I[tid] / (alpha * operand[tid]);
                break;

            default:
                break;
        }
    }
}


template <typename T>
__global__ void kelementwise_inplace(size_t N, T *IO, T alpha, T *operand, ELE_OP op) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        switch (op) {
            case ELE_OP::ADD:
                IO[tid] += (alpha * operand[tid]);
                break;
            case ELE_OP::SUB:
                IO[tid] -= (alpha * operand[tid]);
                break;
            case ELE_OP::MULTIPLY:
                IO[tid] *= (alpha * operand[tid]);
                break;
            case ELE_OP::DIVIDE:
                IO[tid] /= (alpha * operand[tid]);
                break;

            default:
                break;
        }
    }
}