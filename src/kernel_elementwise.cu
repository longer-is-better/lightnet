#pragma once
#include "kernel_elementwise.cuh"

__global__ void kelementwise(size_t N, float *I, float alpha, float *operand, float *O, ELE_OP op) {
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


__global__ void kelementwise_inplace(size_t N, float *IO, float alpha, float *operand, ELE_OP op) {
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