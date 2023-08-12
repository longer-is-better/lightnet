#pragma once

enum class ELE_OP {
    ADD = 0,
    SUB,
    MULTIPLY,
    DIVIDE
};

__global__ void kelementwise(size_t N, float *I, float alpha, float *operand, float *O, ELE_OP op);
__global__ void kelementwise_inplace(size_t N, float *IO, float alpha, float *operand, ELE_OP op);