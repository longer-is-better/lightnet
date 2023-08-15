#pragma once

#include "operator.cuh"

class MatMul: public Operator {
public:


    MatMul(){};
    MatMul(Tensor*A, Tensor*B);
    ~MatMul(){};

    virtual std::string type_str();
    virtual MatMul* copy();
    virtual void set_cudastream(cudaStream_t cudastream);
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    MatMul(const MatMul &mm) = delete;
    MatMul(MatMul &&mm) = delete;
    MatMul &operator=(const MatMul &mm) = delete;
    MatMul &operator=(MatMul &&mm) = delete;
};