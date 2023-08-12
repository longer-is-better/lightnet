#pragma once

#include "operator.cuh"
#include "kernel_reduce.cuh"


class Reduce: public Operator
{
public:
    REDUCE_OP _reduce_op;


    Reduce(){};
    Reduce(Tensor*A, REDUCE_OP op);
    ~Reduce(){};

    virtual Operator* copy();
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    Reduce(const Reduce &Reduce) = delete;
    Reduce(Reduce &&Reduce) = delete;
    Reduce &operator=(const Reduce &Reduce) = delete;
    Reduce &operator=(Reduce &&Reduce) = delete;
};
