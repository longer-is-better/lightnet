#pragma once

#include "operator.cuh"
#include "kernel_reduce.cuh"


class Reduce: public Operator
{
public:

    Reduce(){};
    Reduce(bool end_of_graph);
    Reduce(Tensor*A);
    ~Reduce(){};

    virtual std::string type_str();
    virtual Reduce* copy();
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    Reduce(const Reduce &Reduce) = delete;
    Reduce(Reduce &&Reduce) = delete;
    Reduce &operator=(const Reduce &Reduce) = delete;
    Reduce &operator=(Reduce &&Reduce) = delete;
};
