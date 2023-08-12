#pragma once

#include "operator.cuh"
#include "kernel_elementwise.cuh"


class ElementWise: public Operator
{
public:
    ELE_OP _ele_op;


    ElementWise(){};
    ElementWise(Tensor *A, Tensor *B, ELE_OP op);
    ~ElementWise(){};

    virtual Operator* copy();
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    ElementWise(const ElementWise &elementwise) = delete;
    ElementWise(ElementWise &&elementwise) = delete;
    ElementWise &operator=(const ElementWise &elementwise) = delete;
    ElementWise &operator=(ElementWise &&elementwise) = delete;
};
