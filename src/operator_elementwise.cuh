#pragma once

#include "operator.cuh"
#include "kernel_elementwise.cuh"


class ElementWise: public Operator
{
public:
    ELE_OP _ele_op;


    ElementWise(){};
    ElementWise(ELE_OP op, bool end_of_graph);
    ElementWise(Tensor *A, Tensor *B, ELE_OP op);
    ~ElementWise(){};

    virtual std::string type_str();
    virtual ElementWise* copy();
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    ElementWise(const ElementWise &elementwise) = delete;
    ElementWise(ElementWise &&elementwise) = delete;
    ElementWise &operator=(const ElementWise &elementwise) = delete;
    ElementWise &operator=(ElementWise &&elementwise) = delete;
};
