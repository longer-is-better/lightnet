#pragma once

#include "operator.cuh"
#include "kernel_map.cuh"


class Map: public Operator
{
public:
    MAP_OP _map_op;
    float _operand;

    Map(){};
    Map(Tensor*A, MAP_OP op, float operand = 0.f);
    ~Map(){};

    virtual Operator* copy();
    virtual void infer_shape();
    virtual void forward();
    virtual void backward();

    Map(const Map &Map) = delete;
    Map(Map &&Map) = delete;
    Map &operator=(const Map &Map) = delete;
    Map &operator=(Map &&Map) = delete;
};
