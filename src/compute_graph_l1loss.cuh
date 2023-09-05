#pragma once

#include "compute_graph.cuh"
#include "kernel_reduce.cuh"


class L1LossGraph: public ComputeGraph
{
public:
    L1LossGraph();
    ~L1LossGraph();
};
