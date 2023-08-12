#pragma once

#include "compute_graph.cuh"
#include "kernel_reduce.cuh"


class L1LossGraph: public ComputeGraph
{
public:
    L1LossGraph(REDUCE_OP op = REDUCE_OP::AVG);
    ~L1LossGraph();
};
