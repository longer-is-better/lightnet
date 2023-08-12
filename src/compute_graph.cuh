#pragma once
#include <list>

#include "tensor.cuh"
#include "operator.cuh"


class ComputeGraph
{
public:
    std::vector<Tensor*> _input_tensors;
    std::vector<Tensor*> _weight_tensors;

    std::list<Operator*> _op_seq;  // todo: cache, clear cache after changing graph

    
    std::vector<Tensor*> _output_tensors;  // not very need, tensor which has no following op is output, maybe use for cache
    // todo: may be mark some mid tensor as output tensor

    ComputeGraph();
    ComputeGraph(const ComputeGraph& computegraph) = delete;
    ComputeGraph(ComputeGraph&& computegraph) = delete;
    ComputeGraph& operator=(const ComputeGraph& computegraph) = delete;
    ComputeGraph& operator=(ComputeGraph&& computegraph) = delete;
    ~ComputeGraph();

    std::list<Operator*> &get_op_seq();
    std::vector<Tensor*> get_output_tensors();

    void copy(
        std::vector<Tensor*>& input_tensors,
        std::vector<Tensor*>& weight_tensors
    );
};
