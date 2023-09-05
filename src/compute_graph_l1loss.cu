#include "compute_graph_l1loss.cuh"
#include "operators.cuh"

L1LossGraph::L1LossGraph()
{
    _input_tensors.push_back(new Tensor());  // predict
    _input_tensors.push_back(new Tensor());  // target
    Operator *sub = new ElementWise(_input_tensors[0], _input_tensors[1], ELE_OP::SUB);
    Operator *abs = new Map(sub->_output_tensors[0], MAP_OP::ABS);
    Operator *red = new Reduce(abs->_output_tensors[0]);
    red->_end_of_graph = true;
}

L1LossGraph::~L1LossGraph()
{
}