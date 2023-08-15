#include <set>
#include <map>

// #include "operators.h"
#include "compute_graph.cuh"


ComputeGraph::ComputeGraph() {}

ComputeGraph::~ComputeGraph() {
  // to do
}

std::list<Operator*> &ComputeGraph::get_op_seq() {
    if (_op_seq.empty()) {
        std::set<Operator*> init_ops;
        for (Tensor* t: _input_tensors) {
            for (Operator* op: t->_to) {
                if (init_ops.find(op) == init_ops.end()) {
                    _op_seq.push_back(op);
                    init_ops.insert(op);
                }
            }
        }
        for (Operator* p_op_in_seq: _op_seq) {
            for (std::pair<Operator* const, bool>& pair_next_operator: p_op_in_seq->_nextoperators) {
                Operator* next_operator = pair_next_operator.first;
                next_operator->_prevoperators[p_op_in_seq] = false;
                if (next_operator->indegree() == 0) _op_seq.push_back(next_operator);
            }
        }
        for (auto op: _op_seq) for (auto& p: op->_prevoperators) p.second = true;
    }
    return _op_seq;
}

void ComputeGraph::copy(
    std::vector<Tensor*>& input_tensors,
    std::vector<Tensor*>& weight_tensors
) {
    std::map<Tensor*, Tensor*> tensor_map_on;
    std::map<Operator*, Operator*> operator_map_on;

    for (Tensor* input_tensor: _input_tensors) {
        Tensor* copyed_tensor = new Tensor(*input_tensor);
        input_tensors.push_back(copyed_tensor);
        tensor_map_on[input_tensor] = copyed_tensor;
    }
    for (Tensor* weight_tensor: _weight_tensors) {
        Tensor* copyed_tensor = new Tensor(*weight_tensor);
        weight_tensors.push_back(copyed_tensor);
        tensor_map_on[weight_tensor] = copyed_tensor;
    }

    for (Operator* op: this->get_op_seq()) {
        operator_map_on[op] = op->copy();
        for (Tensor* output_tensor: op->_output_tensors) {
            Tensor* copyed_tensor = new Tensor(*output_tensor);
            tensor_map_on[output_tensor] = copyed_tensor;
        }
    }

    for (std::pair<Tensor*, Tensor*> p: tensor_map_on) {
        p.first->mirror(tensor_map_on, operator_map_on);
    }

    for (std::pair<Operator*, Operator*> p: operator_map_on) {
        p.first->mirror(tensor_map_on, operator_map_on);
    }
}

std::vector<Tensor*> ComputeGraph::get_output_tensors() {
    if (_output_tensors.empty()) {
        for (Operator* op: get_op_seq()) {
            for (Tensor* tensor: op->_output_tensors) {
                if (tensor->_to.empty()) {
                    _output_tensors.push_back(tensor);
                }
            }
        }
    }
    return _output_tensors;
}
