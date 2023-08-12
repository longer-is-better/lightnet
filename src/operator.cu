#include <vector>

#include "operator.cuh"

Operator::Operator(
    std::vector<Tensor*> input_tensors,
    std::vector<Tensor*> output_tensors
):
    _input_tensors(input_tensors),
    _output_tensors(output_tensors)
{
    for (Tensor*t: _input_tensors) {
        t->_to.push_back(this);
        if (t->_p_from) {
            t->_p_from->_nextoperators[this] = true;
            _prevoperators[t->_p_from] = true;
        }
    }
    for (Tensor*t: _output_tensors) {
        t->_p_from = this;
    }
}

Operator::Operator(const Operator& op) {
    _name = op._name;
}


void Operator::set_cudastream(cudaStream_t cudastream) {
    _cudastream = cudastream;
}

void Operator::mirror(
    const std::map<Tensor*, Tensor*>& tensor_map,
    const std::map<Operator*, Operator*>& operator_map
) {
    for (Tensor* tensor: _input_tensors) {
        operator_map.at(this)->_input_tensors.push_back(tensor_map.at(tensor));
    }
    for (Tensor* tensor: _output_tensors) {
        operator_map.at(this)->_output_tensors.push_back(tensor_map.at(tensor));
    }
    for (std::pair<Operator*, bool> p: _prevoperators) {
        Operator* op = p.first;
        operator_map.at(this)->_prevoperators[operator_map.at(op)] = true;
    }
    for (std::pair<Operator*, bool> p: _nextoperators) {
        Operator* op = p.first;
        operator_map.at(this)->_nextoperators[operator_map.at(op)] = true;
    }
}

int Operator::indegree() {
    int ans = 0;
    for (std::pair<Operator *const, bool>& p: _prevoperators) {
        ans += p.second;
    }
    return ans;
}
