#include <glog/logging.h>

#include "network.cuh"
#include "tensor.cuh"


#include "operator_elementwise.cuh"

Network::Network(
    ComputeGraph *computegraph,
    cudaStream_t cudastream
):
    _cudastream(cudastream)
{
    for (Operator *op: get_op_seq()) op->set_cudastream(cudastream);
    computegraph->copy(_input_tensors, _weight_tensors);
}

Network::~Network()
{
}


void Network::to(cudaMemoryType type){
    for (Tensor* input_tensor: _input_tensors) {
        input_tensor->to(type);
    }
    for (Tensor* weight_tensor: _weight_tensors) {
        weight_tensor->to(type);
    }

    for (Operator* op: this->get_op_seq()) {
        for (Tensor* output_tensor: op->_output_tensors) {
            output_tensor->to(type);
        }
    }
}

/// @brief infer tensor shape, tensor alloc memory
/// @param sample_inputs 
/// @param weight_path 
std::vector<Tensor*> Network::init(std::vector<Tensor*> sample_inputs, std::string weight_path) {
    CHECK_EQ(_input_tensors.size(), sample_inputs.size());
    for (int i = 0; i < _input_tensors.size(); i++) {
        *_input_tensors[i] = *sample_inputs[i];
        _input_tensors[i]->alloc_memory();
    }
    for (Tensor* weight_tensor: _weight_tensors) {
        weight_tensor->alloc_memory();
    }
    int i = 0;
    for (Operator *op: get_op_seq()) {
        op->_name = op->type_str() + "_" + std::to_string(i++);
        op->infer_shape();
        for (Tensor *tensor: op->_output_tensors) {
            tensor->alloc_memory();
        }
    }
    return get_output_tensors();
}


std::vector<Tensor*> Network::forward(std::vector<Tensor*> input_tensors){

    CHECK_EQ(_input_tensors.size(), input_tensors.size());
    for (int i = 0; i < _input_tensors.size(); i++) {
        *_input_tensors[i] = *input_tensors[i];
    }
    for (Operator* op: get_op_seq()) {
        op->forward();
    }
    checkCudaErrors(cudaDeviceSynchronize());
    return get_output_tensors();
}



void Network::backward() {
    for (auto it = get_op_seq().rbegin(); it != get_op_seq().rend(); ++it) {
        (*it)->backward();
    }
}


void Network::update_weights(float alpha) {
    for (Tensor* weight_tensor: _weight_tensors) {
        weight_tensor->update_weights(alpha, _cudastream);
    }
}