#include "operator_reduce.cuh"
#include "kernel_reduce.cuh"
#include "kernel_others.cuh"
#include "tools_common.cuh"

Reduce::Reduce(REDUCE_OP op):_reduce_op(op) {}

Reduce::Reduce(Tensor* A, REDUCE_OP op)
    : Operator({A}, {new Tensor()}), _reduce_op(op) {
  ;
}

std::string Reduce::type_str() { return std::string("Reduce"); }

Reduce* Reduce::copy() { return new Reduce(_reduce_op); }

void Reduce::infer_shape() {
    _output_tensors[0]->set_shape({});
}


void Reduce::forward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    if (_input_tensors[0]->_element_count <= 2048) {
        GRID = dim3(1);
        int warp_align = ceil(_input_tensors[0]->_element_count / 2, 32);
        BLOCK = dim3(warp_align);
        shared_mem = warp_align * sizeof(float);

        kreduce<<<GRID, BLOCK, shared_mem, _cudastream>>>(
            _input_tensors[0]->_element_count,
            _input_tensors[0]->_p_data,
            _output_tensors[0]->_p_data,
            _reduce_op
        );
        checkCudaErrors(cudaDeviceSynchronize());
    } else if (2048 < _input_tensors[0]->_element_count && _input_tensors[0]->_element_count <= pow(2048, 2)) {
        LOG(FATAL) << "not implement";
    } else {
        LOG(FATAL) << "not implement";
    }
}


void Reduce::backward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    BLOCK = dim3(32);
    GRID = dim3(ceil(_input_tensors[0]->_element_count, 32) / 32);
    shared_mem = 0;

    float alpha;
    switch (_reduce_op) {
        case REDUCE_OP::SUM:
            alpha = 1.f;
            break;
        case REDUCE_OP::AVG:
            alpha = 1.f / _input_tensors[0]->_element_count;
            break;
        
        default:
            break;
    }
    kmemset_d<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        _input_tensors[0]->_element_count,
        _input_tensors[0]->_p_data,
        alpha,
        _output_tensors[0]->_p_data
    );
    checkCudaErrors(cudaDeviceSynchronize());
}