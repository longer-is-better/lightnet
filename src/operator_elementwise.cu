#include "operator_elementwise.cuh"
#include "kernel_others.cuh"
#include "kernel_map.cuh"
#include "kernel_elementwise.cuh"
#include "tools_common.cuh"

ElementWise::ElementWise(ELE_OP op, bool end_of_graph): Operator(end_of_graph), _ele_op(op){}

ElementWise::ElementWise(Tensor* A, Tensor* B, ELE_OP op)
    : Operator({A, B}, {new Tensor()}), _ele_op(op) {}

std::string ElementWise::type_str() { return std::string("ElementWise"); }

ElementWise* ElementWise::copy() { return new ElementWise(_ele_op, _end_of_graph); }

void ElementWise::infer_shape() {
    CHECK_EQ(_input_tensors.size(), 2);
    CHECK_EQ(_input_tensors[0]->_dim_n, _input_tensors[1]->_dim_n);
    for (size_t i = 0; i < _input_tensors[0]->_dim_n; i++) {
        CHECK_EQ(_input_tensors[0]->_shape[i], _input_tensors[1]->_shape[i]);
    }
    CHECK_STREQ(_input_tensors[0]->_layout.c_str(), _input_tensors[1]->_layout.c_str());
    _output_tensors[0]->set_shape(_input_tensors[0]->_shape);
}


void ElementWise::forward() {
    dim3 BLOCK(32);
    dim3 GRID((_input_tensors[0]->_element_count + BLOCK.x - 1) / BLOCK.x);
    size_t shared_mem = 0;
    kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        _input_tensors[0]->_element_count,
        _input_tensors[0]->_p_data,
        1.f,
        _input_tensors[1]->_p_data,
        _output_tensors[0]->_p_data,
        _ele_op
    );
    D(checkCudaErrors(cudaDeviceSynchronize()));
    D(VLOG(7) << "ElementWise " << _ele_op << " forward output tensor:" << *_output_tensors[0]);
}


void ElementWise::backward() {
    if (_end_of_graph) {
        dim3 BLOCK(_output_tensors[0]->_element_count < 1024 ? _output_tensors[0]->_element_count : 1024);
        dim3 GRID(ceil(_output_tensors[0]->_element_count, 1024) / 1024);
        kmemset<<<GRID, BLOCK, 0, _cudastream>>>(
            _output_tensors[0]->_element_count,
            _output_tensors[0]->_p_gradient,
            1.f
        );
    }
    dim3 BLOCK(32);
    dim3 GRID((_input_tensors[0]->_element_count + BLOCK.x - 1) / BLOCK.x);
    size_t shared_mem = 0;
    switch (_ele_op) {
        case ELE_OP::ADD:
            checkCudaErrors(cudaMemcpyAsync(_input_tensors[0]->_p_gradient, _output_tensors[0]->_p_gradient, _output_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
            checkCudaErrors(cudaMemcpyAsync(_input_tensors[1]->_p_gradient, _output_tensors[0]->_p_gradient, _output_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
            D(checkCudaErrors(cudaDeviceSynchronize()));
            break;
        case ELE_OP::SUB:
            checkCudaErrors(cudaMemcpyAsync(_input_tensors[0]->_p_gradient, _output_tensors[0]->_p_gradient, _output_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
            kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _output_tensors[0]->_p_gradient,
                _input_tensors[1]->_p_gradient,
                MAP_OP::MULTIPLY,
                -1.f
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            break;
        case ELE_OP::MULTIPLY:
            kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[0]->_p_data,
                1.f,
                _output_tensors[0]->_p_gradient,
                _input_tensors[1]->_p_gradient,
                ELE_OP::MULTIPLY
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_data,
                1.f,
                _output_tensors[0]->_p_gradient,
                _input_tensors[0]->_p_gradient,
                ELE_OP::MULTIPLY
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            break;
        case ELE_OP::DIVIDE:
            kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _output_tensors[0]->_p_gradient,
                1.f,
                _input_tensors[1]->_p_data,
                _input_tensors[0]->_p_gradient,
                ELE_OP::DIVIDE
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_data,
                _input_tensors[1]->_p_gradient,
                MAP_OP::POW,
                -2.f
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            kmap_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                MAP_OP::MULTIPLY,
                -1.f
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            kelementwise_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                1.f,
                _input_tensors[0]->_p_data,
                ELE_OP::MULTIPLY
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            kelementwise_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                1.f,
                _output_tensors[0]->_p_gradient,
                ELE_OP::MULTIPLY
            );
            D(checkCudaErrors(cudaDeviceSynchronize()));
            break;

        default:
            break;
    }
    D(Tensor s1 = _input_tensors[0]->grad());
    D(Tensor s2 = _input_tensors[1]->grad());
    D(VLOG(7) << _name << _ele_op << " backward get input tensor[0] grad:" << s1);
    D(VLOG(7) << _name << _ele_op << " backward get input tensor[1] grad:" << s2);
}
