#include "operator_elementwise.cuh"
#include "kernel_others.cuh"
#include "kernel_map.cuh"
#include "kernel_elementwise.cuh"


ElementWise::ElementWise(
    Tensor* A,
    Tensor* B,
    ELE_OP op
):
    Operator({A, B}, {new Tensor()})
{
    ;
}

Operator *ElementWise::copy()
{
    return new ElementWise();
}

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
    check_device_data(_input_tensors[0]->_p_data, _input_tensors[0]->_element_count);
    check_device_data(_input_tensors[1]->_p_data, _input_tensors[1]->_element_count);
    // check_device_data(_output_tensors[0]->_p_data, _output_tensors[0]->_element_count);
    // kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
    //     _input_tensors[0]->_element_count,
    //     _input_tensors[0]->_p_data,
    //     1.f,
    //     _input_tensors[1]->_p_data,
    //     _output_tensors[0]->_p_data,
    //     _ele_op
    // );
    float *a=nullptr, *b=nullptr;
    checkCudaErrors(cudaMalloc(&a, 8));
    checkCudaErrors(cudaMalloc(&b, 8));
    check_device_data(a, 2);
    check_device_data(b, 2);
    VLOG(8) << "bf";
    kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        2,
        a,
        1.f,
        a,
        b,
        _ele_op
    );
    VLOG(8) << "mid";
    checkCudaErrors(cudaDeviceSynchronize());
    check_device_data(a, 2);
    check_device_data(b, 2);
    checkCudaErrors(cudaFree(a));
    checkCudaErrors(cudaFree(b));
    VLOG(8) << "aft";
}


void ElementWise::backward() {
    dim3 BLOCK(32);
    dim3 GRID((_input_tensors[0]->_element_count + BLOCK.x - 1) / BLOCK.x);
    size_t shared_mem = 0;
    switch (_ele_op) {
        case ELE_OP::ADD:
            checkCudaErrors(cudaMemcpyAsync(_input_tensors[0]->_p_gradient, _output_tensors[0]->_p_gradient, _output_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
            checkCudaErrors(cudaMemcpyAsync(_input_tensors[1]->_p_gradient, _output_tensors[0]->_p_gradient, _output_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
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
            kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_data,
                1.f,
                _output_tensors[0]->_p_gradient,
                _input_tensors[0]->_p_gradient,
                ELE_OP::MULTIPLY
            );
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
            kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_data,
                _input_tensors[1]->_p_gradient,
                MAP_OP::POW,
                -2.f
            );
            kmap_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                MAP_OP::MULTIPLY,
                -1.f
            );
            kelementwise_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                1.f,
                _input_tensors[0]->_p_data,
                ELE_OP::MULTIPLY
            );
            kelementwise_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _output_tensors[0]->_total_size,
                _input_tensors[1]->_p_gradient,
                1.f,
                _output_tensors[0]->_p_gradient,
                ELE_OP::MULTIPLY
            );
            break;

        default:
            break;
    }
}