#include "operator_map.cuh"
#include "kernel_map.cuh"
#include "kernel_elementwise.cuh"
#include "kernel_others.cuh"
#include "tools_common.cuh"



Map::Map(
    Tensor* A,
    MAP_OP op,
    float operand
):
    Operator({A}, {new Tensor()}),
    _operand(operand)
{
    ;
}

Operator *Map::copy()
{
    return new Map();
}

void Map::infer_shape() {
    _output_tensors[0]->set_shape(_input_tensors[0]->_shape);
}


void Map::forward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    
    BLOCK = dim3(32);
    GRID = dim3((_input_tensors[0]->_element_count + BLOCK.x - 1) / BLOCK.x);
    shared_mem = 0;

    kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        _input_tensors[0]->_element_count,
        _input_tensors[0]->_p_data,
        _output_tensors[0]->_p_data,
        _map_op,
        _operand
    );
    checkCudaErrors(cudaDeviceSynchronize());

}


void Map::backward() {
    dim3 BLOCK;
    dim3 GRID;
    size_t shared_mem;

    
    BLOCK = dim3(32);
    GRID = dim3((_input_tensors[0]->_element_count + BLOCK.x - 1) / BLOCK.x);
    shared_mem = 0;

    switch (_map_op) {
        case MAP_OP::ADD:
            kmemset<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_gradient,
                1.f
            );
            break;
        case MAP_OP::MULTIPLY:
            kmemset<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_gradient,
                _operand
            );
            break;
        case MAP_OP::POW:
            kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_data,
                _input_tensors[0]->_p_gradient,
                MAP_OP::POW,
                _operand - 1
            );
            kmap_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_gradient,
                MAP_OP::MULTIPLY,
                _operand
            );
            break;
        case MAP_OP::ABS:
            kmap<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_data,
                _input_tensors[0]->_p_gradient,
                MAP_OP::SIGN,
                0.f
            );
            kelementwise_inplace<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _input_tensors[0]->_p_gradient,
                1.f,
                _output_tensors[0]->_p_gradient,
                ELE_OP::MULTIPLY
            );
            break;
        case MAP_OP::LOG:
            kelementwise<<<GRID, BLOCK, shared_mem, _cudastream>>>(
                _input_tensors[0]->_element_count,
                _output_tensors[0]->_p_gradient,
                1.f,
                _input_tensors[0]->_p_data,
                _input_tensors[0]->_p_gradient,
                ELE_OP::DIVIDE
            );
            break;
        
        default:
            break;
    }
}