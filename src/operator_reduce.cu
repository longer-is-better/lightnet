#include "operator_reduce.cuh"
#include "kernel_reduce.cuh"
#include "kernel_map.cuh"
#include "kernel_others.cuh"
#include "tools_common.cuh"

Reduce::Reduce(REDUCE_OP op, bool end_of_graph): Operator(_end_of_graph), _reduce_op(op) {}

Reduce::Reduce(Tensor* A, REDUCE_OP op)
    : Operator({A}, {new Tensor()}), _reduce_op(op) {
  ;
}

std::string Reduce::type_str() { return std::string("Reduce"); }

Reduce* Reduce::copy() { return new Reduce(_reduce_op, _end_of_graph); }

void Reduce::infer_shape() {
    _output_tensors[0]->set_shape({});
}


void Reduce::forward() {
    dim3 GRID;
    dim3 BLOCK = 512;
    size_t shared_mem = BLOCK.x * sizeof(float);


    D(VLOG(7) << "reduce forward input tensor:" << *_input_tensors[0]);

    size_t work_n = _input_tensors[0]->_element_count;
    float *work_space = nullptr;
    checkCudaErrors(cudaMalloc(&work_space, _input_tensors[0]->_total_size));
    checkCudaErrors(cudaMemcpyAsync(work_space, _input_tensors[0]->_p_data, sizeof(float), cudaMemcpyDeviceToDevice, _cudastream));
    while (work_n != 1){
        GRID = ceil(work_n, BLOCK.x * 2) / (BLOCK.x * 2);
        kreduce<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
            _input_tensors[0]->_element_count,
            work_n,
            work_space,
            work_space,
            _reduce_op
        );
        work_n = GRID.x;
    }
    checkCudaErrors(cudaMemcpyAsync(_output_tensors[0]->_p_data, work_space, sizeof(float), cudaMemcpyDeviceToDevice, _cudastream));
    checkCudaErrors(cudaFreeAsync(work_space, _cudastream));

    D(checkCudaErrors(cudaStreamSynchronize(_cudastream)));
    D(VLOG(7) << "reduce forward output tensor:" << *_output_tensors[0]);
}


void Reduce::backward() {
    if (_end_of_graph) {
        dim3 BLOCK(_output_tensors[0]->_element_count < 1024 ? _output_tensors[0]->_element_count : 1024);
        dim3 GRID(ceil(_output_tensors[0]->_element_count, 1024) / 1024);
        kmemset<<<GRID, BLOCK, 0, _cudastream>>>(
            _output_tensors[0]->_element_count,
            _output_tensors[0]->_p_gradient,
            1.f
        );
    }
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
        _input_tensors[0]->_p_gradient,
        alpha,
        _output_tensors[0]->_p_gradient
    );

    checkCudaErrors(cudaDeviceSynchronize());
    float sss[2];
    checkCudaErrors(cudaMemcpy(sss, _input_tensors[0]->_p_gradient, _input_tensors[0]->_total_size, cudaMemcpyDeviceToHost));




    Tensor s = _input_tensors[0]->grad();
    s.to(cudaMemoryTypeHost);
    D(VLOG(7) << _name << _reduce_op << " backward get input tensor[0] grad:" << s);
}