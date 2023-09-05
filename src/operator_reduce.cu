#include "operator_reduce.cuh"
#include "kernel_reduce.cuh"
#include "kernel_map.cuh"
#include "kernel_others.cuh"
#include "tools_common.cuh"

Reduce::Reduce(bool end_of_graph): Operator(end_of_graph) {}

Reduce::Reduce(Tensor* A)
    : Operator({A}, {new Tensor()}) {
  ;
}

std::string Reduce::type_str() { return std::string("Reduce"); }

Reduce* Reduce::copy() {
    return new Reduce(_end_of_graph);
}

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
    checkCudaErrors(cudaMemcpyAsync(work_space, _input_tensors[0]->_p_data, _input_tensors[0]->_total_size, cudaMemcpyDeviceToDevice, _cudastream));
    while (work_n != 1){
        GRID = ceil(work_n, BLOCK.x * 2) / (BLOCK.x * 2);
        kreduce_sum<<<GRID, BLOCK, shared_mem, cudaStreamDefault>>>(
            _input_tensors[0]->_element_count,
            work_n,
            work_space,
            work_space
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

    float alpha = 1.f;

    kmemset_d<<<GRID, BLOCK, shared_mem, _cudastream>>>(
        _input_tensors[0]->_element_count,
        _input_tensors[0]->_p_gradient,
        alpha,
        _output_tensors[0]->_p_gradient
    );

    checkCudaErrors(cudaDeviceSynchronize());


    Tensor s = _input_tensors[0]->grad();
    s.to(cudaMemoryTypeHost);
}