#pragma once

#include "compute_graph.cuh"

class Network: public ComputeGraph
{
public:
    static Network *_trianer;
    bool _train;
    cudaStream_t _cudastream;



    Network();
    Network(ComputeGraph *computegraph, bool train, cudaStream_t cudastream);
    ~Network();

    void to(cudaMemoryType type);
    std::vector<Tensor*> init(std::vector<Tensor*> sample_inputs, std::string weight_path);
    std::vector<Tensor*> forward(std::vector<Tensor*> input_tensors);
    void backward();
    void update_weights(float alpha);


    Network(const Network& network) = delete;
    Network(Network&& network) = delete;
    Network& operator=(const Network& network) = delete;
    Network& operator=(Network&& network) = delete;
};
