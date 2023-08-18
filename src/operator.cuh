#pragma once

#include <map>
#include <utility>

#include "tensor.cuh"

class Operator {
public:
    std::string _name;
    bool _end_of_graph = false;
    cudaStream_t _cudastream = cudaStreamDefault;

    std::map<Operator*, bool> _prevoperators = {};  // bool: exist for topologicalSort
    std::map<Operator*, bool> _nextoperators = {};  // bool: exist
    std::vector<Tensor*> _input_tensors;
    std::vector<Tensor*> _output_tensors;


    Operator(){};
    Operator(bool end_of_graph): _end_of_graph(end_of_graph){
        ;
    };
    Operator(
        std::vector<Tensor*> input_tensors,
        std::vector<Tensor*> output_tensors
    );

    Operator(const Operator &op);
    virtual ~Operator(){for (auto t: _output_tensors) delete t;};




    virtual void mirror(const std::map<Tensor*, Tensor*>& tensor_map, const std::map<Operator*, Operator*>& operator_map);
    virtual int indegree();
    virtual void set_cudastream(cudaStream_t cudastream);
    virtual std::string type_str() = 0;
    virtual Operator* copy() = 0;
    virtual void infer_shape() = 0;
    virtual void forward() = 0;
    virtual void backward() = 0;



    Operator(Operator &&op) = delete;
    Operator& operator = (const Operator &op) = delete;
    Operator& operator = (Operator &&op) = delete;
};