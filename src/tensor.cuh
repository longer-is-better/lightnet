#pragma once
#include <string>
#include <vector>
#include <map>
#include <set>
#include <iostream>

#include "tools_cuda.cuh"

class Operator;

class Tensor
{
public:
    static std::vector<size_t> show_elements;

    cudaMemoryType _data_memorytype = cudaMemoryTypeHost;
    std::string _name = "unnamed";
    size_t _dim_n = 0;
    std::string _layout = "";
    std::vector<size_t> _shape = {};
    std::vector<size_t> _stride = {};
    size_t _element_count = 0;
    size_t _total_size = 0;

    float *_p_data = nullptr;
    float *_p_gradient = nullptr;

    Operator *_p_from = nullptr;
    std::vector<Operator*> _to = {};

    Tensor *_shadow_of = nullptr;
    std::set<Tensor*> _shadows = {};

    Tensor();
    Tensor(Operator *p_from);
    Tensor(std::vector<size_t> shape, cudaMemoryType memtype = cudaMemoryTypeHost, float *data = nullptr);
    Tensor(const Tensor &tensor);
    Tensor(Tensor &&tensor);
    Tensor &operator=(const Tensor &tensor);
    Tensor &operator=(Tensor &&tensor);
    Tensor &operator[](int i);
    Tensor &operator==(const Tensor &tensor) const;
    ~Tensor();

    void set_shape(std::vector<size_t> shape);
    void alloc_memory();
    bool load_data() {LOG(WARNING) << "not implement"; return false;};
    void to(cudaMemoryType memorytype);
    void fill_data_random(float lower_bound, float upper_bound);
    void mirror(const std::map<Tensor*, Tensor*>& tensor_map, const std::map<Operator*, Operator*>& operator_map);
    void update_weights(float alpha, cudaStream_t cudastream);
    friend std::ostream& operator<<(std::ostream& os, Tensor &tensor);
};