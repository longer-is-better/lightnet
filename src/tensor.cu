#include <string>
#include <vector>
#include <map>
#include <random>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <glog/logging.h>

#include "tools_common.cuh"
#include "tools_cuda.cuh"

#include "tensor.cuh"
#include "kernel_elementwise.cuh"
#include "kernel_map.cuh"
#include "kernel_others.cuh"

std::vector<size_t> Tensor::show_elements = {8, 8, 3, 1};


Tensor::Tensor(){
    VLOG(9) << "Tensor default construct";
}


Tensor::Tensor(
    Operator *p_from
):
    _p_from(p_from)
{
    VLOG(9) << "Tensor p_from construct";
}

Tensor::Tensor(std::vector<size_t> shape, cudaMemoryType memtype, float *data):
    _data_memorytype(memtype),
    _layout(shape.size(), '?'),
    _dim_n(shape.size()),
    _shape(shape),
    _stride(shape)
{
    if (_dim_n){
        _stride[_dim_n - 1] = 1;
        for (int i = _dim_n - 1; i > 0; i--) _stride[i - 1] = _stride[i] * _shape[i];

        _element_count = _stride[0] * _shape[0];
    } else {
        _element_count = 1;
    }
    _total_size = _element_count * sizeof(float);

    if (memtype == cudaMemoryTypeHost) {
        CHECK_NOTNULL(_p_data = (float*)malloc(_total_size));
        CHECK_NOTNULL(_p_gradient = (float*)malloc(_total_size));
        if (data) memcpy(_p_data, data, _total_size);
    } else if (memtype == cudaMemoryTypeDevice) {
        checkCudaErrors(cudaMalloc(&_p_data, _total_size));
        checkCudaErrors(cudaMalloc(&_p_gradient, _total_size));
        if (data) checkCudaErrors(cudaMemcpy(_p_data, data, _total_size, cudaMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << "impossible";
    }
    VLOG(9) << "Tensor shape construct";
}




void Tensor::set_shape(std::vector<size_t> shape){
    _layout = std::string(shape.size(), '?');
    _dim_n = shape.size();
    _shape = shape;
    _stride = shape;

    if (_dim_n){
        _stride[_dim_n - 1] = 1;
        for (int i = _dim_n - 1; i > 0; i--) _stride[i - 1] = _stride[i] * _shape[i];

        _element_count = _stride[0] * _shape[0];
    } else {
        _element_count = 1;
    }
    _total_size = _element_count * sizeof(float);
}

Tensor::Tensor(
    const Tensor &tensor
):
    _data_memorytype(tensor._data_memorytype),
    _name(tensor._name),
    _dim_n(tensor._dim_n),
    _layout(tensor._layout),
    _shape(tensor._shape),
    _stride(tensor._stride),
    _element_count(tensor._element_count),
    _total_size(tensor._total_size)
{
    if (tensor._p_data) {
        if (_data_memorytype == cudaMemoryTypeHost) {
            CHECK_NOTNULL(_p_data = (float*)malloc(_total_size));
            memcpy(_p_data, tensor._p_data, _total_size);
        } else if (_data_memorytype == cudaMemoryTypeDevice) {
            checkCudaErrors(cudaMalloc((void**)&_p_data, _total_size));
            checkCudaErrors(cudaMemcpy(_p_data, tensor._p_data, _total_size, cudaMemcpyHostToDevice));
        } else {
            LOG(FATAL) << "not implement.";
        }
    }

    if (tensor._p_gradient) {
        if (_data_memorytype == cudaMemoryTypeHost) {
            CHECK_NOTNULL(_p_gradient = (float*)malloc(_total_size));
            memcpy(_p_gradient, tensor._p_gradient, _total_size);
        } else if (_data_memorytype == cudaMemoryTypeDevice) {
            checkCudaErrors(cudaMalloc((void**)&_p_gradient, _total_size));
            checkCudaErrors(cudaMemcpy(_p_gradient, tensor._p_gradient, _total_size, cudaMemcpyHostToDevice));
        } else {
            LOG(FATAL) << "not implement.";
        }
    }
    VLOG(9) << "Tensor copy construct";
}

Tensor& Tensor::operator=(const Tensor &tensor) {
    if (this != &tensor) {
        if (_data_memorytype == cudaMemoryTypeHost) {
            free(_p_data); _p_data = nullptr;
            free(_p_gradient); _p_gradient = nullptr;
        } else if (_data_memorytype == cudaMemoryTypeDevice) {
            cudaFree(_p_data); _p_data = nullptr;
            cudaFree(_p_gradient); _p_gradient = nullptr;
        } else {
            LOG(FATAL) << "not support";
        }

        _name = tensor._name;
        _dim_n = tensor._dim_n;
        _layout = tensor._layout;
        _shape = tensor._shape;
        _stride = tensor._stride;
        _element_count = tensor._element_count;
        _total_size = tensor._total_size;

        cudaMemcpyKind direct;
        switch (_data_memorytype) {
            case cudaMemoryTypeHost:
                if (tensor._p_data) CHECK_NOTNULL(_p_data = (float*)malloc(_total_size));
                if (tensor._p_gradient) CHECK_NOTNULL(_p_gradient = (float*)malloc(_total_size));
                switch (tensor._data_memorytype) {
                    case cudaMemoryTypeHost:
                        direct = cudaMemcpyHostToHost; break;
                    case cudaMemoryTypeDevice:
                        direct = cudaMemcpyDeviceToHost; break;
                    
                    default:
                        break;
                }
                break;
            case cudaMemoryTypeDevice:
                if (tensor._p_data) checkCudaErrors(cudaMalloc(&_p_data, _total_size));
                if (tensor._p_gradient) checkCudaErrors(cudaMalloc(&_p_gradient, _total_size));
                switch (tensor._data_memorytype) {
                    case cudaMemoryTypeHost:
                        direct = cudaMemcpyHostToDevice; break;
                    case cudaMemoryTypeDevice:
                        direct = cudaMemcpyDeviceToDevice; break;
                    
                    default:
                        break;
                }
                break;
            
            default:
                break;
        }
        if (tensor._p_data) checkCudaErrors(cudaMemcpy(_p_data, tensor._p_data, _total_size, direct));
        if (tensor._p_gradient)checkCudaErrors(cudaMemcpy(_p_gradient, tensor._p_gradient, _total_size, direct));
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor &&tensor) {
    if (this != &tensor) {
        CHECK_EQ(_data_memorytype, tensor._data_memorytype) << "can not move cross-device";

        if (_data_memorytype == cudaMemoryTypeHost) {
            free(_p_data);
            free(_p_gradient);
        } else if (_data_memorytype == cudaMemoryTypeDevice) {
            cudaFree(_p_data);
            cudaFree(_p_gradient);
        } else {
            LOG(FATAL) << "not support";
        }

        _name = tensor._name;
        _dim_n = tensor._dim_n;
        _layout = tensor._layout;
        _shape = tensor._shape;
        _stride = tensor._stride;
        _element_count = tensor._element_count;
        _total_size = tensor._total_size;
        
        _p_data = tensor._p_data;
        tensor._p_data = nullptr;
        _p_gradient = tensor._p_gradient;
        tensor._p_gradient = nullptr;
    }
    return *this;
}


Tensor::Tensor(Tensor &&tensor){
    _data_memorytype = tensor._data_memorytype;
    _name = tensor._name;
    _dim_n = tensor._dim_n;
    _layout = tensor._layout;
    _shape = tensor._shape;
    _stride = tensor._stride;
    _element_count = tensor._element_count;
    _total_size = tensor._total_size;

    _p_data = tensor._p_data;
    tensor._p_data = nullptr;
    _p_gradient = tensor._p_gradient;
    tensor._p_gradient = nullptr;


    _shadow_of = tensor._shadow_of;
    _shadows = tensor._shadows;

    if (_shadow_of) _shadow_of->_shadows.erase(&tensor);
    if (_shadow_of) _shadow_of->_shadows.insert(this);
    VLOG(9) << "Tensor move construct";
}

Tensor::~Tensor(){
    if (_shadow_of == nullptr) {
        LOG_IF(ERROR, _shadows.size())
            << "Tensor " << _name << "'s " << _shadows.size() << " shaows will be breack, todo shadow tree delete";
        if (_data_memorytype == cudaMemoryTypeHost) {
            free(_p_data); _p_data = nullptr;
            free(_p_gradient); _p_gradient = nullptr;
        } else if (_data_memorytype == cudaMemoryTypeDevice) {
            checkCudaErrors(cudaFree(_p_data)); _p_data = nullptr;
            checkCudaErrors(cudaFree(_p_gradient)); _p_gradient = nullptr;
        } else {
            LOG(FATAL) << "not implement.";
        }

    } else {
        _shadow_of->_shadows.erase(this);
        for (Tensor* shadow: _shadows) {
            shadow->_shadow_of = _shadow_of;
            _shadow_of->_shadows.insert(shadow);
        }
    }
}


Tensor &Tensor::operator[](int i) {
    CHECK_NOTNULL(_p_data);
    CHECK_GE(_dim_n, 1);
    Tensor *ans = new Tensor();
    ans->_data_memorytype = _data_memorytype;
    ans->_name = _name + "[" + std::to_string(i) + "]";
    ans->_dim_n = _dim_n - 1;
    ans->_layout = _layout.substr(1);
    ans->_shape = std::vector<size_t>(_shape.begin() + 1, _shape.end());
    ans->_stride = std::vector<size_t>(_stride.begin() + 1, _stride.end());
    ans->_element_count = _stride[0];
    ans->_total_size = ans->_element_count * sizeof(float);

    CHECK_NOTNULL(_p_data);
    ans->_p_data = _p_data + _stride[0] * i;
    CHECK_NOTNULL(_p_gradient);
    ans->_p_gradient = _p_gradient + _stride[0] * i;

    _shadows.insert(ans);
    ans->_shadow_of = this;
    return *ans;
}

// Tensor &Tensor::operator==(const Tensor &tensor) const {
    // TODO: insert return statement here
// }

Tensor Tensor::grad() {
    Tensor ans(*this);
    if (_data_memorytype == cudaMemoryTypeHost) {
        memcpy(ans._p_data, ans._p_gradient, _total_size);
    } else if (_data_memorytype == cudaMemoryTypeDevice) {
        checkCudaErrors(cudaMemcpy(ans._p_data, ans._p_gradient, _total_size, cudaMemcpyDeviceToDevice));
    } else {
        LOG(FATAL) << "no";
    }
    return ans;
}

void Tensor::alloc_memory() {
    if (_data_memorytype == cudaMemoryTypeHost) {
        free(_p_data);
        free(_p_gradient);
        CHECK_NOTNULL(_p_data = (float*)malloc(_total_size));
        CHECK_NOTNULL(_p_gradient = (float*)malloc(_total_size));
        for (int i = 0; i < _element_count; i++) _p_gradient[i] = 1.f;
    } else if (_data_memorytype == cudaMemoryTypeDevice) {
        checkCudaErrors(cudaFree(_p_data));
        checkCudaErrors(cudaFree(_p_gradient));
        checkCudaErrors(cudaMalloc(&_p_data, _total_size));
        checkCudaErrors(cudaMalloc(&_p_gradient, _total_size));
        dim3 BLOCK(_element_count < 1024 ? _element_count : 1024);
        dim3 GRID(ceil(_element_count, 1024) / 1024);
        kmemset<<<GRID, BLOCK>>>(
            _element_count,
            _p_gradient,
            0.f
        );
    } else {
        LOG(FATAL) << "not implement.";
    }
}

void Tensor::to(cudaMemoryType target_memorytype) {
    float *tmp = nullptr;
    if (_data_memorytype == target_memorytype) return;
    else if (target_memorytype == cudaMemoryTypeHost) {
        if (_p_data) {
            CHECK_NOTNULL(tmp = (float*)malloc(_total_size));
            checkCudaErrors(cudaMemcpy(tmp, _p_data, _total_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(_p_data));
            _p_data = tmp;
            if (_shadow_of) _shadow_of->_shadows.erase(this);
            for (Tensor* shadow: _shadows) {
                shadow->_shadow_of = _shadow_of;
                _shadow_of->_shadows.insert(shadow);
            }
            _shadow_of = nullptr;
        } else {
            VLOG(9) << "move tensor with pullptr _p_data";
        }
        if (_p_gradient) {
            CHECK_NOTNULL(tmp = (float*)malloc(_total_size));
            checkCudaErrors(cudaMemcpy(tmp, _p_gradient, _total_size, cudaMemcpyDeviceToHost));
            checkCudaErrors(cudaFree(_p_gradient));
            _p_gradient = tmp;
        } else {
            VLOG(9) << "move tensor with pullptr _p_gradient";
        }
    } else if (target_memorytype == cudaMemoryTypeDevice) {
        if (_p_data) {
            checkCudaErrors(cudaMalloc(&tmp, _total_size));
            checkCudaErrors(cudaMemcpy(tmp, _p_data, _total_size, cudaMemcpyHostToDevice));
            free(_p_data);
            _p_data = tmp;
        } else {
            VLOG(9) << "move tensor with pullptr _p_data";
        }
        if (_p_gradient) {
            checkCudaErrors(cudaMalloc(&tmp, _total_size));
            checkCudaErrors(cudaMemcpy(tmp, _p_gradient, _total_size, cudaMemcpyHostToDevice));
            free(_p_gradient);
            _p_gradient = tmp;
        } else {
            VLOG(9) << "move tensor with pullptr _p_gradient";
        }
    }
    _data_memorytype = target_memorytype;
}

void Tensor::fill_data_random(float lower_bound, float upper_bound){
    CHECK_NOTNULL(_p_data);
    if (_data_memorytype == cudaMemoryTypeHost) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(lower_bound, upper_bound);
        for (int i = 0; i < _element_count; i++) _p_data[i] = dis(gen);
    } else if (_data_memorytype == cudaMemoryTypeDevice) {
        kinitializeRandom<<<(_element_count + 511) / 512, 512>>>(_p_data, _total_size, lower_bound, upper_bound);
    } else {
        LOG(FATAL) << "not implement.";
    }
    VLOG(9) << "random fill tensor[" << _name << "] data.";
}

void Tensor::mirror(
    const std::map<Tensor *, Tensor *> &tensor_map,
    const std::map<Operator *, Operator *> &operator_map
) {
    if (_p_from) tensor_map.at(this)->_p_from = operator_map.at(_p_from);
    for (Operator *op: _to) {
        tensor_map.at(this)->_to.push_back(operator_map.at(op));
    }
}

void Tensor::update_weights(float alpha, cudaStream_t cudastream) {
    CHECK_NOTNULL(_p_data);
    CHECK_NOTNULL(_p_gradient);

    if (_data_memorytype == cudaMemoryTypeDevice) {
        dim3 BLOCK(32);
        dim3 GRID((_element_count + BLOCK.x - 1) / BLOCK.x);
        kelementwise_inplace<<<GRID, BLOCK, 0, cudastream>>>(
            _element_count,
            _p_data,
            alpha,
            _p_gradient,
            ELE_OP::SUB
        );
        checkCudaErrors(cudaDeviceSynchronize());
    } else {
        LOG(FATAL) << "not implement";
    }

}


std::ostream &operator<<(std::ostream &os, Tensor &tensor) {
    if (tensor._data_memorytype != cudaMemoryTypeHost) {
        Tensor show;
        show = tensor;
        os << "[device] " << show;
        return os;
    } else {
        CHECK_EQ(tensor._data_memorytype, cudaMemoryTypeHost);
        if ( tensor._p_data == nullptr ) {
            os << "tensor " << tensor._name << " empty"<< std::endl;
            return os;
        }

        if (tensor._dim_n > 2) {
            for (int i = 0; ; i++) {
                if (i >= tensor.show_elements[tensor._dim_n - 1]) {
                    for (int i = 0; i < tensor._dim_n; i++) {
                        for (int i = 0; i < tensor._dim_n; i++) os << "-";
                        for (int i = 0; i < tensor._dim_n; i++) os << " ";
                    }
                    break;
                } else if (i >= tensor._shape[0]) {
                    break;
                } else {
                    os << tensor[i];
                }
            }
            os << std::endl;
        } else if (tensor._dim_n == 2) {
            os << "Tensor " << tensor._name << std::endl;
            for (int i = 0; ; i++) {
                if (i >= tensor.show_elements[tensor._dim_n - 1]) {
                    os << ".\n.\n.\n";
                    break;
                } else if (i >= tensor._shape[0]) {
                    break;
                } else {
                    os << tensor[i];
                }
            }
        } else if (tensor._dim_n == 1) {
            for (int i = 0; ; i++) {
                if (i >= tensor.show_elements[tensor._dim_n - 1]) {
                    os << "...";
                    break;
                } else if (i >= tensor._shape[0]) {
                    break;
                } else {
                    os << tensor._p_data[i] << " ";
                }
            }
            os << std::endl;
        } else {
            os << "scalar: " << tensor._p_data[0] << std::endl;
        }
        return os;
    }
}