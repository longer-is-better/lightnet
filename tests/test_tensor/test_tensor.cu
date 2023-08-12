// #include<tuple>
// #include<limits>
// #include<functional>
// #include<vector>
#include <gtest/gtest.h>
#include <glog/logging.h>
#include <gflags/gflags.h>


#include "tensor.cuh"

DEFINE_int32(my_param, 0, "Description of my_param");

TEST (tensor, smoke_glog) {
    LOG(ERROR) << "my_param " << FLAGS_my_param;
    Tensor a({1, 3, 7, 5});
    // a.fill_data_random(-10.f, 10.f);
    for (int i = 0; i < a._element_count; i++) a._p_data[i] = i;
    std::cout << a;
}

TEST (tensor, mdata) {
    Tensor b({1, 3, 7, 5});
    for (int i = 0; i < b._element_count; i++) b._p_data[i] = i;
    Tensor a = b;
    std::cout << a;
    std::cout << b;
}