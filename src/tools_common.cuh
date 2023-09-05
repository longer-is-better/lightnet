#pragma once

#include <functional>
#include <cstdlib>
#include <random>

#include <glog/logging.h>



#ifdef DEBUG 
    #define D(x) x
#else 
    #define D(x)
#endif

size_t ceil(size_t in, size_t align);

// std::uniform_int_distribution
template<class DATA_TYPE, template<class> class DISTRIBUTE>
std::function<DATA_TYPE(const std::vector<int>&)> get_rand_data_gen(
    DATA_TYPE lowwer_bound,
    DATA_TYPE upper_bound
) {
    std::random_device rd;
    std::mt19937 gen(rd());
    DISTRIBUTE<DATA_TYPE> dist(lowwer_bound, upper_bound);
    return [dist, gen] (const std::vector<int>& in) mutable {return dist(gen);};
}

template<class DATA_TYPE>
std::ostream& operator<<(std::ostream& os, const std::function<DATA_TYPE(const std::vector<int>&)> &func) {
    os << "rand " << typeid(DATA_TYPE).name();
    return os;
}
