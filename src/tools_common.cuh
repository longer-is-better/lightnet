#pragma once

#include <cstdlib>

#include <glog/logging.h>



#ifdef DEBUG 
    #define D(x) x
#else 
    #define D(x)
#endif

size_t ceil(size_t in, size_t align);