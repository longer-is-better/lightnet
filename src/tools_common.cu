#include "tools_common.cuh"

size_t ceil(size_t in, size_t align) {
    return (in + align - 1) / align * align;
}