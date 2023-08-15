#include <iostream>

#include "kernel_reduce.cuh"

std::ostream& operator<<(std::ostream& os, REDUCE_OP op) {
    switch (op) {
        case REDUCE_OP::SUM:
            os << "REDUCE_OP::SUM";
            break;
        case REDUCE_OP::AVG:
            os << "REDUCE_OP::AVG";
            break;

        default:
            break;
    }
    return os;
}