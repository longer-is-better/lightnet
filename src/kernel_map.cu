#include <iostream>

#include "kernel_map.cuh"

std::ostream& operator<<(std::ostream& os, MAP_OP op) {
    switch (op) {
        case MAP_OP::ADD:
            os << "MAP_OP::ADD";
            break;
        case MAP_OP::MULTIPLY:
            os << "MAP_OP::MULTIPLY";
            break;
        case MAP_OP::POW:
            os << "MAP_OP::POW";
            break;
        case MAP_OP::LOG:
            os << "MAP_OP::LOG";
            break;
        case MAP_OP::ABS:
            os << "MAP_OP::ABS";
            break;
        case MAP_OP::SIGN:
            os << "MAP_OP::SIGN";
            break;

        default:
            break;
    }
    return os;
}
