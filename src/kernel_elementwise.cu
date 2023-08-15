#include <iostream>

#include "kernel_elementwise.cuh"


std::ostream& operator<<(std::ostream& os, ELE_OP op) {
    switch (op) {
        case ELE_OP::ADD:
            os << "ELE_OP::ADD";
            break;
        case ELE_OP::SUB:
            os << "ELE_OP::SUB";
            break;
        case ELE_OP::MULTIPLY:
            os << "ELE_OP::MULTIPLY";
            break;
        case ELE_OP::DIVIDE:
            os << "ELE_OP::DIVIDE";
            break;

        default:
            break;
    }
    return os;
}
