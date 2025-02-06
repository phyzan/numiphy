#include "arrays.hpp"

void arr::_check_index(const int& index, const int& size){
    if (index >= 0){
        if (index < size){return;}
    }
    else{
        if (index > -size-1){return;}
    }
    throw std::out_of_range("Index " + std::to_string(index) + " is out of range for an array of size " + std::to_string(size));
}


int arr::sign(double x){
    if (x>0){
        return 1;
    }
    else if (x<0){
        return -1;
    }
    else{
        return 0;
    }
}