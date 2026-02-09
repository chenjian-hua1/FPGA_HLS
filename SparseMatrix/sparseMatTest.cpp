#include "sparseMat.h"
#include <iostream>

int main() {
    std::cout << "hello world" << std::endl;
    MatType *data, *out;
    sparse_gemm(data,out);
    return 0;
}