#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>
#include "sparse_data.h"

typedef int32_t MatType;

void coo_gemm(MatType *data_ptr, MatType *out);

#endif
