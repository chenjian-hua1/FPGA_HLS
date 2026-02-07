#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>
#include "sparse_data.h"

typedef int32_t MatType;

// MatType GEMM_PARAM[SPARSE_SIZE] = {1, 2, 3, 4, 5, 6, 7};

void coo_gemm(MatType data_ptr[]);

#endif