#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>
#include "sparse_data.h"

typedef int MatType;

// MatType GEMM_PARAM[SPARSE_SIZE] = {1, 2, 3, 4, 5, 6, 7};

void sparseMatGEMM(MatType data_ptr[A_rows*A_cols]);

#endif