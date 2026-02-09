#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>

#define W 5
#define H 5

#define SPARSE_SIZE 7
#define SPARSE_MAX_W 3 
#define SPARSE_MAX_H 3


typedef int MatType;

MatType GEMM_PARAM[SPARSE_SIZE] = {1, 2, 3, 4, 5, 6, 7};

void topModule(MatType *data_ptr);

#endif