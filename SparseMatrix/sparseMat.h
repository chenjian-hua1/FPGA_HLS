#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>
#include "sparse_data.h"

typedef int32_t MatType;

// 每種格式一個代號（整數）
#define FMT_COO   0
#define FMT_CSR   1
#define FMT_SELL  2
#define FMT_BSR   3
#define FMT_ELL   4
// ...之後要加就一直加

// 最後用這個宏指定「本次要用哪個」
// 預設值（也可以不給，強制外部一定要定義）
#ifndef STORAGE_FMT_ID
#define STORAGE_FMT_ID FMT_CSR
#endif

void sparse_gemm(MatType *data_ptr, MatType *out);

#endif
