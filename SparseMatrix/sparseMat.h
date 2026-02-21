#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <stdio.h>
#include "sparse_data.h"
#include "ap_int.h"

// 計算該數字需要使用多少位元
constexpr int LOG2_CEIL(int x) {
    // 定義域：x >= 1
    // ceil(log2(1)) = 0, ceil(log2(2)) = 1, ceil(log2(3)) = 2, ...
    int r = 0;
    int p = 1;
    // 直到 2^r 超過x
    while (p < x) {
        p*=2;
        ++r;
    }
    return r;
}

// 稀疏矩陣高
#define SP_H 5
// 稀疏矩陣寬
#define SP_W 5
// 資料矩陣高
#define DATA_H 5
// 資料矩陣寬
#define DATA_W 5

// 稀疏矩陣最多有幾個非零元素
#define SP_MAX_NNZ (SP_H*SP_W)

// row, col 索引所需要使用的位元數
typedef ap_uint<LOG2_CEIL(SP_H)> rowIdxType;
typedef ap_uint<LOG2_CEIL(SP_W)> colIdxType;
// offset 數值所需要的位元數
typedef ap_uint<LOG2_CEIL(SP_H+1)> offsetIdxType;
typedef ap_uint<LOG2_CEIL(SP_MAX_NNZ)> elemIdxType;

// 8 bits integer -128~127
typedef ap_int<32> matType;
//typedef int matType;


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


void csr_gemm(matType *data_ptr, elemIdxType *row_offset_ptr, colIdxType *col_indices_ptr, matType *values_ptr, matType *out_ptr);

#endif
