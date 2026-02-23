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

constexpr bool IS_POW2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}

// 計算 for index (0..maxium-1) 需要的位元寬度
// 規則：若 maxium 是 2 的冪次方 -> LOG2_CEIL(maxium + 1)
//      否則 -> LOG2_CEIL(maxium)
constexpr int FOR_INDEX_BITS(int maxium) {
    return IS_POW2(maxium) ? LOG2_CEIL(maxium + 1) : LOG2_CEIL(maxium);
}

// 稀疏矩陣高
#define SP_H 5
// 稀疏矩陣寬
#define SP_W 5
// 資料矩陣高
#define DATA_H 5
// 資料矩陣寬
#define DATA_W 5
// 固定稀疏矩陣每行的nnz
#define SP_NNZ_PER_ROW 2
// 稀疏矩陣最多有幾個非零元素
#define SP_MAX_NNZ (SP_NNZ_PER_ROW*SP_H)

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


void csr_gemm(
    matType data_ptr[DATA_H*DATA_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_ptr[SP_H+1],
    ap_uint<LOG2_CEIL(SP_W)> col_indices_ptr[SP_MAX_NNZ],
    matType values_ptr[SP_MAX_NNZ],
    matType out_ptr[SP_H*DATA_W]
);

#endif
