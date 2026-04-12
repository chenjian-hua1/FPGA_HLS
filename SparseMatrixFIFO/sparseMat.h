#ifndef SPARSE_MAT_H
#define SPARSE_MAT_H

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

// ----------------------------------------------------------------
// 工具宏（保留原版）
// ----------------------------------------------------------------
constexpr int LOG2_CEIL(int x) {
    int r = 0, p = 1;
    while (p < x) { p *= 2; ++r; }
    return r;
}
constexpr bool IS_POW2(int x) {
    return x > 0 && ((x & (x - 1)) == 0);
}
constexpr int FOR_IDX_BITS(int maxium) {
    return IS_POW2(maxium) ? LOG2_CEIL(maxium + 1) : LOG2_CEIL(maxium);
}

// ----------------------------------------------------------------
// 矩陣尺寸參數
// ----------------------------------------------------------------
#define SP_H           5    // 稀疏矩陣列數
#define SP_W           5    // 稀疏矩陣行數
#define DATA_H         5    // 稠密矩陣列數（= SP_W）
#define DATA_W         5    // 稠密矩陣行數
#define SP_NNZ_PER_ROW 2    // 每列固定非零數
#define SP_MAX_NNZ     (SP_NNZ_PER_ROW * SP_H) // 稀疏矩陣最大非零元素量

// ----------------------------------------------------------------
// 資料型別
// ----------------------------------------------------------------
typedef ap_int<32> matType;

// ----------------------------------------------------------------
// 寬位元 Package 設計
//
// 每個 packed word = 128-bit，打包一對 (value, col_idx)：
//
//   [127:96] = values[2i+1]      (32-bit matType)
//   [ 95:64] = col_indices[2i+1] (32-bit，zero-extended)
//   [ 63:32] = values[2i]        (32-bit matType)
//   [ 31: 0] = col_indices[2i]   (32-bit，zero-extended)
//
// SP_MAX_NNZ 為奇數時，最後一個 word 高 64-bit 填 0
// ----------------------------------------------------------------
#define VAL_BITS       32
#define COL_BITS       32                        // zero-extend col_idx 至 32-bit
#define PACK_BITS      (2 * (VAL_BITS + COL_BITS))  // = 128
typedef ap_uint<PACK_BITS> packed_nnz_t;         // 128-bit packed (val,col) pair

// NNZ pair 數量（向上取整）
#define       ((SP_MAX_NNZ + 1) / 2)

// ----------------------------------------------------------------
// Stream payload：load → compute 傳遞一對 NNZ 元素
// ----------------------------------------------------------------
struct nnz_pair_t {
    matType                   val0;
    ap_uint<LOG2_CEIL(SP_W)>  col0;
    matType                   val1;
    ap_uint<LOG2_CEIL(SP_W)>  col1;
    bool                      valid1;  // 奇數 NNZ 最後一筆為 false
};

// compute → store 傳遞一個輸出元素
struct out_elem_t {
    matType val;
    uint8_t row;
    uint8_t col;
};

// ----------------------------------------------------------------
// Top-level 宣告
// ----------------------------------------------------------------
void csr_gemm(
    matType      data_ptr[DATA_H * DATA_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_ptr[SP_H + 1],
    packed_nnz_t packed_nnz_ptr[NNZ_PAIRS],      // ← 新增：packed (val,col) 輸入
    matType      out_ptr[SP_H * DATA_W]
);

#endif
