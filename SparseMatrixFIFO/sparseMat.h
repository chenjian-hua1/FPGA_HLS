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
// VAL_BITS：從 matType（ap_int<N>）的 ::width 靜態成員自動推算
// COL_BITS：從 SP_W 透過 LOG2_CEIL 推算，足以索引所有行
//
// 每個 packed word = (VAL_BITS + COL_BITS)-bit，打包單筆 (value, col_idx)：
//
//   [VAL_BITS+COL_BITS-1 : COL_BITS] = value     (VAL_BITS-bit matType)
//   [COL_BITS-1          :         0] = col_index (COL_BITS-bit)
//
// ----------------------------------------------------------------
constexpr int VAL_BITS  = matType::width;         // 與 matType 的位元寬同步
constexpr int COL_BITS  = LOG2_CEIL(SP_W);       // 足以表示 [0, SP_W) 的最小位元數
constexpr int PACK_BITS = VAL_BITS + COL_BITS;   // 實際使用位元數（例：32+3=35）

typedef ap_uint<PACK_BITS> packed_nnz_t;

// 每個 NNZ 一個 word，總數即 SP_MAX_NNZ
#define NNZ_WORDS      SP_MAX_NNZ

// ----------------------------------------------------------------
// Stream payload：load → compute 傳遞單筆 NNZ 元素
// ----------------------------------------------------------------
struct nnz_elem_t {
    matType                   val;
    ap_uint<LOG2_CEIL(SP_W)>  col;
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
    packed_nnz_t packed_nnz_ptr[NNZ_WORDS],      // 64-bit/筆 packed (val, col) 輸入
    matType      out_ptr[SP_H * DATA_W]
);

#endif