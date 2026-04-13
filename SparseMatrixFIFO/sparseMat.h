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
// NNZ packed word 設計
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
constexpr int COL_BITS  = LOG2_CEIL(SP_W);        // 足以表示 [0, SP_W) 的最小位元數
constexpr int PACK_BITS = VAL_BITS + COL_BITS;    // 實際使用位元數（例：32+3=35）

typedef ap_uint<PACK_BITS> packed_nnz_t;

// 每個 NNZ 一個 word，總數即 SP_MAX_NNZ
#define NNZ_WORDS      SP_MAX_NNZ

// ----------------------------------------------------------------
// row_offset packed word 設計
//
// 每個元素需要 LOG2_CEIL(SP_MAX_NNZ + 1) bits 才能表示 [0, SP_MAX_NNZ]
// 相鄰兩個元素 (row_offset[i], row_offset[i+1]) 打包成一個 word：
//
//   [2*RO_BITS-1 : RO_BITS] = row_offset[i+1]   (高位)
//   [RO_BITS-1   :       0] = row_offset[i]      (低位)
//
// 共 SP_H 個 packed word，對應 row 0 ~ SP_H-1
// compute 端讀取 word[r] 即可同一 clock 取得 rs = offset[r]
// 與 re = offset[r+1]，直接計算 nnz_count = re - rs
// ----------------------------------------------------------------
constexpr int RO_BITS        = LOG2_CEIL(SP_MAX_NNZ + 1); // 表示 [0, SP_MAX_NNZ] 所需位元
constexpr int RO_PACK_BITS   = 2 * RO_BITS;               // 打包後位元寬

typedef ap_uint<RO_PACK_BITS> packed_ro_t;

// packed row_offset 陣列長度：SP_H 個 word（對應 row 0 ~ SP_H-1）
#define RO_WORDS  SP_H

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
    matType        data_ptr[DATA_H * DATA_W],
    packed_ro_t    packed_ro_ptr[RO_WORDS],         // packed row_offset，每筆含 (offset[r], offset[r+1])
    packed_nnz_t   packed_nnz_ptr[NNZ_WORDS],
    matType        out_ptr[SP_H * DATA_W]
);

#endif