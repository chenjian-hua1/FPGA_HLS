#include "sparseMat.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

// ================================================================
// 固定稀疏矩陣（SP_H=5, SP_W=5, SP_NNZ_PER_ROW=2）
//
//   row 0: (col=1, val=3), (col=4, val=7)
//   row 1: (col=0, val=5), (col=3, val=2)
//   row 2: (col=2, val=8), (col=4, val=1)
//   row 3: (col=0, val=6), (col=1, val=4)
//   row 4: (col=2, val=9), (col=3, val=3)
// ================================================================
static matType SP_MAT[SP_H][SP_W] = {
    { 0, 3, 0, 0, 7 },
    { 5, 0, 0, 2, 0 },
    { 0, 0, 8, 0, 1 },
    { 6, 4, 0, 0, 0 },
    { 0, 0, 9, 3, 0 }
};
 
// ================================================================
// 固定稠密矩陣（DATA_H=5, DATA_W=5）
// ================================================================
static matType DATA_MAT[DATA_H][DATA_W] = {
    { 1, 2, 3, 4, 5 },
    { 6, 7, 8, 9, 10 },
    { 11, 12, 13, 14, 15 },
    { 16, 17, 18, 19, 20 },
    { 21, 22, 23, 24, 25 }
};
 
// ----------------------------------------------------------------
// 打包工具：將單筆 (val, col) 打包成 packed_nnz_t
//   [COL_BITS+VAL_BITS-1 : COL_BITS] = val
//   [COL_BITS-1          :         0] = col
// ----------------------------------------------------------------
static packed_nnz_t pack_nnz(matType v, int c) {
    packed_nnz_t w = 0;
    ap_uint<VAL_BITS> raw;
    raw = v.range(VAL_BITS - 1, 0);
    w.range(COL_BITS + VAL_BITS - 1, COL_BITS) = raw;
    w.range(COL_BITS - 1, 0) = (ap_uint<COL_BITS>)c;
    return w;
}

// ----------------------------------------------------------------
// 打包工具：將相鄰兩個 row_offset 元素打包成 packed_ro_t
//   高位 [2*RO_BITS-1 : RO_BITS] = hi（= row_offset[r+1]）
//   低位 [RO_BITS-1   :       0] = lo（= row_offset[r]）
// ----------------------------------------------------------------
static packed_ro_t pack_ro(int lo, int hi) {
    packed_ro_t w = 0;
    w.range(RO_BITS - 1,         0) = (ap_uint<RO_BITS>)lo;
    w.range(2 * RO_BITS - 1, RO_BITS) = (ap_uint<RO_BITS>)hi;
    return w;
}

// ----------------------------------------------------------------
// 列印工具
// ----------------------------------------------------------------
template<typename T>
void print_2d(const T* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        cout << "[ ";
        for (int c = 0; c < cols; c++) {
            cout << (int)data[r * cols + c];
            if (c != cols - 1) cout << ", ";
        }
        cout << " ]" << endl;
    }
}

// ----------------------------------------------------------------
// 軟體黃金值
// ----------------------------------------------------------------
template<typename T, int R1, int C1, int C2>
void gemm_sw(T m1[R1][C1], T m2[C1][C2], T res[R1][C2]) {
    for (int r = 0; r < R1; r++)
        for (int c = 0; c < C2; c++) {
            T s = 0;
            for (int k = 0; k < C1; k++) s += m1[r][k] * m2[k][c];
            res[r][c] = s;
        }
}

// ----------------------------------------------------------------
// 生成固定 NNZ 的稀疏矩陣（Fisher-Yates shuffle 確保欄位不重複）
// ----------------------------------------------------------------
template<typename T, int ROWS, int COLS>
void rand_sp_fixed(T mat[ROWS][COLS], int nnz_per_row = SP_NNZ_PER_ROW) {
    int cols_pool[COLS];
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            mat[r][c]    = 0;
            cols_pool[c] = c;
        }
        for (int k = 0; k < nnz_per_row; k++) {
            int swap_idx        = k + rand() % (COLS - k);
            int tmp             = cols_pool[k];
            cols_pool[k]        = cols_pool[swap_idx];
            cols_pool[swap_idx] = tmp;
            mat[r][cols_pool[k]] = (rand() % 20) + 1;
        }
    }
}

// ----------------------------------------------------------------
// 稀疏矩陣 → packed NNZ + packed row_offset
//
// packed_ro[r] 儲存 (row_offset[r], row_offset[r+1])，共 SP_H 個 word
// ----------------------------------------------------------------
void mat2csr_packed(
    matType      sp_mat[SP_H][SP_W],
    packed_ro_t  packed_ro[RO_WORDS],
    packed_nnz_t packed_nnz[NNZ_WORDS])
{
    // 先建出原始 row_offset（SP_H+1 個元素）
    int row_offset[SP_H + 1];
    int nnz = 0;

    row_offset[0] = 0;
    for (int r = 0; r < SP_H; r++) {
        for (int c = 0; c < SP_W; c++) {
            if (sp_mat[r][c] != 0) {
                packed_nnz[nnz] = pack_nnz(sp_mat[r][c], c);
                nnz++;
            }
        }
        row_offset[r + 1] = nnz;
    }

    // 將相鄰兩個 row_offset 打包：packed_ro[r] = (offset[r], offset[r+1])
    for (int r = 0; r < SP_H; r++) {
        packed_ro[r] = pack_ro(row_offset[r], row_offset[r + 1]);
    }
}

// ================================================================
// Main
// ================================================================
int main() {
    srand(time(nullptr));

    // 生成隨機矩陣
    // matType sp_mat[SP_H][SP_W];
    // rand_sp_fixed<matType, SP_H, SP_W>(sp_mat);

    // matType data_mat[DATA_H][DATA_W];
    // for (int r = 0; r < DATA_H; r++)
    //     for (int c = 0; c < DATA_W; c++)
    //         data_mat[r][c] = (rand() % 20);

    // 轉換為 packed NNZ + packed row_offset
    packed_ro_t  packed_ro[RO_WORDS];
    packed_nnz_t packed_nnz[NNZ_WORDS];
    mat2csr_packed(SP_MAT, packed_ro, packed_nnz);

    // 軟體黃金值
    matType sw_result[SP_H][DATA_W];
    gemm_sw<matType, SP_H, SP_W, DATA_W>(SP_MAT, DATA_MAT, sw_result);

    // HLS kernel
    matType hw_result[SP_H * DATA_W];
    csr_gemm(&DATA_MAT[0][0], packed_ro, packed_nnz, hw_result);

    // 驗證
    bool correct = true;
    for (int r = 0; r < SP_H; r++)
        for (int c = 0; c < DATA_W; c++)
            if (hw_result[r * DATA_W + c] != sw_result[r][c]) {
                cout << "MISMATCH [" << r << "][" << c << "]: "
                     << "HW=" << hw_result[r * DATA_W + c]
                     << " SW=" << sw_result[r][c] << endl;
                correct = false;
            }

    cout << (correct ? "PASSED" : "FAILED") << endl;
    return correct ? 0 : 1;
}