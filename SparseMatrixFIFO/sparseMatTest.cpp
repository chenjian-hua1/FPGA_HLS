#include "sparseMat.h"
#include <iostream>
#include <cstdlib>
#include <ctime>
using namespace std;

// ----------------------------------------------------------------
// 打包工具：將兩筆 (val, col) 打包成 128-bit word
//   [127:96]=val1  [95:64]=col1  [63:32]=val0  [31:0]=col0
// ----------------------------------------------------------------
static packed_nnz_t pack_nnz(matType v0, int c0, matType v1, int c1) {
    packed_nnz_t w = 0;
    ap_uint<32> raw0, raw1;
    raw0 = v0.range(31, 0);
    raw1 = v1.range(31, 0);
    w.range( 63, 32) = raw0;
    w.range( 31,  0) = (ap_uint<32>)c0;
    w.range(127, 96) = raw1;
    w.range( 95, 64) = (ap_uint<32>)c1;
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
// 生成固定 NNZ 的稀疏矩陣
// ----------------------------------------------------------------
template<typename T, int ROWS, int COLS>
void rand_sp_fixed(T mat[ROWS][COLS], int nnz_per_row = SP_NNZ_PER_ROW) {
    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) mat[r][c] = 0;
        for (int k = 0; k < nnz_per_row; k++) {
            int col = rand() % COLS;
            mat[r][col] = (rand() % 20) + 1;
        }
    }
}

// ----------------------------------------------------------------
// 稀疏矩陣 → CSR + packed NNZ
// ----------------------------------------------------------------
void mat2csr_packed(
    matType      sp_mat[SP_H][SP_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset[SP_H + 1],
    packed_nnz_t packed_nnz[NNZ_PAIRS])
{
    // 先收集所有 (val, col)
    matType vals[SP_MAX_NNZ];
    int     cols[SP_MAX_NNZ];
    int nnz = 0;

    row_offset[0] = 0;
    for (int r = 0; r < SP_H; r++) {
        for (int c = 0; c < SP_W; c++) {
            if (sp_mat[r][c] != 0) {
                vals[nnz] = sp_mat[r][c];
                cols[nnz] = c;
                nnz++;
            }
        }
        row_offset[r + 1] = nnz;
    }

    // 兩兩打包
    for (int p = 0; p < NNZ_PAIRS; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        bool last_odd = (SP_MAX_NNZ % 2 != 0) && (p == NNZ_PAIRS - 1);

        matType v0 = vals[i0];
        int     c0 = cols[i0];
        matType v1 = last_odd ? matType(0) : vals[i1];
        int     c1 = last_odd ? 0          : cols[i1];

        packed_nnz[p] = pack_nnz(v0, c0, v1, c1);
    }
}

// ================================================================
// Main
// ================================================================
int main() {
    srand(time(nullptr));

    // 生成隨機矩陣
    matType sp_mat[SP_H][SP_W];
    rand_sp_fixed<matType, SP_H, SP_W>(sp_mat);

    matType data_mat[DATA_H][DATA_W];
    for (int r = 0; r < DATA_H; r++)
        for (int c = 0; c < DATA_W; c++)
            data_mat[r][c] = (rand() % 20);

    // 轉換為 CSR + packed NNZ
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset[SP_H + 1];
    packed_nnz_t packed_nnz[NNZ_PAIRS];
    mat2csr_packed(sp_mat, row_offset, packed_nnz);

    // 軟體黃金值
    matType sw_result[SP_H][DATA_W];
    gemm_sw<matType, SP_H, SP_W, DATA_W>(sp_mat, data_mat, sw_result);

    // HLS kernel
    matType hw_result[SP_H * DATA_W];
    csr_gemm(&data_mat[0][0], row_offset, packed_nnz, hw_result);

    // 驗證
    bool correct = true;
    for (int r = 0; r < SP_H; r++)
        for (int c = 0; c < DATA_W; c++)
            if (hw_result[r * DATA_W + c] != sw_result[r][c]) {
                cout << "MISMATCH [" << r << "][" << c << "]: "
                     << "HW=" << hw_result[r*DATA_W+c]
                     << " SW=" << sw_result[r][c] << endl;
                correct = false;
            }

    cout << (correct ? "PASSED" : "FAILED") << endl;
    return correct ? 0 : 1;
}
