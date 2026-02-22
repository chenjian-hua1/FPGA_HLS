#include "sparseMat.h"
#include "sparse_data.h"
#include <iostream>

using namespace std;

template<typename T>
void print_1d(const T* data, int length) {
    cout << "[ ";
    for (int i = 0; i < length; i++) {
        cout << data[i];
        if (i != length - 1)
            cout << ", ";
    }
    cout << " ]" << endl;
}

template<typename T>
void print_2d(const T* data, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        cout << "[ ";
        for (int c = 0; c < cols; c++) {
            cout << data[r * cols + c];
            if (c != cols - 1)
                cout << ", ";
        }
        cout << " ]" << endl;
    }
}

/* 計算矩陣乘法結果 */
template<typename T, int MAT1_H, int MAT1_W, int MAT2_W>
void gemm_sw(T mat1[MAT1_H][MAT1_W], T mat2[MAT1_W][MAT2_W], T result[MAT1_H][MAT2_W]) {

    for (int r = 0; r < MAT1_H; r++) {
        for (int c = 0; c < MAT2_W; c++) {
            T sum = 0;
            for (int k = 0; k < MAT1_W; k++) {
                sum += mat1[r][k] * mat2[k][c];
            }
            result[r][c] = sum;
        }
    }
}

/* 隨機產生資料矩陣 */
template<typename T, int ROWS, int COLS>
void rand_gen_mat(T mat[ROWS][COLS], T min_val = 0, T max_val = 20) {

    // 建議在 main() 只 srand 一次
    // srand(time(nullptr));

    for (int r = 0; r < ROWS; r++) {
        for (int c = 0; c < COLS; c++) {
            mat[r][c] = min_val + (rand() % (max_val - min_val + 1));
        }
    }
}

/* 隨機生成每row固定nnz的稀疏矩陣 */
template<typename T, int SP_ROWS, int SP_COLS>
void rand_gen_sp_mat_fixed_nnz(T sp_mat[SP_ROWS][SP_COLS], 
                               int maximum = 20,
                               int nnz_per_row = 2) {

    for (int r = 0; r < SP_ROWS; r++) {

        // 先全部設 0
        for (int c = 0; c < SP_COLS; c++)
            sp_mat[r][c] = 0;

        // 隨機選 nnz_per_row 個位置
        for (int k = 0; k < nnz_per_row; k++) {
            int col = rand() % SP_COLS;
            sp_mat[r][col] = (rand() % maximum) + 1;
        }
    }
}

/* 隨機生成稀疏矩陣 */
template<typename T, int SP_ROWS, int SP_COLS>
void rand_gen_sp_mat(T sp_mat[SP_ROWS][SP_COLS], int maximum = 20, float density = 0.2f) {

    // 建議在 main() 只 srand 一次
    // srand(time(nullptr));

    for (int r = 0; r < SP_ROWS; r++) {
        for (int c = 0; c < SP_COLS; c++) {

            float prob = static_cast<float>(rand()) / RAND_MAX;

            if (prob < density) {
                // 產生 1 ~ maximum 的值
                sp_mat[r][c] = (rand() % maximum) + 1;
            } else {
                sp_mat[r][c] = 0;
            }
        }
    }
}

template<typename T, int SP_ROWS, int SP_COLS, int MAX_NNZ>
void mat2csr(
    T sp_mat[SP_ROWS][SP_COLS],
    ap_uint<LOG2_CEIL(MAX_NNZ)> row_offset[SP_ROWS + 1],
    ap_uint<LOG2_CEIL(SP_COLS)> col_indices[MAX_NNZ],
    T values[MAX_NNZ]
) {
    int nnz = 0;
    row_offset[0] = 0;

    for (int r = 0; r < SP_ROWS; r++) {
        for (int c = 0; c < SP_COLS; c++) {
            T v = sp_mat[r][c];
            if (v != 0) {
                col_indices[nnz] = c;
                values[nnz]      = v;
                nnz++;
            }
        }
        row_offset[r + 1] = nnz;
    }
}

int main() {
    // 隨機種子 (根據時間)
    srand(time(nullptr));

    // 生成隨機的稀疏矩陣 & 資料矩陣
    matType sp_mat[SP_H][SP_W];
    // rand_gen_sp_mat<matType, SP_H, SP_W>(sp_mat);
    rand_gen_sp_mat_fixed_nnz<matType, SP_H, SP_W>(sp_mat, 20, SP_NNZ_PER_ROW);

    cout << "SP MAT" << endl;
    print_2d(&sp_mat[0][0], SP_H, SP_W);

    matType data_mat[DATA_H][DATA_W];
    rand_gen_mat<matType, DATA_H, DATA_W>(data_mat);
    cout << "DATA MAT" << endl;
    print_2d(&data_mat[0][0], DATA_H, DATA_W);

    // 轉換 CSR 格式
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset[SP_H+1];
    ap_uint<LOG2_CEIL(SP_W)> col_indices[SP_H*SP_W];
    matType values[SP_H*SP_W];


    mat2csr<matType, SP_H, SP_W, SP_MAX_NNZ>(sp_mat, row_offset, col_indices, values);

    cout << "row offset" << endl;
    print_1d(row_offset, SP_H+1);
    cout << "col_indices" << endl;
    print_1d(col_indices, SP_MAX_NNZ);
    cout << "values" << endl;
    print_1d(values, SP_MAX_NNZ);
    
    // 計算正確矩陣乘法結果 (Software)
    matType gemm_result_sw[SP_H][DATA_W];
    gemm_sw<matType, SP_H, SP_W, DATA_W>(sp_mat, data_mat, gemm_result_sw);

    cout << "gemm result sw" << endl;
    print_2d(&gemm_result_sw[0][0], SP_H, DATA_W);

    // 取得電路計算結果
    matType gemm_result_hw[SP_H][DATA_W];
    csr_gemm(&data_mat[0][0], row_offset, col_indices, values, &gemm_result_hw[0][0]);

    cout << "gemm result hw" << endl;
    print_2d(&gemm_result_hw[0][0], SP_H, DATA_W);

    // 比對電路是否計算正確

    return 0;
}
