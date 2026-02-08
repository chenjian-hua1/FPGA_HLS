#include "sparseMat.h"
#include <iostream>

/* 
==========================================================================================================
==================================== Child Module Implement ==============================================
==========================================================================================================
*/
// inner product module
void _inner_product(MatType *vec1, MatType *vec2, MatType &out, const int N) {
#pragma HLS INLINE

    MatType sum;
    for (int i=0; i<N; i++) {
    #pragma HLS UNROLL
        sum += vec1[i]*vec2[i];
    }

    out = sum;
}

// COO (Coordinate) format : save nonzero row, col indices
template<int DATA_ROWS, int DATA_COLS, int SPARSE_MAT_NNZ> // local parameter
void _coo_gemm(
    MatType data_ptr[DATA_ROWS*DATA_COLS],
    MatType out[A_ROWS][DATA_COLS],
    const int row_indices[SPARSE_MAT_NNZ], 
    const int col_indices[SPARSE_MAT_NNZ], // nonzero indices
    const int values[SPARSE_MAT_NNZ] // nonzero values
) {
// #pragma HLS INLINE off
    MatType data[DATA_ROWS][DATA_COLS];
    
    load_data: for (int r=0; r<DATA_ROWS; r++) {
    #pragma HLS UNROLL

        for (int c=0; c<DATA_COLS; c++) {
            data[r][c] = data_ptr[r*DATA_COLS+c];
        }
    }

    int row_offset = 0;
    int current_row = 0;
    gemm: for (int i=0; i<SPARSE_MAT_NNZ; i++) {
    #pragma HLS UNROLL

        // 捕捉每個 row 的邊界, nnz 合成對應大小的乘法電路
        // 感覺已經是 CSR 的概念了！
        if (current_row!=row_indices[i] || i==SPARSE_MAT_NNZ-1) {
            // std::cout << row_indices[i] << std::endl;
            // 最後一個 row 的上邊界在最後一個idx
            const int nnz = (i==SPARSE_MAT_NNZ) ? (i+1-row_offset):(i-row_offset);
            
            // Sparse Row 上的非零 col 索引, 數值
            int row_nz_cols[nnz];
            int row_nz_values[nnz];
            for (int nz_idx=0; nz_idx<nnz; nz_idx++) {
                row_nz_cols[nz_idx] = A_col_idx[row_offset+nz_idx];
                row_nz_values[nz_idx] = A_values[row_offset+nz_idx];
            }

            // (1,nnz)mat * (nnz,DATA_COLS)mat
            for (int c=0; c<DATA_COLS; c++) {
                // 將內積不是對到 0 的部分取出來
                MatType col_elements[nnz];
                for (int nz_idx=0; nz_idx<nnz; nz_idx++)
                    col_elements[nz_idx] = data[nz_idx][c];
                
                _inner_product(row_nz_values,col_elements,out[current_row][c],nnz);
            }

            row_offset = i;
            current_row = row_indices[i];
        }
        else {
            continue;
        }
    }
};


/* 
==========================================================================================================
===================================== Top Module Implement ===============================================
==========================================================================================================
*/
// COO Format Matrix GEMM
void coo_gemm(MatType *data_ptr, MatType *out) {
    MatType gemm_result[A_ROWS][A_COLS];
    _coo_gemm<A_ROWS,A_COLS,A_NNZ>(data_ptr,gemm_result,A_row_idx,A_col_idx,A_values);
}


// int main() {
//     MatType *data, *out;
//     coo_gemm(data, out);
//     return 0;
// }