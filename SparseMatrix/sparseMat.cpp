#include "sparseMat.h"
#include <iostream>

/* 
==========================================================================================================
==================================== Child Module Implement ==============================================
==========================================================================================================
*/
// inner product module
template <int N>
void _inner_product(MatType *vec1, MatType *vec2, MatType &out) {
//#pragma HLS INLINE

    MatType sum = 0;
    for (int i=0; i<N; i++) {
    #pragma HLS UNROLL
        sum += vec1[i]*vec2[i];
    }

    out = sum;
}

// COO (Coordinate) format : save nonzero row, col indices 
template<int DATA_ROWS, int DATA_COLS, int SPARSE_MAT_NNZ> // local parameter
void _coo_gemm(
    MatType data[DATA_ROWS][DATA_COLS],
    MatType out[A_ROWS][DATA_COLS],
    const int row_indices[SPARSE_MAT_NNZ], 
    const int col_indices[SPARSE_MAT_NNZ], // nonzero indices
    const int values[SPARSE_MAT_NNZ] // nonzero values
) {
 #pragma HLS INLINE
    // !!!!!!!!!!!!!!!!!!!!! 錯誤 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // 因為不知道每個 row 的非零元素量，不可能在電路runtime去計算nnz數量
    // 電路 runtime 階段才能知道要設計出多大電路 -> 很奇怪
    // 需要在編譯階段就知道 nnz 等資訊才能設計電路 (轉CSR)
    int row_offset = 0;
    int current_row = 0;

    gemm: for (int i=0; i<SPARSE_MAT_NNZ; i++) {
        if (current_row!=row_indices[i] || i==SPARSE_MAT_NNZ-1) {
            // 最後一個 row 的上邊界在最後一個idx
            bool is_row_end = (i == SPARSE_MAT_NNZ-1) || (row_indices[i+1] != row_indices[i]);
            int end = is_row_end ? (i+1) : i;   // end 是「下一個 row_offset」
            int nnz = end - row_offset;
            
            // Sparse Row 上的非零 col 索引, 數值
            int row_nz_cols[DATA_ROWS];
            MatType row_nz_values[DATA_ROWS];

            for (int nz_idx=0; nz_idx<DATA_ROWS; nz_idx++) {
			#pragma HLS UNROLL
                row_nz_cols[nz_idx] = col_indices[row_offset+nz_idx];
                row_nz_values[nz_idx] = values[row_offset+nz_idx];
            }

            // (1,nnz)mat * (nnz,DATA_COLS)mat
            for (int c=0; c<DATA_COLS; c++) {
			#pragma HLS UNROLL
                // 將內積不是對到 0 的部分取出來
                MatType col_elements[DATA_ROWS] = {0};
                for (int nz_idx=0; nz_idx<DATA_ROWS; nz_idx++) {
				#pragma HLS UNROLL
                	col_elements[nz_idx] = data[row_nz_cols[nz_idx]][c];
                }
                
                _inner_product<DATA_ROWS>(row_nz_values,col_elements,out[current_row][c]);
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
// Sparse Matrix GEMM
void sparse_gemm(MatType *data_ptr, MatType *out) {
    // !!!!!!!!!!!!!!!!!!!!! 錯誤 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // 因為不知道每個 row 的非零元素量，不可能在電路runtime去計算nnz數量
    // 電路 runtime 階段才能知道要設計出多大電路 -> 很奇怪
    // 需要在編譯階段就知道 nnz 等資訊才能設計電路 (轉CSR)

	MatType data[A_ROWS][A_COLS];
	#pragma HLS ARRAY_PARTITION variable=data type=complete dim=0

	load_data: for (int r=0; r<A_ROWS; r++) {
        for (int c=0; c<A_COLS; c++) {
		#pragma HLS PIPELINE II=1
            data[r][c] = data_ptr[r*A_COLS+c];
        }
    }

    MatType gemm_result[A_ROWS][A_COLS];
    // 編譯階段就決定合成哪種電路
    if constexpr (STORAGE_FMT_ID==FMT_CSR) {
        _coo_gemm<A_COLS,A_ROWS,A_NNZ>(data,gemm_result,A_row_idx,A_col_idx,A_values);
    }
        

    write_data: for (int r=0; r<A_ROWS; r++) {
        for (int c=0; c<A_COLS; c++) {
            out[r*A_COLS+c] = gemm_result[r][c];
        }
    }
}


int main() {
    MatType *data, out[A_ROWS*A_COLS];
    sparse_gemm(data, out);
    std::cout << "hello world" << std::endl;
    return 0;
}
