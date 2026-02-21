#include "sparseMat.h"
#include <stdio.h>
#include <iostream>

/* 
==========================================================================================================
==================================== Child Module Implement ==============================================
==========================================================================================================
*/
// inner product module
template <int N, bool UNROLL=true>
void _inner_product(matType vec1[N], matType vec2[N], matType &out) {
#pragma HLS INLINE

    matType sum = 0;
    if constexpr (UNROLL) {
    	inner_product_unroll: for (ap_uint<LOG2_CEIL(N)> i=0; i<N; i++) {
		#pragma HLS UNROLL
			sum += (vec1[i]*vec2[i]);
		}
    }
    else {
    	inner_product_pipe: for (ap_uint<LOG2_CEIL(N)> i=0; i<N; i++) {
		#pragma HLS PIPELINE
			sum += (vec1[i]*vec2[i]);
		}
    }

    out = sum;
}

// COO (Coordinate) format : save nonzero row, col indices 
template<int DATA_ROWS, int DATA_COLS> // local parameter
void _coo_gemm(
    matType data[DATA_ROWS][DATA_COLS],
    matType out[SP_H][DATA_COLS],
    const int row_indices,
    const int col_indices, // nonzero indices
    const int values // nonzero values
) {
 #pragma HLS INLINE
    // !!!!!!!!!!!!!!!!!!!!! 錯誤 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	// 因為不知道 總nnz 所以不知道要取多少資料出來
    // 因為不知道每個 row 的非零元素量，不可能在電路runtime去計算nnz數量
    // 電路 runtime 階段才能知道要設計出多大電路 -> 很奇怪 電路不可能變形

}

// CSR (Coordinate) format : save nonzero row, col indices 
template<int DATA_ROWS, int DATA_COLS, int SP_ROWS, int SP_COLS> // local parameter
void _csr_gemm(
    matType data[DATA_ROWS][DATA_COLS],
    matType out[SP_ROWS][DATA_COLS],
    ap_uint<LOG2_CEIL(SP_ROWS*SP_COLS)> row_offset[SP_ROWS+1],
    ap_uint<LOG2_CEIL(SP_COLS)> col_indices[SP_ROWS*SP_COLS], // nonzero indices
    matType values[SP_ROWS*SP_COLS] // nonzero values
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=out type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=row_offset type=complete
#pragma HLS ARRAY_PARTITION variable=col_indices type=complete
#pragma HLS ARRAY_PARTITION variable=values type=complete

    csr_gemm: for (ap_uint<LOG2_CEIL(SP_ROWS)> r=0; r<SP_ROWS; r++) {
   #pragma HLS UNROLL

        ap_uint<LOG2_CEIL(SP_ROWS*SP_COLS)> row_start = row_offset[r];
    	ap_uint<LOG2_CEIL(SP_ROWS*SP_COLS)> row_end = row_offset[r+1];
        ap_uint<LOG2_CEIL(SP_COLS)> nnz = row_end-row_start;

        matType row_nz_values[DATA_ROWS];
		#pragma HLS ARRAY_PARTITION variable=row_nz_values type=complete
        ap_uint<LOG2_CEIL(SP_COLS)> nz_col_idx[DATA_ROWS];
		#pragma HLS ARRAY_PARTITION variable=nz_col_idx type=complete

        // 抓該 row 的對應範圍的非零 col 索引和數值
        get_nz_data: for (ap_uint<LOG2_CEIL(DATA_ROWS)> data_idx=0; data_idx<DATA_ROWS; data_idx++) {
        #pragma HLS UNROLL
            // 是否超過 nnz
            bool in_nnz_range = (data_idx<nnz);

            // 超過 nnz 代表之後都是 0 -> 多工器去切換 資料,0
            row_nz_values[data_idx] = (in_nnz_range) ? (values[row_start+data_idx]):matType(0);
            nz_col_idx[data_idx] = (in_nnz_range) ? (col_indices[row_start+data_idx]):ap_uint<LOG2_CEIL(SP_COLS)>(0);
        }

        mat_mult: for (ap_uint<LOG2_CEIL(DATA_COLS)> c=0; c<DATA_COLS; c++) {
            // 將內積中不是對到0的部份取出 
            matType col_nz_values[DATA_ROWS];
			#pragma HLS ARRAY_PARTITION variable=col_nz_values type=complete
            load_col_nz: for (ap_uint<LOG2_CEIL(DATA_ROWS)> data_idx=0; data_idx<DATA_ROWS; data_idx++) {
            #pragma HLS UNROLL
                // 是否超過 nnz
                bool in_nnz_range = (data_idx<nnz);
                // 超過就給 0
                col_nz_values[data_idx] = (in_nnz_range) ? (data[nz_col_idx[data_idx]][c]):matType(0);
            }

            _inner_product<DATA_ROWS, true>(row_nz_values,col_nz_values,out[r][c]);
        }
    }
}


/* 
==========================================================================================================
===================================== Top Module Implement ===============================================
==========================================================================================================
*/
// Sparse Matrix GEMM
void csr_gemm(
    matType data_ptr[DATA_H*DATA_W],
    elemIdxType row_offset_ptr[SP_H+1],
    colIdxType col_indices_ptr[SP_MAX_NNZ],
    matType values_ptr[SP_MAX_NNZ],
    matType out_ptr[SP_H*DATA_W]
) {
#pragma HLS INTERFACE m_axi port=data_ptr offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=row_offset_ptr offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=col_indices_ptr offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=values_ptr offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out_ptr offset=slave bundle=gmem
#pragma HLS INTERFACE m_axi port=out_ptr offset=slave bundle=gmem

#pragma HLS INTERFACE s_axilite port=data_ptr        bundle=control
#pragma HLS INTERFACE s_axilite port=row_offset_ptr  bundle=control
#pragma HLS INTERFACE s_axilite port=col_indices_ptr bundle=control
#pragma HLS INTERFACE s_axilite port=values_ptr      bundle=control
#pragma HLS INTERFACE s_axilite port=out_ptr         bundle=control
#pragma HLS INTERFACE s_axilite port=return          bundle=control

    matType data[SP_W][SP_H];
    #pragma HLS ARRAY_PARTITION variable=data type=complete dim=0

    load_data: for (rowIdxType r=0; r<SP_H; r++) {
        for (colIdxType c=0; c<SP_W; c++) {
		#pragma HLS PIPELINE
            data[r][c] = data_ptr[r*SP_W+c];
        }
    }

    elemIdxType row_offset[SP_H+1];
    #pragma HLS ARRAY_PARTITION variable=row_offset type=complete
    load_row_offset: for (offsetIdxType i=0; i<SP_H+1; i++) {
	#pragma HLS PIPELINE
        row_offset[i] = row_offset_ptr[i];
    }

    elemIdxType TOTAL_NNZ = row_offset[SP_H];

    colIdxType col_indices[SP_MAX_NNZ];
    #pragma HLS ARRAY_PARTITION variable=col_indices type=complete
    load_col_indices: for (elemIdxType i=0; i<SP_MAX_NNZ; i++) {
	#pragma HLS PIPELINE
        col_indices[i] = (i<TOTAL_NNZ) ? (col_indices_ptr[i]):colIdxType(0);
    }

    matType values[SP_MAX_NNZ];
    #pragma HLS ARRAY_PARTITION variable=values type=complete
    load_vals: for (elemIdxType i=0; i<SP_MAX_NNZ; i++) {
    #pragma HLS PIPELINE
        values[i] = (i<TOTAL_NNZ) ? (values_ptr[i]):matType(0);
    }
   
    matType gemm_result[SP_H][SP_W];
    #pragma HLS ARRAY_PARTITION variable=gemm_result type=complete dim=1
    _csr_gemm<SP_W,SP_H,SP_H,SP_W>(data, gemm_result, row_offset, col_indices, values);

    // 編譯階段就決定合成哪種電路
//    if constexpr (STORAGE_FMT_ID==FMT_COO) {
//        _coo_gemm<SP_W,SP_H,A_NNZ>(data,gemm_result,A_row_idx,A_col_idx,A_values);
//    }
//    else if (STORAGE_FMT_ID==FMT_CSR) {
//        _csr_gemm<SP_W,SP_H,SP_H,SP_W,A_NNZ>(data,gemm_result,A_row_ptr,A_col_idx,A_values);
//    }
//    else {
//
//    }

    write_data: for (rowIdxType r=0; r<SP_H; r++) {
		for (colIdxType c=0; c<SP_W; c++) {
		#pragma HLS PIPELINE
			out_ptr[r*SP_W+c] = gemm_result[r][c];
		}
	}
}

//int main() {
//  matType data[SP_W*SP_H], out[SP_H*SP_W];
//
//  int row_ptr[SP_H+1];
//  for (int i=0; i<SP_H+1; i++)
//    row_ptr[i] = A_row_idx[i];
//
//  int col_indices[A_NNZ];
//  int values[A_NNZ];
//  for (int i=0; i<A_NNZ; i++) {
//    col_indices[i] = A_col_idx[i];
//    values[i] = A_values[i];
//  }
//
//  csr_gemm(data, row_ptr, col_indices, values, out);
//  std::cout << "hello world" << std::endl;
//  return 0;
//}
