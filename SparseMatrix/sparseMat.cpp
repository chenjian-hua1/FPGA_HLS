#include "sparseMat.h"

/* 
==========================================================================================================
==================================== Child Module Implement ==============================================
==========================================================================================================
*/
// COO (Coordinate) format : save nonzero row, col indices
template<int DATA_ROWS, int DATA_COLS, int SPARSE_MAT_NNZ> // local parameter
void _coo_gemm(
    MatType data_ptr[DATA_ROWS*DATA_COLS],
    const int row_indices[SPARSE_MAT_NNZ], 
    const int col_indices[SPARSE_MAT_NNZ], // nonzero indices
    const int values[SPARSE_MAT_NNZ] // nonzero values
) {
// #pragma HLS INLINE off

    
};


/* 
==========================================================================================================
===================================== Top Module Implement ===============================================
==========================================================================================================
*/
// COO Format Matrix GEMM
void coo_gemm(MatType data_ptr[]) {
    _coo_gemm<A_ROWS,A_COLS,A_NNZ>(data_ptr,A_row_idx,A_col_idx,A_values);
}