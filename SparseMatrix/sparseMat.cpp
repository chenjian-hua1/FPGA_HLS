#include "sparseMat.h"

/* 
==========================================================================================================
==================================== Child Module Implement ==============================================
==========================================================================================================
*/
// COO (Coordinate) format : save nonzero row, col indices
template<int COO_H, int COO_W, int COO_SPARSE_SIZE> // local parameter
void _CooDeconstruct(
    const MatType data[COO_H][COO_W], 
    MatType nz_data[COO_SPARSE_SIZE],
    const int row_indices[COO_SPARSE_SIZE], const int col_indices[COO_SPARSE_SIZE]
) {
#pragma HLS INLINE off

    
};


/* 
==========================================================================================================
===================================== Top Module Implement ===============================================
==========================================================================================================
*/
// Sparse Matrix GEMM
void sparseMatGEMM(MatType data_ptr[H*W]) {
    // input data
    MatType data[H][W];
    // nonzero datas
    MatType nz_data[SPARSE_SIZE];

    // nonzero data indices
    int row_indices[SPARSE_SIZE];
    int col_indices[SPARSE_SIZE];

    load_data: for (int r=0; r<H; r++) {
    // #pragma HLS UNROLL
        
        for (int c=0; c<W; c++) {
            data[r][c] = data_ptr[r*W+c];
        }
    }

    // get nonzero data (COO Format)
    _CooDeconstruct<H,W,SPARSE_SIZE>(data, nz_data, row_indices, col_indices);
};