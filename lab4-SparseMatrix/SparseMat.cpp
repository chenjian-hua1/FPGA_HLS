#include "sparseMat.h"

/* COO (Coordinate) format : save nonzero row, col indices */
template<int COO_H, int COO_W, int COO_SPARSE_SIZE>
void CooDeconstruct(
    const MatType data[COO_H][COO_W], 
    MatType nz_data[COO_SPARSE_SIZE],
    const int row_indices[COO_SPARSE_SIZE], const int col_indices[COO_SPARSE_SIZE]
) {
#pragma HLS INLINE


};

void topModule(MatType *data_ptr) {
    MatType data[H][W];
    MatType nz_data[SPARSE_SIZE];

    int row_indices[SPARSE_SIZE];
    int col_indices[SPARSE_SIZE];

    CooDeconstruct<H,W,SPARSE_SIZE>(data, nz_data, row_indices, col_indices);
};