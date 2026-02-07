#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

static const int32_t A_col_idx[12] = {3, 4, 1, 3, 1, 2, 3, 3, 4, 0, 3, 4};

#define A_cols 5

static const int32_t A_dense[5][5] = {
  {0, 0, 0, 9, 6},
  {0, 5, 0, 3, 0},
  {0, 2, 4, 3, 0},
  {0, 0, 0, 1, 8},
  {9, 0, 0, 3, 7}
};

#define A_nnz 12

static const int32_t A_row_idx[12] = {0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4};

#define A_rows 5

static const int32_t A_values[12] = {9, 6, 5, 3, 2, 4, 3, 1, 8, 9, 3, 7};

#endif // SPARSE_DATA_H_
