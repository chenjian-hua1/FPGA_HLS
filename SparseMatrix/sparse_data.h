#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

static const int32_t A_col_idx[14] = {0, 1, 2, 4, 1, 2, 0, 1, 0, 4, 0, 1, 2, 3};

static const int64_t A_cols = 5;

static const int32_t A_dense[5][5] = {
  {7, 8, 5, 0, 1},
  {0, 9, 7, 0, 0},
  {9, 4, 0, 0, 0},
  {8, 0, 0, 0, 6},
  {8, 9, 9, 2, 0}
};

static const int64_t A_nnz = 14;

static const int32_t A_row_idx[14] = {0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4};

static const int64_t A_rows = 5;

static const int32_t A_values[14] = {7, 8, 5, 1, 9, 7, 9, 4, 8, 6, 8, 9, 9, 2};

#endif // SPARSE_DATA_H_
