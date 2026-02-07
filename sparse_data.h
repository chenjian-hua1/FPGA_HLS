#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

static const int32_t A_col_idx[5] = {4, 4, 0, 1, 4};

static const int64_t A_cols = 5;

static const int32_t A_dense[5][5] = {
  {0, 0, 0, 0, 0},
  {0, 0, 0, 0, 1},
  {0, 0, 0, 0, 7},
  {9, 5, 0, 0, 0},
  {0, 0, 0, 0, 7}
};

static const int64_t A_nnz = 5;

static const int32_t A_row_idx[5] = {1, 2, 3, 3, 4};

static const int64_t A_rows = 5;

static const int32_t A_values[5] = {1, 7, 9, 5, 7};

#endif // SPARSE_DATA_H_
