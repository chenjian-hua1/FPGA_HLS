#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

static const int32_t A_col_idx[9] = {4, 0, 4, 0, 1, 2, 3, 1, 3};

static const int64_t A_cols = 5;

static const int32_t A_dense[5][5] = {
  {0, 0, 0, 0, 4},
  {8, 0, 0, 0, 7},
  {1, 0, 0, 0, 0},
  {0, 8, 9, 7, 0},
  {0, 1, 0, 8, 0}
};

static const int64_t A_nnz = 9;

static const int32_t A_row_idx[9] = {0, 1, 1, 2, 3, 3, 3, 4, 4};

static const int64_t A_rows = 5;

static const int32_t A_values[9] = {4, 8, 7, 1, 8, 9, 7, 1, 8};

#endif // SPARSE_DATA_H_
