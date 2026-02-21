#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

#define A_COLS 5

#define A_NNZ 8

#define A_ROWS 5

static const int32_t A_col_idx[8] = {0, 3, 3, 3, 0, 1, 2, 1};

static const int32_t A_dense[5][5] = {
  {9, 0, 0, 4, 0},
  {0, 0, 0, 3, 0},
  {0, 0, 0, 6, 0},
  {9, 6, 9, 0, 0},
  {0, 2, 0, 0, 0}
};

static const int32_t A_row_idx[8] = {0, 0, 1, 2, 3, 3, 3, 4};

static const int32_t A_row_ptr[6] = {0, 2, 3, 4, 7, 8};

static const int32_t A_values[8] = {9, 4, 3, 6, 9, 6, 9, 2};

#endif // SPARSE_DATA_H_
