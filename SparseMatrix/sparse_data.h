#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

#define A_COLS 5

#define A_NNZ 9

#define A_ROWS 5

static const int32_t A_col_idx[9] = {1, 4, 1, 3, 0, 2, 0, 2, 4};

static const int32_t A_dense[5][5] = {
  {0, 3, 0, 0, 2},
  {0, 3, 0, 7, 0},
  {7, 0, 2, 0, 0},
  {3, 0, 0, 0, 0},
  {0, 0, 4, 0, 4}
};

static const int32_t A_row_idx[9] = {0, 0, 1, 1, 2, 2, 3, 4, 4};

static const int32_t A_values[9] = {3, 2, 3, 7, 7, 2, 3, 4, 4};

#endif // SPARSE_DATA_H_
