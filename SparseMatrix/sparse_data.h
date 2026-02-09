#ifndef SPARSE_DATA_H_
#define SPARSE_DATA_H_

#include <cstdint>

#define A_COLS 5

#define A_NNZ 10

#define A_ROWS 5

static constexpr int32_t A_col_idx[10] = {0, 4, 0, 2, 0, 3, 4, 1, 3, 4};

static constexpr const int32_t A_dense[5][5] = {
  {3, 0, 0, 0, 1},
  {4, 0, 3, 0, 0},
  {5, 0, 0, 4, 1},
  {0, 0, 0, 0, 0},
  {0, 6, 0, 4, 4}
};

static constexpr int32_t A_row_idx[10] = {0, 0, 1, 1, 2, 2, 2, 4, 4, 4};

static constexpr int32_t A_row_ptr[6] = {0, 2, 4, 7, 7, 10};

static constexpr int32_t A_values[10] = {3, 1, 4, 3, 5, 4, 1, 6, 4, 4};

#endif // SPARSE_DATA_H_
