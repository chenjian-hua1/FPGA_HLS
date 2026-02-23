#ifndef MATRIX_MULT_H
#define MATRIX_MULT_H

#define N 4
#define M 4
#define P 4

#define BLOCKSIZE 2
// 無條件進位 : -1 是為當 x/y整除時 + 1(y/y) 邊界會多一個
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

#define BLOCK_N CEIL_DIV(N, BLOCKSIZE)
#define BLOCK_P CEIL_DIV(P, BLOCKSIZE)
#define BLOCK_M (M * BLOCKSIZE)

typedef int MatrixType;

void adder_tree(MatrixType A[], MatrixType &out, int length);
void parallel_mult(MatrixType A[], MatrixType B[], MatrixType out[], int length, int blockSize);
void BlockMatrixMult(MatrixType A[BLOCK_N][BLOCK_M], MatrixType B[BLOCK_M][BLOCK_P], MatrixType C[N][P]);
void BlockMatrixMultAXI(MatrixType* A_Addr, MatrixType* B_Addr, MatrixType* C_Addr);

#endif
