#include "MatrixMult.h"
#include <iostream>
#include <cstdlib> // rand
#include <string>
using namespace std;

void matrixMult(MatrixType A[N][M], MatrixType B[M][P], MatrixType C[N][P]) {
    row: for (int r=0; r<N; r++) {
        col: for (int c=0; c<P; c++) {
            // A row and B col inner product
            MatrixType sum = 0;
            for (int k=0; k<M; k++)
                sum+=(A[r][k]*B[k][c]);

            C[r][c] = sum;
        }
    }
}

void randMatrix(MatrixType *Mat, int h, int w, int mod = 20) {
    for (int r=0; r<h; r++) {
        for (int c=0; c<w; c++)
            Mat[r * w + c] = rand() % mod;
    }
}

template <typename T>
void printMatrix(T *Mat, int h, int w) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            cout << Mat[r * w + c] << "\t";
        }
        cout << endl;
    }
}

void mat2blockMat(MatrixType *mat, MatrixType *blockMat, int h, int w, int blockH, int blockW, int blockSize=BLOCKSIZE, int dim=0) {
    if (dim==0) {
        // row major
        for (int block=0; block<blockH; block++) {
            int writeBlockStart = block*blockW;
            // Scan Row Block Line
            for (int r=0; r<blockSize; r++) {
                // int readRowIdx = block*blockSize + r;
                int readRowStart = (block*blockSize + r) * w;
                int writeColStart = r*w;

                for (int c=0; c<w; c++) {
                    int readIdx = readRowStart + c;
                    int writeIdx = writeBlockStart+(writeColStart+c);

                    blockMat[writeIdx] = mat[readIdx];
                }
            }
        }
    }
    else {
        // col major
        for (int block=0; block<blockW; block++) {
            // Scan Row Block Line
            for (int c=0; c<blockSize; c++) {
                int readStart = block*blockSize + c;
                int writeStart = block + (c*h)*blockW;

                for (int r=0; r<h; r++) {
                    int readIdx = readStart+(r*w);
                    int writeIdx = writeStart+(r*blockW);

                    blockMat[writeIdx] = mat[readIdx];
                }
            }
        }
    }
}

/* === ... = <tilte> === ...= */
void printTitle(string title = "1", int length = 100) {
    // title 兩側加空白
    string mid = " " + title + " ";

    // 若 length 太小，直接印 title
    if (length <= (int)mid.size()) {
        cout << mid << endl;
        return;
    }

    int remain = length - mid.size();
    int left  = remain / 2;
    int right = remain - left;  // 處理奇數

    cout << string(left, '=') << mid << string(right, '=') << endl;
}


int main() {
    // ------- Create Matrix ---------------------
    MatrixType A[N][M];
    randMatrix(&A[0][0], N, M);
    printTitle("Matrix A");
    printMatrix(&A[0][0], N, M);

    MatrixType B[M][P];
    randMatrix(&B[0][0], M, P);
    printTitle("Matrix B");
    printMatrix(&B[0][0], M, P);

    // ------- Reshape Block Mat ------------------
    MatrixType blockA[BLOCK_N][BLOCK_M];

    mat2blockMat(&A[0][0], &blockA[0][0], N, M, BLOCK_N, BLOCK_M, BLOCKSIZE, 0);
    printTitle("A : Mat -> Block Mat");
    printMatrix(&blockA[0][0], BLOCK_N, BLOCK_M);

    MatrixType blockB[BLOCK_M][BLOCK_P];
    mat2blockMat(&B[0][0], &blockB[0][0], M, P, BLOCK_M, BLOCK_P, BLOCKSIZE, 1);
    printTitle("B : Mat -> Block Mat");
    printMatrix(&blockB[0][0], BLOCK_M, BLOCK_P);

    // ------- Test HLS IP ------------------------
    MatrixType C_HW[N][P];
    // BlockMatrixMult(blockA, blockB, C_HW);
    BlockMatrixMultAXI(&blockA[0][0], &blockB[0][0], &C_HW[0][0]);
    printTitle("HW : Mat A * Mat B");
    printMatrix(&C_HW[0][0], N, P);

    // ------- Calculate Truth --------------------
    MatrixType C_SW[N][P];
    matrixMult(A, B, C_SW);
    printTitle("SW : Mat A * Mat B");
    printMatrix(&C_SW[0][0], N, P);

    // ------- Compare Result ---------------------
    char check[N][P];
    bool all_correct = true;

    for (int r=0; r<N; r++) {
        for (int c=0; c<P; c++) {
            // HLS IP Result equal SoftWare Truth
            bool correct = (C_HW[r][c]==C_SW[r][c]);
            // if (find one wrong) -> false
            all_correct = all_correct && correct;
            // mark correct or wrong
            check[r][c] = (correct) ? 'V':'X';
        }
    }

    printTitle("Check Result");
    printMatrix(&check[0][0], N, P);

    string check_msg;
    if (all_correct)
        check_msg = " HLS IP Test Correct ";
    else
        check_msg = " HLS IP Test Error ";

    printTitle(check_msg);

    return all_correct ? 0:1;
}
