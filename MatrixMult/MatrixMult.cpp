#include "MatrixMult.h"


/* ------------------------------------------------ Top Module ------------------------------------------------ */
void MatrixMult(MatrixType A[N][M], MatrixType B[M][P], MatrixType C[N][P]) {
	/* A[N][M] * B[M][P] = C[N][P] */

#pragma HLS ARRAY_PARTITION variable=A type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=B type=complete dim=1
//#pragma HLS ARRAY_PARTITION variable=C type=complete dim=2

// close all PIPELINE
//#pragma HLS PIPELINE off

    row: for (int r = 0; r < N; r++) {
    	MatrixType A_row[M];
		#pragma HLS ARRAY_PARTITION variable=A_row type=complete

    	// move data to new array
    	for (int i=0; i<M; i++) {
		#pragma HLS UNROLL
    		A_row[i] = A[r][i];
    	}

        col: for (int c = 0; c < P; c++) {
		#pragma HLS PIPELINE

        	//------------------ Stage 1 : Parallel Multiplication -----------------------------------
        	MatrixType mult_result[M];
			#pragma HLS ARRAY_PARTITION variable=mult_result type=complete
        	MatrixType B_col[M];
			#pragma HLS ARRAY_PARTITION variable=B_col type=complete

        	// move data to new array
        	for (int i=0; i<M; i++) {
			#pragma HLS UNROLL
        		B_col[i] = B[i][c];
        	}

        	parallel_mult(A_row, B_col, mult_result, M, M);

//            parallel_mult: for (int k = 0; k < M; k++) {
//			#pragma HLS UNROLL
//            	// element-wise mult first
//            	mult_result[k] = A[r][k]*B[k][c];
//            }

        	//------------------  Stage 2 : Adder Tree -----------------------------------------------
            MatrixType sum = 0;
            adder_tree(mult_result, sum, M);

            C[r][c] = sum;
        }
    }
}


void BlockMatrixMult(MatrixType A[BLOCK_N][BLOCK_M], MatrixType B[BLOCK_M][BLOCK_P], MatrixType C[N][P]) {

#pragma HLS ARRAY_PARTITION variable=A type=complete dim=2
#pragma HLS ARRAY_PARTITION variable=B type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=C type=complete dim=0

	blockRow: for (int block_r=0; block_r<BLOCK_N; block_r++) {

		// load Row Block
		MatrixType rowBlock[BLOCK_M];

		#pragma HLS ARRAY_PARTITION variable=rowBlock type=complete

		for (int k=0; k<BLOCK_M; k++) {
		#pragma HLS UNROLL
			rowBlock[k] = A[block_r][k];
		}

		// Scan Column Block
		blockCol: for (int block_c=0; block_c<BLOCK_P; block_c++) {
		#pragma HLS PIPELINE

			// load Column Block
			MatrixType colBlock[BLOCK_M];
			#pragma HLS ARRAY_PARTITION variable=colBlock type=complete

			 for (int k=0; k<BLOCK_M; k++) {
			 #pragma HLS UNROLL
			 	 colBlock[k] = B[k][block_c];
			 }


			// Block Operation : parrellel Many Row,Col inner product
			BlockOp: for (int r=0; r<BLOCKSIZE; r++) {
			#pragma HLS UNROLL

				// Out Array Row Index
				int writeRowIdx = block_r*BLOCKSIZE + r;

				// inner product use row elements start position of block
				int rowElementStart = r*M;

				// get Row elements (Wire)
				MatrixType rowElements[M];
				#pragma HLS ARRAY_PARTITION variable=rowElements type=complete

				for (int k=0; k<M; k++) {
				#pragma HLS UNROLL
					rowElements[k] = rowBlock[rowElementStart+k];
				}

				// Scan all col
				for (int c=0; c<BLOCKSIZE; c++) {
				#pragma HLS UNROLL

					// Out Array Column Index
					int writeColIdx = block_c*BLOCKSIZE + c;

					// inner product use column elements start position of block
					int colElementStart = c*M;

					// get Column elements (Wire)
					MatrixType colElements[M];
					#pragma HLS ARRAY_PARTITION variable=colElements type=complete

					for (int k=0; k<M; k++) {
					#pragma HLS UNROLL
						colElements[k] = colBlock[colElementStart+k];
					}

					// over boundary detect
					if (writeRowIdx<N && writeColIdx<P) {
						// ------------ inner Product Stage 1 : Parallel Mult --------------------
						MatrixType elementsMult[M];
						#pragma HLS ARRAY_PARTITION variable=elementsMult type=complete

						parallel_mult(rowElements, colElements, elementsMult, M, M);
						// ------------ inner Product Stage 2 : Summation ------------------------
						MatrixType sum = 0;
						adder_tree(elementsMult, sum, M);

						C[writeRowIdx][writeColIdx] = sum;
					}
				}

			}
		}
	}
}

void BlockMatrixMultAXI(MatrixType* A_Addr, MatrixType* B_Addr, MatrixType* C_Addr) {
// PL Master -> PS Slave
#pragma HLS INTERFACE m_axi port=A_Addr offset=slave bundle=gmem0 depth=N*M \
  max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=B_Addr offset=slave bundle=gmem1 depth=M*P \
  max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=C_Addr offset=slave bundle=gmem0 depth=N*P \
  max_write_burst_length=64 num_write_outstanding=16

// PS Master -> PL Slave
#pragma HLS INTERFACE s_axilite port=A_Addr      bundle=control
#pragma HLS INTERFACE s_axilite port=B_Addr      bundle=control
#pragma HLS INTERFACE s_axilite port=C_Addr      bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

	MatrixType A[BLOCK_N][BLOCK_M];
	MatrixType B[BLOCK_M][BLOCK_P];
	MatrixType C[N][P];

	#pragma HLS ARRAY_PARTITION variable=A type=complete dim=2
	#pragma HLS ARRAY_PARTITION variable=B type=complete dim=1
	#pragma HLS ARRAY_PARTITION variable=C type=complete dim=0

	// row major
	load_A_Addr: for (int row=0; row<BLOCK_N; row++) {
	#pragma HLS UNROLL

		int row_start = row*BLOCK_M;
		for (int col=0; col<BLOCK_M; col++) {
		#pragma HLS UNROLL
			A[row][col] = A_Addr[row_start+col];
		}
	}

	load_B_Addr: for (int row=0; row<BLOCK_M; row++) {
	#pragma HLS UNROLL
		int row_start = row*BLOCK_P;

		for (int col=0; col<BLOCK_P; col++) {
		#pragma HLS UNROLL
			B[row][col] = B_Addr[row_start + col];
		}
	}

	blockRowOP: for (int block_r=0; block_r<BLOCK_N; block_r++) {
		// load Row Block
		MatrixType rowBlock[BLOCK_M];

		#pragma HLS ARRAY_PARTITION variable=rowBlock type=complete

		for (int k=0; k<BLOCK_M; k++) {
		#pragma HLS UNROLL
			rowBlock[k] = A[block_r][k];
		}

		// Scan Column Block
		blockColOP: for (int block_c=0; block_c<BLOCK_P; block_c++) {
		#pragma HLS PIPELINE

			// load Column Block
			MatrixType colBlock[BLOCK_M];
			#pragma HLS ARRAY_PARTITION variable=colBlock type=complete

			 for (int k=0; k<BLOCK_M; k++) {
			 #pragma HLS UNROLL
			 	 colBlock[k] = B[k][block_c];
			 }

			// Block Operation : parrellel Many Row,Col inner product
			BlockOp: for (int r=0; r<BLOCKSIZE; r++) {
			#pragma HLS UNROLL

				// Out Array Row Index
				int writeRowIdx = block_r*BLOCKSIZE + r;

				// inner product use row elements start position of block
				int rowElementStart = r*M;

				// get Row elements (Wire)
				MatrixType rowElements[M];
				#pragma HLS ARRAY_PARTITION variable=rowElements type=complete

				for (int k=0; k<M; k++) {
				#pragma HLS UNROLL
					rowElements[k] = rowBlock[rowElementStart+k];
				}

				// Scan all col
				for (int c=0; c<BLOCKSIZE; c++) {
				#pragma HLS UNROLL

					// Out Array Column Index
					int writeColIdx = block_c*BLOCKSIZE + c;

					// inner product use column elements start position of block
					int colElementStart = c*M;

					// get Column elements (Wire)
					MatrixType colElements[M];
					#pragma HLS ARRAY_PARTITION variable=colElements type=complete

					for (int k=0; k<M; k++) {
					#pragma HLS UNROLL
						colElements[k] = colBlock[colElementStart+k];
					}

					// over boundary detect
					if (writeRowIdx<N && writeColIdx<P) {
						// ------------ inner Product Stage 1 : Parallel Mult --------------------
						MatrixType elementsMult[M];
						#pragma HLS ARRAY_PARTITION variable=elementsMult type=complete

						parallel_mult(rowElements, colElements, elementsMult, M, M);
						// ------------ inner Product Stage 2 : Summation ------------------------
						MatrixType sum = 0;
						adder_tree(elementsMult, sum, M);

						C[writeRowIdx][writeColIdx] = sum;
					}
				}

			}
		}
	}


	// row major
	write_C_Addr: for (int row=0; row<N; row++) {
	#pragma HLS UNROLL
		for (int col=0; col<P; col++) {
		#pragma HLS UNROLL
			C_Addr[row*P+col] = C[row][col];
		}
	}
}


/* ------------------------------------------------ Child Function ------------------------------------------------ */
void adder_tree(MatrixType A[], MatrixType &out, int length) {
#pragma HLS INLINE

	for (int k = 0; k < length; k++) {
	#pragma HLS UNROLL
        out += A[k];
    }
}

void parallel_mult(MatrixType A[], MatrixType B[], MatrixType out[], int length, int blockSize) {
#pragma HLS INLINE

	for (int start = 0; start < length; start+=blockSize) {
		int end = start+blockSize;
		end = (end>length) ? length:end;

		for (int k=start; k<end; k++){
		#pragma HLS UNROLL

			// element-wise mult
			out[k] = A[k]*B[k];
		}
	}
}
