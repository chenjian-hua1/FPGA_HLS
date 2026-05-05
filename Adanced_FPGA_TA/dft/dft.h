/*
Reference :
https://github.com/bol-edu/2022-fall-ntu/tree/main
https://github.com/kthohr/gcem/tree/master
*/

#ifndef DFT_H
#define DFT_H

#include <hls_stream.h>
#include <ap_fixed.h>
#include <array>
#include "table.h"

constexpr int LOG2_CEIL(int x) {
    // 定義域：x >= 1
    // ceil(log2(1)) = 0, ceil(log2(2)) = 1, ceil(log2(3)) = 2, ...
    int r = 0;
    int p = 1;
    // 直到 2^r 超過x
    while (p < x) {
        p*=2;
        ++r;
    }
    return r;
}

// TOTAL_BITS have one sign bit so -1   
// 2^(TOTAL_BITS-1)**2 = 2^((TOTAL_BITS-1)*2)
// add one sign bit
constexpr int MULT_TOTAL_BITS = (TOTAL_BITS-1)*2+1;

// depend bits of dec point  
// if INT_BITS==1 (only decimal) : MULT_INT_BITS = 1 
// else : MULT_INT_BITS = INT_BITS*2   ( 2^INT_BITS*2^INT_BITS = 2^(INT_BITS+INT_BITS) )
constexpr int MULT_INT_BITS = (INT_BITS==1)?(1):(INT_BITS*2);

// mult out data type
typedef ap_fixed<MULT_TOTAL_BITS, MULT_INT_BITS> MULT_DTYPE; // 只有小數跟正負號

// ((x1+x2) + (x3+x4)) + ... (x(n-1)+xn)
// add 2^n times -> generate extra bits log2(n)
constexpr int SUM_EXTRA_BITS = LOG2_CEIL(COEFF_SIZE);

// data_bits+floor(log2(add_times))
constexpr int SUM_TOTAL_BITS = MULT_TOTAL_BITS+SUM_EXTRA_BITS;

// DATA_INT_BITS+log2(add_times)
constexpr int SUM_INT_BITS = MULT_INT_BITS+SUM_EXTRA_BITS;

// sum out data type
typedef ap_fixed<SUM_TOTAL_BITS, SUM_INT_BITS> SUM_DTYPE;


/**
 * @brief Discrete Fourier Transform (DFT) IP core.
 *
 * Computes the N-point DFT of a complex input sequence using direct
 * matrix-vector multiplication form:
 *     X(k) = sum_{n=0}^{N-1} x(n) * exp(-j*2*pi*k*n/N),  k = 0..N-1
 *
 * The twiddle factors cos(2*pi*k*n/N) and sin(2*pi*k*n/N) are read from
 * precomputed lookup tables indexed by (k*n) mod N, exploiting the
 * periodicity of the complex exponential.
 *
 * @param[in]  real_samples  Real parts of the time-domain input,  length N.
 * @param[in]  imag_samples  Imag parts of the time-domain input,  length N.
 * @param[out] real_outs     Real parts of the frequency-domain output, length N.
 * @param[out] imag_outs     Imag parts of the frequency-domain output, length N.
 *
 * @note  N is defined by the macro SIZE (default 256) and must be a power of 2
 *        so that (k*n) mod N can be implemented as a bitwise AND.
 * @note  Computational complexity is O(N^2). For larger N, use the FFT IP
 *        instead.
 * @note  Input arrays are not modified; results are written to separate output
 *        arrays. real_outs / imag_outs are expected to be zero-initialized
 *        by the caller (or accumulation will be incorrect).
 */
void dft(DTYPE real_samples[COEFF_SIZE], DTYPE imag_samples[COEFF_SIZE], SUM_DTYPE real_outs[COEFF_SIZE], SUM_DTYPE imag_outs[COEFF_SIZE]);

#endif // DFT_H