#include "dft.h"
#include <iostream>

constexpr int UNROLL_FACTOR = 4;

void dft(DTYPE real_samples[COEFF_SIZE], DTYPE imag_samples[COEFF_SIZE], SUM_DTYPE real_outs[COEFF_SIZE], SUM_DTYPE imag_outs[COEFF_SIZE])
{
    #pragma HLS INTERFACE mode=s_axilite port=return
    #pragma HLS INTERFACE mode=m_axi bundle=A port=real_samples
    #pragma HLS INTERFACE mode=m_axi bundle=B port=imag_samples
    #pragma HLS INTERFACE mode=m_axi bundle=C port=real_outs
    #pragma HLS INTERFACE mode=m_axi bundle=D port=imag_outs

    #pragma HLS BIND_STORAGE variable=SIN_TABLE type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=COS_TABLE type=ROM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=SIN_TABLE cyclic factor=UNROLL_FACTOR dim=1
    #pragma HLS ARRAY_PARTITION variable=COS_TABLE cyclic factor=UNROLL_FACTOR dim=1

    // constexpr auto SIN_TABLE = make_sin_table<SIZE>();
    // constexpr auto COS_TABLE = make_cos_table<SIZE>();

    // init_table_loop: for (int i = 0; i < SIZE; i++) {
    //     #pragma HLS PIPELINE II=1
    //     SIN_TABLE[i] = (DTYPE)SIN_TABLE_D[i];
    //     COS_TABLE[i] = (DTYPE)COS_TABLE_D[i];
    // }

    // ---------- 輸出陣列 partition（讓多個元素同時寫）----------
    #pragma HLS ARRAY_PARTITION variable=real_outs cyclic factor=UNROLL_FACTOR dim=1
    #pragma HLS ARRAY_PARTITION variable=imag_outs cyclic factor=UNROLL_FACTOR dim=1

    // ---------- 初始化輸出累加器 ----------
    init_out_loop: for (int k = 0; k < COEFF_SIZE; k++) {
        #pragma HLS UNROLL // factor=UNROLL_FACTOR
        real_outs[k] = 0;
        imag_outs[k] = 0;
    }

    // ---------- 主迴圈 ----------
    outer_n_loop: for (int n = 0; n < COEFF_SIZE; n++) {
        DTYPE real = real_samples[n];
        DTYPE imag = imag_samples[n];

        inner_k_loop: for (int k = 0; k < COEFF_SIZE; k++) {
            #pragma HLS UNROLL // factor=UNROLL_FACTOR

            int index = (k * n) & (COEFF_SIZE - 1);  // = (k*n) % 256

            DTYPE c = COS_TABLE[index];
            DTYPE s = SIN_TABLE[index];

            DTYPE rc = real * c;
            DTYPE is = imag * s;
            DTYPE rs = real * s;
            DTYPE ic = imag * c;

            real_outs[k] += (rc + is);
            imag_outs[k] += (rs - ic);
        }
    }
}

int main() {
    DTYPE real_samples[COEFF_SIZE]; 
    DTYPE imag_samples[COEFF_SIZE]; 
    SUM_DTYPE real_outs[COEFF_SIZE];
    SUM_DTYPE imag_outs[COEFF_SIZE];

    dft(real_samples, imag_samples, real_outs, imag_outs);
    std::cout << "success" << std::endl;
    return 0;
}
