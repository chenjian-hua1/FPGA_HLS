#include "dft.h"
#include <ap_int.h>
#include <iostream>

constexpr int UNROLL_FACTOR = 4;

void dft(DTYPE real_samples[COEFF_SIZE],
         DTYPE imag_samples[COEFF_SIZE],
         SUM_DTYPE real_outs[COEFF_SIZE],
         SUM_DTYPE imag_outs[COEFF_SIZE])
{
    #pragma HLS INTERFACE mode=s_axilite port=return
    // PL -> PS  write data to address (base on s_axilite)
    #pragma HLS INTERFACE mode=m_axi bundle=A port=real_samples offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=B port=imag_samples offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=C port=real_outs    offset=slave
    #pragma HLS INTERFACE mode=m_axi bundle=D port=imag_outs    offset=slave
    // PS -> PL  data address
    #pragma HLS INTERFACE mode=s_axilite port=real_samples
    #pragma HLS INTERFACE mode=s_axilite port=imag_samples
    #pragma HLS INTERFACE mode=s_axilite port=real_outs 
    #pragma HLS INTERFACE mode=s_axilite port=imag_outs

    // ---------- ROM 表 ----------
    #pragma HLS BIND_STORAGE variable=SIN_TABLE type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=COS_TABLE type=ROM_1P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=SIN_TABLE cyclic factor=UNROLL_FACTOR dim=1
    #pragma HLS ARRAY_PARTITION variable=COS_TABLE cyclic factor=UNROLL_FACTOR dim=1

    // ---------- 內部累加器（重點改動）----------
    SUM_DTYPE local_re[COEFF_SIZE];
    SUM_DTYPE local_im[COEFF_SIZE];
    #pragma HLS BIND_STORAGE variable=local_re type=RAM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=local_im type=RAM_2P impl=BRAM
    #pragma HLS ARRAY_PARTITION variable=local_re cyclic factor=1 dim=1
    #pragma HLS ARRAY_PARTITION variable=local_im cyclic factor=1 dim=1

    // 同時把輸入 sample 也搬到本地，避免內迴圈每輪都走 AXI
    DTYPE local_re_in[COEFF_SIZE];
    DTYPE local_im_in[COEFF_SIZE];
    #pragma HLS BIND_STORAGE variable=local_re_in type=RAM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=local_im_in type=RAM_1P impl=BRAM

    // ---------- 1. 從 m_axi 讀進輸入（burst）----------
    // log2ceil(COEFF_SIZE) bits 只能表示 0 ~ log2ceil(COEFF_SIZE)-1
    load_input_loop: for (ap_uint<LOG2_CEIL(COEFF_SIZE)> n = 0; n < COEFF_SIZE; n++) {
        #pragma HLS PIPELINE II=1
        
        local_re_in[n] = real_samples[n];
        local_im_in[n] = imag_samples[n];
    }

    // ---------- 2. 累加器清零 ----------
    init_out_loop: for (ap_uint<LOG2_CEIL(COEFF_SIZE)> k = 0; k < COEFF_SIZE; k++) {
        #pragma HLS PIPELINE II=1
        local_re[k] = 0;
        local_im[k] = 0;
    }

    // ---------- 3. 主計算迴圈 ----------
    outer_n_loop: for (ap_uint<LOG2_CEIL(COEFF_SIZE+1)> n = 0; n < COEFF_SIZE; n++) {
        DTYPE real = local_re_in[n];
        DTYPE imag = local_im_in[n];

        inner_k_loop: for (ap_uint<LOG2_CEIL(COEFF_SIZE)> k = 0; k < COEFF_SIZE; k++) {
            // #pragma HLS PIPELINE II=4

            // TODO

            

        }
    }

    // ---------- 4. 寫回 m_axi（burst）----------
    write_back_loop: for (ap_uint<LOG2_CEIL(COEFF_SIZE+1)> k = 0; k < COEFF_SIZE; k++) {
        #pragma HLS PIPELINE II=1
        real_outs[k] = local_re[k];
        imag_outs[k] = local_im[k];
    }
}
