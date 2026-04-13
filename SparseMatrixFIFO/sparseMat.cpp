#include "sparseMat.h"

// ================================================================
// Stage 1 — load_nnz
//
//  讀取 packed word，拆出單筆 (val, col_idx)
//  打包格式：
//    [COL_BITS+VAL_BITS-1:COL_BITS] = val
//    [COL_BITS-1: 0] = col_idx
// ================================================================
static void load_nnz(
    packed_nnz_t              packed_nnz_ptr[NNZ_WORDS],
    hls::stream<nnz_elem_t>  &st_nnz,
    int                       total_nnz)
{
    load_nnz_loop: for (int i = 0; i < total_nnz; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=NNZ_WORDS

        packed_nnz_t word = packed_nnz_ptr[i];

        nnz_elem_t ne;
        // ap_uint<VAL_BITS> raw_val = word.range(COL_BITS + VAL_BITS - 1, COL_BITS);
        ap_uint<VAL_BITS> raw_val = word.range(COL_BITS + VAL_BITS-1, COL_BITS-1);

        ne.val = raw_val;
        // ne.col = word.range(COL_BITS - 1, 0);
        ne.col = word.range(COL_BITS,0);

        st_nnz.write(ne);
    }
}

// ================================================================
// Stage 2 — load_data
//
//  將稠密矩陣 DATA 從 DRAM 搬入片上 BRAM/FF
// ================================================================
static void load_data(
    matType  data_ptr[DATA_H * DATA_W],
    matType  data_local[DATA_H][DATA_W])
{
    load_data_r: for (int r = 0; r < DATA_H; r++) {
        load_data_c: for (int c = 0; c < DATA_W; c++) {
        #pragma HLS PIPELINE II=1
            data_local[r][c] = data_ptr[r * DATA_W + c];
        }
    }
}

// ================================================================
// Stage 3 — load_row_offset
//
//  讀取 packed row_offset 陣列至片上
//
//  每個 packed_ro_t word 儲存相鄰兩個 row_offset 元素：
//    高位 [2*RO_BITS-1 : RO_BITS] = row_offset[r+1]
//    低位 [RO_BITS-1   :       0] = row_offset[r]
//
//  compute 端讀取 ro_local[r] 即可同一 clock 取得
//  rs = offset[r] 與 re = offset[r+1]
// ================================================================
static void load_row_offset(
    packed_ro_t  packed_ro_ptr[RO_WORDS],
    packed_ro_t  ro_local[RO_WORDS])
{
    load_row_off: for (int i = 0; i < RO_WORDS; i++) {
    #pragma HLS PIPELINE II=1
        ro_local[i] = packed_ro_ptr[i];
    }
}

// ================================================================
// Stage 4 — compute
//
//  從 stream 依序接收 nnz_elem，根據 ro_local 判斷列歸屬
//
//  每次迭代從 ro_local[r] 單一讀取即得 rs、re，
//  不需要兩次存取，消除相鄰 index 的讀取衝突
// ================================================================
static void compute(
    hls::stream<nnz_elem_t>  &st_nnz,
    matType                   data_local[DATA_H][DATA_W],
    packed_ro_t               ro_local[RO_WORDS],
    hls::stream<out_elem_t>  &st_out,
    int                       total_nnz)
{
    // 片上累加器，按列完全分割（SP_H 個獨立存取埠）
    matType localOut[SP_H][DATA_W];
    #pragma HLS ARRAY_PARTITION variable=localOut   dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=data_local dim=1 complete

    // 初始化
    init_r: for (int r = 0; r < SP_H; r++) {
    #pragma HLS UNROLL
        init_c: for (int c = 0; c < DATA_W; c++) {
        #pragma HLS UNROLL
            localOut[r][c] = 0;
        }
    }

    // ----------------------------------------------------------
    // 預建 nnz → row 反查表
    //
    // 外層迴圈走所有可能的 NNZ index（固定邊界 SP_MAX_NNZ），
    // 內層迴圈走所有 row（固定邊界 SP_H），兩層皆為靜態 trip count。
    //
    // 對每個 NNZ index i，從 ro_local[r] 同一 clock 取得
    // rs = offset[r] 與 re = offset[r+1]，判斷 rs <= i < re
    // 即可確定 i 屬於 row r，寫入 nnz_row[i]。
    //
    // 消除動態邊界迴圈，HLS 可正確展開並合成，cosim 行為一致。
    // ----------------------------------------------------------
    int nnz_row[SP_MAX_NNZ];
    #pragma HLS ARRAY_PARTITION variable=nnz_row complete

    build_row_map: for (int i = 0; i < SP_MAX_NNZ; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=SP_MAX_NNZ max=SP_MAX_NNZ

        build_row_inner: for (int r = 0; r < SP_H; r++) {
        #pragma HLS UNROLL

            // 一次讀取 packed word，同一 clock 取得 rs 與 re
            packed_ro_t word = ro_local[r];

            // int rs = (int)(ap_uint<RO_BITS>)word.range(RO_BITS - 1,         0);
            // int re = (int)(ap_uint<RO_BITS>)word.range(2 * RO_BITS - 1, RO_BITS);

            // if (rs <= i && i < re) nnz_row[i] = r;

            int rs = (int)(ap_uint<RO_BITS>)word.range(2*RO_BITS-1, RO_BITS);
            int re = (int)(ap_uint<RO_BITS>)word.range(RO_BITS-1, 0);

            nnz_row[i] = r;
        }
    }

    // ----------------------------------------------------------
    // 主累加迴圈：每 clock 處理一筆 NNZ
    // ----------------------------------------------------------
    mac_loop: for (int i = 0; i < total_nnz; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=NNZ_WORDS

        nnz_elem_t ne = st_nnz.read();

        // ----------------- TODO -------------------------------



    }

    // ----------------------------------------------------------
    // 結果排出到 stream（row-major）
    // ----------------------------------------------------------
    drain_r: for (int r = 0; r < SP_H; r++) {
        drain_c: for (int c = 0; c < DATA_W; c++) {
        #pragma HLS PIPELINE II=1
            out_elem_t e;
            e.val = localOut[r][c];
            e.row = r;
            e.col = c;
            st_out.write(e);
        }
    }
}

// ================================================================
// Stage 5 — store
//
//  從 stream 接收結果，按 (row,col) 寫回 DRAM
// ================================================================
static void store(
    hls::stream<out_elem_t> &st_out,
    matType                  out_ptr[SP_H * DATA_W])
{
    store_loop: for (int i = 0; i < SP_H * DATA_W; i++) {
    #pragma HLS PIPELINE II=1
        out_elem_t e = st_out.read();
        out_ptr[e.row * DATA_W + e.col] = e.val;
    }
}

// ================================================================
// Top-level：DATAFLOW 串接所有 stage
// ================================================================
void csr_gemm(
    matType      data_ptr[DATA_H * DATA_W],
    packed_ro_t  packed_ro_ptr[RO_WORDS],
    packed_nnz_t packed_nnz_ptr[NNZ_WORDS],
    matType      out_ptr[SP_H * DATA_W])
{
// --- AXI 介面：各 port 用不同 bundle 避免頻寬競爭 ---
#pragma HLS INTERFACE m_axi port=data_ptr        bundle=gmem0 depth=DATA_H*DATA_W
#pragma HLS INTERFACE m_axi port=packed_ro_ptr   bundle=gmem1 depth=RO_WORDS
#pragma HLS INTERFACE m_axi port=packed_nnz_ptr  bundle=gmem2 depth=NNZ_WORDS
#pragma HLS INTERFACE m_axi port=out_ptr         bundle=gmem3 depth=SP_H*DATA_W
#pragma HLS INTERFACE s_axilite port=return      bundle=control

#pragma HLS DATAFLOW

    // 片上緩衝
    matType     data_local[DATA_H][DATA_W];
    #pragma HLS ARRAY_PARTITION variable=data_local dim=1 complete

    packed_ro_t ro_local[RO_WORDS];
    #pragma HLS ARRAY_PARTITION variable=ro_local complete

    // stream
    hls::stream<nnz_elem_t>  st_nnz("st_nnz");
    hls::stream<out_elem_t>  st_out("st_out");
    #pragma HLS STREAM variable=st_nnz depth=NNZ_WORDS
    #pragma HLS STREAM variable=st_out depth=SP_H*DATA_W

    const int total_nnz = SP_NNZ_PER_ROW * SP_H;

    // 五個 stage 同時執行
    load_data      (data_ptr,        data_local);
    load_row_offset(packed_ro_ptr,   ro_local);
    load_nnz       (packed_nnz_ptr,  st_nnz, total_nnz);
    compute        (st_nnz, data_local, ro_local, st_out, total_nnz);
    store          (st_out, out_ptr);
}
