#include "sparseMat.h"

// ================================================================
// Stage 1 — load_nnz
//
//  讀取 64-bit packed word，拆出單筆 (val, col_idx)
//  打包格式：
//    [COL_BITS+VAL_BITS-1:COL_BITS] = val
//    [COL_BITS-1: 0] = col_idx
//
//  每個 NNZ 對應一個 64-bit word，共 NNZ_WORDS 個
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
        ap_uint<VAL_BITS> raw_val = word.range(COL_BITS + VAL_BITS - 1, COL_BITS);
        ne.val = raw_val;
        ne.col = word.range(COL_BITS - 1, 0);

        st_nnz.write(ne);
    }
}

// ================================================================
// Stage 2 — load_data
//
//  將稠密矩陣 DATA 從 DRAM 搬入片上 BRAM/FF
//  pipeline II=1，搬完後供 compute 隨機存取
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
//  讀取 CSR row_offset 陣列至片上
// ================================================================
static void load_row_offset(
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_ptr[SP_H + 1],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_local[SP_H + 1])
{
    load_row_off: for (int i = 0; i <= SP_H; i++) {
    #pragma HLS PIPELINE II=1
        row_offset_local[i] = row_offset_ptr[i];
    }
}

// ================================================================
// Stage 4 — compute
//
//  從 stream 依序接收 nnz_elem，根據 row_offset 判斷列歸屬
//  對每個非零元素：累加 val * data[col][c]（對所有 DATA_W 行）
//
//  設計重點：
//    - localOut 按 dim=1（列）完全分割 → SP_H 個輸出可同時寫
//    - 每 clock 處理一筆 NNZ
//    - 用全域 nnz_counter 追蹤目前處理到第幾個 NNZ，
//      配合 row_offset 判斷列邊界
// ================================================================
static void compute(
    hls::stream<nnz_elem_t>         &st_nnz,
    matType                          data_local[DATA_H][DATA_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)>   row_offset[SP_H + 1],
    hls::stream<out_elem_t>         &st_out,
    int                              total_nnz)
{
    // 片上累加器，按列完全分割（SP_H 個獨立存取埠）
    matType localOut[SP_H][DATA_W];
    #pragma HLS ARRAY_PARTITION variable=localOut    dim=1 complete
    #pragma HLS ARRAY_PARTITION variable=data_local  dim=1 complete

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
    // ----------------------------------------------------------
    int nnz_row[SP_MAX_NNZ];
    #pragma HLS ARRAY_PARTITION variable=nnz_row complete

    build_row_map: for (int r = 0; r < SP_H; r++) {
        int rs = row_offset[r];
        int re = row_offset[r + 1];
        for (int k = rs; k < re; k++) {
        #pragma HLS PIPELINE II=1
            if (k < SP_MAX_NNZ) nnz_row[k] = r;
        }
    }

    // ----------------------------------------------------------
    // 主累加迴圈：每 clock 處理一筆 NNZ
    // ----------------------------------------------------------
    mac_loop: for (int i = 0; i < total_nnz; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=1 max=NNZ_WORDS

        nnz_elem_t ne = st_nnz.read();

        int row = nnz_row[i];
        int col = ne.col;

        mac_cols: for (int c = 0; c < DATA_W; c++) {
        #pragma HLS UNROLL
            localOut[row][c] += ne.val * data_local[col][c];
        }
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
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_ptr[SP_H + 1],
    packed_nnz_t packed_nnz_ptr[NNZ_WORDS],
    matType      out_ptr[SP_H * DATA_W])
{
// --- AXI 介面：各 port 用不同 bundle 避免頻寬競爭 ---
#pragma HLS INTERFACE m_axi port=data_ptr        bundle=gmem0 depth=DATA_H*DATA_W
#pragma HLS INTERFACE m_axi port=row_offset_ptr  bundle=gmem1 depth=SP_H+1
#pragma HLS INTERFACE m_axi port=packed_nnz_ptr  bundle=gmem2 depth=NNZ_WORDS
#pragma HLS INTERFACE m_axi port=out_ptr         bundle=gmem3 depth=SP_H*DATA_W
#pragma HLS INTERFACE s_axilite port=return      bundle=control

#pragma HLS DATAFLOW

    // 片上緩衝（load_data / load_row_offset 寫入，compute 讀取）
    matType data_local[DATA_H][DATA_W];
    #pragma HLS ARRAY_PARTITION variable=data_local dim=1 complete

    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_local[SP_H + 1];
    #pragma HLS ARRAY_PARTITION variable=row_offset_local complete

    // NNZ stream（load_nnz → compute）
    hls::stream<nnz_elem_t>  st_nnz("st_nnz");
    hls::stream<out_elem_t>  st_out("st_out");
    #pragma HLS STREAM variable=st_nnz depth=NNZ_WORDS
    #pragma HLS STREAM variable=st_out depth=SP_H*DATA_W

    const int total_nnz = SP_NNZ_PER_ROW * SP_H;  // 已知為固定值

    // 五個 stage 同時執行
    load_data      (data_ptr,       data_local);
    load_row_offset(row_offset_ptr, row_offset_local);
    load_nnz       (packed_nnz_ptr, st_nnz, total_nnz);
    compute        (st_nnz, data_local, row_offset_local, st_out, total_nnz);
    store          (st_out, out_ptr);
}