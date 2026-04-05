#include "sparseMat.h"

// ================================================================
// Stage 1 — load_nnz
//
//  讀取 128-bit packed word，拆出兩筆 (val, col_idx)
//  打包格式：
//    [127:96]=val1  [95:64]=col1  [63:32]=val0  [31:0]=col0
//
//  奇數 NNZ 時，最後一個 word 的高 64-bit 由呼叫端填 0，
//  load_stage 設 valid1=false 通知下游不累加
// ================================================================
static void load_nnz(
    packed_nnz_t             packed_nnz_ptr[NNZ_PAIRS],
    hls::stream<nnz_pair_t> &st_nnz,
    int                      total_nnz)
{
    const int PAIRS   = (total_nnz + 1) / 2;
    const bool IS_ODD = (total_nnz % 2 != 0);

    load_nnz_loop: for (int p = 0; p < PAIRS; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=NNZ_PAIRS

        packed_nnz_t word = packed_nnz_ptr[p];

        nnz_pair_t np;
        // 低 64-bit → 第 0 筆
        ap_uint<VAL_BITS> raw_val0 = word.range(63, 32);
        np.val0 = raw_val0;
        // np.col0 = word.range(31,  0).range(LOG2_CEIL(SP_W)-1, 0);
        np.col0 = word.range(LOG2_CEIL(SP_W)-1, 0);

        // 高 64-bit → 第 1 筆
        ap_uint<VAL_BITS> raw_val1 = word.range(127, 96);
        np.val1 = raw_val1;
        // np.col1 = word.range(95, 64).range(LOG2_CEIL(SP_W)-1, 0);
        np.col1 = word.range(64+LOG2_CEIL(SP_W)-1, 64);

        // 奇數 NNZ 的最後一對：第 1 筆無效
        np.valid1 = !(IS_ODD && (p == PAIRS - 1));

        st_nnz.write(np);
    }
}

// ================================================================
// Stage 2 — load_data
//
//  將稠密矩陣 DATA 從 DRAM 搬入片上 BRAM/FF
//  pipeline II=1，搬完後供 compute 隨機存取
//  （data 需要隨機存取 col_idx 對應行，不適合用 stream）
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
//  從 stream 依序接收 nnz_pair，根據 row_offset 判斷列歸屬
//  對每個非零元素：累加 val * data[col][c]（對所有 DATA_W 列）
//
//  設計重點：
//    - localOut 按 dim=1（列）完全分割 → SP_H 個輸出可同時寫
//    - 每 clock 處理兩筆 NNZ（一個 pair），兩筆可能在同列或跨列
//    - 用全域 nnz_counter 追蹤目前處理到第幾個 NNZ，
//      配合 row_offset 判斷列邊界
// ================================================================
static void compute(
    hls::stream<nnz_pair_t>         &st_nnz,
    matType                          data_local[DATA_H][DATA_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)>   row_offset[SP_H + 1],
    hls::stream<out_elem_t>         &st_out,
    int                              total_nnz)
{
    // 片上累加器，按列完全分割（SP_H 個獨立存取埠）
    matType localOut[SP_H][DATA_W];
#pragma HLS ARRAY_PARTITION variable=localOut dim=1 complete
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
    // 預建 nnz → row 反查表（compile-time 靜態展開）
    // row_offset 在片上，查詢零延遲
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
    // 主累加迴圈：每 clock 處理一個 NNZ pair
    // 每個 pair 對 DATA_W 所有行做 MAC
    // ----------------------------------------------------------
    const int PAIRS = (total_nnz + 1) / 2;

    mac_pairs: for (int p = 0; p < PAIRS; p++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=NNZ_PAIRS

        nnz_pair_t np = st_nnz.read();

        int row0 = nnz_row[p * 2];
        int col0 = np.col0;

        // 第 0 筆 NNZ：對所有 DATA_W 行累加
        mac_cols0: for (int c = 0; c < DATA_W; c++) {
#pragma HLS UNROLL
            localOut[row0][c] += np.val0 * data_local[col0][c];
        }

        // 第 1 筆 NNZ（需 valid1）：對所有 DATA_W 行累加
        if (np.valid1) {
            int row1 = nnz_row[p * 2 + 1];
            int col1 = np.col1;
            mac_cols1: for (int c = 0; c < DATA_W; c++) {
#pragma HLS UNROLL
                localOut[row1][c] += np.val1 * data_local[col1][c];
            }
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
//
//  注意：load_data 和 load_row_offset 不產生 stream，
//        直接寫入片上陣列後供 compute 使用。
//        DATAFLOW 仍然有效：load_nnz 與 load_data 同時進行，
//        compute 一收到第一個 nnz_pair 就可以開始累加。
// ================================================================
void csr_gemm(
    matType      data_ptr[DATA_H * DATA_W],
    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_ptr[SP_H + 1],
    packed_nnz_t packed_nnz_ptr[NNZ_PAIRS],
    matType      out_ptr[SP_H * DATA_W])
{
// --- AXI 介面：各 port 用不同 bundle 避免頻寬競爭 ---
#pragma HLS INTERFACE m_axi port=data_ptr        bundle=gmem0 depth=DATA_H*DATA_W
#pragma HLS INTERFACE m_axi port=row_offset_ptr  bundle=gmem1 depth=SP_H+1
#pragma HLS INTERFACE m_axi port=packed_nnz_ptr  bundle=gmem2 depth=NNZ_PAIRS
#pragma HLS INTERFACE m_axi port=out_ptr         bundle=gmem3 depth=SP_H*DATA_W
#pragma HLS INTERFACE s_axilite port=return      bundle=control

#pragma HLS DATAFLOW

    // 片上緩衝（load_data / load_row_offset 寫入，compute 讀取）
    matType data_local[DATA_H][DATA_W];
#pragma HLS ARRAY_PARTITION variable=data_local dim=1 complete

    ap_uint<LOG2_CEIL(SP_MAX_NNZ)> row_offset_local[SP_H + 1];
#pragma HLS ARRAY_PARTITION variable=row_offset_local complete

    // NNZ stream（load_nnz → compute）
    hls::stream<nnz_pair_t>  st_nnz("st_nnz");
    hls::stream<out_elem_t>  st_out("st_out");
#pragma HLS STREAM variable=st_nnz depth=NNZ_PAIRS
#pragma HLS STREAM variable=st_out depth=SP_H*DATA_W

    const int total_nnz = SP_NNZ_PER_ROW * SP_H;  // 已知為固定值

    // 五個 stage 同時執行
    load_data      (data_ptr,       data_local);
    load_row_offset(row_offset_ptr, row_offset_local);
    load_nnz       (packed_nnz_ptr, st_nnz, total_nnz);
    compute        (st_nnz, data_local, row_offset_local, st_out, total_nnz);
    store          (st_out, out_ptr);
}