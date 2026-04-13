/*
 * main.c: Test application for csr_gemm HLS IP
 *
 * Sparse Matrix (CSR) × Dense Matrix GEMM accelerator
 * SP (5×5, 2 NNZ/row) × DATA (5×5) → OUT (5×5)
 *
 * 依賴：
 *   xcsr_gemm.h      — driver function 宣告
 *   xcsr_gemm_hw.h   — register map (需由 Vitis HLS export 產生)
 *
 * 注意：原 sparseMat.h 的所有參數已直接展開於此檔，不再 include。
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"

#include "xcsr_gemm.h"   /* driver */

/* ================================================================
 * 矩陣尺寸參數（來自 sparseMat.h）
 * ================================================================ */
#define SP_H            5   /* 稀疏矩陣列數 */
#define SP_W            5   /* 稀疏矩陣行數 */
#define DATA_H          5   /* 稠密矩陣列數（= SP_W） */
#define DATA_W          5   /* 稠密矩陣行數 */
#define SP_NNZ_PER_ROW  2   /* 每列固定非零數 */
#define SP_MAX_NNZ      (SP_NNZ_PER_ROW * SP_H)   /* = 10 */

/* ================================================================
 * NNZ packed word 參數
 *
 *   VAL_BITS  = 32          (matType = ap_int<32> → int32_t)
 *   COL_BITS  = LOG2_CEIL(SP_W=5) = 3
 *   PACK_BITS = 35          → 以 uint64_t 儲存
 *
 *   打包格式：
 *     [34:3] = value   (32-bit)
 *     [ 2:0] = col_idx ( 3-bit)
 *
 *   NNZ_WORDS = SP_MAX_NNZ = 10
 * ================================================================ */
#define VAL_BITS   32
#define COL_BITS   3        /* LOG2_CEIL(5) */
#define PACK_BITS  (VAL_BITS + COL_BITS)   /* = 35 */
#define NNZ_WORDS  SP_MAX_NNZ              /* = 10 */

typedef uint64_t packed_nnz_t;   /* 35-bit payload，用 uint64_t 儲存 */

/* ================================================================
 * row_offset packed word 參數
 *
 *   RO_BITS      = LOG2_CEIL(SP_MAX_NNZ+1) = LOG2_CEIL(11) = 4
 *   RO_PACK_BITS = 2 * RO_BITS = 8  → 以 uint16_t 儲存
 *
 *   打包格式（每個 word 含相鄰兩個 row_offset 元素）：
 *     [7:4] = row_offset[r+1]  (高位)
 *     [3:0] = row_offset[r]    (低位)
 *
 *   RO_WORDS = SP_H = 5（row 0 ~ row 4 各一個 word）
 *
 *   !! 型態寬度必須與 HLS 一致 !!
 *   HLS: ap_uint<8> = 1 byte/element。
 *   C 端必須用 uint8_t，不可用 uint16_t。
 *   若用 uint16_t，AXI m_axi 按 1 byte 步進讀取時，
 *   會讀入元素間的補零位元組，row1/row3 的 packed word
 *   變成 0x00 → (rs=0,re=0) 空範圍 → 對應 NNZ 無列命中
 *   → nnz_row[] 殘留初始值 0 → 全部錯誤累加進 row0。
 * ================================================================ */
#define RO_BITS       4     /* LOG2_CEIL(SP_MAX_NNZ+1 = 11) */
#define RO_PACK_BITS  (2 * RO_BITS)   /* = 8 */
#define RO_WORDS      SP_H            /* = 5 */

typedef uint8_t packed_ro_t;     /* 必須與 HLS ap_uint<8> 同為 1 byte！
                                  * 若用 uint16_t，AXI 每次讀 1 byte 時
                                  * 會讀到補零位元組，造成 row1/row3 的
                                  * row_offset 範圍變成 (0,0) 空範圍，
                                  * NNZ 累加到錯誤的列。                 */

/* ================================================================
 * 固定稀疏矩陣（SP_H=5, SP_W=5, SP_NNZ_PER_ROW=2）
 *
 *   row 0: (col=1, val=3), (col=4, val=7)
 *   row 1: (col=0, val=5), (col=3, val=2)
 *   row 2: (col=2, val=8), (col=4, val=1)
 *   row 3: (col=0, val=6), (col=1, val=4)
 *   row 4: (col=2, val=9), (col=3, val=3)
 * ================================================================ */
static int32_t SP_MAT[SP_H][SP_W] = {
    { 0, 3, 0, 0, 7 },
    { 5, 0, 0, 2, 0 },
    { 0, 0, 8, 0, 1 },
    { 6, 4, 0, 0, 0 },
    { 0, 0, 9, 3, 0 }
};

/* ================================================================
 * 固定稠密矩陣（DATA_H=5, DATA_W=5）
 * ================================================================ */
static int32_t DATA_MAT[DATA_H][DATA_W] = {
    {  1,  2,  3,  4,  5 },
    {  6,  7,  8,  9, 10 },
    { 11, 12, 13, 14, 15 },
    { 16, 17, 18, 19, 20 },
    { 21, 22, 23, 24, 25 }
};

/* ================================================================
 * 打包工具
 *
 * pack_nnz：將 (val, col_idx) 打包成 packed_nnz_t
 *   [COL_BITS+VAL_BITS-1 : COL_BITS] = val（32-bit）
 *   [COL_BITS-1          :         0] = col_idx（COL_BITS-bit）
 *
 * pack_ro：將相鄰兩個 row_offset 打包成 packed_ro_t
 *   高位 [2*RO_BITS-1 : RO_BITS] = hi（= row_offset[r+1]）
 *   低位 [RO_BITS-1   :       0] = lo（= row_offset[r]）
 * ================================================================ */

static packed_nnz_t pack_nnz(int32_t val, int col)
{
    packed_nnz_t w = 0;
    /* val 放高位 */
    w  = ((packed_nnz_t)(uint32_t)val) << COL_BITS;
    /* col_idx 放低位 */
    w |= (packed_nnz_t)(col & ((1 << COL_BITS) - 1));
    return w;
}

static packed_ro_t pack_ro(int lo, int hi)
{
    packed_ro_t w = 0;
    w  = (packed_ro_t)(lo & ((1 << RO_BITS) - 1));
    w |= (packed_ro_t)((hi & ((1 << RO_BITS) - 1)) << RO_BITS);
    return w;
}

/* ================================================================
 * mat2csr_packed
 *   稀疏矩陣 → packed NNZ array + packed row_offset array
 * ================================================================ */
static void mat2csr_packed(
    int32_t        sp_mat[SP_H][SP_W],
    packed_ro_t    packed_ro[RO_WORDS],
    packed_nnz_t   packed_nnz[NNZ_WORDS])
{
    int row_offset[SP_H + 1];
    int nnz = 0;

    row_offset[0] = 0;
    for (int r = 0; r < SP_H; r++) {
        for (int c = 0; c < SP_W; c++) {
            if (sp_mat[r][c] != 0) {
                packed_nnz[nnz] = pack_nnz(sp_mat[r][c], c);
                nnz++;
            }
        }
        row_offset[r + 1] = nnz;
    }

    /* 相鄰兩個 row_offset 打包：packed_ro[r] = (offset[r], offset[r+1]) */
    for (int r = 0; r < SP_H; r++) {
        packed_ro[r] = pack_ro(row_offset[r], row_offset[r + 1]);
    }
}

/* ================================================================
 * gemm_sw：軟體黃金值（SP × DATA → OUT）
 * ================================================================ */
static void gemm_sw(
    int32_t sp[SP_H][SP_W],
    int32_t data[DATA_H][DATA_W],
    int32_t res[SP_H][DATA_W])
{
    for (int r = 0; r < SP_H; r++) {
        for (int c = 0; c < DATA_W; c++) {
            int32_t s = 0;
            for (int k = 0; k < SP_W; k++)
                s += sp[r][k] * data[k][c];
            res[r][c] = s;
        }
    }
}

/* ================================================================
 * 列印工具
 * ================================================================ */
static void print_matrix(const int32_t *data, int rows, int cols,
                         const char *label)
{
    xil_printf("\r\n%s:\r\n", label);
    for (int r = 0; r < rows; r++) {
        xil_printf("  [");
        for (int c = 0; c < cols; c++) {
            xil_printf(" %5d", (int)data[r * cols + c]);
        }
        xil_printf(" ]\r\n");
    }
}

/* ================================================================
 * Main
 * ================================================================ */
int main(void)
{
    int Status;

    /* ── 1. 初始化平台 ──────────────────────────────────────── */
    init_platform();
    Xil_DCacheDisable();
    xil_printf("\r\n=== csr_gemm HLS IP Test ===\r\n");
    xil_printf("Data cache disabled.\r\n");

    /* ── 2. 初始化 IP Driver ────────────────────────────────── */
    XCsr_gemm           CsrGemmInst;
    XCsr_gemm_Config   *ConfigPtr;

    ConfigPtr = XCsr_gemm_LookupConfig(XPAR_CSR_GEMM_0_DEVICE_ID);
    if (!ConfigPtr) {
        xil_printf("ERROR: LookupConfig failed.\r\n");
        return XST_FAILURE;
    }

    Status = XCsr_gemm_CfgInitialize(&CsrGemmInst, ConfigPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: CfgInitialize failed.\r\n");
        return XST_FAILURE;
    }
    xil_printf("IP driver initialized OK.\r\n");

    /* ── 3. 準備輸入資料 ────────────────────────────────────── */

    /* 3-a. DATA 矩陣（稠密，flatten to 1-D） */
    static int32_t       data_flat[DATA_H * DATA_W];
    for (int r = 0; r < DATA_H; r++)
        for (int c = 0; c < DATA_W; c++)
            data_flat[r * DATA_W + c] = DATA_MAT[r][c];

    /* 3-b. 稀疏矩陣 → packed NNZ + packed row_offset */
    static packed_ro_t   packed_ro [RO_WORDS];
    static packed_nnz_t  packed_nnz[NNZ_WORDS];
    mat2csr_packed(SP_MAT, packed_ro, packed_nnz);

    /* 3-c. 輸出緩衝區 */
    static int32_t hw_result[SP_H * DATA_W];
    memset(hw_result, 0, sizeof(hw_result));

    /* ── 4. 設定 AXI-Lite 位址暫存器 ───────────────────────── */
    /*
     * 注意：以下 cast 假設系統為 32-bit 位址空間（Zynq PS/PL）。
     * 若使用 64-bit 平台，請改用 64-bit 版本的 Set_xxx API。
     *
     * 各 port 對應的 bundle（gmem0~gmem3）需已在 Vivado 中
     * 連接至同一個或不同的 HP/ACP slave port。
     */
    XCsr_gemm_Set_data_ptr      (&CsrGemmInst, (u32)(uintptr_t)data_flat);
    XCsr_gemm_Set_packed_ro_ptr (&CsrGemmInst, (u32)(uintptr_t)packed_ro);
    XCsr_gemm_Set_packed_nnz_ptr(&CsrGemmInst, (u32)(uintptr_t)packed_nnz);
    XCsr_gemm_Set_out_ptr       (&CsrGemmInst, (u32)(uintptr_t)hw_result);
    xil_printf("AXI-Lite address registers set.\r\n");

    /* ── 5. 確保 D-Cache 一致性後啟動 IP ───────────────────── */
    /*
     * 若 D-Cache 已關閉（上方 Xil_DCacheDisable），此步驟可省略。
     * 若 D-Cache 保持開啟，務必在啟動前 flush 輸入、
     * 在讀回結果前 invalidate 輸出。
     *
     * Xil_DCacheFlushRange((INTPTR)data_flat,   sizeof(data_flat));
     * Xil_DCacheFlushRange((INTPTR)packed_ro,   sizeof(packed_ro));
     * Xil_DCacheFlushRange((INTPTR)packed_nnz,  sizeof(packed_nnz));
     * Xil_DCacheFlushRange((INTPTR)hw_result,   sizeof(hw_result));
     */

    xil_printf("Starting IP...\r\n");
    XCsr_gemm_Start(&CsrGemmInst);

    /* ── 6. 輪詢等待完成（Polling，無中斷） ────────────────── */
    while (!XCsr_gemm_IsDone(&CsrGemmInst))
        ;   /* busy-wait */

    xil_printf("IP done.\r\n");

    /* 若使用中斷，可改為：
     *   XCsr_gemm_InterruptEnable (&CsrGemmInst, 1);
     *   XCsr_gemm_InterruptGlobalEnable(&CsrGemmInst);
     *   ... 等待中斷 ISR 設旗標 ...
     *   XCsr_gemm_InterruptClear  (&CsrGemmInst, 1);
     */

    /* ── 7. 軟體黃金值 ──────────────────────────────────────── */
    static int32_t sw_result[SP_H][DATA_W];
    gemm_sw(SP_MAT, DATA_MAT, sw_result);

    /* ── 8. 驗證 ────────────────────────────────────────────── */
    int correct = 1;
    for (int r = 0; r < SP_H; r++) {
        for (int c = 0; c < DATA_W; c++) {
            int32_t hw_val = hw_result[r * DATA_W + c];
            int32_t sw_val = sw_result[r][c];
            if (hw_val != sw_val) {
                xil_printf("MISMATCH [%d][%d]: HW=%d SW=%d\r\n",
                           r, c, (int)hw_val, (int)sw_val);
                correct = 0;
            }
        }
    }

    /* ── 9. 列印結果 ────────────────────────────────────────── */
    print_matrix((int32_t *)sw_result, SP_H, DATA_W, "SW golden");
    print_matrix(hw_result,            SP_H, DATA_W, "HW result");

    xil_printf("\r\n>>> %s <<<\r\n\r\n", correct ? "PASSED" : "FAILED");

    /* ── 10. 清除中斷旗標（好習慣，即使未開中斷） ──────────── */
    XCsr_gemm_InterruptClear(&CsrGemmInst, 1);

    cleanup_platform();
    return correct ? 0 : 1;
}
