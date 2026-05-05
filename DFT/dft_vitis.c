/*
 * main.c : DFT HLS IP test application (Vitis / Bare-metal)
 *
 * 對應 HLS 端設定（來自 table.h / dft.h）：
 *   COEFF_SIZE      = 256
 *   DTYPE           = ap_fixed<16, 2>     -> Q2.14 ，PS 端用 int16_t
 *   SUM_DTYPE       = ap_fixed<39, 12>    -> Q12.27，PS 端用 int64_t
 *
 * HLS IP 介面：m_axi master
 *   driver 的 Set_*_samples / Set_*_outs 傳入「實體記憶體位址」
 *   IP 透過 AXI 主動讀寫 DDR
 *
 * Cache 處理：
 *   採用最簡單的「關閉 D-Cache」做法。如果你想保留 cache 提高效能，
 *   就改用 Xil_DCacheFlushRange(輸入位址, 大小) / InvalidateRange(輸出位址, 大小)。
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xdft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ============================================================
 *  IP 參數（必須與 HLS 端 dft.h / table.h 完全一致）
 * ============================================================ */
#define COEFF_SIZE      256

/* DTYPE = ap_fixed<16, 2>  -> Q2.14 */
#define TOTAL_BITS      16
#define INT_BITS        2
#define DEC_BITS        (TOTAL_BITS - INT_BITS)   /* = 14 */

/* SUM_DTYPE = ap_fixed<MULT_TOTAL_BITS + log2(N), MULT_INT_BITS + log2(N)>
 *           = ap_fixed<31 + 8, 4 + 8>
 *           = ap_fixed<39, 12>  -> Q12.27
 * 在 m_axi 上會被 pack 成 64-bit element，因此 PS 端用 int64_t 陣列。
 */
#define MULT_TOTAL_BITS ((TOTAL_BITS - 1) * 2 + 1)        /* = 31 */
#define MULT_INT_BITS   (INT_BITS * 2)                    /* = 4  */
#define SUM_EXTRA_BITS  8                                 /* log2(256) */
#define SUM_TOTAL_BITS  (MULT_TOTAL_BITS + SUM_EXTRA_BITS)/* = 39 */
#define SUM_INT_BITS    (MULT_INT_BITS   + SUM_EXTRA_BITS)/* = 12 */
#define SUM_DEC_BITS    (SUM_TOTAL_BITS  - SUM_INT_BITS)  /* = 27 */

/* PS 端定點型別 */
typedef int16_t DTYPE_T;        /* 存 Q2.14         */
typedef int64_t SUM_DTYPE_T;    /* 存 Q12.27 (39b 用 64b 容納) */

/* float <-> 定點 轉換 */
static inline DTYPE_T float_to_dtype(float v)
{
    /* Q2.14：範圍 [-2, 2)，飽和處理 */
    float scaled = v * (float)(1 << DEC_BITS);
    if (scaled >  32767.0f) scaled =  32767.0f;
    if (scaled < -32768.0f) scaled = -32768.0f;
    return (DTYPE_T)(scaled >= 0 ? (scaled + 0.5f) : (scaled - 0.5f));
}

static inline float sum_dtype_to_float(SUM_DTYPE_T v)
{
    /* Q12.27 -> float：除以 2^27 */
    return (float)((double)v / (double)(1LL << SUM_DEC_BITS));
}

/* ============================================================
 *  輸入 / 輸出緩衝區
 *  - HLS IP 透過 m_axi 直接存取，必須是合法位址
 *  - 64-byte 對齊有助於 AXI burst 效能（非必要，但建議）
 *  - 已關閉 D-Cache，PS 寫入後 PL 可立即看到
 * ============================================================ */
static DTYPE_T     real_in [COEFF_SIZE] __attribute__((aligned(64)));
static DTYPE_T     imag_in [COEFF_SIZE] __attribute__((aligned(64)));
static SUM_DTYPE_T real_out[COEFF_SIZE] __attribute__((aligned(64)));
static SUM_DTYPE_T imag_out[COEFF_SIZE] __attribute__((aligned(64)));

/* float 參考 DFT 用 */
static float real_f [COEFF_SIZE];
static float imag_f [COEFF_SIZE];
static float ref_re [COEFF_SIZE];
static float ref_im [COEFF_SIZE];

/* IP 實例 */
static XDft Dft;

/* ============================================================
 *  Float 參考 DFT（與 HLS 公式一致）
 *    real_outs[k] += real*cos + imag*sin
 *    imag_outs[k] += imag*cos - real*sin
 * ============================================================ */
static void dft_float_ref(const float real_samples[COEFF_SIZE],
                          const float imag_samples[COEFF_SIZE],
                          float real_outs[COEFF_SIZE],
                          float imag_outs[COEFF_SIZE])
{
    static float cos_tbl[COEFF_SIZE];
    static float sin_tbl[COEFF_SIZE];
    static int initialized = 0;
    int i, k, n;

    if (!initialized) {
        for (i = 0; i < COEFF_SIZE; i++) {
            double theta = 2.0 * M_PI * (double)i / (double)COEFF_SIZE;
            cos_tbl[i] = (float)cos(theta);
            sin_tbl[i] = (float)sin(theta);
        }
        initialized = 1;
    }

    for (k = 0; k < COEFF_SIZE; k++) {
        real_outs[k] = 0.0f;
        imag_outs[k] = 0.0f;
    }

    for (n = 0; n < COEFF_SIZE; n++) {
        float r  = real_samples[n];
        float im = imag_samples[n];
        for (k = 0; k < COEFF_SIZE; k++) {
            int index = (k * n) & (COEFF_SIZE - 1);
            float c = cos_tbl[index];
            float s = sin_tbl[index];
            real_outs[k] += (r  * c + im * s);
            imag_outs[k] += (im * c - r  * s);
        }
    }
}

/* ============================================================
 *  訊號產生
 * ============================================================ */
static void gen_signal_sine(int freq_bin, float amplitude)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        double v = (double)amplitude *
                   cos(2.0 * M_PI * (double)freq_bin * (double)n / (double)COEFF_SIZE);
        real_f [n] = (float)v;
        imag_f [n] = 0.0f;
        real_in[n] = float_to_dtype((float)v);
        imag_in[n] = 0;
    }
}

/* 簡單 LCG：固定 seed 結果可重現，不依賴 stdlib rand 的實作差異 */
static uint32_t g_rand_state = 42;
static void my_srand(uint32_t s) { g_rand_state = s; }
static uint32_t my_rand(void)
{
    g_rand_state = g_rand_state * 1103515245u + 12345u;
    return (g_rand_state >> 1) & 0x7FFFFFFF;
}
#define MY_RAND_MAX 0x7FFFFFFF

static void gen_signal_random(float range)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        float r = ((float)my_rand() / (float)MY_RAND_MAX * 2.0f - 1.0f) * range;
        float i = ((float)my_rand() / (float)MY_RAND_MAX * 2.0f - 1.0f) * range;
        real_f [n] = r;
        imag_f [n] = i;
        real_in[n] = float_to_dtype(r);
        imag_in[n] = float_to_dtype(i);
    }
}

static void gen_signal_dc(float value)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        real_f [n] = value;
        imag_f [n] = 0.0f;
        real_in[n] = float_to_dtype(value);
        imag_in[n] = 0;
    }
}

static void gen_signal_impulse(float value)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        real_f [n] = 0.0f;
        imag_f [n] = 0.0f;
        real_in[n] = 0;
        imag_in[n] = 0;
    }
    real_f [0] = value;
    real_in[0] = float_to_dtype(value);
}

/* ============================================================
 *  IP 控制：設位址 -> Start -> 等 Done
 * ============================================================ */
static void run_dft_on_ip(void)
{
    int k;

    /* 清空輸出 buffer（IP 是累加，必須先歸零） */
    for (k = 0; k < COEFF_SIZE; k++) {
        real_out[k] = 0;
        imag_out[k] = 0;
    }

    /* 把陣列實體位址傳給 IP（m_axi 介面） */
    XDft_Set_real_samples(&Dft, (u64)(UINTPTR)real_in);
    XDft_Set_imag_samples(&Dft, (u64)(UINTPTR)imag_in);
    XDft_Set_real_outs   (&Dft, (u64)(UINTPTR)real_out);
    XDft_Set_imag_outs   (&Dft, (u64)(UINTPTR)imag_out);

    /* 啟動 IP */
    XDft_Start(&Dft);

    /* Polling 等待完成 */
    while (!XDft_IsDone(&Dft)) {
        /* busy wait */
    }

    /* 清中斷旗標 */
    XDft_InterruptClear(&Dft, 1);
}

/* ============================================================
 *  比較 IP 結果與 float 參考
 * ============================================================ */
typedef struct {
    double max_abs_err;
    double rmse;
    int    max_err_index;
} ErrorStats;

static ErrorStats compare_results(const SUM_DTYPE_T fx_re[COEFF_SIZE],
                                  const SUM_DTYPE_T fx_im[COEFF_SIZE],
                                  const float ref_re_[COEFF_SIZE],
                                  const float ref_im_[COEFF_SIZE])
{
    ErrorStats st;
    double sum_sq = 0.0;
    int k;

    st.max_abs_err   = 0.0;
    st.rmse          = 0.0;
    st.max_err_index = -1;

    for (k = 0; k < COEFF_SIZE; k++) {
        double er = (double)sum_dtype_to_float(fx_re[k]) - (double)ref_re_[k];
        double ei = (double)sum_dtype_to_float(fx_im[k]) - (double)ref_im_[k];
        double mag = sqrt(er * er + ei * ei);
        if (mag > st.max_abs_err) {
            st.max_abs_err   = mag;
            st.max_err_index = k;
        }
        sum_sq += er * er + ei * ei;
    }
    st.rmse = sqrt(sum_sq / (double)COEFF_SIZE);
    return st;
}

/* xil_printf 不支援 %f：把 double 拆成符號 + 整數 + 4 位小數 */
static void split_float(double v, int *sign, int *int_part, int *frac4)
{
    double a;
    *sign = (v < 0.0) ? -1 : 1;
    a = (v < 0.0) ? -v : v;
    *int_part = (int)a;
    *frac4    = (int)((a - (double)(*int_part)) * 10000.0 + 0.5);
    if (*frac4 >= 10000) {  /* 進位修正 */
        *frac4 = 0;
        (*int_part)++;
    }
}

static void print_float(double v)
{
    int sign, ip, fp;
    split_float(v, &sign, &ip, &fp);
    if (sign < 0) xil_printf("-%d.%04d", ip, fp);
    else          xil_printf( "%d.%04d", ip, fp);
}

/* ============================================================
 *  單個測試
 * ============================================================ */
static int run_one_test(const char *name, double tolerance, int dump)
{
    ErrorStats st;
    int pass;

    /* 跑 IP */
    run_dft_on_ip();

    /* 跑 float 參考 */
    dft_float_ref(real_f, imag_f, ref_re, ref_im);

    /* 比較 */
    st = compare_results(real_out, imag_out, ref_re, ref_im);
    pass = (st.max_abs_err <= tolerance);

    xil_printf("[%s] max_err=", name);
    print_float(st.max_abs_err);
    xil_printf(" (k=%d), RMSE=", st.max_err_index);
    print_float(st.rmse);
    xil_printf(", tol=");
    print_float(tolerance);
    xil_printf("  %s\r\n", pass ? "PASS" : "FAIL");

    if (dump) {
        int n_dump = (COEFF_SIZE < 8) ? COEFF_SIZE : 8;
        int k;
        xil_printf("    k |    ip_real      ip_imag    |   ref_real     ref_imag\r\n");
        for (k = 0; k < n_dump; k++) {
            xil_printf("  %3d | ", k);
            print_float((double)sum_dtype_to_float(real_out[k]));
            xil_printf("   ");
            print_float((double)sum_dtype_to_float(imag_out[k]));
            xil_printf("  | ");
            print_float((double)ref_re[k]);
            xil_printf("   ");
            print_float((double)ref_im[k]);
            xil_printf("\r\n");
        }
    }
    return pass;
}

/* ============================================================
 *  main
 * ============================================================ */
int main(void)
{
    int Status;
    int pass = 0, total = 0;
    int t;
    double tol;
    XDft_Config *CfgPtr;

    /* 1. 平台初始化 */
    init_platform();

    /* 2. 關閉 Data Cache（IP 透過 m_axi 直接讀寫 DDR） */
    Xil_DCacheDisable();

    xil_printf("\r\n=== DFT HLS IP Test (N=%d) ===\r\n", COEFF_SIZE);
    xil_printf("Data Cache has been disabled.\r\n");

    /* 3. 初始化 IP */
    CfgPtr = XDft_LookupConfig(XPAR_DFT_0_DEVICE_ID);
    if (!CfgPtr) {
        xil_printf("ERROR: XDft_LookupConfig failed.\r\n");
        cleanup_platform();
        return XST_FAILURE;
    }
    Status = XDft_CfgInitialize(&Dft, CfgPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: XDft_CfgInitialize failed (Status=%d).\r\n", Status);
        cleanup_platform();
        return XST_FAILURE;
    }
    xil_printf("XDft initialized.\r\n");

    /* 4. 印出參數資訊 */
    xil_printf("DTYPE     = ap_fixed<%d,%d>  (Q%d.%d)\r\n",
               TOTAL_BITS, INT_BITS, INT_BITS, DEC_BITS);
    xil_printf("SUM_DTYPE = ap_fixed<%d,%d>  (Q%d.%d), stored as int64\r\n",
               SUM_TOTAL_BITS, SUM_INT_BITS, SUM_INT_BITS, SUM_DEC_BITS);
    xil_printf("Buffer addr: real_in=0x%08x imag_in=0x%08x real_out=0x%08x imag_out=0x%08x\r\n",
               (unsigned)(UINTPTR)real_in,
               (unsigned)(UINTPTR)imag_in,
               (unsigned)(UINTPTR)real_out,
               (unsigned)(UINTPTR)imag_out);

    /* 5. 容忍度
     *    DTYPE 量化步長 = 2^-14 ≈ 6.1e-5
     *    最壞情況 N 次累加 ≈ N * 步長 = 256 * 6.1e-5 ≈ 0.0156
     *    再放寬 2 倍給 cos/sin 表量化，並至少 0.1
     */
    tol = (double)COEFF_SIZE * pow(2.0, -(double)DEC_BITS) * 2.0;
    if (tol < 0.1) tol = 0.1;
    xil_printf("Tolerance = ");
    print_float(tol);
    xil_printf("\r\n\r\n");

    /* ---------- Test 1: 純 cos，bin=4，amp=0.5 ---------- */
    gen_signal_sine(4, 0.5f);
    if (run_one_test("Cos bin=4 amp=0.5", tol, 1)) pass++;
    total++;

    /* ---------- Test 2: 高頻 cos，bin=N/4 ---------- */
    gen_signal_sine(COEFF_SIZE / 4, 0.5f);
    if (run_one_test("Cos bin=N/4", tol, 0)) pass++;
    total++;

    /* ---------- Test 3: DC=0.5 ---------- */
    gen_signal_dc(0.5f);
    if (run_one_test("DC=0.5", tol, 0)) pass++;
    total++;

    /* ---------- Test 4: 5 組 random ---------- */
    my_srand(42);
    for (t = 0; t < 5; t++) {
        char name[16];
        gen_signal_random(0.5f);
        name[0]='R'; name[1]='a'; name[2]='n'; name[3]='d';
        name[4]='o'; name[5]='m'; name[6]=' ';
        name[7]='0' + t; name[8]='\0';
        if (run_one_test(name, tol, 0)) pass++;
        total++;
    }

    /* ---------- Test 5: Impulse ---------- */
    gen_signal_impulse(0.9f);
    if (run_one_test("Impulse", tol, 0)) pass++;
    total++;

    /* ---------- 總結 ---------- */
    xil_printf("\r\n=== %d / %d passed ===\r\n", pass, total);

    cleanup_platform();
    return (pass == total) ? 0 : 1;
}