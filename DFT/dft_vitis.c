/*
 * main.c : DFT HLS IP test application (Vitis / Bare-metal)
 *
 * 對應 HLS 端設定：
 *   COEFF_SIZE = 256
 *   DTYPE      = ap_fixed<16, 2>   -> Q2.14 ，PS 端用 int16_t
 *   SUM_DTYPE  = ap_fixed<39, 12>  -> Q12.27，PS 端用 int64_t
 *
 * ★ 關鍵：ap_fixed<39, 12> 在 m_axi 上會被 pack 到 64-bit element
 *   但 IP 只寫 bit 0~38，bit 39~63 不會被觸碰（維持 buffer 原值，通常是 0）
 *   所以 PS 端讀回後必須對「bit 38」做 sign extension，才能還原負數
 *
 * 不依賴 libm（不需要 -lm）：cos/sin 直接用整數表，sqrt 用整數 Newton 法
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xdft.h"

/* ============================================================
 *  IP 參數
 * ============================================================ */
#define COEFF_SIZE      256

#define TOTAL_BITS      16
#define INT_BITS        2
#define DEC_BITS        (TOTAL_BITS - INT_BITS)   /* = 14 */
#define ONE_Q14         (1 << DEC_BITS)           /* = 16384 */

#define MULT_TOTAL_BITS ((TOTAL_BITS - 1) * 2 + 1)         /* = 31 */
#define MULT_INT_BITS   (INT_BITS * 2)                     /* = 4  */
#define SUM_EXTRA_BITS  8                                  /* log2(256) */
#define SUM_TOTAL_BITS  (MULT_TOTAL_BITS + SUM_EXTRA_BITS) /* = 39 */
#define SUM_INT_BITS    (MULT_INT_BITS   + SUM_EXTRA_BITS) /* = 12 */
#define SUM_DEC_BITS    (SUM_TOTAL_BITS  - SUM_INT_BITS)   /* = 27 */

typedef int16_t DTYPE_T;
typedef int64_t SUM_DTYPE_T;

/* ============================================================
 *  ★★★ 關鍵修正：39-bit sign extension ★★★
 *
 *  ap_fixed<39,12> 在 m_axi 上被 pack 到 64-bit slot：
 *    bit 0~38  = 有效資料（39-bit 兩補數）
 *    bit 39~63 = IP 不寫，維持 buffer 原值
 *
 *  PS 寫入時若初始化為 0，bit 39~63 會是 0。
 *  IP 寫入負數時（bit 38=1），PS 讀回會誤以為是「巨大正數」。
 *  必須手動把 bit 38 廣播到 bit 39~63 來還原 64-bit signed 表示。
 * ============================================================ */
static inline int64_t sign_extend_39(int64_t v)
{
    const int64_t MASK39  = ((int64_t)1 << SUM_TOTAL_BITS) - 1;     /* bit 0~38 */
    const int64_t SIGNBIT = (int64_t)1 << (SUM_TOTAL_BITS - 1);     /* bit 38 */
    int64_t x = v & MASK39;
    if (x & SIGNBIT) {
        x |= ~MASK39;   /* 高位填 1 */
    }
    return x;
}

/* ============================================================
 *  cos/sin Q2.14 整數表（與 HLS table.h 對齊）
 * ============================================================ */
static const int16_t COS_TABLE_INT[COEFF_SIZE] = {
    16383,16379,16364,16340,16305,16261,16207,16143,
    16069,15986,15893,15791,15679,15557,15426,15286,
    15137,14979,14811,14635,14449,14256,14053,13842,
    13623,13395,13160,12916,12665,12406,12140,11866,
    11585,11297,11003,10702,10394,10080, 9760, 9434,
     9102, 8765, 8423, 8076, 7723, 7366, 7005, 6639,
     6270, 5897, 5520, 5139, 4756, 4370, 3981, 3590,
     3196, 2801, 2404, 2006, 1606, 1205,  804,  402,
        0, -402, -804,-1205,-1606,-2006,-2404,-2801,
    -3196,-3590,-3981,-4370,-4756,-5139,-5520,-5897,
    -6270,-6639,-7005,-7366,-7723,-8076,-8423,-8765,
    -9102,-9434,-9760,-10080,-10394,-10702,-11003,-11297,
   -11585,-11866,-12140,-12406,-12665,-12916,-13160,-13395,
   -13623,-13842,-14053,-14256,-14449,-14635,-14811,-14979,
   -15137,-15286,-15426,-15557,-15679,-15791,-15893,-15986,
   -16069,-16143,-16207,-16261,-16305,-16340,-16364,-16379,
   -16383,-16379,-16364,-16340,-16305,-16261,-16207,-16143,
   -16069,-15986,-15893,-15791,-15679,-15557,-15426,-15286,
   -15137,-14979,-14811,-14635,-14449,-14256,-14053,-13842,
   -13623,-13395,-13160,-12916,-12665,-12406,-12140,-11866,
   -11585,-11297,-11003,-10702,-10394,-10080, -9760, -9434,
    -9102, -8765, -8423, -8076, -7723, -7366, -7005, -6639,
    -6270, -5897, -5520, -5139, -4756, -4370, -3981, -3590,
    -3196, -2801, -2404, -2006, -1606, -1205,  -804,  -402,
        0,   402,   804,  1205,  1606,  2006,  2404,  2801,
     3196,  3590,  3981,  4370,  4756,  5139,  5520,  5897,
     6270,  6639,  7005,  7366,  7723,  8076,  8423,  8765,
     9102,  9434,  9760, 10080, 10394, 10702, 11003, 11297,
    11585, 11866, 12140, 12406, 12665, 12916, 13160, 13395,
    13623, 13842, 14053, 14256, 14449, 14635, 14811, 14979,
    15137, 15286, 15426, 15557, 15679, 15791, 15893, 15986,
    16069, 16143, 16207, 16261, 16305, 16340, 16364, 16379
};

static const int16_t SIN_TABLE_INT[COEFF_SIZE] = {
        0,  402,  804, 1205, 1606, 2006, 2404, 2801,
     3196, 3590, 3981, 4370, 4756, 5139, 5520, 5897,
     6270, 6639, 7005, 7366, 7723, 8076, 8423, 8765,
     9102, 9434, 9760,10080,10394,10702,11003,11297,
    11585,11866,12140,12406,12665,12916,13160,13395,
    13623,13842,14053,14256,14449,14635,14811,14979,
    15137,15286,15426,15557,15679,15791,15893,15986,
    16069,16143,16207,16261,16305,16340,16364,16379,
    16383,16379,16364,16340,16305,16261,16207,16143,
    16069,15986,15893,15791,15679,15557,15426,15286,
    15137,14979,14811,14635,14449,14256,14053,13842,
    13623,13395,13160,12916,12665,12406,12140,11866,
    11585,11297,11003,10702,10394,10080, 9760, 9434,
     9102, 8765, 8423, 8076, 7723, 7366, 7005, 6639,
     6270, 5897, 5520, 5139, 4756, 4370, 3981, 3590,
     3196, 2801, 2404, 2006, 1606, 1205,  804,  402,
        0, -402, -804,-1205,-1606,-2006,-2404,-2801,
    -3196,-3590,-3981,-4370,-4756,-5139,-5520,-5897,
    -6270,-6639,-7005,-7366,-7723,-8076,-8423,-8765,
    -9102,-9434,-9760,-10080,-10394,-10702,-11003,-11297,
   -11585,-11866,-12140,-12406,-12665,-12916,-13160,-13395,
   -13623,-13842,-14053,-14256,-14449,-14635,-14811,-14979,
   -15137,-15286,-15426,-15557,-15679,-15791,-15893,-15986,
   -16069,-16143,-16207,-16261,-16305,-16340,-16364,-16379,
   -16383,-16379,-16364,-16340,-16305,-16261,-16207,-16143,
   -16069,-15986,-15893,-15791,-15679,-15557,-15426,-15286,
   -15137,-14979,-14811,-14635,-14449,-14256,-14053,-13842,
   -13623,-13395,-13160,-12916,-12665,-12406,-12140,-11866,
   -11585,-11297,-11003,-10702,-10394,-10080, -9760, -9434,
    -9102, -8765, -8423, -8076, -7723, -7366, -7005, -6639,
    -6270, -5897, -5520, -5139, -4756, -4370, -3981, -3590,
    -3196, -2801, -2404, -2006, -1606, -1205,  -804,  -402
};

/* ============================================================
 *  整數開根號（floor sqrt）
 * ============================================================ */
static uint64_t isqrt_u64(uint64_t x)
{
    uint64_t r, t;
    if (x < 2) return x;
    r = 1; t = x;
    while (t > 1) { t >>= 2; r <<= 1; }
    for (;;) {
        uint64_t nr = (r + x / r) >> 1;
        if (nr >= r) return r;
        r = nr;
    }
}

/* ============================================================
 *  輸入 / 輸出緩衝區
 * ============================================================ */
static DTYPE_T     real_in [COEFF_SIZE] __attribute__((aligned(64)));
static DTYPE_T     imag_in [COEFF_SIZE] __attribute__((aligned(64)));
static SUM_DTYPE_T real_out[COEFF_SIZE] __attribute__((aligned(64)));
static SUM_DTYPE_T imag_out[COEFF_SIZE] __attribute__((aligned(64)));

static int32_t real_in_q14 [COEFF_SIZE];
static int32_t imag_in_q14 [COEFF_SIZE];
static int64_t ref_re_q27  [COEFF_SIZE];
static int64_t ref_im_q27  [COEFF_SIZE];

static XDft Dft;

/* ============================================================
 *  整數參考 DFT，輸出 Q12.27
 * ============================================================ */
static void dft_int_ref(const int32_t real_samples[COEFF_SIZE],
                        const int32_t imag_samples[COEFF_SIZE],
                        int64_t real_outs[COEFF_SIZE],
                        int64_t imag_outs[COEFF_SIZE])
{
    int k, n;
    for (k = 0; k < COEFF_SIZE; k++) {
        real_outs[k] = 0;
        imag_outs[k] = 0;
    }
    for (n = 0; n < COEFF_SIZE; n++) {
        int32_t r  = real_samples[n];
        int32_t im = imag_samples[n];
        for (k = 0; k < COEFF_SIZE; k++) {
            int idx = (k * n) & (COEFF_SIZE - 1);
            int32_t c = COS_TABLE_INT[idx];
            int32_t s = SIN_TABLE_INT[idx];
            int64_t pr = (int64_t)r  * c + (int64_t)im * s;  /* Q4.28 */
            int64_t pi = (int64_t)im * c - (int64_t)r  * s;
            real_outs[k] += pr;
            imag_outs[k] += pi;
        }
    }
    /* Q4.28 -> Q12.27 (右移 1) */
    for (k = 0; k < COEFF_SIZE; k++) {
        real_outs[k] = real_outs[k] / 2;
        imag_outs[k] = imag_outs[k] / 2;
    }
}

/* ============================================================
 *  訊號產生
 * ============================================================ */
static void gen_signal_sine(int freq_bin, int amp_q14)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        int idx = (freq_bin * n) & (COEFF_SIZE - 1);
        int64_t v = ((int64_t)amp_q14 * (int64_t)COS_TABLE_INT[idx]) >> DEC_BITS;
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        real_in[n]     = (DTYPE_T)v;
        imag_in[n]     = 0;
        real_in_q14[n] = (int32_t)v;
        imag_in_q14[n] = 0;
    }
}

static void gen_signal_dc(int value_q14)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        real_in[n]     = (DTYPE_T)value_q14;
        imag_in[n]     = 0;
        real_in_q14[n] = value_q14;
        imag_in_q14[n] = 0;
    }
}

static void gen_signal_impulse(int value_q14)
{
    int n;
    for (n = 0; n < COEFF_SIZE; n++) {
        real_in[n]     = 0;
        imag_in[n]     = 0;
        real_in_q14[n] = 0;
        imag_in_q14[n] = 0;
    }
    real_in[0]     = (DTYPE_T)value_q14;
    real_in_q14[0] = value_q14;
}

static uint32_t g_rand_state = 42;
static void my_srand(uint32_t s) { g_rand_state = s; }
static uint32_t my_rand(void)
{
    g_rand_state = g_rand_state * 1103515245u + 12345u;
    return (g_rand_state >> 1) & 0x7FFFFFFF;
}

static void gen_signal_random(int range_q14)
{
    int n;
    int span = 2 * range_q14;
    for (n = 0; n < COEFF_SIZE; n++) {
        int r = (int)(my_rand() % (uint32_t)span) - range_q14;
        int i = (int)(my_rand() % (uint32_t)span) - range_q14;
        real_in[n]     = (DTYPE_T)r;
        imag_in[n]     = (DTYPE_T)i;
        real_in_q14[n] = r;
        imag_in_q14[n] = i;
    }
}

/* ============================================================
 *  IP 控制
 * ============================================================ */
static void run_dft_on_ip(void)
{
    int k;
    for (k = 0; k < COEFF_SIZE; k++) {
        real_out[k] = 0;
        imag_out[k] = 0;
    }
    XDft_Set_real_samples(&Dft, (u64)(UINTPTR)real_in);
    XDft_Set_imag_samples(&Dft, (u64)(UINTPTR)imag_in);
    XDft_Set_real_outs   (&Dft, (u64)(UINTPTR)real_out);
    XDft_Set_imag_outs   (&Dft, (u64)(UINTPTR)imag_out);
    XDft_Start(&Dft);
    while (!XDft_IsDone(&Dft)) { /* busy wait */ }
    XDft_InterruptClear(&Dft, 1);
}

/* ============================================================
 *  比較（讀取 IP 結果時做 39-bit sign extension）
 * ============================================================ */
typedef struct {
    int64_t  max_err_q27;
    int64_t  rmse_q27;
    int      max_err_index;
} ErrorStats;

static ErrorStats compare_results(const SUM_DTYPE_T fx_re[COEFF_SIZE],
                                  const SUM_DTYPE_T fx_im[COEFF_SIZE],
                                  const int64_t     ref_re_[COEFF_SIZE],
                                  const int64_t     ref_im_[COEFF_SIZE])
{
    ErrorStats st;
    const int SHIFT = 8;
    uint64_t sum_sq = 0;
    int k;

    st.max_err_q27   = 0;
    st.rmse_q27      = 0;
    st.max_err_index = -1;

    for (k = 0; k < COEFF_SIZE; k++) {
        /* ★ sign extension：把 IP 寫入的 39-bit 值還原成正確的 64-bit signed */
        int64_t ip_re = sign_extend_39((int64_t)fx_re[k]);
        int64_t ip_im = sign_extend_39((int64_t)fx_im[k]);

        int64_t er = ip_re - ref_re_[k];
        int64_t ei = ip_im - ref_im_[k];

        /* 縮小避免 (er^2 + ei^2) 溢位 int64 */
        int64_t er_s = er >> SHIFT;
        int64_t ei_s = ei >> SHIFT;
        uint64_t sq = (uint64_t)(er_s * er_s + ei_s * ei_s);
        sum_sq += sq;

        uint64_t mag_sq_s = (uint64_t)(er_s * er_s + ei_s * ei_s);
        uint64_t mag_s    = isqrt_u64(mag_sq_s);
        int64_t  mag      = (int64_t)mag_s << SHIFT;
        if (mag > st.max_err_q27) {
            st.max_err_q27   = mag;
            st.max_err_index = k;
        }
    }
    {
        uint64_t mean = sum_sq / COEFF_SIZE;
        uint64_t rmse_scaled = isqrt_u64(mean);
        st.rmse_q27 = (int64_t)(rmse_scaled << SHIFT);
    }
    return st;
}

/* ============================================================
 *  Q12.27 -> 顯示
 * ============================================================ */
static void print_q27(int64_t v)
{
    int neg = (v < 0);
    uint64_t a = neg ? (uint64_t)(-v) : (uint64_t)v;
    uint64_t int_part  = a >> SUM_DEC_BITS;
    uint64_t frac_part = a & (((uint64_t)1 << SUM_DEC_BITS) - 1);
    uint64_t frac4 = (frac_part * 10000ULL) >> SUM_DEC_BITS;
    if (neg && (int_part != 0 || frac4 != 0))
        xil_printf("-%u.%04u", (unsigned)int_part, (unsigned)frac4);
    else
        xil_printf( "%u.%04u", (unsigned)int_part, (unsigned)frac4);
}

/* ============================================================
 *  單個測試
 * ============================================================ */
static int run_one_test(const char *name, int64_t tol_q27, int dump)
{
    ErrorStats st;
    int pass;

    run_dft_on_ip();
    dft_int_ref(real_in_q14, imag_in_q14, ref_re_q27, ref_im_q27);

    st = compare_results(real_out, imag_out, ref_re_q27, ref_im_q27);
    pass = (st.max_err_q27 <= tol_q27);

    xil_printf("[%s] max_err=", name);
    print_q27(st.max_err_q27);
    xil_printf(" (k=%d), RMSE=", st.max_err_index);
    print_q27(st.rmse_q27);
    xil_printf(", tol=");
    print_q27(tol_q27);
    xil_printf("  %s\r\n", pass ? "PASS" : "FAIL");

    if (dump) {
        int n_dump = (COEFF_SIZE < 8) ? COEFF_SIZE : 8;
        int k;
        xil_printf("    k |    ip_real      ip_imag    |   ref_real     ref_imag\r\n");
        for (k = 0; k < n_dump; k++) {
            int64_t ip_re = sign_extend_39((int64_t)real_out[k]);
            int64_t ip_im = sign_extend_39((int64_t)imag_out[k]);
            xil_printf("  %3d | ", k);
            print_q27(ip_re);
            xil_printf("   ");
            print_q27(ip_im);
            xil_printf("  | ");
            print_q27(ref_re_q27[k]);
            xil_printf("   ");
            print_q27(ref_im_q27[k]);
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
    int64_t tol_q27;
    XDft_Config *CfgPtr;

    init_platform();
    Xil_DCacheDisable();

    xil_printf("\r\n=== DFT HLS IP Test (N=%d) ===\r\n", COEFF_SIZE);
    xil_printf("Data Cache has been disabled.\r\n");

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

    xil_printf("DTYPE     = ap_fixed<%d,%d>  (Q%d.%d)\r\n",
               TOTAL_BITS, INT_BITS, INT_BITS, DEC_BITS);
    xil_printf("SUM_DTYPE = ap_fixed<%d,%d>  (Q%d.%d), 39b in 64b slot (sign-ext on read)\r\n",
               SUM_TOTAL_BITS, SUM_INT_BITS, SUM_INT_BITS, SUM_DEC_BITS);
    xil_printf("Buffer addr: real_in=0x%08x imag_in=0x%08x real_out=0x%08x imag_out=0x%08x\r\n",
               (unsigned)(UINTPTR)real_in,
               (unsigned)(UINTPTR)imag_in,
               (unsigned)(UINTPTR)real_out,
               (unsigned)(UINTPTR)imag_out);

    tol_q27 = ((int64_t)1 << SUM_DEC_BITS) / 2;   /* 0.5 */
    xil_printf("Tolerance = ");
    print_q27(tol_q27);
    xil_printf("\r\n\r\n");

    gen_signal_sine(4, ONE_Q14 / 2);
    if (run_one_test("Cos bin=4 amp=0.5", tol_q27, 1)) pass++;
    total++;

    gen_signal_sine(COEFF_SIZE / 4, ONE_Q14 / 2);
    if (run_one_test("Cos bin=N/4", tol_q27, 0)) pass++;
    total++;

    gen_signal_dc(ONE_Q14 / 2);
    if (run_one_test("DC=0.5", tol_q27, 0)) pass++;
    total++;

    my_srand(42);
    for (t = 0; t < 5; t++) {
        char name[16];
        gen_signal_random(ONE_Q14 / 2);
        name[0]='R'; name[1]='a'; name[2]='n'; name[3]='d';
        name[4]='o'; name[5]='m'; name[6]=' ';
        name[7]='0' + t; name[8]='\0';
        if (run_one_test(name, tol_q27, 0)) pass++;
        total++;
    }

    gen_signal_impulse((int)(0.9f * ONE_Q14 + 0.5f));
    if (run_one_test("Impulse", tol_q27, 0)) pass++;
    total++;

    xil_printf("\r\n=== %d / %d passed ===\r\n", pass, total);

    cleanup_platform();
    return (pass == total) ? 0 : 1;
}