#include "dft.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- Float 參考 DFT（與 HLS 版本同公式、同表）----------
void dft_float_ref(const float real_samples[COEFF_SIZE],
                   const float imag_samples[COEFF_SIZE],
                   float real_outs[COEFF_SIZE],
                   float imag_outs[COEFF_SIZE])
{
    static float cos_tbl[COEFF_SIZE];
    static float sin_tbl[COEFF_SIZE];
    static bool initialized = false;
    if (!initialized) {
        for (int i = 0; i < COEFF_SIZE; i++) {
            double theta = 2.0 * M_PI * (double)i / (double)COEFF_SIZE;
            cos_tbl[i] = (float)std::cos(theta);
            sin_tbl[i] = (float)std::sin(theta);
        }
        initialized = true;
    }

    for (int k = 0; k < COEFF_SIZE; k++) {
        real_outs[k] = 0.0f;
        imag_outs[k] = 0.0f;
    }

    for (int n = 0; n < COEFF_SIZE; n++) {
        float real = real_samples[n];
        float imag = imag_samples[n];
        for (int k = 0; k < COEFF_SIZE; k++) {
            int index = (k * n) & (COEFF_SIZE - 1);
            float c = cos_tbl[index];
            float s = sin_tbl[index];
            // 跟 dft.cpp 的虛部公式 (ic - rs) 對應
            real_outs[k] += (real * c + imag * s);
            imag_outs[k] += (imag * c - real * s);
        }
    }
}

// ---------- 訊號產生 ----------
void gen_signal_sine(DTYPE real_fx[COEFF_SIZE], DTYPE imag_fx[COEFF_SIZE],
                     float real_f[COEFF_SIZE], float imag_f[COEFF_SIZE],
                     int freq_bin, float amplitude)
{
    for (int n = 0; n < COEFF_SIZE; n++) {
        double v = amplitude * std::cos(2.0 * M_PI * freq_bin * n / COEFF_SIZE);
        real_f[n]  = (float)v;
        imag_f[n]  = 0.0f;
        real_fx[n] = (DTYPE)v;
        imag_fx[n] = (DTYPE)0;
    }
}

void gen_signal_random(DTYPE real_fx[COEFF_SIZE], DTYPE imag_fx[COEFF_SIZE],
                       float real_f[COEFF_SIZE], float imag_f[COEFF_SIZE],
                       float range)
{
    for (int n = 0; n < COEFF_SIZE; n++) {
        float r = ((float)std::rand() / RAND_MAX * 2.0f - 1.0f) * range;
        float i = ((float)std::rand() / RAND_MAX * 2.0f - 1.0f) * range;
        real_f[n]  = r;
        imag_f[n]  = i;
        real_fx[n] = (DTYPE)r;
        imag_fx[n] = (DTYPE)i;
    }
}

// ---------- 比較 ----------
struct ErrorStats { double max_abs_err; double rmse; int max_err_index; };

ErrorStats compare_results(const SUM_DTYPE fx_re[COEFF_SIZE],
                           const SUM_DTYPE fx_im[COEFF_SIZE],
                           const float ref_re[COEFF_SIZE],
                           const float ref_im[COEFF_SIZE])
{
    ErrorStats st = {0.0, 0.0, -1};
    double sum_sq = 0.0;
    for (int k = 0; k < COEFF_SIZE; k++) {
        double er = (double)(float)fx_re[k] - (double)ref_re[k];
        double ei = (double)(float)fx_im[k] - (double)ref_im[k];
        double mag = std::sqrt(er * er + ei * ei);
        if (mag > st.max_abs_err) { st.max_abs_err = mag; st.max_err_index = k; }
        sum_sq += er * er + ei * ei;
    }
    st.rmse = std::sqrt(sum_sq / COEFF_SIZE);
    return st;
}

// ---------- 單個測試 ----------
bool run_one_test(const char* name,
                  DTYPE real_fx[COEFF_SIZE], DTYPE imag_fx[COEFF_SIZE],
                  float real_f[COEFF_SIZE], float imag_f[COEFF_SIZE],
                  double tolerance, bool dump = false)
{
    // 顯式清零（即使函式內部會清，也保險起見）
    SUM_DTYPE fx_re[COEFF_SIZE];
    SUM_DTYPE fx_im[COEFF_SIZE];
    for (int k = 0; k < COEFF_SIZE; k++) { fx_re[k] = 0; fx_im[k] = 0; }

    float ref_re[COEFF_SIZE];
    float ref_im[COEFF_SIZE];

    dft(real_fx, imag_fx, fx_re, fx_im);
    dft_float_ref(real_f, imag_f, ref_re, ref_im);

    ErrorStats st = compare_results(fx_re, fx_im, ref_re, ref_im);

    bool pass = (st.max_abs_err <= tolerance);
    std::cout << "[" << name << "] "
              << "max_err=" << std::fixed << std::setprecision(4) << st.max_abs_err
              << " (k=" << st.max_err_index << "), "
              << "RMSE=" << st.rmse
              << ", tol=" << tolerance
              << "  " << (pass ? "PASS" : "FAIL") << std::endl;

    if (dump) {
        int n_dump = std::min(8, (int)COEFF_SIZE);
        std::cout << "  k |   fx_real      fx_imag    |   ref_real     ref_imag" << std::endl;
        for (int k = 0; k < n_dump; k++) {
            std::cout << "  " << std::setw(2) << k << " | "
                      << std::setw(10) << std::setprecision(4) << (float)fx_re[k] << "  "
                      << std::setw(10) << (float)fx_im[k] << "  | "
                      << std::setw(10) << ref_re[k] << "  "
                      << std::setw(10) << ref_im[k] << std::endl;
        }
    }
    return pass;
}

int main() {
    std::cout << "=== DFT Fixed-point vs Float Reference ===" << std::endl;
    std::cout << "COEFF_SIZE       = " << COEFF_SIZE << std::endl;
    std::cout << "TOTAL_BITS       = " << TOTAL_BITS << std::endl;
    std::cout << "INT_BITS         = " << INT_BITS << std::endl;
    std::cout << "MULT_TOTAL_BITS  = " << MULT_TOTAL_BITS
              << ", MULT_INT_BITS = " << MULT_INT_BITS << std::endl;
    std::cout << "SUM_EXTRA_BITS   = " << SUM_EXTRA_BITS << std::endl;
    std::cout << "SUM_TOTAL_BITS   = " << SUM_TOTAL_BITS
              << ", SUM_INT_BITS  = " << SUM_INT_BITS << std::endl;
    std::cout << std::endl;

    // 容忍度：依資料寬度調整
    // DTYPE 量化步長 ≈ 2^-DEC_BITS；累加 N 次最壞約 N × 步長
    double tol = (double)COEFF_SIZE * std::pow(2.0, -(double)DEC_BITS) * 2.0;
    if (tol < 0.1) tol = 0.1;   // 至少給點空間給 cos/sin 表的量化

    DTYPE real_fx[COEFF_SIZE], imag_fx[COEFF_SIZE];
    float real_f[COEFF_SIZE],  imag_f[COEFF_SIZE];
    int pass = 0, total = 0;

    // 1. 純 cos 訊號
    gen_signal_sine(real_fx, imag_fx, real_f, imag_f, 4, 0.5f);
    if (run_one_test("Cos bin=4, amp=0.5", real_fx, imag_fx, real_f, imag_f, tol, true)) pass++;
    total++;

    // 2. 高頻
    gen_signal_sine(real_fx, imag_fx, real_f, imag_f, COEFF_SIZE / 4, 0.5f);
    if (run_one_test("Cos bin=N/4", real_fx, imag_fx, real_f, imag_f, tol)) pass++;
    total++;

    // 3. DC
    for (int n = 0; n < COEFF_SIZE; n++) {
        real_fx[n] = (DTYPE)0.5f; imag_fx[n] = (DTYPE)0;
        real_f[n]  = 0.5f;        imag_f[n]  = 0.0f;
    }
    if (run_one_test("DC=0.5", real_fx, imag_fx, real_f, imag_f, tol)) pass++;
    total++;

    // 4. 隨機
    std::srand(42);
    for (int t = 0; t < 5; t++) {
        gen_signal_random(real_fx, imag_fx, real_f, imag_f, 0.5f);
        std::string name = "Random " + std::to_string(t);
        if (run_one_test(name.c_str(), real_fx, imag_fx, real_f, imag_f, tol)) pass++;
        total++;
    }

    // 5. Impulse
    for (int n = 0; n < COEFF_SIZE; n++) {
        real_fx[n] = (DTYPE)0; imag_fx[n] = (DTYPE)0;
        real_f[n]  = 0.0f;     imag_f[n]  = 0.0f;
    }
    real_fx[0] = (DTYPE)0.9f; real_f[0] = 0.9f;
    if (run_one_test("Impulse", real_fx, imag_fx, real_f, imag_f, tol)) pass++;
    total++;

    std::cout << std::endl
              << "=== " << pass << " / " << total << " passed ===" << std::endl;
    return (pass == total) ? 0 : 1;
}