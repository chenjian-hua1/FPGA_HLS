/*
Reference :
https://github.com/bol-edu/2022-fall-ntu/tree/main
https://github.com/kthohr/gcem/tree/master
*/

#include <hls_stream.h>
#include <ap_fixed.h>
#include <array>

#define SIZE 256


// ============================================================
// 編譯期 sin/cos（精簡自 gcem 演算法）
// ============================================================
namespace const_trig {
    constexpr double PI      = 3.14159265358979323846;
    constexpr double TWO_PI  = 6.28318530717958647692;
    constexpr double HALF_PI = 1.57079632679489661923;

    constexpr double tan_cf(double z, int depth) noexcept {
        double result = static_cast<double>(2 * depth - 1);
        double zz = z * z;
        for (int d = depth - 1; d >= 1; --d) {
            result = static_cast<double>(2 * d - 1) - zz / result;
        }
        return z / result;
    }

    constexpr int tan_depth(double abs_x) noexcept {
        return (abs_x > 1.4) ? 45 : (abs_x > 1.0) ? 35 : 25;
    }

    constexpr double sin(double x) noexcept {
        while (x >  PI) x -= TWO_PI;
        while (x < -PI) x += TWO_PI;
        if (x == 0.0) return 0.0;
        if (x ==  HALF_PI) return  1.0;
        if (x == -HALF_PI) return -1.0;
        if (x ==  PI || x == -PI) return 0.0;
        double h = x / 2.0;
        double absh = (h < 0) ? -h : h;
        double t = tan_cf(h, tan_depth(absh));
        return (2.0 * t) / (1.0 + t * t);
    }

    constexpr double cos(double x) noexcept {
        while (x >  PI) x -= TWO_PI;
        while (x < -PI) x += TWO_PI;
        if (x == 0.0) return 1.0;
        if (x ==  HALF_PI || x == -HALF_PI) return 0.0;
        if (x ==  PI || x == -PI) return -1.0;
        double h = x / 2.0;
        double absh = (h < 0) ? -h : h;
        double t = tan_cf(h, tan_depth(absh));
        return (1.0 - t * t) / (1.0 + t * t);
    }
}

// ============================================================
// 編譯期建表
// ============================================================
template <int N>
constexpr std::array<double, N> make_sin_table() {
    std::array<double, N> t{};
    for (int i = 0; i < N; ++i) {
        t[i] = const_trig::sin(const_trig::TWO_PI * static_cast<double>(i)
                                                  / static_cast<double>(N));
    }
    return t;
}

template <int N>
constexpr std::array<double, N> make_cos_table() {
    std::array<double, N> t{};
    for (int i = 0; i < N; ++i) {
        t[i] = const_trig::cos(const_trig::TWO_PI * static_cast<double>(i)
                                                  / static_cast<double>(N));
    }
    return t;
}



// Calculate sum n times out data need how many bits
template<int W, int I, int N>
struct SumType {
    static constexpr int log2_ceil(int n, int p = 0) {
        return (1 << p) >= n ? p : log2_ceil(n, p + 1);
    }
    static constexpr int EXTRA = log2_ceil(N);
    typedef ap_fixed<W + EXTRA, I + EXTRA> type;
};

// data type
typedef ap_fixed<16, 8> DTYPE;
// sum out data type
typedef SumType<16, 8, SIZE>::type SUM_DTYPE;


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
void dft(DTYPE real_samples[SIZE], DTYPE imag_samples[SIZE], SUM_DTYPE real_outs[SIZE], SUM_DTYPE imag_outs[SIZE]);
