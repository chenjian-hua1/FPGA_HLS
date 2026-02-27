#include "cordic_2.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

struct Rmse {
    int num_sq;
    double sum_sq;  // 使用 double 提高精度
    double error;

    Rmse() : num_sq(0), sum_sq(0), error(0) {}

    double add_value(double d_n) {  // 使用 double
        num_sq++;
        sum_sq += (d_n * d_n);
        error = sqrt(sum_sq / num_sq);
        return error;
    }
};

Rmse rmse_sin, rmse_cos;

void run_test(THETA_TYPE theta, COS_SIN_TYPE golden_sin, COS_SIN_TYPE golden_cos) {
    COS_SIN_TYPE s, c;

    cordic(theta, s, c);

    std::cout << "Test: theta=" << theta
              << " golden_sin=" << golden_sin
              << " golden_cos=" << golden_cos
              << " your_sin=" << s
              << " your_cos=" << c
              << std::endl;
    //使用 double 進行計算並在最後轉換
    rmse_sin.add_value(static_cast<double>(s) - static_cast<double>(golden_sin));
    rmse_cos.add_value(static_cast<double>(c) - static_cast<double>(golden_cos));
}

int main() {
    std::cout << "--- Testing CORDIC Algorithm ---\n";

    double test_angles_degrees[] = {0.000, 0.262, 0.524, 0.785, 1.047, 1.571};
    int num_tests = sizeof(test_angles_degrees) / sizeof(THETA_TYPE);

    for (int i = 0; i < num_tests; i++) {
        // 使用 THETA_TYPE
        THETA_TYPE theta = test_angles_degrees[i];    // 直接進行轉換

        COS_SIN_TYPE golden_sin = sin(static_cast<double>(theta));  // 先轉為 double
        COS_SIN_TYPE golden_cos = cos(static_cast<double>(theta));

        run_test(theta, golden_sin, golden_cos);
    }

    std::cout << "RMSE(Sin): " << rmse_sin.error << "\n";
    std::cout << "RMSE(Cos): " << rmse_cos.error << "\n";

    return (rmse_sin.error < 0.1 && rmse_cos.error < 0.1) ? 0 : 1;
}
