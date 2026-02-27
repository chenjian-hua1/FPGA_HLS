/*
 * main.c: Test application for XCordic HLS IP (Fixed Point Version)
 */

#include <stdio.h>
#include <stdint.h>
#include <math.h>         // 用於 M_PI 常數 (若無此檔可手動定義 PI)
#include "platform.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xparameters.h"
#include "xcordic.h"

// =============================================================
// 定點數轉換巨集 (對應 ap_fixed<16,4>)
// =============================================================
// 定義小數位數為 12 (因為 16總位元 - 4整數位元 = 12)
#define FRAC_BITS 12
#define SCALING_FACTOR (1 << FRAC_BITS) // 相當於 4096

// 將浮點數轉為定點數整數表示 (寫入用)
// 邏輯: 乘以 4096 後強制轉型為 short
#define FLOAT_TO_FIXED(x) ((int16_t)((x) * SCALING_FACTOR))

// 將定點數整數表示轉回浮點數 (讀取用)
// 邏輯: 轉為 float 後除以 4096
#define FIXED_TO_FLOAT(x) ((float)(x) / SCALING_FACTOR)

// =============================================================
// 全域變數
// =============================================================
XCordic HlsCordic;

// 接收結果的變數 (必須是 16-bit 整數以配合 IP 輸出的寬度)
// 使用 volatile 避免被優化
volatile int16_t Raw_Sin = 0;
volatile int16_t Raw_Cos = 0;

int main()
{
    int Status;

    init_platform();

    // 關閉 Cache 以確保 CPU 讀到 IP 寫入記憶體的最新值
    Xil_DCacheDisable();
    xil_printf("\r\n--- XCordic IP Fixed-Point Test ---\r\n");

    // IP 初始化
    XCordic_Config *ConfigPtr = XCordic_LookupConfig(XPAR_XCORDIC_0_DEVICE_ID);
    if (!ConfigPtr) {
        xil_printf("ERROR: LookupConfig failed.\r\n");
        return XST_FAILURE;
    }

    Status = XCordic_CfgInitialize(&HlsCordic, ConfigPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: CfgInitialize failed.\r\n");
        return XST_FAILURE;
    }

    // ============================================================
    // 設定測試數值 (弧度)
    // ============================================================
    // 測試 PI/4 (約 0.78539 弧度，即 45 度)
    // 預期結果: Sin=0.707, Cos=0.707
    float theta_radians = 0.78539f;

    // 1. 軟體轉換: Float -> Fixed Point Binary
    int16_t theta_fixed = FLOAT_TO_FIXED(theta_radians);

    // 2. 顯示轉換資訊 (Debug用)
    // printf 支援 %f, 但 xil_printf 不支援 %f (若用 xil_printf 需只印整數部分)
    printf("Input Theta (Float): %f radians\r\n", theta_radians);
    printf("Input Theta (Raw Hex): 0x%04x (Decimal: %d)\r\n", (uint16_t)theta_fixed, theta_fixed);

    // ============================================================
    // 寫入 IP 與執行
    // ============================================================

    // 設定 Theta (需轉型為 u32 傳入，因為暫存器是 32-bit，但有效位是低 16-bit)
    XCordic_Set_theta(&HlsCordic, (u32)theta_fixed);

    // 設定輸出位址 (轉型為 u64 傳入指標)
    XCordic_Set_s(&HlsCordic, (u64)(uintptr_t)&Raw_Sin);
    XCordic_Set_c(&HlsCordic, (u64)(uintptr_t)&Raw_Cos);

    // 啟動與等待
    XCordic_Start(&HlsCordic);
    while (!XCordic_IsDone(&HlsCordic));
    XCordic_InterruptClear(&HlsCordic, 1);

    // ============================================================
    // 讀取結果與還原
    // ============================================================

    // 1. 從記憶體取得原始二進位資料 (Raw Bits)
    int16_t val_sin_raw = Raw_Sin;
    int16_t val_cos_raw = Raw_Cos;

    // 2. 軟體轉換: Fixed Point Binary -> Float
    float result_sin_float = FIXED_TO_FLOAT(val_sin_raw);
    float result_cos_float = FIXED_TO_FLOAT(val_cos_raw);

    // 3. 顯示結果
    printf("Result Raw Hex - Sin: 0x%04x, Cos: 0x%04x\r\n", (uint16_t)val_sin_raw, (uint16_t)val_cos_raw);
    printf("Result Float   - Sin: %f\r\n", result_sin_float);
    printf("Result Float   - Cos: %f\r\n", result_cos_float);

    cleanup_platform();
    return 0;
}
