/*
 * main.c: Rewritten Test application for XFir HLS IP (ap_int<16>)
 */

#include <stdio.h>
#include <stdint.h>      // 【必要】引用此檔以使用 int16_t
#include "platform.h"
#include "xil_printf.h"
#include "xparameters.h"
#include "xfir.h"
#include "xil_cache.h"

XFir FirInstance;

// 定義測試長度 (比 FIR 的 Tap 數 11 再多一點，以觀察完整波形)
#define TEST_LEN 20

// 【修正1】使用 int16_t 對應 HLS 的 ap_int<16>
// 若用 int (32bit) 會導致 High/Low byte 讀取錯誤
int16_t output_buffer[TEST_LEN] __attribute__((aligned(64)));

int main()
{
    int Status;

    // 1. 初始化平台
    init_platform();

    // 2. 關閉 Cache (Debug 模式推薦)
    Xil_DCacheDisable();
    xil_printf("Data Cache has been disabled.\r\n");

    xil_printf("\r\n--- Start FIR IP Test ---\r\n");

    // 3. 初始化 IP (使用 LookupConfig 比較穩健)
    XFir_Config *ConfigPtr = XFir_LookupConfig(XPAR_FIR_0_DEVICE_ID);
    if (!ConfigPtr) {
        xil_printf("ERROR: LookupConfig failed.\r\n");
        return XST_FAILURE;
    }

    Status = XFir_CfgInitialize(&FirInstance, ConfigPtr);
    if (Status != XST_SUCCESS) {
        xil_printf("ERROR: CfgInitialize failed.\r\n");
        return XST_FAILURE;
    }

    // --- 準備 Output Buffer ---
    // 填入 0x7FFF (髒數據) 以驗證 IP 是否真的有寫入
    for(int i = 0; i < TEST_LEN; i++) {
        output_buffer[i] = 0x7FFF;
    }

    xil_printf("Starting Impulse Response Test (Input: 100, 0, 0...)\r\n");

    // ============================================================
    // 執行 FIR 運算迴圈
    // ============================================================
    for (int i = 0; i < TEST_LEN; i++) {

        // 【修正3】建立脈衝訊號 (Impulse)
        // 第 0 次輸入 100，之後都輸入 0。這樣輸出結果應該等於「係數 * 100」
        int16_t input_x_val = (i == 0) ? 100 : 0;

        // 設定 x
        XFir_Set_x(&FirInstance, input_x_val);

        // 【修正2】動態調整寫入位址
        // 我們希望第 i 次運算的結果，寫入到 output_buffer 的第 i 格
        // (u64)(UINTPTR) 是為了相容 32/64 位元系統的標準寫法
        u64 current_y_addr = (u64)(UINTPTR)&output_buffer[i];

        XFir_Set_y(&FirInstance, current_y_addr);

        // 啟動 IP
        XFir_Start(&FirInstance);

        // 等待完成
        while (!XFir_IsDone(&FirInstance));

        // 清除中斷旗標 (雖然沒開中斷，但清一下是好習慣)
        XFir_InterruptClear(&FirInstance, 1);
    }

    xil_printf("FIR Processing Complete.\r\n");

    // ============================================================
    // 驗證與顯示結果
    // ============================================================
    xil_printf("Result Verification:\r\n");
    xil_printf("   i | Input | Output (Expected Coef * 100)\r\n");
    xil_printf("-------------------------------------------\r\n");

    for (int i = 0; i < TEST_LEN; i++) {
        int input_display = (i == 0) ? 100 : 0;

        // 讀取結果
        int16_t result = output_buffer[i];

        xil_printf(" %3d |  %3d  | %d\r\n", i, input_display, result);
    }

    cleanup_platform();
    return 0;
}
