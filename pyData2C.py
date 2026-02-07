import numpy as np
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

# -------- dtype 推斷：numpy dtype -> C/C++ type --------
def npdtype_to_ctype(dt: np.dtype) -> str:
    dt = np.dtype(dt)

    # bool
    if dt.kind == "b":
        return "bool"

    # signed / unsigned int
    if dt.kind in ("i", "u"):
        bits = dt.itemsize * 8
        prefix = "" if dt.kind == "i" else "u"
        # 用 stdint 的固定寬度型別
        return f"{prefix}int{bits}_t"

    # float
    if dt.kind == "f":
        bits = dt.itemsize * 8
        if bits == 32:
            return "float"
        if bits == 64:
            return "double"
        # 其他比較少見的浮點（如 float16/float128）
        return "double"

    # complex：C++ 才比較合理（std::complex）
    if dt.kind == "c":
        bits = dt.itemsize * 8
        # complex64 -> 2x float, complex128 -> 2x double
        return "std::complex<float>" if bits == 64 else "std::complex<double>"

    # fallback
    return "int32_t"


def _ensure_ndarray(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def _format_0d_scalar(val: Any) -> str:
    # bool
    if isinstance(val, (np.bool_, bool)):
        return "true" if bool(val) else "false"
    # float
    if isinstance(val, (np.floating, float)):
        # 用 repr 盡量保精度
        return repr(float(val))
    # int
    if isinstance(val, (np.integer, int)):
        return str(int(val))
    # complex
    if isinstance(val, (np.complexfloating, complex)):
        return f"{{{repr(val.real)}, {repr(val.imag)}}}"
    # string/other
    return str(val)


def _format_1d(arr: np.ndarray) -> str:
    if arr.size == 0:
        return ""
    return ", ".join(_format_0d_scalar(v) for v in arr.tolist())


def _format_2d(arr: np.ndarray) -> str:
    lines = []
    for r in range(arr.shape[0]):
        row = ", ".join(_format_0d_scalar(v) for v in arr[r].tolist())
        lines.append(f"  {{{row}}}")
    return ",\n".join(lines)


def export_arrays_to_c_header(
    out_path: Union[str, Path],
    variables: Dict[str, Any],
    *,
    array_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    header_guard: Optional[str] = None,
    includes: Optional[Iterable[str]] = None,
    storage: str = "static const",  # "static const" (header-only) 或 "extern const"
    default_int_ctype: str = "int32_t",
    sort_keys: bool = True,
) -> Path:
    """
    通用匯出器：把多個 numpy array / list / scalar 匯出成 C/C++ 可 include 的 .h

    Args:
        out_path: 輸出 .h 路徑
        variables: dict，key=你想要的 C/C++ 變數名，value=scalar/1D/2D array/list
        array_overrides: 可針對單一變數覆寫設定，例如：
            {
              "sp_values": {"ctype": "int16_t"},
              "A": {"ctype": "float"},
            }
          支援欄位：
            - ctype: 強制指定 C/C++ 型別
            - reshape: tuple，例如 (5,5) 把 1D 改成 2D
        header_guard: 自訂 include guard（不給就用檔名產生）
        includes: 額外 include，例如 ["<vector>"]；預設會自動加 <cstdint>，必要時加 <complex>
        storage: "static const" 直接定義在 header；或用 "extern const" 只宣告
        default_int_ctype: 若推斷失敗時整數預設型別
        sort_keys: 是否依 key 排序輸出，方便比對

    Returns:
        header 檔案 Path
    """
    out_path = Path(out_path).with_suffix(".h")

    array_overrides = array_overrides or {}
    include_set = set(includes or [])
    include_set.add("<cstdint>")  # 固定寬度整數常用

    # 先把所有變數整理成 ndarray + 推斷型別 + 形狀
    entries = []
    need_complex = False

    keys = sorted(variables.keys()) if sort_keys else list(variables.keys())
    for var_name in keys:
        raw = variables[var_name]
        ov = array_overrides.get(var_name, {})

        arr = _ensure_ndarray(raw)

        # reshape override（可選）
        if "reshape" in ov and ov["reshape"] is not None:
            arr = arr.reshape(tuple(ov["reshape"]))

        # 推斷 ctype（可被 override）
        if "ctype" in ov and ov["ctype"]:
            ctype = str(ov["ctype"])
        else:
            ctype = npdtype_to_ctype(arr.dtype)
            if ctype == "int32_t" and arr.dtype.kind in ("i", "u") and default_int_ctype:
                # 若你想強制整數全部用某個型別，可改這邊策略
                pass

        if "std::complex" in ctype:
            need_complex = True

        entries.append((var_name, arr, ctype))

    if need_complex:
        include_set.add("<complex>")
    # 你也可以加上 bool 時 include <stdbool.h>，但 C++ 不需要；這裡不強制。

    # header guard
    if header_guard is None:
        guard = f"{out_path.stem.upper()}_H_"
        guard = "".join(ch if ch.isalnum() else "_" for ch in guard)
    else:
        guard = header_guard

    lines = []
    lines += [f"#ifndef {guard}", f"#define {guard}", ""]
    for inc in sorted(include_set):
        lines.append(f"#include {inc}")
    lines += [""]

    # 逐一輸出
    for var_name, arr, ctype in entries:
        if arr.ndim == 0:
            # scalar
            v = arr.item()
            init = _format_0d_scalar(v)
            lines.append(f"{storage} {ctype} {var_name} = {init};")
            lines.append("")
            continue

        if arr.ndim == 1:
            n = arr.shape[0]
            init = _format_1d(arr)
            if storage.startswith("extern"):
                lines.append(f"{storage} {ctype} {var_name}[{n}];")
            else:
                lines.append(f"{storage} {ctype} {var_name}[{n}] = {{{init}}};")
            lines.append("")
            continue

        if arr.ndim == 2:
            r, c = arr.shape
            init2 = _format_2d(arr)
            if storage.startswith("extern"):
                lines.append(f"{storage} {ctype} {var_name}[{r}][{c}];")
            else:
                lines.append(f"{storage} {ctype} {var_name}[{r}][{c}] = {{\n{init2}\n}};")
            lines.append("")
            continue

        raise ValueError(f"Only support 0D/1D/2D arrays. '{var_name}' has shape {arr.shape} (ndim={arr.ndim})")

    lines += [f"#endif // {guard}", ""]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path

if __name__ == "__main__":
    # ================================
    # Example: Sparse Matrix → COO → C Header
    # ================================

    np.random.seed(0)

    # ---------- 1. 產生稀疏矩陣 ----------
    rows, cols = 5, 5
    density = 0.4

    mat = (np.random.rand(rows, cols) < density) * np.random.randint(1, 10, (rows, cols))
    mat = mat.astype(np.int32)

    print("Sparse Matrix:")
    print(mat)
    print()

    # ---------- 2. 轉 COO ----------
    row_idx = []
    col_idx = []
    values = []

    for r in range(rows):
        for c in range(cols):
            if mat[r, c] != 0:
                row_idx.append(r)
                col_idx.append(c)
                values.append(mat[r, c])

    row_idx = np.array(row_idx, dtype=np.int32)
    col_idx = np.array(col_idx, dtype=np.int32)
    values  = np.array(values,  dtype=np.int32)

    print("COO Format:")
    print("row_idx :", row_idx.tolist())
    print("col_idx :", col_idx.tolist())
    print("values  :", values.tolist())
    print("nnz     :", len(values))
    print()

    # ---------- 3. 匯出成 C/C++ Header ----------
    export_arrays_to_c_header(
        out_path="sparse_data.h",
        variables={
            # scalar
            "A_rows": rows,
            "A_cols": cols,
            "A_nnz":  len(values),

            # COO
            "A_row_idx": row_idx,
            "A_col_idx": col_idx,
            "A_values":  values,

            # dense matrix（debug / golden 用）
            "A_dense": mat,
        },
        array_overrides={
            # index / value 型別你可以在這裡完全控制
            "A_row_idx": {"ctype": "int32_t"},
            "A_col_idx": {"ctype": "int32_t"},
            "A_values":  {"ctype": "int32_t"},
            "A_dense":   {"ctype": "int32_t"},
        },
        includes=[
            "<cstdint>",   # 固定寬度整數
        ],
        storage="static const",  # header-only，include 即用
    )

    print("Export finished -> sparse_data.h")