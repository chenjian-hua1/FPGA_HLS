"""
gen_coefficients.py
-------------------
Generate C++ header file containing precomputed cos/sin lookup tables
for DFT/FFT twiddle factors.

The angles are uniformly sampled over [0, 2*pi):
    theta[i] = 2 * pi * i / N,  i = 0, 1, ..., N-1

so that any phase 2*pi*k*n/N in the DFT formula can be obtained by
indexing the table at (k*n) mod N.

Usage examples:
    python gen_coefficients.py                       # default: N=256, float
    python gen_coefficients.py --size 1024           # N=1024
    python gen_coefficients.py --size 256 --dtype double
    python gen_coefficients.py --size 256 --output coefficients256.h
"""

import argparse
import math
from pathlib import Path


# ---------- table generation ----------
def generate_tables(n: int):
    """Return (cos_values, sin_values), each of length n.

    Uses the negative-sign convention of the forward DFT:
        W_N^k = exp(-j*2*pi*k/N) = cos(2*pi*k/N) - j*sin(2*pi*k/N)

    The stored sin values are sin(2*pi*k/N) (positive), since the sign
    is handled in the DFT/FFT computation itself (e.g. `imag -= sample*s`).
    """
    cos_vals = [math.cos(2.0 * math.pi * k / n) for k in range(n)]
    sin_vals = [math.sin(2.0 * math.pi * k / n) for k in range(n)]
    return cos_vals, sin_vals


# ---------- formatting ----------
def format_value(v: float, dtype: str, precision: int) -> str:
    """Format one float value for C++, with 'f' suffix for single precision."""
    # Clean up tiny floating-point noise near zero so the table looks nice.
    if abs(v) < 10 ** -(precision + 2):
        v = 0.0

    s = f"{v:.{precision}f}"
    if dtype == "float":
        s += "f"
    return s


def format_array(name: str, values, dtype: str, precision: int,
                 per_line: int = 4) -> str:
    """Render a C++ const array declaration with values laid out per_line per row."""
    lines = [f"const {dtype} {name}[{len(values)}] = {{"]
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        formatted = ", ".join(format_value(v, dtype, precision) for v in chunk)
        # Trailing comma except for the very last chunk
        suffix = "," if i + per_line < len(values) else ""
        lines.append(f"    {formatted}{suffix}")
    lines.append("};")
    return "\n".join(lines)


# ---------- header file assembly ----------
def build_header(n: int, dtype: str, precision: int, guard: str) -> str:
    cos_vals, sin_vals = generate_tables(n)

    parts = [
        "// =====================================================================",
        f"// Auto-generated DFT/FFT twiddle-factor lookup tables (N = {n}).",
        "// Do NOT edit by hand. Regenerate with gen_coefficients.py.",
        "//",
        "// Each entry i corresponds to the angle theta = 2*pi*i / N.",
        f"// To get cos/sin(2*pi*k*n/N), index the table at (k*n) & {n - 1}.",
        "// =====================================================================",
        "",
        f"#ifndef {guard}",
        f"#define {guard}",
        "",
        f"#define COEFF_SIZE {n}",
        "",
        format_array("cos_coefficients_table", cos_vals, dtype, precision),
        "",
        format_array("sin_coefficients_table", sin_vals, dtype, precision),
        "",
        f"#endif // {guard}",
        "",
    ]
    return "\n".join(parts)


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(
        description="Generate cos/sin lookup tables as a C++ header file."
    )
    parser.add_argument("--size", "-n", type=int, default=256,
                        help="Number of samples N (default: 256). "
                             "Must be a power of 2 for `(k*n) & (N-1)` indexing.")
    parser.add_argument("--dtype", choices=["float", "double"], default="float",
                        help="C++ floating-point type (default: float).")
    parser.add_argument("--precision", "-p", type=int, default=10,
                        help="Decimal digits to keep (default: 10).")
    parser.add_argument("--per-line", type=int, default=4,
                        help="Values per line in the output array (default: 4).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename (default: coefficients<N>.h).")
    args = parser.parse_args()

    # Sanity check: warn (not error) if N isn't a power of 2.
    n = args.size
    if n & (n - 1) != 0:
        print(f"[WARN] N={n} is not a power of 2. The DFT/FFT code uses "
              f"`(k*n) & {n - 1}` as `% {n}`, which only works correctly "
              "when N is a power of 2.")

    output_path = Path(args.output or f"coefficients{n}.h")
    guard = f"COEFFICIENTS{n}_H"

    header_text = build_header(n, args.dtype, args.precision, guard)
    output_path.write_text(header_text, encoding="utf-8")

    print(f"[OK] Wrote {output_path}  (N={n}, dtype={args.dtype}, "
          f"precision={args.precision})")


if __name__ == "__main__":
    main()