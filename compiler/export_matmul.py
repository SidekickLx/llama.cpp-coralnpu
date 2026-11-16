#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the static F32 M1 ABI as StableHLO text; does not compile or execute it."""

import argparse
from pathlib import Path


def positive_dimension(value: str) -> int:
    result = int(value)
    if not 0 < result <= 2**31 - 1:
        raise argparse.ArgumentTypeError("dimension must be in [1, 2^31-1]")
    return result


def export_matmul(m: int, k: int, n: int) -> str:
    # Input order is X, W, although GGML MUL_MAT stores W in src[0].
    # Contract both K axes directly, avoiding an explicit transpose operation.
    abi = f"v1;f32;row-major;X={m}x{k};W={n}x{k};Y={m}x{n}"
    return f"""// Generated M1 ABI. Weight VALUES remain runtime parameters.
module @m1 {{
  func.func @matmul(
      %x: tensor<{m}x{k}xf32>,
      %w: tensor<{n}x{k}xf32>) -> tensor<{m}x{n}xf32>
      attributes {{iree.reflection = {{"coral.m1.abi" = "{abi}"}}}} {{
    %y = stablehlo.dot_general %x, %w,
        contracting_dims = [1] x [1],
        precision = [DEFAULT, DEFAULT]
        : (tensor<{m}x{k}xf32>, tensor<{n}x{k}xf32>)
          -> tensor<{m}x{n}xf32>
    return %y : tensor<{m}x{n}xf32>
  }}
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=positive_dimension, default=1)
    parser.add_argument("--k", type=positive_dimension, default=16)
    parser.add_argument("--n", type=positive_dimension, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for rows, cols in ((args.m, args.k), (args.n, args.k), (args.m, args.n)):
        if rows * cols * 4 > 2**31 - 1:
            parser.error("an M1 tensor must fit in signed 32-bit byte addressing")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing experiment accidentally.
    with args.output.open("x", encoding="utf-8", newline="\n") as output:
        output.write(export_matmul(args.m, args.k, args.n))


if __name__ == "__main__":
    main()
