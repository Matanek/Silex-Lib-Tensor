# Tensor release notes

## [Unreleased]

- Adam avoids recomputing zero moments after their first initialization on
  CPU and GPU. The formulas, moment values, and deferred initialization of
  parameters remain unchanged. Subsequent GPU steps remove two passes and
  two temporary allocations per active parameter.

  The [MLP/CNN/RNN qualification](https://github.com/Matanek/Silex-Benchmarks/blob/5599f77a79b065a830740e757501ecad5b5ab167/Sources/TensorStableCompute/Baselines/2026-10-04-macos-arm64-adam-warm-moments/Summary.md) compares Tensor `175c383` and
  `e742c9a` in Release with Silex 0.47.1 on macOS arm64 / Metal M3 Pro. It
  retains five pairs per family, their dispersion, and a same-binary control;
  no performance gain outside this configuration is claimed.
