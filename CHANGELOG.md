# Tensor release notes

## [Unreleased]

- `Tensor.Autograd.grad` returns detached CPU/GPU gradients without changing
  slots, with an explicit seed and optional zeros for unused inputs. The graph
  is consumed after success. `Autograd.gradcheck` checks a deterministic function’s
  complete Jacobian on float32 CPU data, with explicit epsilon and tolerances
  and a localized first discrepancy. See the [guide](Docs/EN/Gradients.md).

- Add `DType.boolean()` masks, broadcast comparisons, and logic on CPU/GPU.
  `Tensor.where` selects numeric branches with broadcast gradients and no
  implicit GPU transfers; `masked_select` compacts on the CPU and differentiates
  its source. Booleans stay distinct from numbers with typed extraction and
  exact transfers. See the [mask guide](Docs/EN/Masks.md).

- Add `Tensor.Profile` and `Tensor.ProfileStatistics`: Tensor GPU costs per
  device and period, immutable snapshots and independent resets. Transfers,
  live/peak buffers and bytes, staging, fences, submissions, compilations,
  passes, bindings and dispatches are observable without implicit readback or
  waits. Bytes describe requested storage sizes, not physical VRAM; periods
  follow calls from the calling thread. See the [profiling guide](Docs/EN/Profiling.md).

- Adam avoids recomputing zero moments after their first initialization on
  CPU and GPU. The formulas, moment values, and deferred initialization of
  parameters remain unchanged. Subsequent GPU steps remove two passes and
  two temporary allocations per active parameter.

  The [MLP/CNN/RNN qualification](https://github.com/Matanek/Silex-Benchmarks/blob/5599f77a79b065a830740e757501ecad5b5ab167/Sources/TensorStableCompute/Baselines/2026-10-04-macos-arm64-adam-warm-moments/Summary.md) compares Tensor `175c383` and
  `e742c9a` in Release with Silex 0.47.1 on macOS arm64 / Metal M3 Pro. It
  retains five pairs per family, their dispersion, and a same-binary control;
  no performance gain outside this configuration is claimed.
