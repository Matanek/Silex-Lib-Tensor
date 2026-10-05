# Tensor release notes

## [Unreleased]

- Add `Tensor.topk` with explicit axis, ordered values and detached int32
  indices in `TopKResult`. Exact CPU numeric comparisons and resident float32
  GPU selection preserve first-index ties and NaN ordering; selected values
  differentiate through `gather`. k0 and empty outputs are defined, with checked
  shapes, sizes and indices. See the [top-k guide](Docs/EN/TopK.md).

- Add `argmin` and `argmax`: detached int32 indices in logical order, exact
  CPU integer comparisons and resident float32 GPU input. Ties and NaNs retain
  their first index; axes/domains/index widths are validated. Selected values
  differentiate through `gather`. See the [index guide](Docs/EN/IndexedReductions.md).

- Add CPU-generated exclusive-stop int32 `Tensor.arange` and explicit-count
  float32 `Tensor.linspace`. Bounds/steps/cardinality/int32 values are checked
  before allocation; int64 extremes and interpolation between large finite
  bounds avoid artificial overflow. GPU transfer is explicit. See the
  [sequence guide](Docs/EN/Sequences.md).

- Add float32 CPU/GPU `variance` and `standard_deviation` with explicit integer
  correction, validated axes/domains, centered computation, and resident gradients.
  Standard deviation chooses a zero gradient when computed variance is zero; NaN remains
  observable. See the [statistics guide](Docs/EN/Statistics.md).

- Add exact `squeeze`, `unsqueeze`, and `expand` views, and materialized
  `repeat` tiles, for all nine numeric dtypes and booleans on CPU/GPU. Resident
  float32 gradients, validated axes/factors/shapes, and defined empty domains.
  Reshape backward accepts noncontiguous seeds. See the [shape guide](Docs/EN/Shapes.md).

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
