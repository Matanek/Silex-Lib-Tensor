# Tensor release notes

## [Unreleased]

- Add batch-first reset-after r/z/n GRU with explicit state and sequence, four named parameters, CPU/Metal oracles, chunking and exact resumption.

- `Neural.TransformerBlock` composes attention, residuals, LayerNorm and MLP with explicit pre/post-norm and reproducible dropout. An external causal model with separate positions, learning and exact CPU/Metal resume is qualified; see the [guide](Docs/EN/Transformer.md).

- `Neural.MultiHeadAttention` adds four trainable projections to Layer registration, self/cross-attention and explicit seed/iteration dropout. Queries without keys stay zero after projection, optional applied weights, shared state and CPU/Metal resume are qualified; see the [guide](Docs/EN/MultiHeadAttention.md).

- `Neural.attention` composes Q/K/V products and masked softmax into batch-first rank3/4 self/cross-attention. Differentiable optional weights, shared gradients, empty shapes and resident CPU/Metal computation; see the [guide](Docs/EN/Attention.md).

- `Tensor.masked_softmax(mask,axis)` normalizes finite visible scores and defines zero outputs/gradients for excluded positions and fully masked rows. Excluded nonfinite values, empty shapes and resident CPU/Metal computation are qualified by oracle and gradcheck; see the [guide](Docs/EN/MaskedSoftmax.md).

- Sequence masks `Neural.sequence_mask` and `Neural.causal_mask`: CPU booleans, exact prefixes and positions, explicit offset, composition and GPU transfer; selection gradients qualified on CPU/Metal.

- `Neural.Embedding` learns a seeded float32 table from batch-first int32 indices, with repeated dense gradients and a stopped padding derivative. Layer composition, state/resume and resident CPU/Metal lookup are qualified; see the [guide](Docs/EN/Embeddings.md).

- `Autograd.Operation` and `Autograd.apply` connect an external rule built from public Tensor operations to the first-order graph. Leaf gradients are prepared before publication and retained values are released on consumption/destruction. See the [guide](Docs/EN/CustomOperations.md).

- `Data.BatchCursor(...,prefetch:N)` prepares detached CPU batches through STD.Threading, preserves consumed snapshots and error order, and joins workers on close/destruction. Explicit source eligibility, CPU/GPU checkpoint resumption and documented batch bounds.

- `Neural.TrainingCheckpoint` exactly saves and restores the registered model, SGD/Adam/AdamW, cursor, step/schedule, modes, trainability and dropout. Prevalidated CPU loading uses recoverable errors; optimizer snapshots expose explicit GPU movement.

- BatchNorm normalizes dense or NCHW batches on CPU/GPU with gradients, optional affine state, detached running statistics, and an exact CPU int64 counter. Population/unbiased conventions and train/eval modes are documented in the [guide](Docs/EN/BatchNorm.md).

- Buffers can declare BufferPlacement.cpu to retain exact CPU state during model transfers. The model policy remains the compatible default; conflicting direct transfers are rejected without conversion or hidden readback. See the [guide](Docs/EN/HostState.md).

- AdamW applies decay independently of moments, with distinct named state and atomic restoration. LearningRateSchedule provides constant, step, exponential, and bounded cosine rates from an explicit step; finite positive rates and rate/update/increment order are documented in the [guide](Docs/EN/OptimizerUpdates.md).

- SGD and Adam capture and restore named state with recoverable errors and atomic validation, without hidden transfers. The current rate can change without discarding moments. Adam bias correction is bounded for large restored counters, and overflow is rejected before mutation. See the [resume guide](Docs/EN/OptimizerState.md).

- `Tensor.Data.Source`, `TensorSource`, and `BatchCursor` provide synchronous supervised batches. Stack or named callback collation has recoverable errors, advances after success, resumes the next batch, and preserves placement/dtypes/gradients. See the [guide](Docs/EN/SupervisedBatches.md).

- Add `Tensor.Data.BatchSampler`: ordered or shuffled int32 batches, explicit seed/epoch, final partial batch kept by default, and immutable reconstructible cursor state. Ordered sampling avoids a full permutation; CPU metadata performs no GPU work. See the [batch guide](Docs/EN/BatchSampling.md).

- Layer propagates train/eval, placement, freezing/unfreezing, and zero_grad to
  unique registered identities, including shared diamonds. Protected local
  mode/placement hooks and shared is_training with Dropout; names/gradients/devices
  are prevalidated before mutation or transfer. Older manual boundaries without
  registration remain compatible. See the [state guide](Docs/EN/ModelState.md).

- Layers explicitly register parameters, buffers, and named children. Inherited
  collection preserves order and names, deduplicates shared identities, and rejects
  cycles and collisions; Sequential collects shared state once while executing
  the layer at each occurrence. See the
  [registration guide](Docs/EN/Registration.md).

- Add detached, named and shared `Neural.Buffer` values with checked replacement
  and exact CPU/GPU transfers. `Parameter.freeze/unfreeze/is_trainable` suspend
  new gradients, updates, decay, clipping and SGD/Adam state, even when a graph
  built earlier produces a gradient. Unfreezing retains moments. See the
  [buffer and freezing guide](Docs/EN/State.md).

- External classes can derive from `Neural.Layer` through its protected
  constructor and share existing layers/parameters. Virtual state hooks rely
  on the compiler’s family-wide mutation inference; remove the obsolete base
  marker. CPU/GPU residual values and shared gradients are qualified. See the
  [composition guide](Docs/EN/Composition.md).

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
