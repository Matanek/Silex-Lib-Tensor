# Diagnostic fixtures

Run each `.sx` file explicitly from the `SilexProject` root with `silex run`.
Every fixture must fail and mention its intended contract:

- `Extraction.sx`: matching dtype extractor or explicit cast;
- `GPUExtraction.sx`: `cpu()` before value extraction;
- `CastRange.sx`: invalid controlled numeric conversion;
- `GPUInteger.sx`: GPU scalar arithmetic accepts only `float32`;
- `CrossDevice.sx`: explicit CPU staging between devices;
- `Placement.sx`: matching placements;
- `DTypeMismatch.sx`: matching dtypes;
- `CopyTensor.sx`: a value reaching a `noncopyable` class cannot be cloned;
- `NegativeShape.sx`: shape dimensions are non-negative;
- `ShapeOverflow.sx`: shape cardinality overflow is rejected before allocation;
- `ByteSizeOverflow.sx`: physical byte-size overflow is rejected before allocation;
- `ReshapeInference.sx`, `ReshapeEmpty.sx`, `ReshapeMismatch.sx`,
  `ReshapeNegative.sx`, and `ReshapeStrided.sx`: reshape inference,
  cardinality, dimensions, and contiguity rules;
- `PermuteDuplicate.sx`, `PermuteAxis.sx`, `NegativeAxis.sx`, and
  `PermuteRank.sx`: a permutation contains every in-range axis exactly once;
- `TransposeRank.sx`: the transpose shortcut is rank-2 only;
- `SelectBounds.sx`, `SelectAxis.sx`, `NarrowBounds.sx`, `NarrowAxis.sx`, and
  `NarrowNegative.sx`: selected axes, indices, and ranges are valid;
- `IndexRank.sx`, `IndexBounds.sx`, `NegativeIndex.sx`, and `GPUIndex.sx`:
  scalar indexing supplies every non-negative in-range axis and remains
  CPU-only;
- `GPUViewCompute.sx`: GPU arithmetic requires an explicit `contiguous()` for
  a strided view.
