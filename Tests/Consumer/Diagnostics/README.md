# Diagnostic fixtures

Run each `.sx` file explicitly from the `SilexProject` root with `silex run`.
Every fixture must fail and mention its intended contract:

- `Extraction.sx`: matching dtype extractor or explicit cast;
- `GPUExtraction.sx`: `cpu()` before value extraction;
- `CastRange.sx`: invalid controlled numeric conversion;
- `GPUInteger.sx` and `GPUIntegerPair.sx`: GPU arithmetic accepts only
  `float32`, for scalar and tensor operands alike;
- `BroadcastShape.sx`: the diagnostic names the operation and both incompatible shapes;
- `IntegerOverflow.sx`, `IntegerMultiplyOverflow.sx`, `IntegerDivideZero.sx`,
  `SignedDivideOverflow.sx`, `SignedNegateOverflow.sx`, `UnsignedOverflow.sx`,
  and `UnsignedUnderflow.sx`: checked Silex integer arithmetic remains
  observable;
- `UnsignedNegate.sx`: negation requires a floating or signed dtype;
- `FloatOnlyUnary.sx`: transcendental operations require `float32`;
- `ReductionAxis.sx` and `ReductionDuplicate.sx`: reduction axes are unique and
  inside the tensor rank;
- `EmptyMean.sx`, `EmptyMin.sx`, and `EmptyMax.sx`: reductions without an
  identity reject an empty domain;
- `IntegerMean.sx`: `mean` requires an explicit conversion to `float32`;
- `GPUIntegerReduction.sx`, `GPUIntegerDot.sx`, and
  `GPUIntegerMatmul.sx`: integer reductions and linear algebra remain CPU-only;
- `DotRank.sx`, `DotShape.sx`, `MatmulRank.sx`, and `MatmulShape.sx`: linear
  algebra validates ranks and dimensions before calculation;
- `LinearDType.sx` and `LinearPlacement.sx`: both operands retain one dtype and
  one placement;
- `SignedSumOverflow.sx`, `UnsignedSumOverflow.sx`, `SignedDotOverflow.sx`,
  and `UnsignedMatmulOverflow.sx`: reduction and multiply-accumulate overflow
  remains checked in the tensor dtype;
- `CrossDevice.sx`: explicit CPU staging between devices;
- `Placement.sx`: matching placements;
- `DTypeMismatch.sx`: matching dtypes;
- `ScalarDTypeMismatch.sx`: a scalar has the exact dtype selected by its Silex
  type;
- `CopyTensor.sx`: a value reaching a `nocopy` class cannot be cloned;
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
- `NeuralAxis.sx` and `NeuralDType.sx`: neural activations validate axes and
  require `float32`;
- `NeuralTargetShape.sx`: dense cross-entropy targets match the logits shape;
- `GatherShape.sx`: gather indices match all non-selected dimensions;
- `InitializerBounds.sx` and `DropoutProbability.sx`: random initialization
  bounds and dropout probabilities are valid;
- `ConvolutionShape.sx`: convolution channels agree before calculation;
- `AutogradInteger.sx`: tracked leaves require `float32`;
- `AutogradNonScalar.sx`, `AutogradSeed.sx`, `AutogradSeedDType.sx`, and
  `AutogradSeedPlacement.sx`: backward requires either a scalar result or an
  explicit seed with the exact output shape, dtype, placement, and device;
- `AutogradConsumed.sx`: a completed graph cannot be traversed twice;
- `AutogradTransfer.sx` and `AutogradCast.sx`: placement and dtype changes are
  outside a tracked graph and require `detach()` first;
- `AutogradUntracked.sx`: backward requires provenance from a tracked leaf;
- transposed GPU views are covered as successful stride-aware computation in
  `Tests/GPUCompute.sx`.
