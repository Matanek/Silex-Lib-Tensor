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
- `ParameterName.sx`: trainable parameters require a stable non-empty name;
- `OptimizerDuplicateName.sx`: one optimizer rejects duplicate parameter names;
- `SGDLearningRate.sx` and `AdamBeta.sx`: optimizer hyperparameters are finite
  and remain inside their documented domains;
- `GradientClipNonFinite.sx`: the rejecting clipping policy diagnoses a
  non-finite global norm before changing any gradient;
- `OptimizerStatePlacement.sx`: moving a parameter after Adam initialized its
  state fails before an update instead of partially migrating that state;
- `DenseDimensions.sx`: a dense layer validates the final input dimension;
- `CheckpointSchema.sx`, `CheckpointMissing.sx`,
  `CheckpointAdditional.sx`, `CheckpointDType.sx`, and `CheckpointShape.sx`:
  checkpoint schema and the complete named parameter contract are validated
  before any model value is replaced;
- transposed GPU views are covered as successful stride-aware computation in
  `Tests/GPUCompute.sx`.

- `ProfileSnapshotMutation.sx`: public profile snapshots have immutable fields.

Boolean masks add the following fixtures:

- `BooleanArithmetic`, `BooleanUnary`, `BooleanReduction`, `BooleanLinear`:
  numeric operations reject booleans, including an identity reduction;
- `BooleanCast`, `NumericBooleanCast`: no bool/number cast;
- `BooleanOrder`, `BooleanExtraction`, `NumericLogic`: exact dtype contracts;
- `WhereMask`, `WhereBranchDType`, `WhereBooleanBranch`: boolean condition and
  matching numeric branches;
- `MaskBroadcast`, `MaskPlacement`, `MaskCrossDevice`, `MaskGPUShapeRank`:
  shape, placement, device, and rank validation;
- `MaskGPUInteger`, `MaskGPUExtraction`: GPU compute dtype and explicit readback;
- `MaskGPUCompaction`: dynamic compaction requires the CPU explicitly.

Each name above refers to the corresponding `.sx` file.

Functional gradients add the following fixtures:

- `GradUnused`, `GradUntrackedInput`, `GradUntrackedOutput`, `GradEmptyInputs`,
  `GradOutputDType`: tracked float32 provenance and explicit unused-input policy;
- `GradSeedMissing`, `GradSeedShape`, `GradSeedDType`, `GradSeedPlacement`,
  `GradSeedDevice`: exact seed shape, dtype, placement, and device;
- `GradConsumed`, `GradConsumedAncestor`, `BackwardAfterGrad`, `GradAfterBackward`:
  graph consumption is shared by both reverse-mode entry points;
- `GradDetachedResult`: returned derivatives cannot be differentiated again;
- `GradcheckEmptyInputs`, `GradcheckDType`, `GradcheckGPU`, `GradcheckOutputDType`:
  nonempty float32 CPU input lists and float32 CPU outputs;
- `GradcheckEpsilon`, `GradcheckTolerance`, `GradcheckNonfiniteTolerance`,
  `GradcheckPerturbation`, `GradcheckNonfiniteInput`: finite numerical options,
  input values, and effective perturbations;
- `GradcheckShape`: the callback keeps a stable output shape.

Each name refers to the corresponding `.sx` file.

Shape adaptation adds `UnsqueezeAxis`, `UnsqueezeNegative`, `SqueezeAxis`, and
`SqueezeDimension` for valid nonnegative axes and unit dimensions;
`ExpandRank`, `ExpandDimension`, `ExpandEmpty`, and `ExpandNegative` for exact
broadcast views; `ExpandOverflow` and `ExpandBytes` for logical sizes;
`RepeatRank`, `RepeatNegative`, `RepeatOverflow`, and `RepeatBytes` for checked
tiling factors and sizes. Each name refers to its `.sx` fixture.

Statistics add `VarianceCorrection`, `VarianceDomain`, `VarianceEmpty`,
`VarianceInteger`, `VarianceBoolean`, `VarianceAxis`, `VarianceDuplicate`,
`DeviationCorrection`, `DeviationDomain`, `DeviationDType`, and `VarianceGPURank`:
explicit nonnegative correction below domain size, float32 data, valid axes,
and the existing GPU rank limit. Each name refers to its `.sx` fixture.

Sequence constructors add `ArangeStep`, `ArangeValue`, `ArangeLast`, and
`ArangeCardinality` for nonzero steps, int32 outputs, and checked cardinality;
`LinspaceCount`, `LinspaceNaN`, `LinspaceInfinity`, and `LinspaceBytes` for
nonnegative counts, finite bounds, and physical sizes. Each name is its `.sx` fixture.

State primitives add `BufferName` for a nonempty name, `BufferDType`,
`BufferShape`, and `BufferPlacement` for replacement compatibility,
`BufferOptimizer` for the distinct Buffer/Parameter types, and
`ParameterFreezeGradient`/`ParameterUnfreezeGradient` for an empty slot before
changing trainability. Each name refers to its `.sx` fixture.

Layer registration adds `RegistryChildName`, `RegistryChildCollision`,
`RegistryDirectCycle`, `RegistryIndirectCycle`, `RegistryParameterCollision`,
`RegistryBufferCollision`, `RegistryMixedCollision`, `RegistryTransitiveCollision`,
and `RegistryLaterCollision` for nonempty local aliases, unique names, acyclic
registered edges, and unique state names even after child changes. Each name
refers to its `.sx` fixture.

Recursive state adds `LayerFreezeGradient`, `LayerUnfreezeGradient`,
`LayerToGradient`, and `LayerCPUGradient` for empty slots before mutation;
`LayerCrossDeviceBuffer`/`LayerCrossDeviceParameter` for explicit CPU staging;
`LayerModeCollision` for collecting and validating all names before a mode hook.
Each name refers to its `.sx` fixture.
