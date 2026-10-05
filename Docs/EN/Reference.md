# Tensor 0.2.0 compatibility reference

Use this page to choose a dtype, placement, and operation family. The
[Tensor guide](README.md) details shapes, views, axes, and each operation's
contract.

## Execution matrix

| Family | CPU dtypes | GPU dtypes | Differentiable |
| --- | --- | --- | --- |
| Construction, views, `reshape`, `permute`, `select`, `narrow` | all nine + booleans | metadata for all nine + booleans | tracked `float32` views |
| `add`, `subtract`, `multiply`, `divide` | all nine | `float32` | `float32` |
| `negate` | `float32`, signed integers | `float32` | `float32` |
| `abs` | all nine | `float32` | `float32` away from zero |
| `exp`, `log`, `sqrt` | `float32` | `float32` | yes |
| `sum`, `min`, `max` | all nine | `float32` | `sum` only |
| `mean`, `dot`, `matmul` | `float32`; `dot`/`matmul` also integers | `float32` | `float32` |
| `concatenate`, `stack` | all nine + booleans | all nine + booleans when resident | `float32` |
| `gather` | any source, `int32` indices | `float32` source, `int32` indices | `float32` source |
| numeric comparisons | all nine | `float32` | no, detached boolean output |
| boolean equality, logic | booleans | booleans | no |
| `where` | branches: all nine; boolean mask | `float32` branches; boolean mask | `float32` branches |
| `masked_select` | all nine + booleans | explicit rejection | `float32` source |
| unit dimensions, `expand`, `repeat` | all nine + booleans | all nine + booleans | float32 |
| `variance`, `standard_deviation` | `float32` | resident `float32` | yes, explicit correction |
| `topk` | all nine | resident `float32` | float32 values; detached int32 indices |
| `argmin`, `argmax` | all nine | resident `float32` | no, detached int32 indices |
| `arange`, `linspace` | int32 / float32, CPU data | explicit transfer | no |
| `Autograd.grad` | `float32` graphs | resident `float32` graphs | first order, detached results |
| `Autograd.gradcheck` | `float32` inputs/output | explicit CPU check | Jacobian by central differences |
| activations, losses, normalization, convolution, pooling, dropout | `float32` | `float32` | yes where defined |

“All nine” means `float32`, `int8`, `uint8`, `int16`, `uint16`, `int32`,
`uint32`, `int64`, and `uint64`. Transfer availability does not imply a compute
kernel: all integer GPU arithmetic is rejected before pipeline creation or
submission.

## Conversions and extraction

Constructors infer dtype from the Silex array. `float`, `int`, and `uint`
become `float32`, `int64`, and `uint64` respectively. `cast(dtype)` performs a
checked numerical conversion on the CPU; a GPU tensor follows the explicit
`tensor.cpu().cast(dtype).to(device)` path.

`values()`, `item()`, and `at()` read `float32` only. Every integer has exact
forms such as `int32_values()`, `int32_item()`, and `int32_at(indices)`.
Extracting another dtype fails without conversion.
`bool_values()`, `bool_item()`, and `bool_at(indices)` extract booleans.
`cast` does not convert between booleans and numbers. The [mask guide](Masks.md)
details comparisons, gradients, and compaction limits. GPU extraction first
requires `cpu()`.

## Overflow and floating values

CPU integer operations remain in their operands' dtype. Addition,
subtraction, multiplication, negation, division, sum, dot product, and matrix
product fail whenever a result is not representable. Division by zero and the
minimum signed value divided by `-1` also fail.

`float32` preserves observable IEEE rules: NaNs and infinities propagate,
division by zero is not converted into a Tensor error, and `min`/`max`
distinguish both signs of zero. CPU/GPU and differential comparisons use
operation-specific absolute and relative tolerances. The
[numeric report](../../Tools/Oracle/REPORT.md) and
[neural report](../../Tools/Oracle/NEURAL_REPORT.md) record the exact values
used by the tests.

## Placement, synchronization, and lifetime

`to(device)` is an explicit upload; `cpu()` is a download and waits for every
command on which the result depends. GPU operations remain ordered and
resident up to that boundary. Moving between two devices is written
`tensor.cpu().to(other_device)`.

An autograd graph is consumed by a successful `backward()`. Gradients
accumulate on leaves until `zero_grad()`. Parameters, gradients, and optimizer
state remain on one placement; partial movement is rejected after state has
been created.

`Neural.Parameter.replace(value)` replaces a value only after `zero_grad()` and
preserves dtype, shape, and placement. The supplied value becomes a new detached
leaf; any optimizer state is not reset.

## Structural errors

Before mutation or submission, Tensor diagnoses incompatible shapes, invalid
or duplicate axes, out-of-bounds CPU indices, mixed dtypes, placements or
devices, invalid hyperparameters, already-consumed graphs, and incompatible
checkpoints. GPU `gather` preserves its resident path: an out-of-range index
produces a NaN sentinel instead of a hidden readback.

Model and checkpoint limits are summarized in the
[0.2.0 direction](Direction.md). Complete workflows are available in the
[training recipes](Recipes/Training.md).

## GPU profiling

`Tensor.Profile` defines a period on one device; `snapshot()` returns immutable
`Tensor.ProfileStatistics` and `reset()` starts a new period. See
[attribution, counters and lifetime](Profiling.md).

The [functional gradient guide](Gradients.md) explains preserved slots, seeds,
graph consumption, and Jacobian checking.

[Shapes, views, and tiles](Shapes.md): axes, empty dimensions, and gradients.

[Variance and standard deviation](Statistics.md): correction, domains, and constant gradients.

[Progressions and grids](Sequences.md): bounds, steps, cardinality, and dtypes.

[Buffers and parameter freezing](State.md): detachment, identity, transfers, and retained optimizer state.

[Layer registration](Registration.md): named children, order, identities, cycles, and collisions.
