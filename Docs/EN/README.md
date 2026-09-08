# Tensor

`Tensor` represents a dense numerical value without exposing its storage. The
same public type covers `float32` and signed and unsigned 8-, 16-, 32-, and
64-bit integers on both CPU and GPU.

## Create and inspect

`scalar`, `vector`, `matrix`, and the shape-values constructor infer the dtype
from the supplied Silex type. `float` becomes `float32`, `int` becomes `int64`,
and `uint` becomes `uint64`.

```sx
use Tensor
use Tensor.DType

var samples:float[] = [1.0, 2.0, 3.0]
var labels:int32[] = [2 as int32, 1 as int32, 0 as int32]

let x = Tensor.vector(samples)
let y = Tensor(labels, [3])

assert(x.dtype() == DType.float32())
assert(y.dtype() == DType.int32())
assert(y.int32_values()[0] == 2 as int32)
```

The nine `DType` values are `float32`, `int8`, `uint8`, `int16`, `uint16`,
`int32`, `uint32`, `int64`, and `uint64`. They expose only their width and
numeric family. `DType` belongs directly to the `Tensor` module, with no
intermediate child module: `use Tensor.DType` selects that exact declaration.
`zeros`, `ones`, and the `full` overloads cover common filled constructions.

`values()` and `item()` extract only `float32`. Each integer dtype has matching
extractors, such as `int32_values()` and `int32_item()`. A mismatched extraction
fails instead of converting silently. `cast(dtype)` performs a checked numeric
conversion on the CPU and preserves the shape.

## Transform shapes and views

A shape contains positive or zero dimensions. A scalar uses `[]`, and a zero
dimension produces an empty tensor. `shape()`, `strides()`, `offset()`,
`rank()`, `count()`, and `is_contiguous()` inspect the layout without reading
elements, including on the GPU. Strides describe a dense row-major layout for
a new tensor.

`reshape(shape)` and `flatten()` do not copy a contiguous tensor. A reshape
shape may contain exactly one inferred `-1`; every other dimension remains
positive or zero. Inference made ambiguous by a zero cardinality is rejected.

`permute(axes)` reorders every axis. `transpose()` is its matrix shortcut.
`select(axis, index)` removes an axis, while `narrow(axis, start, count)` keeps
the axis and selects a unit-step range. These operations produce immutable
views that share their storage and keep it alive:

```sx
use Tensor

var values:int32[] = [
    1 as int32, 2 as int32, 3 as int32,
    4 as int32, 5 as int32, 6 as int32,
]

let matrix = Tensor(values, [2, 3])
let columns = matrix.transpose()
let compact = columns.contiguous()

assert(columns.shape()[0] == 3)
assert(columns.strides()[0] == 1)
assert(columns.int32_at([2, 1]) == 6 as int32)
assert(compact.offset() == 0)
```

Indices are zero-based and non-negative, and scalar access supplies one index
per dimension. `at()` reads `float32`; `int32_at()` and the matching typed
forms preserve the exact dtype. Like array extractors, these accesses are
CPU-only: call `cpu()` explicitly before reading a GPU tensor.

`contiguous()` preserves a tensor that already covers its canonical storage;
otherwise, it materializes the logical values into new dense storage without
changing the dtype. On the GPU this remains a GPU copy and causes no readback.
Elementwise operations in this release address view strides directly;
`contiguous()` remains available when a consumer explicitly requires canonical
dense storage.

## Broadcasting and elementwise computation

`add`, `subtract`, `multiply`, and `divide` align dimensions from the right.
Two dimensions are compatible when they are equal or either is `1`; a zero
dimension is therefore compatible with `0` or `1`. A scalar with shape `[]`
follows the same rule:

```sx
use Tensor

var samples:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
var bias:float[] = [10.0, 20.0, 30.0]

let matrix = Tensor(samples, [2, 3])
let shifted = matrix.add(Tensor.vector(bias)).multiply(2.0)

assert(shifted.shape()[0] == 2 && shifted.shape()[1] == 3)
assert(shifted.at([1, 2]) == 72.0)
```

Both tensors in a binary operation retain the same dtype and placement. On the
CPU, all four verbs accept all nine dtypes. Integers retain their dtype and
integer division; overflow, non-representable subtraction, division by zero,
and a non-representable signed quotient fail before computation. A scalar also
has an exact type: write `tensor.add(1 as int32)` for an `int32` tensor.

`negate` accepts `float32` and signed integers. `abs` accepts all nine dtypes
and leaves unsigned integers unchanged. `exp`, `log`, and `sqrt` are restricted
to `float32`. IEEE behavior remains observable for `float32`: division by zero,
NaN, and infinities are not converted into Tensor errors.

The GPU executes these nine verbs only for `float32`, including strided views
and broadcast shapes from rank 0 through 5. Integers may reside on the GPU and
return bit-exactly to the CPU, but any integer GPU computation fails before
pipeline creation or submission. Elementwise pipelines are reused throughout
a resident chain.

## Reductions and linear algebra

`sum`, `mean`, `min`, and `max` always return a `Tensor`. With no argument they
reduce every axis and produce the scalar shape `[]`. The optional `axes`
parameter selects dimensions to reduce; `[]` performs no reduction, while
`keep_dimensions:true` replaces each reduced dimension with `1`:

```sx
use Tensor

var values:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
let matrix = Tensor(values, [2, 3])

let columns = matrix.sum([0])
let rows = matrix.mean([1], true)

assert(columns.shape()[0] == 3 && columns.values()[2] == 9.0)
assert(rows.shape()[0] == 2 && rows.shape()[1] == 1)
```

`sum`, `min`, and `max` accept all nine dtypes; `mean` is restricted to
`float32`. A sum over an empty domain is zero. `mean`, `min`, and `max` reject
an empty domain when an output value exists. Out-of-range and duplicate axes
are also rejected. NaN propagates, and `min`/`max` select `-0.0` and `+0.0`
respectively when both signed zeros occur. Integer sums remain in their dtype
and fail on overflow.

`dot` accepts two vectors of the same length and returns a scalar Tensor.
`matmul` accepts two matrices whose inner dimensions match:

```sx
var left:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
var right:float[] = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]

let product = Tensor(left, [2, 3]).matmul(Tensor(right, [3, 2]))
assert(product.at([1, 1]) == 154.0)
```

Both operands retain the same dtype and placement. On the CPU, `dot` and
`matmul` accept `float32` and all eight integer dtypes; integer products and
sums are checked within the dtype. On the GPU, reductions, `dot`, and `matmul`
accept only `float32`, return a GPU Tensor, and cause no implicit readback.
The floating CPU/GPU comparisons in the tests use an absolute tolerance of
`1e-5 × term count` and a relative tolerance of `1e-5 × |reference|`, taking
the larger value.

`matmul` also accepts batches of matrices with rank 2 or greater. Leading
dimensions follow the usual broadcasting rules, while the last two dimensions
hold rows, columns, and the shared inner dimension. CPU and GPU paths address
batches and strided views directly.

## Compose and select

`Tensor.concatenate(tensors, axis)` joins a non-empty list along one axis. All
other dimensions, dtype, placement, and device must match.
`Tensor.stack(tensors, axis)` first adds an axis and then joins identical
shapes. Both operations stay on the current placement and accept all nine
dtypes.

`source.gather(axis, indices)` selects one value per position outside the
chosen axis. `indices` is an `int32` Tensor shaped like the source without that
axis: logits `[batch, classes]` are therefore selected by classes `[batch]`.
On GPU, this operation is the authorized `int32` addressing exception for a
`float32` source; it does not enable general integer computation.
Indices must belong to the selected axis. The CPU path validates them before
access; to keep the GPU path resident, an out-of-range GPU index produces a
NaN sentinel in the output instead of causing a hidden readback.

## Eager neural computation

`relu`, `sigmoid`, `tanh`, `softmax(axis)`, and `log_softmax(axis)` operate on
`float32` tensors. Softmax and log-softmax subtract the axis maximum before the
exponential. NaNs propagate; an empty or invalid axis follows the same failure
rules as reductions.

`layer_norm(dimensions, epsilon)` normalizes the trailing dimensions without
trainable state in this foundational version. `mse_loss(target)` returns the
scalar MSE. `cross_entropy(target, axis)` accepts either a dense `float32`
target with the same shape or an `int32` class Tensor whose shape omits the
class axis. These operations compose ordinary Tensors and remain GPU-resident
until `cpu()`.

2D convolution has one public layout: NCHW input and OIHW kernel.
`conv2d(kernel, bias, stride, padding)` accepts a scalar stride and symmetric
padding. `max_pool2d` and `average_pool2d` use a square window; stride defaults
to the window size. Average pooling counts padding cells as zero in its
divisor. Dilation, groups, and transposed convolution are outside 0.1.0.

```sx
use Tensor

let logits = Tensor.ones([4, 10]).matmul(Tensor.ones([10, 3]))
let probabilities = logits.softmax(1)
var labels:int32[] = [0 as int32, 1 as int32, 2 as int32, 0 as int32]
let classes = probabilities.gather(1, Tensor.vector(labels))

let features = Tensor.ones([1, 3, 16, 16])
let kernels = Tensor.ones([8, 3, 3, 3])
let pooled = features.conv2d(kernels, stride:1, padding:1).relu().max_pool2d(2)
```

## Initialize and apply dropout

The `uniform`, `normal`, `xavier_uniform`, and `he_uniform` constructors take
an explicit `STD.Randomizer`. The same seed and call sequence reproduce the
values; a subsequent call advances the source. Initializations are created on
CPU and transferred explicitly when needed.

```sx
use STD.Randomizer
use Tensor

var randomizer = Randomizer(42)
let weights = Tensor.xavier_uniform([64, 128], randomizer)
let noise = Tensor.normal([64], randomizer, standard_deviation:0.01)
```

`dropout(probability, seed, iteration, training)` requires a seed and exposes
the iteration. The same seed-iteration pair reproduces the mask; another
iteration refreshes it. In evaluation, `training:false` keeps the Tensor. On
GPU, a private counter generates the mask in the compute pass: no mask upload
or global RNG is hidden.

## Move to the GPU

Create the device with `GFX.GPU`, then place the tensor explicitly:

```sx
use GFX.GPU
use Tensor
use Tensor.DType

var device = GPU.Device()
var values:float[] = [1.0, 2.0, 3.0]

let gpu = Tensor.vector(values).to(device)
let result = gpu.add(2.0).multiply(3.0)

assert(result.is_gpu())
let cpu = result.cpu()
assert(cpu.values()[0] == 9.0)
```

Shape and placement metadata cause no transfer. `cpu()` is the first
synchronization point in a GPU chain. Value, item, and scalar-index extraction
is CPU-only.

This release runs the binary and unary elementwise operations described above
for `float32` on the GPU. All nine dtypes transfer without loss, but integer GPU
computation is rejected before submission. Migration between two devices
remains explicit: `tensor.cpu().to(other_device)`.

## Choose CPU or GPU

The September 8, 2026 [public Release campaign](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)
measured Tensor on an 18 GiB Apple M3 Pro under macOS 26.6.2 and Metal. These
thresholds describe only that machine and Tensor `bdcd062`, GFX.GPU `bb23787`,
and Silex `1c310ce`; they do not predict another CPU, GPU, driver, or backend.

For already-resident compute, hot GPU execution overtakes the CPU at 65,536
elements for addition and global sum on the measured grid. Square `matmul`
overtakes the CPU at side 32, although its first pass includes pipeline
creation. The `add -> multiply -> matmul -> sum -> add` chain remains slower at
side 32 when transfers are included; its observed crossover is side 128. At
that point, the median is 91.889 ms on the CPU, 11.892 ms for hot GPU compute,
and 18.268 ms including both uploads and the final download, a 5.03× end-to-end
speedup. The report retains full ranges and widely dispersed CPU scheduling
regimes without removing samples; these speedup factors must not be
generalized.

For 262,144 bytes, the observed median throughput includes typed conversion,
allocation, submission, and completion wait:

| Dtype | Upload MiB/s | Download MiB/s |
| --- | ---: | ---: |
| `float32` | 30.99 | 73.13 |
| `int8` | 22.50 | 23.73 |
| `uint8` | 19.97 | 22.67 |
| `int16` | 25.70 | 39.99 |
| `uint16` | 26.48 | 41.68 |
| `int32` | 36.17 | 62.68 |
| `uint32` | 28.84 | 69.71 |
| `int64` | 41.36 | 89.63 |
| `uint64` | 39.39 | 93.89 |

Keep small isolated operations on the CPU. The GPU becomes useful when inputs
remain resident across several operations or compute explicitly amortizes
upload and download. These transfer measurements make no integer GPU compute
acceleration claim.

## Value semantics

A `Tensor` can remain in a `let`. Ordinary assignment shares its immutable
storage and safe lifetime. A deep `copy` is rejected when the tensor reaches a
non-clonable GPU resource; no native handle is duplicated.

## Development

From the `SilexProject` root:

```text
silex test Packages/Tensor/Tests/Consumer
silex test Packages/Tensor/Tests/PerformanceGuards.sx
silex check Packages/Tensor
```

The normal suite is hermetic. It reads the committed Silex fixture without
Python, downloads, or network access. The corpus covers all nine dtypes and
separates exact results, tolerated `float32` computations, bit-exact GPU
transfers, and expected errors. Comparisons handle NaN, infinities, and the sign
of zero explicitly; absolute and relative tolerances are fixed per operation
family in the [differential report](../../Tools/Oracle/REPORT.md).

Regeneration is a deliberate maintenance operation. It runs pinned NumPy,
PyTorch, TensorFlow, and JAX versions in a disposable Python environment,
requires at least two oracles to agree on every common numeric result, and then
requires review of the resulting diff. The
[generator guide](../../Tools/Oracle/README.md) records the exact commands,
seed, and license rules. Updating a framework never changes the Tensor 0.1.0
contract automatically.
