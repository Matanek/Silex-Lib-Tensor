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
numeric family. `zeros`, `ones`, and the `full` overloads cover common filled
constructions.

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

## Value semantics

A `Tensor` can remain in a `let`. Ordinary assignment shares its immutable
storage and safe lifetime. A deep `copy` is rejected when the tensor reaches a
non-clonable GPU resource; no native handle is duplicated.

## Development

From the `SilexProject` root:

```text
silex test Packages/Tensor/Tests/Consumer
silex check Packages/Tensor
```
