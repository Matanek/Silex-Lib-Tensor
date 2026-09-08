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

`shape()`, `rank()`, `count()`, `dtype()`, `is_cpu()`, and `is_gpu()` cause no
transfer. `cpu()` is the first synchronization point in a GPU chain. Value and
item extraction is CPU-only.

This release runs only `add(float)` and `multiply(float)` for `float32` on the
GPU. All nine dtypes transfer without loss, but integer GPU computation is
rejected before submission. Migration between two devices remains explicit:
`tensor.cpu().to(other_device)`.

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
