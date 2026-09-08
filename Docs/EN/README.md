# Tensor

`Tensor` represents multidimensional numerical values without exposing their
storage. This first version executes operations eagerly on the CPU and uses the
Silex `float` type.

## Create a tensor

Values use row-major order and their count must match the declared shape:

```sx
use Tensor

let image = Tensor([
    0.1, 0.2, 0.3,
    0.4, 0.5, 0.6
], [2, 3])

assert(image.rank() == 2)
assert(image.at([1, 2]) == 0.6)
```

Factories cover common intentions:

```sx
let bias = Tensor.ones([3])
let weights = Tensor.zeros([3, 4])
let temperature = Tensor.scalar(0.7)
```

`shape()` and `values()` return detached copies. A `Tensor` has no public
mutation and preserves Silex value semantics. A future storage optimization
will not change this guarantee.

## Compute

Element-wise operations currently require identical shapes. A scalar value can
be added, subtracted, multiplied, or divided without creating an intermediate
tensor:

```sx
let left = Tensor.vector([1.0, 2.0, 3.0])
let right = Tensor.vector([4.0, 5.0, 6.0])
let centered = left.add(right).divide(2.0)

assert(centered.sum() == 10.5)
assert(centered.mean() == 3.5)
```

`matmul()` and `transpose()` currently carry the explicit contract of rank-2
matrices:

```sx
let inputs = Tensor.matrix([1.0, 2.0, 3.0, 4.0], 2, 2)
let weights = Tensor.matrix([2.0, 0.0, 0.0, 3.0], 2, 2)
let outputs = inputs.matmul(weights)

assert(outputs.at([1, 1]) == 12.0)
```

A negative shape, inconsistent value count, invalid index, or incompatible
shapes terminates the program with a diagnostic. `mean()` rejects an empty
tensor.

## Direction

The contract favors the readable eager use popularized by PyTorch, with
immutable values close to JAX's functional approach. Graph, autodiff, dtype,
and device concerns remain outside this first surface so a backend can be
chosen later without leaking it into basic usage.

Broadcasting, slices, and higher-rank linear algebra will be added with their
own consumer proofs. `AI` may depend on `Tensor`; the inverse would be an
incorrect domain dependency.

## Development

From the `SilexProject` root:

```text
silex link Packages/Tensor
silex link Packages/Tensor --workspace Packages/Tensor/Tests/Consumer
silex test Packages/Tensor/Tests/Consumer
```
