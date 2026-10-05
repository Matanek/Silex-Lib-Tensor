# Select values with a mask

Use a comparison to choose which values participate in a computation.
`Tensor.where` keeps a broadcast shape and differentiates both numeric branches.
`masked_select` compacts values on the CPU.

This program keeps positive values and replaces the others with zero:

```sx
use Tensor
use Tensor.Autograd.Variable

func main() {
    var values:float[] = [-2.0, 1.0, 3.0]
    var input = Variable(Tensor.vector(values))
    let mask = input.value().greater_than(0.0)
    let selected = Tensor.where(mask, input.value(), Tensor.zeros([3]))
    selected.sum().backward()
    let result = selected.detach().values()
    assert(result[0] == 0.0 && result[1] == 1.0 && result[2] == 3.0)
    if let gradient = input.gradient() {
        let gradient_values = gradient.values()
        assert(gradient_values[0] == 0.0 && gradient_values[1] == 1.0 && gradient_values[2] == 1.0)
    } else { panic("expected gradient") }
    let positive = input.value().detach().masked_select(mask).values()
    assert(positive.count() == 2 && positive[0] == 1.0 && positive[1] == 3.0)
}
```

The mask is a detached boolean value. The comparison creates no gradient path;
`where` passes the gradient to each numeric branch at positions where it is
selected. Broadcasting sums contributions along repeated dimensions, and two
branches sharing a leaf accumulate their contributions. An unselected NaN does
not contaminate the output or the gradients computed by `where`.

## Build and combine masks

A Silex `bool[]` array constructs a Tensor of `DType.boolean()`. The `scalar`,
`vector`, `matrix`, `full`, and values-shape constructors accept booleans.
`zeros(shape, DType.boolean())` produces `false`, and `ones` produces `true`.
`bool_values()`, `bool_item()`, and `bool_at(indices)` are exact CPU extractors.
On the GPU, call `cpu()` first.

The `equal`, `not_equal`, `less_than`, `less_equal`, `greater_than`, and
`greater_equal` comparisons accept a Tensor or a scalar of the same dtype.
All nine numeric dtypes are available on the CPU; GPU comparisons support
`float32`. Equality and inequality also accept booleans. `logical_not`,
`logical_and`, and `logical_or` operate on masks; `logical_and/or` also accept
a `bool` scalar.

All these operations use broadcasting aligned from the right and preserve
strided views. For floats, NaN equals nothing and makes ordered comparisons
false; `not_equal` is true with NaN. Infinities follow the usual order, and
both signs of zero are equal. 64-bit integers are compared exactly without
passing through `float32`.

Booleans have private logical storage of one byte per element. `is_boolean()`
distinguishes them from numbers; `is_integer()` and `is_floating()` both return
`false`. Views, exact transfers, `contiguous`, `concatenate`, and `stack` preserve
this dtype. Arithmetic, numeric reductions, and ordered comparisons reject
booleans. `cast` does not convert between booleans and numbers; no implicit
conversion exists.

## Choose a shape and placement

`Tensor.where(mask, when_true, when_false)` requires a boolean mask and two
numeric branches of the same dtype. All three shapes broadcast together.
All three values stay on the same placement and, on the GPU, the same device.
CPU branches accept all nine numeric dtypes; GPU branches accept `float32`.
GPU computation and scalar uniforms perform no hidden transfers. The GPU
supports ranks up to five and buffer sizes representable as `uint32`; an empty
shape submits no computation.

`source.masked_select(mask)` broadcasts both shapes and returns a vector of
selected values in logical order, including from a view. An all-false mask
returns an empty vector. The gradient of a `float32` source accumulates repeated
contributions from broadcasting. CPU gather also accepts a boolean source.
The source cardinality must fit `int32` indices.

Dynamic compaction stays on the CPU: on the GPU, it fails before pipeline
creation or submission. To request it explicitly, use
`source.cpu().masked_select(mask.cpu())`, with `detach()` first if the source
is tracked. To preserve a differentiable GPU chain, use `where` with a known
shape.

See the [compatibility matrix](Reference.md) for other operations and
[GPU profiling](Profiling.md) to observe explicit transfers.
