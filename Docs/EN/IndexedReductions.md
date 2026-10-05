# Find and use an extremum index

`argmin` and `argmax` find the position of a minimum or maximum. Use that index
with `gather` to select a value and retain its gradient. The first index wins
when several values are equal.

This program selects the first maximum observation:

```sx
use Tensor
use Tensor.Autograd

func main() {
    var observations:float[] = [2.0,7.0,7.0]
    var input = Autograd.Variable(Tensor.vector(observations))
    let index = input.value().argmax().reshape([1])
    let selected = input.value().gather(0,index).sum()
    assert(index.int32_values()[0] == 1 && selected.detach().item() == 7.0)
    var gradients = Autograd.grad(selected,[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 0.0 && actual[1] == 1.0 && actual[2] == 0.0)
    assert(input.gradient() == null)
}
```

The index is 1, the selected value 7, and its gradient is 1 only at position 1.
Indices are detached int32 data; the gradient flows through `gather` to the
observations without changing the Variable slot.

## Choose the domain and shape

`argmin(axis:int?=null, keep_dimensions:bool=false)` and `argmax` share the same
rules. Without an axis, they return the index in logical row-major flattened
order, even for a noncontiguous view. With an axis, they return the coordinate
on that axis for each combination of the other coordinates. The axis must be
nonnegative and less than the rank.

Reduced dimensions disappear; `keep_dimensions:true` retains them with size 1.
Without an axis, the output is scalar, or retains all unit dimensions when
requested. A scalar input returns index 0. An empty reduced domain is rejected,
even when the output would be empty. Empty outputs are accepted for nonempty
reduced domains: `[0,3]` with axis 1 returns `[0]`.

## Predict ties and nonfinite values

Ties retain the first logical index; +0 and −0 are equal. If the domain contains
NaNs, its first NaN is selected by both operations. Otherwise infinities follow
their usual numeric order. This index rule differs from the signed zero choice
made by `min` and `max`.

## Choose placement

All nine CPU numeric dtypes are accepted, with exact integer comparison through
int64/uint64. Booleans are rejected. GPU input must be float32; output remains
int32 on the same device without implicit transfers. Read it with
`result.cpu().int32_values()` when needed. Views with offsets, strides and
expanded dimensions are supported; the GPU retains its rank 5 and uint32
metadata limits.

The last index must fit int32: at most 2^31 elements in the domain. Axes, sizes
and metadata are checked before allocation/submission. Indices do not
differentiate; selected values can. No performance advantage over a value
reduction is promised. See the [gradient guide](Gradients.md) and
[shapes and views](Shapes.md) to compose the selection.
