# Build indices and a numerical grid

`Tensor.arange` creates an int32 progression with an exclusive stop.
`Tensor.linspace` builds a float32 grid with an explicit point count.
Both create CPU data, which you can then transfer through `to(device)`.

This program selects observations and weights them with a grid:

```sx
use Tensor
use Tensor.Autograd

func main() {
    var data:float[] = [0.0,10.0,20.0,30.0,40.0,50.0]
    var input = Autograd.Variable(Tensor.vector(data))
    let indices = Tensor.arange(1,6,2)
    let weights = Tensor.linspace(0.0,1.0,3)
    let selected = input.value().gather(0,indices)
    let loss = selected.multiply(weights).sum()
    assert(loss.detach().item() == 65.0)
    var gradients = Autograd.grad(loss,[input.value()])
    let actual = gradients[0].values()
    assert(actual[1] == 0.0 && actual[3] == 0.5 && actual[5] == 1.0)
    assert(input.gradient() == null)
}
```

Indices are [1,3,5], selected values [10,30,50], and weights [0,0.5,1]. The weighted
sum is65. Indices and weights are detached data; the gradient traverses selection
and preserves the Variable's slot.

## Build an integer progression

`arange(stop)` starts at zero with a step of one.
`arange(start, stop, step:1)` takes `int` parameters and returns an int32 vector.
The stop is exclusive. A positive step advances toward a larger stop; a negative
step moves toward a smaller stop. A direction that does not reach stop, or equal
bounds, returns an empty vector. A zero step is rejected even for an empty progression.

Cardinality and the last element are calculated without signed subtraction
that could overflow at int64 extremes. Every returned value must fit int32,
and cardinality/sizes are validated before allocation. Bounds and steps may
exceed int32 when returned values remain valid; an empty progression contains
no values to check. These rules prevent silent conversion and an unnecessary
last increment beyond the interval.

## Interpolate floating values

`linspace(start, stop, count)` requires finite bounds and a nonnegative count.
Bounds are checked even for count0. Count0 returns an empty vector; count1
returns start. With at least two points, start and stop are preserved exactly,
including signed zeros. A grid may ascend, descend, or remain constant.

Interpolation avoids computing an overflowing stop−start for opposite-sign
bounds and keeps every result between the endpoints. Float32 rounding may
merge neighboring points; no extra precision is promised. Physical size is
validated before filling values.

These constructors track no Tensor parameter: they create reusable data,
not a differentiable transformation of scalar bounds. On the GPU, only the
consumer's explicit `to(device)` performs an upload. No automatic placement
or implicit download is introduced.

See [batch shapes](Shapes.md) and the [gradient guide](Gradients.md) to use
these data in differentiable computation.
