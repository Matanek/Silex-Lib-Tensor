# Calculate and check gradients without changing slots

`Autograd.grad` returns detached gradients without changing the gradients
stored in `Variable` or `Parameter` values. `Autograd.gradcheck` compares a
Silex function's derivatives with central differences on the CPU.

This program calculates a gradient, preserves the stored gradient, and checks
a quadratic function:

```sx
use Tensor
use Tensor.Autograd

func energy(inputs:Tensor[]) Tensor {
    return inputs[0].multiply(inputs[0]).sum()
}

func main() {
    var data:float[] = [1.0, 2.0, 3.0]
    var input = Autograd.Variable(Tensor.vector(data))
    input.value().multiply(7.0).sum().backward()
    var gradients = Autograd.grad(energy([input.value()]), [input.value()])
    let values = gradients[0].values()
    assert(values[0] == 2.0 && values[1] == 4.0 && values[2] == 6.0)
    if let stored = input.gradient() { assert(stored.values()[0] == 7.0) }
    else { panic("expected stored gradient") }
    let check = Autograd.gradcheck(energy, [Tensor.vector(data)],
        epsilon:0.001, absolute_tolerance:0.003, relative_tolerance:0.001)
    assert(check.passed && check.checked_derivatives == 3)
}
```

## Request a graph's derivatives

`Autograd.grad(output, inputs, seed:null, allow_unused:false)` takes a nonempty
list of tracked Tensor values: Variable values, Parameter values, or intermediate
results. Results follow the requested inputs' order, shape, and placement.
Repeating an input repeats its gradient; it does not multiply its derivative.
Existing slots stay identical, absent slots stay absent, and results do not
support higher-order derivatives.

Without a seed, the output must be scalar and receives an initial gradient of
one. For any output shape, provide a `float32` seed with the same shape,
placement, and device. The result is the transposed Jacobian multiplied by
that seed. The seed is treated as detached data. CPU/GPU computation uses the
same first-order rules as `backward`, including views, broadcasting, shared
branches, and masked selection.

An unused input fails by default. With `allow_unused:true`, it receives a real
zero on its own placement; a constant output is also allowed. An untracked value
is not a differentiable input, even with this option. GPU computation stays
resident without implicit transfers; use `cpu()` to observe a result.

The operation graph is consumed after success, as with `backward`. It cannot
serve another `grad` or `backward`. Leaves remain available to build a new graph.
Provenance, seed, unused requests, and consumed nodes are validated before any
consumption; functional computation never changes a slot, including on rejection.

## Check a numerical function

`Autograd.gradcheck(evaluate, inputs, epsilon, absolute_tolerance,
relative_tolerance)` takes a `func(Tensor[]) Tensor` callback, `float32` CPU
inputs, and three explicit numerical parameters. The callback must be
deterministic, depend on its input data, and keep the same output shape,
without external mutable state. The output may be scalar, vector, or another
fixed shape: the check covers the complete Jacobian.

Each analytical derivative is compared with the central quotient
`(f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)` using the rule
`abs(a - n) <= absolute_tolerance + relative_tolerance * max(abs(a), abs(n))`.
Choose epsilon and tolerances for `float32` rounding, and avoid nonsmooth points
or mask boundaries. Epsilon must be finite, positive, have a finite double,
and produce two finite perturbations distinct from the initial value.
Tolerances must be finite and nonnegative; initial data must be finite.

`GradcheckResult` exposes `passed`, `checked_derivatives`, and `failure`. The
check stops at the first discrepancy. `GradcheckFailure` gives the logical
`input`, `element`, and `output` indices, then the `analytical` and `numerical`
values. Elements and outputs use flattened logical order, starting at zero.
A nonfinite derivative produces a failed result, while an invalid dtype,
placement, parameter, or changing shape produces a diagnostic. An empty output
passes with zero checked derivatives; this result establishes no nonempty derivative.

Temporary probes and leaves are independent of the supplied inputs. The check
does not change their slots. For N input values and M output values, it checks
up to N × M derivatives and reevaluates the callback for each probe: use small
cases. No GPU Tensor is downloaded implicitly. To compare GPU gradients, call
`grad` on the GPU and observe its results through explicit transfers, then
compare them with an independent CPU check.

See [differentiable masks](Masks.md) and the
[compatibility reference](Reference.md) to select graph operations.
