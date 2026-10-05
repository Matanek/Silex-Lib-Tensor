# Add a differentiable operation

Implement `Autograd.Operation`, then call `Autograd.apply(operation, inputs)` to
connect your calculation to tracked leaves. `forward` and `backward` use only
Tensor's public API; `gradcheck` checks the rule on CPU.

This program defines an elementwise square and verifies its gradient:

```sx
use Tensor
use Tensor.Autograd

struct Square:Autograd.Operation {
    func forward(inputs:Tensor[]) Tensor {
        return inputs[0].multiply(inputs[0])
    }
    func backward(inputs:Tensor[], output:Tensor, gradient:Tensor) Tensor[] {
        return [inputs[0].multiply(2.0).multiply(gradient)]
    }
}
func energy(inputs:Tensor[]) Tensor { return Autograd.apply(Square(), inputs).sum() }
func main() {
    var values:float[] = [2.0, 3.0]
    var input = Autograd.Variable(Tensor.vector(values))
    energy([input.value()]).backward()
    if let gradient = input.gradient() {
        assert(gradient.values()[0] == 4.0 && gradient.values()[1] == 6.0)
    } else { panic("missing gradient") }
    let check = Autograd.gradcheck(energy, [Tensor.vector(values)],
        epsilon:0.001, absolute_tolerance:0.003, relative_tolerance:0.002)
    assert(check.passed && check.checked_derivatives == 2)
    print("Custom gradient: 4, 6; gradcheck: pass")
}
```

It prints `Custom gradient: 4, 6; gradcheck: pass`.
The [complete example](../../Tests/Consumer/Examples/CustomOperation.sx) is an
external package consumer.

## Define the rule

`forward(inputs:Tensor[]) Tensor` receives detached input values and returns a
detached Tensor. `backward(inputs:Tensor[], output:Tensor,
gradient:Tensor) Tensor[]` receives the same values, the detached forward output
and the detached upstream gradient. It returns **one contribution per input**,
in the same order, including untracked inputs.

Each contribution has exactly its input's shape, placement and device, and uses
`float32`. The forward output may change shape; the input list is nonempty and
all inputs use `float32` on the same placement and device. No conversion,
CPU/GPU copy or gradient adaptation is implicit. A broadcasting rule must
therefore explicitly reduce contributions to input shapes. Views and empty
shapes follow public operations.

Repeating an input in apply adds its contributions; an untracked input participates
in the calculation without receiving a slot. With no tracked input, apply returns
the detached output directly and retains no graph rule. Seeds for backward and grad
keep their usual conventions. Contributions are detached: only first-order
derivatives are supported. A consumed Tensor graph remains rejected.

## Preserve the graph and accumulators

Detached inputs, the output and the operation value are owned by the graph until
successful consumption or destruction. Consumption releases these references even
while the calculated output remains alive. References explicitly held by the
application have their own lifetime.

Leaf gradient additions are prepared before publishing slots. When a graph
contains a custom rule, all its rules are also validated before consuming
operations. Graphs composed only of built-in primitives keep their progressive
release of values. Invalid cardinality, shape, dtype, provenance or placement
produces a diagnostic without partial engine accumulation. `Autograd.grad`
publishes no slots. Errors follow the fatal behavior of Tensor operations; no
new recoverable result is introduced.

The rule must depend on its input data and must not modify Variable, Parameter
or external state during differentiation. The engine does not restore arbitrary
callback effects or captured state mutations. For gradcheck, both the rule and
forward must also be deterministic between probes.

See [gradient checking](Gradients.md) for seeds, the complete Jacobian and
tolerances, and the [reference](Reference.md) for placements.
