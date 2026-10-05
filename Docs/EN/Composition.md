# Compose a layer in your application

Derive from `Tensor.Neural.Layer` to give your model a common contract. Its
protected constructor is available to derived classes outside the package.
Override `forward` using public Tensor operations and layers.

This program composes two residual branches sharing one Dense:

```sx
use Tensor
use Tensor.Autograd
use Tensor.Neural
use GFX.GPU

nocopy class Residual:Neural.Layer {
    private var block:Neural.Dense

    init(block:Neural.Dense) { self.block = block }
    override func forward(input:Tensor) Tensor { return input.add(self.block.forward(input)) }
    override func parameters() Neural.Parameter[] { return self.block.parameters() }
    override func to(device:GPU.Device) { self.block.to(device) }
    override func cpu() { self.block.cpu() }
}

func main() {
    var block = Neural.Dense("shared",1,1,seed:0)
    var parameters = block.parameters()
    parameters[0].replace(Tensor.full([1,1],2.0))
    parameters[1].replace(Tensor.full([1],1.0))
    var left:Neural.Layer = Residual(block)
    var right:Neural.Layer = Residual(block)
    assert(left.parameters()[0] == right.parameters()[0])
    var observations:float[] = [1.0,2.0]
    var input = Autograd.Variable(Tensor(observations,[2,1]))
    let output = left.forward(input.value()).add(right.forward(input.value())).sum()
    assert(output.detach().item() == 22.0)
    var gradients = Autograd.grad(output,[parameters[0].value(),parameters[1].value(),input.value()])
    assert(gradients[0].item() == 6.0 && gradients[1].item() == 4.0)
    for value in gradients[2].values() { assert(value == 6.0) }
    assert(parameters[0].gradient() == null && parameters[1].gradient() == null && input.gradient() == null)
}
```

The weight is 2 and the bias 1. Each residual computes x + (2x + 1); together
the two branches yield [8,14] for [1,2], then a sum of 22. The shared weight
receives a gradient of 6, the bias 4, and each input 6. Sharing retains Parameter
identity: the branches do not create two weights.

## Forward the layer methods

`forward` is the computation entry point. `parameters` returns parameters to
optimize. For a layer containing children, also forward `train`, `eval`,
`to(device)` and `cpu()` according to their state and placement. In this example,
Dense has no train/eval state; only parameters and transfers are forwarded.
Base implementations return an empty parameter list and no-op state hooks.
Layer is intended for derivation; direct construction is rejected.

Methods dispatch according to the actual class, even through a Layer variable
or Sequential. The compiler infers the mutable contract of the whole virtual
family when an override modifies state; a base with an empty hook is sufficient.
A stateful layer therefore uses a `var` receiver, without an artificial marker
in the base class.

## Keep gradients and placement explicit

Composition uses ordinary differentiable Tensor operations. `Autograd.grad`
can differentiate both shared parameters and inputs without filling their slots.
On GPU, computation remains resident; only explicitly requested `to(device)`
and `cpu()` perform transfers.

Passing an existing layer to several parents retains its identity. For the
optimizer, collect each shared Parameter once; current Sequential still requires
unique names in its collection. Automatic recursive registration
still requires manual forwarding. The [buffer and freezing guide](State.md)
explains the available state primitives. The [training recipe](Recipes/Training.md)
explains the existing loop; the [gradient guide](Gradients.md) details slots and
graph consumption.
