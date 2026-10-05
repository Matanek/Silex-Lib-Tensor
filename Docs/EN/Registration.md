# Register and share a layer’s state

A class derived from `Neural.Layer` registers its parameters, buffers, and children
once in its constructor. The inherited `parameters()` and `buffers()` methods
collect that state recursively, retaining each identity once. Computation remains
free: your methods may accept multiple inputs or outputs.

These two residuals share a Dense layer and a nontrainable coefficient:

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use STD.Math

nocopy class RegisteredResidual:Neural.Layer {
    private var block:Neural.Dense
    private var scale:Neural.Buffer
    init(block:Neural.Dense, scale:Neural.Buffer) {
        self.block = block
        self.scale = scale
        self.register_child("block",block)
        self.register_buffer(scale)
    }
    override func forward(input:Tensor) Tensor { return input.add(self.block.forward(input).multiply(self.scale.value())) }
}

func main() {
    var block = Neural.Dense("shared",1,1,seed:0)
    var weights = block.parameters()
    weights[0].replace(Tensor.full([1,1],2.0)); weights[1].replace(Tensor.full([1],1.0))
    var scale = Neural.Buffer("scale",Tensor.scalar(1.0))
    var left = RegisteredResidual(block,scale)
    var right = RegisteredResidual(block,scale)
    var branches:Neural.Layer[] = [left,right]
    var model = Neural.Sequential(branches)
    assert(model.parameters().count() == 2 && model.buffers().count() == 1)
    assert(model.parameters()[0] == weights[0] && model.buffers()[0] == scale)
    var optimizer = Optim.SGD(model.parameters(),learning_rate:0.1)
    left.forward(Tensor.ones([1,1])).add(right.forward(Tensor.ones([1,1]))).sum().backward()
    optimizer.step()
    assert(Math.abs(weights[0].value().item()-1.8) < 0.00001)
    assert(scale.value().item() == 1.0)
}
```

Collection yields two parameters, the Dense weight and bias, and one Buffer.
Each branch contributes to the weight gradient; SGD receives one reference and
performs one step, from 2 to approximately 1.8. The coefficient stays at 1.

## Give state an owner

The three protected methods are available inside your derived class:

- `register_parameter(parameter)` registers an existing Parameter;
- `register_buffer(buffer)` registers an existing Buffer;
- `register_child(name, child)` registers a layer under a nonempty local name.

Call them after initializing your class’s fields. The registry retains existing
identities and their lifetimes; it does not clone layers or storage. Repeating
the same registration does nothing. A child may have several local aliases.
An existing alias cannot refer to a different layer.

Parameter and Buffer names remain those provided at construction: no implicit
prefix changes them. They are unique in the model’s collected state, including
across parameters and buffers. Two references to the same object are allowed;
two distinct objects with the same name are rejected. Name independent instances,
for example `encoder.weight` and `decoder.weight`.

## Inspect a shared model

`children()` returns direct children in registration order with unique identities.
`child(name)` returns the layer at that local alias, or `null` if absent.
`parameters()` and `buffers()` return owned objects before children’s objects,
with depth-first traversal in registration order. An already visited layer is
not collected again in a shared diamond.

Returned lists may be changed locally without changing the registry. Their objects
retain shared identity. Collection is recomputed: state subsequently added to a
child is visible on the next call. If that addition causes a collision in a parent,
its next collection rejects it. Cycles visible through registered children are
rejected before adding the edge.

Dense, Conv2D, LayerNorm, SimpleRNN, and Sequential use this collection without
changing existing names. Sequential accepts several occurrences of the same layer,
executes it at each forward occurrence, and collects its state once. The current
checkpoint keeps its parameter schema: a shared parameter appears once; Buffer
and frozen state remain outside that format.

## Keep state hooks explicit

A layer adopting registration inherits `parameters()` and `buffers()` unchanged.
Older layers without registration can keep a manual override: their list is read
as an opaque boundary. The registry does not discover children such an override
keeps only in its fields.

Registration automates collection here. `train`, `eval`, `to(device)`, and `cpu()`
remain hooks to forward explicitly inside a custom layer. Moving the example’s
model to GPU therefore requires `block.to(device)` and `scale.to(device)` before
the forward; collection triggers no transfers. The [composition guide](Composition.md)
explains those hooks and the [buffer and freezing guide](State.md) explains
individual state.
