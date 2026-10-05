# Keep a buffer and freeze a parameter

A `Neural.Buffer` holds model data that must not be optimized.
A `Neural.Parameter` holds a trainable weight; `freeze()` suspends training and
`unfreeze()` resumes it with the optimizer’s existing moments.

This program trains a weight, freezes it, changes its coefficient, and resumes:

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use STD.Math

func main() {
    var weight = Neural.Parameter("weight",Tensor.scalar(2.0))
    var scale = Neural.Buffer("scale",Tensor.scalar(3.0))
    var optimizer = Optim.SGD([weight],learning_rate:0.1,momentum:0.5)
    weight.value().multiply(scale.value()).sum().backward()
    optimizer.step(); optimizer.zero_grad()
    assert(Math.abs(weight.value().item()-1.7) < 0.00001 && scale.value().item() == 3.0)
    weight.freeze()
    assert(!weight.is_trainable())
    optimizer.step()
    assert(Math.abs(weight.value().item()-1.7) < 0.00001)
    scale.replace(Tensor.scalar(4.0))
    weight.unfreeze()
    weight.value().multiply(scale.value()).sum().backward()
    optimizer.step()
    assert(Math.abs(weight.value().item()-1.15) < 0.00001)
}
```

The first step yields a weight of 1.7. Freezing preserves that value. After
replacing the coefficient and unfreezing, SGD resumes with its retained momentum
and yields approximately 1.15. The weight’s name and identity stay unchanged.

## Keep data outside the graph

`Buffer(name, value)` requires a nonempty name and detaches the supplied value.
`name()` and `value()` expose its stable name and current value. All nine numeric
dtypes and booleans are supported on CPU and GPU. A Buffer is distinct from a
Parameter: optimizers only accept Parameter values.

`replace(value)` preserves dtype, shape, and placement, including the GPU device,
and detaches the new value before storing it. The previous Tensor value remains
valid: its storage is immutable. References to the same Buffer observe the
replacement through their shared identity.

With the default `BufferPlacement.model` policy, `to(device)` and `cpu()`
explicitly move the Buffer’s value. Transfers remain
exact for every dtype; integers and booleans gain no additional numeric GPU
computation. Extracting elements requires a CPU value.

## Suspend and resume training

A Parameter is trainable when created. `is_trainable()` exposes its state.
Changing that state requires an empty gradient: call `parameter.zero_grad()` or
`optimizer.zero_grad()` before `freeze()` or `unfreeze()`. Repeating the current
state does nothing, even when a gradient is present.

After freezing, new `value()` calls return a detached value. Gradients of other
inputs still propagate. A graph built before freezing retains its meaning and
may still fill the Parameter’s slot; `gradient()` remains observable and
`zero_grad()` clears it normally.

SGD and Adam ignore frozen parameters, including their earlier gradients:
no updates, decay, moment creation or evolution, or Adam counter increments.
`clip_grad_norm` excludes their gradients from the norm and leaves their slots
unchanged. Unfreezing preserves existing states and resumes at the next active
step. Optimizer state placement checks still apply; freezing does not allow
moving existing moments separately.

`replace`, `to(device)`, and `cpu()` remain explicit Parameter actions with their
usual empty-gradient, dtype, shape, and placement checks. To observe its GPU
value, use `parameter.value().detach().cpu()`.

The [layer composition guide](Composition.md) explains the hooks to forward
manually. [Layer registration](Registration.md)
collects explicitly registered objects; the current checkpoint only saves its
parameters, without buffers, frozen state,
or optimizer moments.

[Model state operations](ModelState.md) propagate mode, freezing, and placement to registered objects.

[CPU model metadata](HostState.md): explicit CPU buffers and recursive transfers.
