# Propagate a model’s mode, placement, and freezing

Layer registration lets one instruction change the entire model’s state:
`train()`, `eval()`, `to(device)`, `cpu()`, `freeze()`, `unfreeze()`, and `zero_grad()`.
Shared registered identities are handled once, including when two branches
converge on the same layer.

This program counts shared layer notifications. It also qualifies explicit
transfers with a GFX.GPU-compatible GPU:

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use GFX.GPU

nocopy class StateModel:Neural.Layer {
    init() {}
    func attach(name:str, child:Neural.Layer) { self.register_child(name,child) }
    func add_parameter(parameter:Neural.Parameter) { self.register_parameter(parameter) }
    func add_buffer(buffer:Neural.Buffer) { self.register_buffer(buffer) }
    override func forward(input:Tensor) Tensor { return input }
}

nocopy class CountedLayer:Neural.Layer {
    private var mode_calls:int
    private var gpu_calls:int
    private var cpu_calls:int
    init(weight:Neural.Parameter, scale:Neural.Buffer) {
        self.mode_calls = 0; self.gpu_calls = 0; self.cpu_calls = 0
        self.register_parameter(weight); self.register_buffer(scale)
    }
    override func training_changed(training:bool) {
        assert(self.is_training() == training)
        self.mode_calls++
    }
    override func moved_to(device:GPU.Device) { self.gpu_calls++ }
    override func moved_to_cpu() { self.cpu_calls++ }
    func calls() int[] { return [self.mode_calls,self.gpu_calls,self.cpu_calls] }
    override func forward(input:Tensor) Tensor {
        return input.multiply(self.parameters()[0].value()).multiply(self.buffers()[0].value())
    }
}

func main() {
    var weight = Neural.Parameter("weight",Tensor.scalar(2.0)); var scale = Neural.Buffer("scale",Tensor.scalar(3.0))
    var leaf = CountedLayer(weight,scale)
    var first = StateModel(); var second = StateModel(); var root = StateModel()
    first.attach("shared",leaf); second.attach("shared",leaf); root.attach("first",first); root.attach("second",second)
    root.eval(); assert(!leaf.is_training() && leaf.calls()[0] == 1)
    root.freeze(); assert(!weight.is_trainable())
    root.unfreeze(); assert(weight.is_trainable())
    assert(GPU.Device.is_supported())
    var device = GPU.Device(GPU.DeviceSettings(debug:true))
    root.to(device); assert(weight.value().is_gpu() && scale.value().is_gpu() && leaf.calls()[1] == 1)
    root.cpu(); assert(weight.value().is_cpu() && scale.value().is_cpu() && leaf.calls()[2] == 1)
}
```

Each model call yields one local notification to the leaf despite its two paths.
The Buffer coefficient and weight move together. Freezing changes neither their
values nor the weight’s identity; the Buffer remains nontrainable.

## Define a local hook

With registration, inherit the public traversal methods and override only the
protected notifications you need:

- `training_changed(training:bool)` receives the mode after it has changed;
- `moved_to(device:GPU.Device)` is called after state moves to that GPU;
- `moved_to_cpu()` is called after state moves to CPU.

These hooks describe only your layer’s local state. Do not forward children
inside them: central traversal already visits them. A registered layer therefore
does not override train/eval/to/cpu to recursively call its children.
`is_training()` returns Layer’s mode, initially `true`. Each mode request notifies
layers, even if that mode is already set. Dropout uses the same mode; its iteration
counter only advances during training.

An older layer without registration keeps its manual public overrides: registered
traversal calls that opaque boundary once. Children hidden inside the override
remain its responsibility. A layer adopting registration migrates its manual hooks
to local notifications; inherited operations then handle all its declared state.

## Keep transfers explicit

Collection validates names before mutation. `to(device)` and `cpu()` require empty
gradients for every collected parameter; call `model.zero_grad()` first. For to,
the devices of all GPU parameters and buffers are checked before the first
transfers. Moving between devices explicitly requires `model.cpu()` followed by
`model.to(other_device)`.

Each unique registered Parameter and Buffer moves once. Buffers support all
storage dtypes, including booleans, without conversion. Collection, mode changes,
and freezing perform no readback. Moving parameters does not automatically move
an optimizer’s moments: its consistency checks still apply.

## Freeze and resume

`freeze()` detaches new reads of the model’s parameters. `unfreeze()` makes them
trainable again. An actual change requires empty slots; every affected slot is
checked before changing the first parameter. Repeating the current state does
nothing. `zero_grad()` clears all collected slots, including those filled by
graphs built before freezing.

SGD and Adam ignore frozen parameters and retain their moments; input gradients
still propagate. See the [individual Buffer and Parameter contract](State.md).
[Names, aliases, cycles, and identities](Registration.md) define collection.
The current checkpoint still saves parameters only: buffers, mode, freezing,
and optimizer state are outside its schema.
