# Train a GRU with explicit state

`Neural.GRU(name,input_features,hidden_features,seed)` unrolls one unidirectional
batch-first layer. `forward(input)` returns the final state;
`unroll(input,initial_state:null,return_sequence:false)` returns a `GRUResult`
with required `state` and optional `sequence`. No state persists between calls.
Feature dimensions are positive and the initialization seed fits uint32.
Float32 parameters start on CPU and move explicitly.

## Shapes and state

Float32 input has shape `[batch,time,input_features]`. An optional float32
initial state has shape `[batch,hidden_features]`; by default it is zero on the
input placement. The final state is `[batch,hidden_features]`, and a requested
sequence is `[batch,time,hidden_features]` in time order. With `time > 0`, the
final state equals its last element. Input, initial state and all four parameters
must share placement and GPU device. Non-contiguous views are accepted without
fallback or implicit readback.

With `time == 0`, the supplied state returns unchanged or a zero state is created;
a requested sequence has shape `[batch,0,hidden_features]`. Unused input and
parameters remain unconnected to autograd. Request their mathematical zero
explicitly with `Autograd.grad(...,allow_unused:true)`. A tracked initial state's
gradient follows the identity operation. An empty batch with nonzero time retains
shapes and zero gradients of executed operations. Dtype, shape and placement
diagnostics also apply when time is zero.

Pass `result.state` to the next chunk. Keeping it tracked preserves outputs and
gradients of the whole sequence. After consuming a graph, recompute the forward;
use `detach()` explicitly when intending to cut gradients between chunks.
The layer never detaches state automatically.

## Gates and parameters

This GRU uses **reset-after** with gates `r,z,n`:

```text
r = sigmoid(x Wr + bir + h Ur + bhr)
z = sigmoid(x Wz + biz + h Uz + bhz)
n = tanh(x Wn + bin + r * (h Un + bhn))
h_next = (1 - z) * n + z * h
```

`parameters()` returns `name.input_weight` with shape
`[input_features,3*hidden_features]`, `name.recurrent_weight` with shape
`[hidden_features,3*hidden_features]`, then `name.input_bias` and
`name.recurrent_bias` with shape `[3*hidden_features]`, in that order.
Packed columns are `r`, `z`, then `n`; matrices are row-major `[input,output]`.
Matrices use Xavier uniform initialization and biases start at zero.
Weight conversion must preserve this gate order, orientation and reset-after
convention. The two biases remain separate because the recurrent `n` bias is
multiplied by `r`. A different gate convention requires more than renaming.
Cardinalities and byte sizes are checked before allocation.

The layer joins the [recursive registry](Registration.md): four parameters,
no buffer or child, identity-based sharing, placement and freezing.
`train()`/`eval()` propagate without changing this dropout-free calculation.
The seed only initializes parameters; there is no runtime random counter.

The [full checkpoint](TrainingCheckpoint.md) restores parameters, mode,
optimizer and batches; configuration includes dimensions and gate convention.
Explicit recurrent state belongs to the application and is not saved as hidden
GRU state. Parameter-only checkpoints retain their existing contract.

## Complete public example

This model classifies six asymmetric sequences with identical final tokens,
requiring information from earlier steps. Forty AdamW steps and CPU/Metal
profiles qualify this synthetic example only. Exact resumption is tested on
the same backend; bit-identical CPU and Metal trajectories are not promised.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use GFX.GPU

nocopy class GRUClassifier:Neural.Layer {
    private var memory:Neural.GRU
    private var classifier:Neural.Dense
    init(seed:int,hidden_features:int = 4) {
        self.memory=Neural.GRU("tiny.memory",2,hidden_features,seed)
        self.classifier=Neural.Dense("tiny.classifier",hidden_features,3,seed+1)
        self.register_child("memory",self.memory)
        self.register_child("classifier",self.classifier)
    }
    override func forward(input:Tensor) Tensor {return self.classifier.forward(self.memory.forward(input))}
}
func gru_training_source() Data.TensorSource {
    var samples:float[]=[0.7,0.1,0.3,0.2,0.05,0.1,0.9,0.2,0.4,0.1,0.05,0.1,-0.5,0.4,-0.3,0.6,0.05,0.1,-0.7,0.3,-0.4,0.5,0.05,0.1,0.1,-0.8,-0.2,-0.5,0.05,0.1,0.2,-0.6,-0.1,-0.4,0.05,0.1]
    var targets:int32[]=[0,0,1,1,2,2]
    return Data.TensorSource(Tensor(samples,[6,3,2]),Tensor(targets,[6]))
}
func gru_learns(device:GPU.Device?) {
    var model=GRUClassifier(seed:17);var source=gru_training_source();var cursor=Data.BatchCursor(source,6,shuffle:false)
    var batch=try cursor.next() else error {panic(error)}
    if var value=batch {
        if let gpu=device {model.to(gpu);value.input=value.input.to(gpu);value.target=value.target.to(gpu)}
        let initial=model.forward(value.input).cross_entropy(value.target,1).detach().cpu().values()[0]
        var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.04)
        var step=0
        while step<40 {
            var profile:Profile?=null
            if let gpu=device {profile=Profile(gpu)}
            model.forward(value.input).cross_entropy(value.target,1).backward();optimizer.step();optimizer.zero_grad()
            if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
            step++
        }
        let final=model.forward(value.input).cross_entropy(value.target,1).detach().cpu().values()[0]
        print("GRU synthetic loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.25)
    } else {panic("empty GRU learning source")}
}

func main() {
    gru_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));gru_learns(device)}
}
```

From `SilexProject`, run
`silex run Packages/Tensor/Tests/Consumer/Examples/TrainGRU.sx`, or add
`--release`. Ordinary tests compare outputs and six VJPs against an
[independent Decimal60 oracle](../../Tools/Oracle/GRU_REPORT.json) with absolute
tolerance `2e-6`, then check Jacobians, chunking and exact resumption.
Bidirectional or stacked layers with dropout, and implicit state, are not provided.
