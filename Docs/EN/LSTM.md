# Train an LSTM with two explicit states

`Neural.LSTM(name,input_features,hidden_features,seed)` unrolls one unidirectional
batch-first layer. `forward(input)` returns the final hidden Tensor.
`unroll(input,initial_state:null,return_sequence:false)` returns `LSTMResult`:
required `state` is `LSTMState(hidden:Tensor,cell:Tensor)` and `sequence` is
optional. No state persists between calls. Dimensions are positive and the seed
fits uint32; float32 initialization takes place on CPU.

## Shapes and state transfer

Float32 input is `[batch,time,input_features]`. Both float32 initial states are
`[batch,hidden_features]`. Supply the complete pair, or use `initial_state:null`
for two zeros on the input placement. Final states and a requested hidden
sequence have shapes `[batch,hidden_features]` and `[batch,time,hidden_features]`.
With `time > 0`, final hidden equals the sequence's last element. A sequence of
cells is not returned. Input, hidden, cell and all four parameters must share
placement and GPU device. Non-contiguous views and their gradients work without
fallback or implicit readback. Move parameters through Layer methods.

Passing `result.state` to the next chunk preserves both gradient paths. Outputs,
states and seven VJPs then match the whole sequence. The layer never detaches
state automatically. After graph consumption, recompute the forward; detach
both hidden and cell explicitly only when intending to cut gradients between
chunks. Passing hidden alone would discard the memory stored in cell.

With `time == 0`, the supplied pair returns unchanged or two zero states are
created; a requested sequence is `[batch,0,hidden_features]`. Tracked initial
state gradients follow identity operations, while unused input and parameters
remain unconnected. Use `Autograd.grad(...,allow_unused:true)` to request their
mathematical zeros explicitly. Empty batches with nonzero time retain shapes
and zero gradients of executed operations. Diagnostics for both states and
placements also apply when time is zero.

## Gates and weights

Packed columns follow `i,f,g,o`:

```text
a = x W + h U + input_bias + recurrent_bias
i = sigmoid(a_i)
f = sigmoid(a_f)
g = tanh(a_g)
o = sigmoid(a_o)
cell_next = f * cell + i * g
hidden_next = o * tanh(cell_next)
```

`parameters()` returns `name.input_weight` `[input_features,4*hidden_features]`,
`name.recurrent_weight` `[hidden_features,4*hidden_features]`, then
`name.input_bias` and `name.recurrent_bias` `[4*hidden_features]`, in that order.
Matrices are row-major `[input,output]` with Xavier uniform initialization.
Both biases start at zero, including the forget gate. Explicitly convert other
orientations or gate orders before `Parameter.replace`. Biases remain separate
parameters. Cardinalities and byte sizes are checked before any allocation.
Peepholes, projections, integrated bidirectionality and inter-layer dropout are
not provided.

The layer joins the [recursive registry](Registration.md): four parameters,
no buffer/child, identity-based sharing and freezing. `train()`/`eval()` propagate
without changing this dropout-free computation. The seed only initializes
parameters; there is no runtime random counter.

The [full checkpoint](TrainingCheckpoint.md) stores parameters, mode, optimizer
and batches; configuration includes dimensions and gates. Recurrent state belongs
to the application and is not registered as hidden LSTM state. Parameter-only
checkpoints retain their contract.

## Complete public example

The six sequences share identical final tokens; classes depend on earlier steps.
This model combines an LSTM with a Dense classifier, six registered parameters
and forty AdamW steps. Profiles verify resident CPU/Metal steps; this is a local
synthetic qualification. Exact resumption is tested on the same backend without
promising bit-identical CPU and Metal trajectories.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use GFX.GPU

nocopy class LSTMClassifier:Neural.Layer {
    private var memory:Neural.LSTM
    private var classifier:Neural.Dense
    init(seed:int,hidden_features:int = 4) {
        self.memory=Neural.LSTM("tiny.memory",2,hidden_features,seed)
        self.classifier=Neural.Dense("tiny.classifier",hidden_features,3,seed+1)
        self.register_child("memory",self.memory)
        self.register_child("classifier",self.classifier)
    }
    override func forward(input:Tensor) Tensor {return self.classifier.forward(self.memory.forward(input))}
}
func lstm_training_source() Data.TensorSource {
    var samples:float[]=[0.7,0.1,0.3,0.2,0.05,0.1,0.9,0.2,0.4,0.1,0.05,0.1,-0.5,0.4,-0.3,0.6,0.05,0.1,-0.7,0.3,-0.4,0.5,0.05,0.1,0.1,-0.8,-0.2,-0.5,0.05,0.1,0.2,-0.6,-0.1,-0.4,0.05,0.1]
    var targets:int32[]=[0,0,1,1,2,2]
    return Data.TensorSource(Tensor(samples,[6,3,2]),Tensor(targets,[6]))
}
func lstm_learns(device:GPU.Device?) {
    var model=LSTMClassifier(seed:17);var source=lstm_training_source();var cursor=Data.BatchCursor(source,6,shuffle:false)
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
        print("LSTM synthetic loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.25)
    } else {panic("empty LSTM learning source")}
}

func main() {
    lstm_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));lstm_learns(device)}
}
```

From `SilexProject`, run
`silex run Packages/Tensor/Tests/Consumer/Examples/TrainLSTM.sx`, or add
`--release`. Hermetic tests compare outputs/sequence and seven VJPs with an
[independent Decimal60 oracle](../../Tools/Oracle/LSTM_REPORT.json), absolute
tolerance `2e-6`, then check Jacobians and explicit chunking.
