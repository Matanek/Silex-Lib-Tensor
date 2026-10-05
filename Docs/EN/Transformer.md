# Train a small causal Transformer

`Neural.TransformerBlock` composes multi-head attention, two residuals, two
LayerNorms and a Dense/ReLU/Dense MLP. Choose `norm_first:true` for pre-norm
or `false` for post-norm. The block derives from Layer: state, sharing, freezing,
mode and placement follow the [registry](Registration.md).

This program trains an external model on six sequences of three tokens.
Its local cyclic target maps 1→2, 2→3 and 3→1. Embedding and classification
projection surround the block; the application chooses sin/cos positions and
causal visibility in two registered buffers.

```sx
use Tensor
use Tensor.Neural
use Tensor.Data
use Tensor.Optim
use Tensor.Profile
use STD.Math
use STD.File
use STD.FileSystem
use GFX.GPU

nocopy class TinyTransformer:Neural.Layer {
    private var tokens:Neural.Embedding
    private var positions:Neural.Buffer
    private var visibility:Neural.Buffer
    private var block:Neural.TransformerBlock
    private var classifier:Neural.Dense

    init(seed:int, norm_first:bool = true, dropout_probability:float = 0.0) {
        self.tokens=Neural.Embedding("language.tokens",4,4,seed,padding_index:0)
        self.block=Neural.TransformerBlock("language.block",4,2,8,seed+1,norm_first:norm_first,dropout_probability:dropout_probability)
        self.classifier=Neural.Dense("language.classifier",4,4,seed+2)
        var positions:float[]=[]
        var index=0
        while index<3 {
            positions.append(Math.sin(index as float));positions.append(Math.cos(index as float))
            positions.append(Math.sin((index as float)*0.01));positions.append(Math.cos((index as float)*0.01))
            index++
        }
        self.positions=Neural.Buffer("language.positions",Tensor(positions,[1,3,4]))
        self.visibility=Neural.Buffer("language.visibility",Neural.causal_mask(3,3))
        self.register_child("tokens",self.tokens)
        self.register_child("block",self.block)
        self.register_child("classifier",self.classifier)
        self.register_buffer(self.positions);self.register_buffer(self.visibility)
    }
    override func forward(input:Tensor) Tensor {return self.predict(input,0)}
    func predict(input:Tensor,iteration:int) Tensor {
        let embeddings=self.tokens.forward(input).add(self.positions.value())
        return self.classifier.forward(self.block.encode(embeddings,mask:self.visibility.value(),iteration:iteration).output)
    }
}
func transformer_source() Data.TensorSource {
    var ids:int32[]=[1,2,3,2,3,1,3,1,2,1,2,3,2,3,1,3,1,2]
    var next:int32[]=[2,3,1,3,1,2,1,2,3,2,3,1,3,1,2,1,2,3]
    return Data.TensorSource(Tensor(ids,[6,3]),Tensor(next,[6,3]))
}
func transformer_learns(device:GPU.Device?) {
    var model=TinyTransformer(17)
    var cursor=Data.BatchCursor(transformer_source(),6,shuffle:false)
    if var batch=try cursor.next() else error {panic(error)} {
        if let gpu=device {model.to(gpu);batch.input=batch.input.to(gpu);batch.target=batch.target.to(gpu)}
        model.eval()
        let initial=model.forward(batch.input).cross_entropy(batch.target,2).detach().cpu().values()[0]
        var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.04)
        model.train()
        var step=0
        while step<30 {
            var profile:Profile?=null;if let gpu=device {profile=Profile(gpu)}
            model.predict(batch.input,step).cross_entropy(batch.target,2).backward();optimizer.step();optimizer.zero_grad()
            if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
            step++
        }
        model.eval()
        let final=model.forward(batch.input).cross_entropy(batch.target,2).detach().cpu().values()[0]
        print("Tiny Transformer loss: ",initial," -> ",final)
        assert(Math.is_finite(final) && final<initial*0.4)
    } else {panic("empty Transformer learning source")}
}
func main() {
    transformer_learns(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));transformer_learns(device)}
}
```

The program prints loss before and after thirty AdamW steps and checks its
decrease on CPU, then on Metal when available. This test covers the small
synthetic target; it qualifies neither a tokenizer nor a pretrained model.
Logits have shape `[batch,sequence,vocabulary]`, int32 targets
`[batch,sequence]`: `cross_entropy(target,2)` preserves these shapes directly.
Transfers precede training. Each step checks that forward, backward and the
optimizer add no upload or readback.

## Choose where to normalize

With `norm_first:true`, the block computes `r=x+dropout(attention(norm1(x)))`,
then `y=r+dropout(MLP(norm2(r)))`. With `false`, it computes
`r=norm1(x+dropout(attention(x)))`, then `y=norm2(r+dropout(MLP(r)))`.
LayerNorm uses population variance across features and adds epsilon before
the square root. On GPU float32, this pair is fused into one pass; mean,
variance and differentiation keep their order. Evidence and performance limits
are described in [profiling](Profiling.md).

Visibility follows [attention masks](Attention.md): `true` allows,
padding and causality are explicit. It applies only to the attention branch.
A fully masked query therefore keeps its residual and MLP contribution;
it does not automatically become a zero output position. The application
adds positional encoding before the block, as shown here.

## Call the block and resume

`TransformerBlock(name,features,heads,feed_forward_features,seed,
norm_first:true,dropout_probability:0.0,epsilon:0.00001)` requires positive
dimensions, features divisible by heads, a uint32 seed, finite probability in
`[0,1)` and finite strictly positive epsilon. Float32 input has shape
`[batch,sequence,features]`; batch and sequence may be zero. Parameters,
input and mask share placement and device; no fallback is added.
Values and visible scores must stay finite in float32.

`encode(input,mask:null,iteration:0,return_weights:false)` returns a
`TransformerResult` with `output` and optional `weights`. Output keeps input
shape; `[batch,heads,sequence,sequence]` weights are those of attention after
its dropout. The two residual dropouts do not change those weights.
`forward(input)` provides Layer's shortcut without a mask at iteration0.

The uint32 iteration explicitly selects draws; repeating a call on the
same backend reproduces the result. Seeds derived from the construction seed
separate attention and residual branches. Evaluation disables all three dropouts.
To train with dropout, call `encode` with `iteration:progress.step`, as used
by the model's `predict` call above.

The four children are `attention`, `attention_norm`, `feed_forward_norm` and
`feed_forward`. The sixteen parameters follow that order: eight from attention,
then weight/bias of both norms and the `name.feed_forward.input` and
`name.feed_forward.output` Dense layers. Dense matrices are `[input,output]`;
norm vectors and biases follow feature order. A shared instance retains one
collection and one update per parameter.

The [complete checkpoint](TrainingCheckpoint.md) preserves parameters, buffers,
modes, seeds of all three dropouts, optimizer moments and progress/cursor.
It rejects an incompatible pre/post-norm convention before mutation. GPU resume
explicitly transfers model and moments as described in that guide. A parameter-only
checkpoint retains its distinct role. Model tests qualify exact resume after
three steps and five following steps, with CPU prefetch.
