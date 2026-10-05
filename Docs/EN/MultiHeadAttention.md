# Train multi-head attention

`Neural.MultiHeadAttention` learns four named query, key, value and output
projections. `attend(query,key,value,...)` returns output and optional weights;
`forward(input)` provides self-attention without a mask at iteration 0. The layer
derives from [Layer](Composition.md): children, parameters, mode, placement and
freezing follow the [shared registry](Registration.md).

This program combines query and key padding with causality, then performs one
AdamW step on CPU and Metal. The third position is excluded on both sides:
its output and gradient stay zero, including after the final projection.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Optim
use Tensor.Profile
use GFX.GPU

func demonstrate_multihead(device:GPU.Device?) {
    var model=Neural.MultiHeadAttention("attention",4,2,seed:17,dropout_probability:0.1)
    var data:float[]=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
    var input=Tensor(data,[1,3,4])
    let valid=Neural.sequence_mask([2],3)
    var mask=valid.reshape([1,1,3,1]).logical_and(valid.reshape([1,1,1,3])).logical_and(Neural.causal_mask(3,3))
    if let gpu=device {model.to(gpu);input=input.to(gpu);mask=mask.to(gpu)}
    var values=Autograd.Variable(input)
    var optimizer=Optim.AdamW(model.parameters(),learning_rate:0.01)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let result=model.attend(values.value(),values.value(),values.value(),mask:mask,iteration:0,return_weights:true)
    result.output.sum().backward();optimizer.step();optimizer.zero_grad()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let output=result.output.detach().cpu().values()
    assert(result.output.shape()[2]==4)
    var i=8;while i<12 {assert(output[i]==0.0);i++}
    if let weights=result.weights {assert(weights.shape()[1]==2 && weights.shape()[2]==3 && weights.shape()[3]==3)} else {panic("expected attention weights")}
    if let gradient=values.gradient() {
        let gradient_values=gradient.cpu().values();var j=8;while j<12 {assert(gradient_values[j]==0.0);j++}
    } else {panic("expected input gradient")}
}
func main() {
    demonstrate_multihead(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_multihead(device)}
    print("Multi-head attention: causal padding and resident AdamW step")
}
```

Define both length masks when padded queries must also be excluded. A key-only
mask still allows those queries to read visible keys. Transfers precede
computation; forward, backward and the optimizer step add no upload or readback.

## Project and merge heads

`MultiHeadAttention(name,features,heads,seed,dropout_probability:0.0)` requires
positive features and heads, with features divisible by heads. Q/K/V have shape
`[batch,sequence,features]`, float32 dtype, matching batch, width, placement and
device. Queries may have a different length from keys; K and V have the same
length. All four weights have shape `[features,features]` in `[input,output]`
order, with `[features]` biases.

Parameter names are `name.query.weight`, `name.query.bias`, followed by key,
value and output pairs in that order. Children are named query, key, value and
output. Each projection places contiguous feature groups in
`[batch,heads,sequence,features/heads]`. Merging reorders axes and materializes
them on the same placement before the output projection.

The output shape is `[batch,queries,features]`. Optional weights retain every
head: `[batch,heads,queries,keys]`. Mask and scale follow the
[attention computation](Attention.md), with default scale
`1/sqrt(features/heads)`. A query with no visible key in any head has zero output
and gradient contribution, even with a nonzero output bias. Batch, queries or
keys dimensions may be zero. Values and visible scores must remain finite in
float32.

## Control dropout and resume

Dropout applies to probabilities in train mode. The construction seed and
`attend`'s `iteration` argument, both in `[0,4294967295]`, explicitly select the
random draw. Repeating the same iteration reproduces the draw on that backend.
Dropout is disabled in evaluation. Returned weights are those applied to values,
after dropout; their sum may differ from 1. A row emptied by dropout retains the
output bias, while a query without a visible key stays zero.

Use `attend` with `iteration:progress.step` to advance draws during training.
`forward(input)` retains Layer's iteration 0 shortcut. The
[complete checkpoint](TrainingCheckpoint.md) restores weights, moments, modes,
seed and cursor; TrainingProgress provides the next call's iteration. Resuming
on GPU explicitly transfers weights and moments as described in that guide.
An incompatible head count or probability is rejected before restoration.
The parameter-only checkpoint keeps its distinct role.

A shared layer has one collection of eight parameters and one update. Freeze
the model after `zero_grad`; input gradients remain possible. This layer expresses
self/cross-attention; applications separately compose residuals, normalization,
MLP and positional encoding.
