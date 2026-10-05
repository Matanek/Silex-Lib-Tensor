# Attend to values

`Neural.attention(query,key,value,mask:null,scale:null,return_weights:false)`
weights values from query/key products. It returns `AttentionResult.output`
and, on request, `AttentionResult.weights`. [Boolean masks](SequenceMasks.md)
allow `true` positions; a query with no visible key has zero output and zero
gradient contribution.

This program applies two queries to three values, excluding the last key.
It checks the resulting averages and value gradients on CPU and Metal.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Profile
use GFX.GPU

func demonstrate_attention(device:GPU.Device?) {
    var samples:float[]=[2.0,4.0,6.0,8.0,10.0,12.0]
    var q=Tensor.zeros([1,2,2]);var k=Tensor.zeros([1,3,2]);var v=Tensor(samples,[1,3,2])
    var mask=Neural.sequence_mask([2],3).reshape([1,1,3])
    if let gpu=device {q=q.to(gpu);k=k.to(gpu);v=v.to(gpu);mask=mask.to(gpu)}
    var values=Autograd.Variable(v)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let result=Neural.attention(q,k,values.value(),mask:mask,return_weights:true)
    result.output.sum().backward()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let output=result.output.detach().cpu().values()
    assert(output[0]==4.0 && output[1]==6.0 && output[2]==4.0 && output[3]==6.0)
    if let weights=result.weights {assert(weights.detach().cpu().values()[2]==0.0)}
    else {panic("expected attention weights")}
    if let gradient=values.gradient() {
        let data=gradient.cpu().values()
        assert(data[0]==1.0 && data[1]==1.0 && data[2]==1.0 && data[3]==1.0 && data[4]==0.0 && data[5]==0.0)
    } else {panic("expected value gradient")}
}
func main() {
    demonstrate_attention(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_attention(device)}
    print("Attention: visible values averaged and gradients preserved")
}
```

Both outputs are `[4,6]`. Each visible value receives gradient `[1,1]` because
it contributes half of each output; the excluded value receives `[0,0]`.
Uploads precede the profile; computation and backward stay resident.

## Choose shapes and visibility

Q, K and V use `float32`, matching rank3 or4, placement and device. Batch-first
shapes are `[batch,queries,features]`, `[batch,keys,features]` and
`[batch,keys,value_features]`. For heads, add an axis after batch:
`[batch,heads,sequence,features]`. Batch and heads match exactly, without
broadcasting between Q/K/V; Q/K features match and are strictly positive.
K and V have the same number of keys. The output retains batch, heads and
queries axes, with V's features.

The boolean mask broadcasts to `[batch,queries,keys]` or
`[batch,heads,queries,keys]` without enlarging those shapes. `null` allows every
key on the inputs' placement. Compose length and causality explicitly with
`logical_and`; for heads, reshape a `[batch,keys]` padding mask to
`[batch,1,1,keys]`. Transfer the mask before a GPU call. Optional weights have
the scores' shape and remain differentiable.

Self-attention uses the same source for Q/K/V; cross-attention allows different
query and key lengths. Sharing a variable accumulates all three paths' gradient
contributions. `Autograd.grad` accepts a nonuniform seed; a consumed graph
cannot be reused for a second backward.

## Scale, stability and empty shapes

The default scale is `1/sqrt(features)`; `scale` replaces it with a finite
scalar, including zero or negative values. Values and visible scores produced
by multiplication and scaling must remain finite in float32. The
[stable masked softmax](MaskedSoftmax.md) subtracts a maximum before
exponentiation and defines fully excluded rows as zero. Rounding remains float32.

Batch, heads, queries, keys and value_features may be zero. Output and gradient
shapes are preserved; zero keys give zero output even when queries exist.
Strided views are supported. This function has no trainable projection or
dropout: applications choose projections and assemble operations with
[composable layers](Composition.md).
