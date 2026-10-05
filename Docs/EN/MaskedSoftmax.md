# Normalize visible scores

`Tensor.masked_softmax(mask,axis)` normalizes only positions allowed by a boolean
mask. A row with no visible position returns zeros with zero gradients. The mask
uses the `true` convention of [sequence masks](SequenceMasks.md).

This program builds probabilities for three sequences of lengths 0, 2 and 4.
It also checks gradients and runs the same path on CPU and Metal when the GPU
is available.

```sx
use Tensor
use Tensor.Neural
use Tensor.Autograd
use Tensor.Profile
use GFX.GPU

func demonstrate_masked_probabilities(device:GPU.Device?) {
    var scores=Tensor.ones([3,4])
    var mask=Neural.sequence_mask([0,2,4],4)
    if let gpu=device {scores=scores.to(gpu);mask=mask.to(gpu)}
    var input=Autograd.Variable(scores)
    var profile:Profile?=null
    if let gpu=device {profile=Profile(gpu)}
    let probabilities=input.value().masked_softmax(mask,axis:1)
    probabilities.sum().backward()
    if let p=profile {assert(p.snapshot().uploads==0 && p.snapshot().downloads==0)}
    let values=probabilities.detach().cpu().values()
    assert(values[0]==0.0 && values[1]==0.0 && values[2]==0.0 && values[3]==0.0)
    assert(values[4]==0.5 && values[5]==0.5 && values[6]==0.0 && values[7]==0.0)
    assert(values[8]==0.25 && values[9]==0.25 && values[10]==0.25 && values[11]==0.25)
    if let gradient=input.gradient() {for value in gradient.cpu().values() {assert(value==0.0)}}
    else {panic("expected masked softmax gradient")}
}
func main() {
    demonstrate_masked_probabilities(null)
    if GPU.Device.is_supported() {var device=GPU.Device(GPU.DeviceSettings(debug:true));demonstrate_masked_probabilities(device)}
    print("Masked probabilities: empty row and gradients preserved")
}
```

The rows produce `[0,0,0,0]`, `[0.5,0.5,0,0]` and
`[0.25,0.25,0.25,0.25]` respectively. A visible row’s probabilities sum to 1;
its gradient is zero when the loss is that sum. The mask itself is detached data,
never a differentiable parameter.

## Shapes and stability

Scores must be `float32`. The axis must belong to the input rank. The boolean
mask must broadcast to the scores’ shape without enlarging it; result and
gradients preserve shape and placement. Strided views and axes other than the
last are supported. An empty input returns an empty value of the same shape with
an empty gradient, including a zero-length axis.

The qualified numerical domain contains finite visible scores, including large
translations and finite float32 extremes. Excluded positions may contain NaN or
infinity: they produce zero probability and derivative without contaminating
visible positions. Nonfinite visible values remain outside this domain; no
readback is added to inspect them on GPU.

An entirely `false` row produces zeros, never a uniform distribution. On a visible
row, a nonuniform seed produces the usual normalization gradient, with zeros at
excluded positions. `Autograd.grad` preserves existing slots; `backward` follows
the ordinary graph consumption rules.

Computation retains float32 rounding; compare normalization and gradients
with a numerical tolerance.

## Place and compose

Scores and mask must use the same placement and device. Explicitly transfer the
mask before computation; softmax and backward remain resident without implicit
upload or readback. Add batch/head axes with `unsqueeze` as in the
[visibility guide](SequenceMasks.md).

This operation normalizes scores. The query/key product, scale and application
of probabilities to values remain separate Tensor computations. Required
constants and resources are released with the graph through the ordinary
workflow; no new stateful layer is created.
