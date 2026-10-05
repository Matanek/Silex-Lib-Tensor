# Normalize a batch then use its statistics for evaluation

`Neural.BatchNorm` normalizes each feature of a dense `[N,F]` batch, or each
channel of a NCHW `[N,C,H,W]` input. During training it computes batch statistics
and updates its named buffers. During evaluation it uses those buffers without
changing them. Float32 values and gradients are supported on CPU and GPU.

This example computes affine gradients for a batch, clears them, then evaluates
with the running statistics. GPU observations explicitly use `detach().cpu()`.

```sx
use STD.Math
use Tensor
use Tensor.Neural
use GFX.GPU

func main() {
    var model=Neural.BatchNorm("normalization",2,momentum:0.5)
    var values:float[]=[1.0,2.0,3.0,6.0];var input=Tensor(values,[2,2])
    if GPU.Device.is_supported() {
        var device=GPU.Device();model.to(device);input=input.to(device)
    }
    let output=model.forward(input)
    output.multiply(output).sum().backward();model.zero_grad()
    let training_values=output.detach().cpu().values()
    assert(Math.abs(training_values[0]+0.999995)<0.00001)
    assert(model.buffers()[2].value().is_cpu() && model.buffers()[2].value().int64_item()==1)
    model.eval();let evaluated=model.forward(input).detach().cpu().values()
    assert(Math.abs(evaluated[0])<0.00001 && Math.abs(evaluated[3]-1.885616)<0.00001)
    assert(model.buffers()[2].value().int64_item()==1)
}
```

## Choose the batch conventions

The constructor is
`BatchNorm(name,features,epsilon:0.00001,momentum:0.1,affine:true)`.
The name must be nonempty, `features` positive and equal to axis 1 size,
`epsilon` finite and strictly positive, and `momentum` finite in `[0,1]`.

The initial mode is train. Reduced axes are `[0]` for dense input and `[0,2,3]`
for NCHW. With `m` observations per feature, the output is
`(input - mean) / sqrt(population_variance + epsilon)`, followed by
`weight * output + bias` when `affine` is true. Normalization uses population
variance; the running variance update uses unbiased variance
`population_variance * m / (m - 1)`. Train therefore requires `m > 1`.

Updates are `(1-momentum)*old + momentum*batch`. `momentum:0.0` preserves
statistics for a finite batch but still counts the forward;
`momentum:1.0` replaces statistics with those of the latest batch. Nonfinite
input data propagates through the Tensor operations; zero momentum is not a
NaN filter.

## Keep named state

For the name `normalization`, `parameters()` returns, in order,
`normalization.weight` initialized to one and `normalization.bias` initialized
to zero. `affine:false` creates no parameters.

`buffers()` returns, in order:

| Name | Initial value | Placement |
| --- | --- | --- |
| `normalization.running_mean` | float32 `[features]`, zeros | Follows model |
| `normalization.running_variance` | float32 `[features]`, ones | Follows model |
| `normalization.batches_tracked` | int64 scalar, zero | Explicit CPU |

Statistics are detached: they retain no autograd graph. The counter remains
exact on CPU through `to(device)` and `cpu()`. It advances once per successful
train forward, and int64 overflow is rejected before any update. Collection,
sharing, and modes follow the [layer registry](Registration.md). Freezing affine
parameters suspends their new gradients while preserving input derivatives and
running-statistics updates in train mode.

In eval, initial statistics are zero mean and unit variance. The counter and
statistics do not change; empty and singleton batches are accepted. The input
remains differentiable, as do trainable affine parameters. Call `eval()` for
inference independently of [parameter freezing](State.md).

The schema 1 model checkpoint preserves only parameters: it does not save
BatchNorm buffers. To retain these statistics with the current API, capture
their detached values and restore them explicitly with `Buffer.replace` using
the same dtype, shape, and placement.

## Keep computations on their placement

Explicitly move model and input to the same CPU or GPU device. Forward validates
dtype, rank, features, all data placements, counter, and population before
changing buffers. Views with nonnegative strides are accepted. GPU forward and
backward perform no uploads or downloads; only an explicit observation reads
data on CPU. The CPU counter is control metadata, not a conversion of the
computation. SyncBatchNorm and input ranks other than 2 or 4 are unavailable.
