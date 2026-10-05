# Prepare a repeated GPU forward

`model.prepare_forward(example)` prepares the model forward for a GPU `float32`
input with a fixed layout. Then call its usual `forward`: it uses a compatible
plan, or eager evaluation with an observable reason. The model owns its
preparation; the application does not construct a graph.

## Prepare the existing model

```sx
use Tensor
use Tensor.Neural
use GFX.GPU

func main() {
    assert(GPU.Device.is_supported())
    var device = GPU.Device()
    var layers:Neural.Layer[] = [
        Neural.Dense("hidden", 2, 4, 101),
        Neural.Activation.tanh(),
        Neural.Dense("output", 4, 2, 102)
    ]
    var model = Neural.Sequential(layers)
    model.to(device)
    let input = Tensor.ones([16, 2]).to(device)
    match model.prepare_forward(input) {
        failure(reason) => { panic(reason) }
        success => {}
    }
    let output = model.forward(input)
    assert(output.shape()[0] == 16 && output.shape()[1] == 2)
    let statistics = model.forward_plan_statistics()
    assert(statistics.captures == 1 && statistics.executions == 1)
    model.clear_forward_plan()
    assert(model.forward_plan_statistics().cached_plans == 0)
}
```

The three methods belong to `Neural.Layer` and are inherited by concrete layers.
`prepare_forward` returns `Result<void,str>`; failure explains why this model
or example is unsupported.

## Supported family

The first family includes the exact concrete classes `Sequential`, `Dense`,
`Activation` (`relu`, `sigmoid`, `tanh`), and `LayerNorm`. A Sequential may
compose them and repeat the same layer: every occurrence executes, using the
same shared parameter handles.

Other classes and their subclasses are rejected before calling their `children`,
`parameters`, or `forward` hooks. Preparation executes no application forward,
callback, randomness, transfer, or weight update. Any model construction hooks
precede this preparation.

The example must be nonempty GPU `float32`, rank at most 5, with shapes accepted
by its layers. Preparation is bounded to 256 unique layers, 128 unique parameters,
and 256 operations. Each prepared layer owns at most one plan; no example or
output storage is captured.

## Reuse, state, and fallback

The plan checks input shape, strides, offset, dtype, and device, model mode,
then current parameters before encoding any work. A different signature uses
eager evaluation and records `last_fallback`. This evaluation retains the usual
operand compatibility rules: also move the model before forwarding a CPU input
or an input on another device.

A successful new preparation replaces the previous one. Rejection preserves
the previous plan, parameters, buffers, and gradients. Repeating a still
compatible preparation does not recapture. Fallback never automatically recaptures.

Compatible `Parameter.replace` values are read on the next execution, including
changed strides or offsets. Current parameter layouts are handled without
recapture. `freeze`, `unfreeze`, and input provenance determine gradients for
the current forward. Backward uses eager autograd rules; a forward plan does
not prepare a whole training step.

`clear_forward_plan()` releases this layer's cache without changing its numeric
state. Retained outputs remain independent and usable.

## Observe and measure

`ForwardPlanStatistics` snapshots cumulative counters:

| Field | Observation |
|---|---|
| `preparations`, `rejections` | Explicit calls and rejected preparations |
| `captures`, `compilations` | New prepared representations; plan compilations, separate from GPU pipelines |
| `executions`, `fallbacks` | Plan executions and eager fallbacks when a plan exists |
| `cached_plans`, `prepared_operations` | Current cache presence and size |
| `cache_capacity_plans` | Fixed capacity: 1 |
| `last_fallback` | Latest reason, cleared by compatible execution or new capture |

An unprepared model does not increment `fallbacks`. Clearing the cache does
not reset cumulative counters. Use [Profile](Profiling.md) to observe actual
pipeline compilations, transfers, passes, and submissions.

The plan preserves eager operations and their order, then batches their
submission. The example MLP keeps five passes with one shared submission/fence;
LayerNorm keeps eight passes, including six normalization passes. Values and
all gradients are qualified against GPU and CPU eager evaluation.

Across three Dense–tanh–Dense Release workloads measured on Metal macOS ARM64,
hot medians decrease by 40.15%, 33.42%, and 19.58%, with 16 favorable pairs out
of 16 and valid A/A controls. Initial preparation, including pipelines, is
observed separately around 1.09 ms on this harness. These measurements establish
neither a startup gain nor a promise for all models or architectures. Profiling
overhead remains indeterminate in this series. Sources, controls, and raw data
are preserved in `TensorStableCompute/PreparedForward.sx` and its sealed campaign.

[Back to Tensor](README.md) · [Model state](ModelState.md) ·
[Functional gradients](Gradients.md)
