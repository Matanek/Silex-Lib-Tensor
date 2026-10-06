# Prepare an MSE training step with AdamW

`Neural.TrainingStep.mse(model, optimizer, input, target)` prepares a closed step
for an existing model and AdamW. `run(input, target)` performs zero-grad,
forward, scalar MSE, backward and update in eager order. Preparation is explicit;
the ordinary training loop remains available.

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim
use GFX.GPU

func main() {
    assert(GPU.Device.is_supported())
    var device=GPU.Device()
    var layers:Neural.Layer[]=[Neural.Dense("hidden",2,4,101),Neural.Activation.tanh(),Neural.Dense("output",4,2,102)]
    var model=Neural.Sequential(layers)
    model.to(device)
    var optimizer=Optim.AdamW(model.parameters(),epsilon:0.01)
    let input=Tensor.ones([4,2]).to(device)
    let target=Tensor.zeros([4,2]).to(device)
    var step=try Neural.TrainingStep.mse(model,optimizer,input,target) else error {panic(error)}
    for iteration in 0...12 {
        var result=try step.run(input,target) else error {panic(error)}
        assert(result.output.shape()[0]==4 && result.loss.rank()==0)
    }
    let statistics=step.statistics()
    assert(statistics.captures==1 && statistics.executions==12 && statistics.rejections==0)
}
```

Preparation returns `Result<TrainingStep,str>`; execution returns
`Result<TrainingStepResult,str>`. `output` and `loss` are detached observations
from before the update. They retain their values across subsequent steps;
their backward graph cannot be replayed.

## Supported domain and current state

The family accepts exact Sequential, Dense, Activation (relu/sigmoid/tanh) and
LayerNorm classes, composed without callbacks. Exact AdamW must own the same
Parameter handles as the model. Unknown classes, subclasses, buffers, dropout,
other losses/optimizers and clipping are rejected boundaries. Loaders,
checkpoints and schedulers remain outside preparation. Backward uses eager
rules and is not captured.

Input and target are nonempty GPU float32, rank ≤ 5, with fixed device and layout.
The target exactly matches the output shape, without implicit broadcasting.
Parameters preserve shape/dtype/device; their strides/offset may change.
Current learning rate, betas, epsilon, weights, moments, counters and trainability
are read each time. Switching between zero and nonzero decay requires a new
preparation. Backward requires a tracked input, target or trainable parameter.
Input/target autograd provenance must be pure and unconsumed. Custom callbacks
and graphs exceeding 4,096 nodes are rejected before preparation or zero-grad.
Variables and pure views remain supported.

## Rejection, another preparation and ownership

`run` validates signatures and state before clearing gradients. Failure leaves
weights, moments, AdamW counters and gradients unchanged. Statistics count the
rejection and expose its reason. A changed signature requires another explicit
plan; the application may instead use its ordinary eager loop with compatible
operands. There is no hidden recapture or fallback. Ordinary zero-grad rules
before trainability/placement changes and state restoration still apply.

Each TrainingStep owns one program. `statistics()` exposes captures/compilations,
executions/rejections, forward/loss operations and optimizer parameters.
Bounds are 256 layers, 128 parameters, 256 forward operations, three loss operations
and at most 17 cold AdamW operations per parameter. Two plans remain independent.
The plan keeps its model and optimizer alive until released; neither owns it.
The program retains no example, result or previous graph and overwrites no
result buffer.

## Choose using a complete measurement

Grouping reduces submissions but may increase intermediate allocations, and
does not guarantee a gain at every size. Measure the complete synchronized hot
step after warmup, separately from preparation. Qualified results and
counter-evidence for the candidate live in Silex-Benchmarks; performance outside
macOS arm64/Metal remains open. All six families are qualified against eager
with explicit rejections outside the prepared domain. The float16 gate remains indeterminate,
with no mixed API; see the [capability and precision reference](Reference.md). See also [prepared forward](ForwardPlans.md)
and [profiling](Profiling.md).

On Apple M3 Pro/Metal in Release, October 6, 2026: 16 pairs and 8 A/A controls per
arm admit 6.21% on MLP(2,2,4,2) and 20.66% on the normalized model (4,2,4,2),
with 16/16 favorable pairs. MLP(16,16,32,8) regresses by 6.49%, with 0/16
favorable pairs; keep eager for performance on this workload. A/A controls
pass; preparation and RSS are separate. 192 processes and 229,312 GPU scalars
agree, CPU oracle tolerance 2e-5, identical sources/binary. These numbers measure
the entire step, not isolated GPU time. The recomputable campaign lives under
`Silex-Benchmarks/Sources/TensorStableCompute/Baselines/` in
`2026-10-06-macos-arm64-training-step`.
