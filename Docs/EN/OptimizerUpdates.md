# Choose decay and the learning rate

`Tensor.Optim.AdamW` separates weight decay from moment computation.
`LearningRateSchedule` calculates a rate from an explicit step. This example
first checks decay alone, then resumes identical AdamW updates with a cosine
schedule and a reconstructed model value.

```sx
use STD.Math
use Tensor
use Tensor.Neural.Parameter
use Tensor.Optim

func main() {
    var parameter=Parameter("weight",Tensor.scalar(2.0))
    var optimizer=Optim.AdamW([parameter],learning_rate:0.1,weight_decay:0.5)
    parameter.value().multiply(0.0).backward();optimizer.step();optimizer.zero_grad()
    assert(Math.abs(parameter.value().detach().item()-1.9)<0.000001)
    var saved=optimizer.state()
    var resumed_parameter=Parameter("weight",parameter.value().detach())
    var resumed=Optim.AdamW([resumed_parameter])
    try resumed.restore(saved) else error {panic(error)}
    let schedule=Optim.LearningRateSchedule.cosine(0.1,duration:4,minimum_rate:0.01)
    var step=1
    while step<6 {
        let rate=schedule.value_at(step)
        optimizer.set_learning_rate(rate);resumed.set_learning_rate(rate)
        parameter.value().multiply(3.0).backward();resumed_parameter.value().multiply(3.0).backward()
        optimizer.step();resumed.step();optimizer.zero_grad();resumed.zero_grad()
        assert(parameter.value().detach().item()==resumed_parameter.value().detach().item())
        step++
    }
}
```

## Choose Adam or AdamW

`Adam` retains coupled L2 decay: the gradient used by its moments includes
`weight_decay * parameter`. `AdamW` computes moments from the gradient alone,
multiplies the old value by `1 - learning_rate * weight_decay`, then subtracts
the bias-corrected adaptive update. A zero gradient is still active: moments
advance and AdamW applies decay. A missing gradient or frozen parameter advances
neither value, moment, nor counter.

AdamW accepts the same options and domains as Adam; its defaults are
`learning_rate:0.001`, `beta1:0.9`, `beta2:0.999`, `epsilon:0.00000001`, and
`weight_decay:0.01`. With zero decay, both optimizers produce identical updates.
`zero_grad`, `clip_grad_norm`, `learning_rate`, and `set_learning_rate` have the
same contracts. GPU computation keeps values and moments resident.

`AdamW.state()` returns an `AdamWState` distinct from `AdamState`, with the same
fields and `AdamMomentState` entries. `restore(AdamWState)` validates everything
before mutation, and `state.cpu()` explicitly reads moments. Rules for names,
shapes, absent gradients, and placements follow the
[resume guide](OptimizerState.md). These snapshots do not save model values or
global training progress.

## Calculate the next update rate

A schedule is a value with no mutable counter. `value_at(step)` depends only on
its configuration and the supplied nonnegative integer step. Repeated or
reordered calls do not change any result.

| Construction | Rate at step s |
| --- | --- |
| `constant(learning_rate)` | initial rate |
| `step(learning_rate, step_size, gamma)` | initial × gamma to the integer quotient s/step_size |
| `exponential(learning_rate, gamma)` | initial × gamma to the power s |
| `cosine(learning_rate, duration, minimum_rate)` | minimum + (initial − minimum) × (1 + cos(π × s/duration))/2, then minimum from duration onward |

Initial rate, gamma, and minimum rate must be finite and strictly positive.
`step_size` and `duration` are strictly positive integers. The cosine minimum is
supplied explicitly, must not exceed the initial rate, and stays constant after
the duration. Gamma above one gives an increasing schedule; gamma equal to one
keeps the rate constant.

The result is a host float32 scalar. Underflow to zero or overflow to a nonfinite
value is diagnosed before that rate can be passed to the optimizer. Integer
powers have bounded computation even at the `int` limit. Cosine returns exactly
the initial rate at zero and the minimum from duration onward; intermediate
rounding follows float32 precision.

For resumption, retain the step **of the next update**. At step s: calculate
`value_at(s)`, apply `set_learning_rate`, execute `optimizer.step`, then increment
s after success. The schedule never increments anything automatically.
An optimizer snapshot captures the current rate, but you must retain the global
step and reconstruct the same schedule configuration.
