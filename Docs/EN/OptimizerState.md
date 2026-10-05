# Resume an optimizer

`SGD.state()` and `Adam.state()` capture the current rate, hyperparameters, and
named moments. After reconstructing model values, `restore(state)` resumes the
next update with the same moments and counters. This example explicitly copies
the parameter value, then checks that both training paths produce exactly the
same value at the second update.

```sx
use Tensor
use Tensor.Neural.Parameter
use Tensor.Optim

func main() {
    var parameter=Parameter("weight",Tensor.scalar(2.0))
    var optimizer=Optim.Adam([parameter],learning_rate:0.1,beta1:0.5,beta2:0.5,epsilon:0.0001)
    parameter.value().multiply(3.0).backward();optimizer.step();optimizer.zero_grad()
    var saved=optimizer.state()
    var restored_parameter=Parameter("weight",parameter.value().detach())
    var restored=Optim.Adam([restored_parameter])
    try restored.restore(saved) else error {panic(error)}
    parameter.value().multiply(4.0).backward();optimizer.step()
    restored_parameter.value().multiply(4.0).backward();restored.step()
    assert(parameter.value().detach().item()==restored_parameter.value().detach().item())
    assert(optimizer.state().parameters[0].step_count==2 && restored.state().parameters[0].step_count==2)
}
```

## Capture moments

`SGDState` contains `learning_rate`, `momentum`, `weight_decay`, and a
`parameters` array of `SGDMomentumState`. Each entry contains `name`, `shape`,
and `momentum:Tensor?`. An absent buffer means momentum is uninitialized; with
a zero coefficient, it stays absent even after active updates.

`AdamState` contains `learning_rate`, `beta1`, `beta2`, `epsilon`, `weight_decay`,
and a `parameters` array of `AdamMomentState`. Each entry contains `name`,
`shape`, `step_count`, `first_moment:Tensor?`, and `second_moment:Tensor?`.
At zero, both moments are absent; after an active update, both are present.
The counter belongs to the parameter: a missing gradient or frozen parameter
does not advance it.

Snapshots construct new entry and dimension arrays. Their detached Tensor
values share immutable storage. Later updates therefore leave captured values
unchanged. Arrays and Tensor fields remain editable to construct a state to
restore. Snapshots do not contain model values, buffers, gradients, modes,
freezing states, batches, or RNG state. The [parameter checkpoint](README.md)
remains a separate format; these snapshots are not a complete training-resume
file.

## Restore without replacing a partial state

`restore(snapshot) Result<void,str>` requires the same set of unique names and
the same shapes, including parameters without moments. Array order may differ.
Call `zero_grad()` first: every gradient slot must be absent, even for a frozen
parameter and an old graph.

All hyperparameters, names, shapes, counters, and placements are validated before
changing the optimizer. A recoverable error leaves all its moments, counters,
and hyperparameters unchanged. Success replaces that state without changing
parameter values, gradients, identities, or freezing states. Supplied moments
are detached even when they came from a tracked graph.

Every moment must be float32, match the parameter shape, and reside on the same
CPU or GPU device. Numeric contents are not inspected: moment NaNs/infinities
are preserved and validation hides no readback. Hyperparameters are host scalars
and must be finite. Rate and epsilon must be strictly positive, Adam betas belong
to `[0,1)`, and momentum and weight_decay must be nonnegative. Adam rejects a
negative counter, moments at zero, or only one moment after zero; SGD rejects a
momentum buffer when the saved coefficient is zero.

## Choose placement and the current rate

`state()` and `restore()` preserve resident placement without transfers.
`snapshot.cpu()` explicitly copies moments to CPU; `snapshot.to(device)`
explicitly transfers them to a device. Each copy has independent entry and
shape arrays. Move the model as well before restoring its GPU moments.
The [complete checkpoint](TrainingCheckpoint.md) also saves the model and progress.

`learning_rate()` reads the current rate; `set_learning_rate(value)` requires a
finite, strictly positive scalar and preserves every moment and counter.
Apply a new rate before the update that should use it. A snapshot captures that
rate; restoring it also replaces the other hyperparameters.

Adam accepts nonnegative counters up to the `int` limit. Its bias correction
computes powers by squaring in at most 63 iterations, even after a long resume.
Grouping multiplications may change floating-point rounding compared with linear
multiplication. Trying to advance a maximal counter with an active gradient
fails before any parameter update; without an active gradient, it stays unchanged.
