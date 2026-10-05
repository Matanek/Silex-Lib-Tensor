# Resume a complete training run

`Neural.TrainingCheckpoint` saves and restores the registered model, optimizer
and batch cursor together. `save` returns `Result<void,str>`; `load` returns
`Result<Neural.TrainingProgress,str>`. Overloads accept `SGD`, `Adam` or `AdamW`.
Recreate the same model and source, then load on CPU before explicitly moving
to the GPU.

This example runs one step, saves, reconstructs with another seed, and compares
the next step with continuous training. It prints
`Exact continuation after step 2` and removes its temporary file.

```sx
use STD.File
use STD.FileSystem
use Tensor
use Tensor.Data
use Tensor.Neural
use Tensor.Optim

func model(seed:int) Neural.Sequential {
    return Neural.Sequential([
        Neural.Dense("dense",2,2,seed:seed),
        Neural.BatchNorm("norm",2),Neural.Dropout(0.25,seed:seed)
    ])
}
func train_step(model:Neural.Layer,optimizer:Optim.AdamW,cursor:Data.BatchCursor,progress:Neural.TrainingProgress) Neural.TrainingProgress {
    var active=model;var optim=optimizer;var reading=cursor
    if let batch=try reading.next() else error {panic(error)} {
        if let schedule=progress.schedule {optim.set_learning_rate(schedule.value_at(progress.step))}
        let error=active.forward(batch.input).subtract(batch.target)
        error.multiply(error).mean().backward();optim.step();optim.zero_grad()
        return Neural.TrainingProgress(step:progress.step+1,schedule:progress.schedule)
    }
    panic("example corpus exhausted")
}
func main() {
    var inputs:float[]=[1.0,2.0,2.0,4.0,3.0,6.0,4.0,8.0]
    var source=Data.TensorSource(Tensor(inputs,[4,2]),Tensor.ones([4,2]))
    var original=model(42);var cursor=Data.BatchCursor(source,2,shuffle:true,seed:42)
    var optimizer=Optim.AdamW(original.parameters(),learning_rate:0.01)
    var progress=Neural.TrainingProgress(step:0,schedule:Optim.LearningRateSchedule.step(0.01,step_size:2,gamma:0.5))
    progress=train_step(original,optimizer,cursor,progress)
    let path="tensor-training-example.sxtc"
    try Neural.TrainingCheckpoint.save(original,optimizer,cursor,progress,path) else error {panic(error)}
    var restored=model(93);var restored_cursor=Data.BatchCursor(source,2,shuffle:true,seed:42)
    var restored_optimizer=Optim.AdamW(restored.parameters())
    var restored_progress=try Neural.TrainingCheckpoint.load(restored,restored_optimizer,restored_cursor,path) else error {panic(error)}
    progress=train_step(original,optimizer,cursor,progress)
    restored_progress=train_step(restored,restored_optimizer,restored_cursor,restored_progress)
    var expected=original.parameters();var actual=restored.parameters();var index=0
    while index<expected.count() {
        let left=expected[index].value().detach().values();let right=actual[index].value().detach().values();var element=0
        while element<left.count() {assert(left[element]==right[element]);element++}
        index++
    }
    assert(progress.step==restored_progress.step && cursor.state()==restored_cursor.state())
    try FileSystem.remove_file(path) else error {panic("cannot remove example checkpoint")}
    print("Exact continuation after step $(restored_progress.step)")
}
```

## State that resumes

Training format version 1 preserves exact tensor bits: float32, signed/unsigned
8/16/32/64-bit integers and booleans. Strides are not stored: values are saved
in logical order and restored as contiguous CPU tensors. Subnormals, signed
zeros and nonfinite numeric state are preserved; host hyperparameters must
remain valid and finite.

Parameters retain names, values and trainability; buffers retain names, values
and placement policies. Registered layers retain local modes, including eval
children inside a train parent. Dropout recovers its seed and next iteration.
BatchNorm recovers its parameters and three buffers, including its CPU int64
counter. Optimizer moments, counters and hyperparameters, the step and optional
schedule, and cursor seed/epoch/position are restored together. The optimizer
must own exactly the model's parameter identities, including frozen parameters
and entries with uninitialized moments.

`TrainingProgress(step:...,schedule:...)` describes the next step. Set the rate
using `schedule.value_at(progress.step)` before `optimizer.step()`, then increment
the step after success. The saved current rate belongs to the last completed
step; the restored schedule supplies the next rate. The cursor is captured at
its current position: save between completed batches, after `zero_grad()`, to
resume at the next batch.

A cursor can enable [CPU prefetch](BatchPrefetch.md): the checkpoint preserves
consumed progress without serializing prepared batches. Rebuild the same sampler
configuration with or without prefetch.

## Preconditions and recoverable errors

The target model has matching registered names and order, shapes, dtypes,
buffer policies, named children and identity sharing. Built-in layer
configurations are compared; weight construction seeds may differ because
saved values replace them. Target parameters and buffers are on CPU and all
model gradients are absent. The cursor is inactive, its source still reports
a stable count, and its count/batch_size/shuffle/seed/drop_last configuration
matches the file.

Loading prepares and validates the whole document and every target before
applying it. A `failure` leaves parameters, buffers, modes, trainability, RNG,
optimizer and cursor unchanged. File, version, encoding, cardinality,
configuration, name, placement and moment errors are recoverable.
`maximum_bytes` is positive and defaults to 16 MiB; larger files are rejected.
Use model, optimizer and cursor exclusively from the loop's thread during
save/load. Arbitrary user callback/hook effects and fatal allocation failures
are not rolled back by restoration.

The file reconstructs no code, autograd graph, device, source or callback.
Internal signatures validate registered links and configurations; they do not
rebuild the architecture. Recreate the same code for external layers; mutable
state needed for continuation belongs in their registered parameters and
buffers. Recreated source contents remain the caller's responsibility: matching
cursor configuration does not prove dataset identity. Pending gradients and
unregistered external state are not saved.

## Resume on GPU

GPU saving performs explicit readbacks. Loading stays on CPU with no hidden
transfer or automatic device choice. After successful loading:

```sx
var moments=optimizer.state().to(device)
model.to(device)
try optimizer.restore(moments) else error {panic(error)}
```

`SGDState.to(device)`, `AdamState.to(device)` and `AdamWState.to(device)` copy
metadata arrays and explicitly transfer present moments. CPU-policy buffers
remain on CPU. Moving only the model is insufficient: initialized moments must
be on its device before the next step. Exact continuation is qualified separately
on CPU and Metal with the same code and placement; it does not guarantee identity
between backends.

Dropout has 2^32 training iterations. Checkpoints also retain the exhausted
position 2^32, which remains usable in eval; another train pass is rejected
before computation. This counter does not silently wrap.

## Legacy model checkpoint

`Neural.Checkpoint` and its schema 1 remain compatible and separate. They store
only float32 parameters using their historical text representation; they are
not complete training resumes and do not promise bit fidelity. Use this guide
for continuation, or [optimizer snapshots](OptimizerState.md) for in-memory
restoration managed by your application.
