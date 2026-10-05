# Prepare CPU batches ahead

`Data.BatchCursor(source, batch_size, prefetch:2)` prepares at most two CPU batches
ahead through `STD.Threading`. `prefetch:0`, the default, keeps synchronous reading.
`Data.TensorSource` is eligible when both tensors are CPU and detached. An external
source explicitly opts into this contract by overriding `supports_prefetch()`;
`Data.Source` returns `false` by default.

This example resumes the second batch from the consumed position, then closes
the cursor. It prints `Prefetched batches resume from consumed position 2`.

```sx
use Tensor
use Tensor.Data

nocopy class Samples:Data.Source {
    init():super(5) {}
    override func supports_prefetch() bool {return true}
    override func observation(index:int) Result<Data.Observation,str> {
        return Result<Data.Observation,str>.success(Data.Observation(input:Tensor.scalar(index),target:Tensor.scalar(index*10)))
    }
}
func main() {
    var source=Samples()
    var cursor=Data.BatchCursor(source,2,prefetch:2)
    assert(cursor.state().position==0)
    if let first=try cursor.next() else error {panic(error)} {assert(first.input.int64_values()[0]==0)}
    let saved=cursor.state()
    assert(saved.position==2)
    if let second=try cursor.next() else error {panic(error)} {assert(second.input.int64_values()[0]==2)}
    cursor.restore(saved)
    if let repeated=try cursor.next() else error {panic(error)} {assert(repeated.input.int64_values()[0]==2)}
    cursor.close()
    print("Prefetched batches resume from consumed position $(saved.position)")
}
```

## Same traversal, consumed progress

Prefetch starts at construction. One worker owns source and collation calls while
preparing; it stops at the first failure or epoch end. Batches and their errors
are delivered in sampler order. An error does not advance consumed progress;
`next()` can retry that entire batch from its first observation. Shuffle, incomplete
final batches, `drop_last`, dtypes and values follow the
[synchronous traversal](SupervisedBatches.md).

`state()` refers only to consumed batches, even when preparation is further ahead.
`restore()` and `set_epoch()` wait for accepted work, discard prepared batches and
restart from the new position. Abandoned preparation may have performed reads:
external source/callback effects are not rolled back. Their observations must
not depend on consumption order.

The bound is `prefetch` prepared batches plus the currently returned batch;
collation also uses one batch's transient observations. Source contents, shuffle
indices and batches retained by the application remain its responsibility.
This bounds the number of batches rather than their byte size.

## Ownership and errors

Use the cursor from one thread. While it prefetches, do not use or modify its
source, collation captures or their dependencies from other code. Sources and
callbacks must stay on CPU, produce detached tensors and never create/manipulate
autograd graphs or GPU devices. They must not call the cursor again. The package
validates observations before collation and outputs before consumption; an
arbitrary callback is not executed in an environment that prohibits system effects.

An external source returning `true` promises this contract for `count()` and
`observation()`. A custom collator shares it when the caller enables `prefetch`.
GPU or tracked TensorSource inputs are rejected before starting a worker, with
no implicit detachment, transfer or fallback. Source, collation and validation
errors remain `Result.failure`. Negative `prefetch` or an ineligible source is
a fatal construction diagnostic.

`close()` waits for accepted work, releases prepared batches and joins workers.
It is idempotent and terminal: `next()` then returns an error, `restore()` and
`set_epoch()` are rejected, and checkpoint loading fails without changing its
target. Cursor destruction also ensures shutdown; references retained by the
application extend its lifetime. `close()` lets you reuse the source immediately
after stopping early. A blocked source read must finish before closing can return.

## Resume CPU or GPU training

The [complete checkpoint](TrainingCheckpoint.md) records consumed progress without
serializing prepared batches, workers or the `prefetch` choice. It waits for
preparation before checking stable source count. Rejected loading preserves the
next prepared batches and all progress; successful restoration reconstructs them
from the saved snapshot. The caller can rebuild a synchronous or prefetched
cursor with the same sampler configuration.

Batches stay on CPU. For GPU training, explicitly move `batch.input` and
`batch.target` to the device in the training loop, along with the model and its
moments as described in the guide. Tests compare complete continuous/resumed state
with BatchNorm, Dropout, AdamW, schedules and epoch changes, separately on CPU and
Metal. This qualification promises no throughput improvement.
