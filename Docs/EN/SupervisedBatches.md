# Iterate supervised batches

This feature requires the Silex compiler fix for `try` in mutating methods,
described in its `Unreleased` notes.

`Data.TensorSource(inputs, targets)` reads observations along the leading axis
of two Tensors. They have the same observation count and share CPU placement or
a GPU device; remaining shapes and dtypes may differ. Both Tensors have at least
one axis, including an empty source.

`Data.BatchCursor(source, batch_size, shuffle:false, seed:0, drop_last:false,
collate:Data.stack_batch)` owns its index cursor. `next()` returns
`Result<Data.Batch?,str>`: success with `Batch(input, target)`, success with
`null` at epoch end, or a recoverable error. By default each input/target column
is stacked along batch axis0. Observations within a column share shape, dtype,
and placement; input and target share placement.

```sx
use Tensor
use Tensor.Data

func same_values<T>(left:T[],right:T[]) bool {
    if left.count()!=right.count() {return false}
    var index=0
    while index<left.count() {
        if left[index]!=right[index] {return false}
        index++
    }
    return true
}

func main() {
    var inputs:float[] = [1.0,2.0,3.0,4.0,5.0,6.0]
    var labels:int32[] = [0 as int32,1 as int32,2 as int32]
    var source = Data.TensorSource(Tensor(inputs,[3,2]),Tensor.vector(labels))
    var cursor = Data.BatchCursor(source,2)
    if let batch = try cursor.next() else error { panic(error) } {
        assert(same_values(batch.input.shape(),[2,2]) && same_values(batch.target.shape(),[2]))
        var expected:float[] = [1.0,2.0,3.0,4.0]
        assert(same_values(batch.input.values(),expected))
        var expected_labels:int32[] = [0 as int32,1 as int32]
        assert(same_values(batch.target.int32_values(),expected_labels))
    } else { panic("missing first supervised batch") }
    let saved = cursor.state()
    var resumed = Data.BatchCursor(source,2)
    resumed.restore(saved)
    if let batch = try resumed.next() else error { panic(error) } {
        var expected:float[] = [5.0,6.0]
        assert(same_values(batch.input.values(),expected) && batch.target.int32_item() == 2 as int32)
    } else { panic("missing final supervised batch") }
    assert((try resumed.next() else error { panic(error) }) == null)
}
```

`state()`, `restore(state)`, and `set_epoch(epoch)` follow the
[sampler guarantees](BatchSampling.md). Partial final batches are kept by default,
shuffling depends on seed/epoch/count, and exhaustion never advances the epoch.
The snapshot describes the next batch without keeping Tensors or a source.
To resume the same data, reconstruct the same source and collation: matching
sampler configuration does not prove matching data content.

## External source and custom collation

Derive from `Data.Source`, call its protected constructor `super(count)`, and
override `observation(index:int) Result<Data.Observation,str>`. The advertised
count stays stable during traversal; the author guarantees bounds and observation
stability. `Observation` contains two Tensor fields, `input` and `target`.
Source and BatchCursor references use `var` bindings in Silex.

A named `func(Data.Observation[]) Result<Data.Batch,str>` function chooses
collation. It receives observations in sampler order. This complete external
example explicitly pads variable-length inputs:

```sx
use Tensor
use Tensor.Data

func same_values<T>(left:T[],right:T[]) bool {
    if left.count()!=right.count() {return false}
    var index=0
    while index<left.count() {
        if left[index]!=right[index] {return false}
        index++
    }
    return true
}

nocopy class Samples:Data.Source {
    init():super(3) {}
    override func observation(index:int) Result<Data.Observation,str> {
        if index < 0 || index >= self.count() {return Result<Data.Observation,str>.failure("sample index outside source")}
        var shape:int[]=[index+1]
        return Result<Data.Observation,str>.success(Data.Observation(input:Tensor.full(shape,(index+1) as float),target:Tensor.scalar(index)))
    }
}

func padded_batch(observations:Data.Observation[]) Result<Data.Batch,str> {
    var width=0
    for observation in observations {if observation.input.count()>width {width=observation.input.count()}}
    var padded:Data.Observation[]=[]
    for observation in observations {
        var padding_shape:int[]=[width-observation.input.count()]
        var parts:Tensor[]=[observation.input,Tensor.zeros(padding_shape)]
        padded.append(Data.Observation(input:Tensor.concatenate(parts,0),target:observation.target))
    }
    return Data.stack_batch(padded)
}

func main() {
    var source=Samples()
    var cursor=Data.BatchCursor(source,2,collate:padded_batch)
    if let batch=try cursor.next() else error {panic(error)} {
        var expected:float[]=[1.0,0.0,2.0,2.0]
        assert(same_values(batch.input.shape(),[2,2]) && same_values(batch.input.values(),expected))
    } else {panic("missing padded batch")}
}
```

Collation returns input and target with a leading axis equal to the received
observation count and matching placement/device. It may choose padding,
remaining shapes, and dtypes. `Data.stack_batch` can complete a transformation
without reimplementing default collation. A directly supplied empty list returns
an error from stack_batch.

## Errors and progress

Source and collation errors propagate their `str` message. Shape/dtype/placement
incompatibilities in stack_batch, a changed count, and inconsistent collation
outputs are recoverable. The sampler advances only after the entire batch
succeeds. An error leaves its snapshot unchanged; the next call rereads the same
batch from its first observation. External read or callback effects are not undone.

The cursor rejects recursive reads. Epoch and restoration cannot change during
a read. Invalid construction/restoration configurations and changed counts during
those operations remain fatal diagnostics, as with the sampler. Use a cursor
from one thread; synchronous reads and callbacks execute on that thread.

## Placement, gradients, and lifetime

TensorSource keeps the supplied values; observation selects a view. Collation
produces Tensors under the `Tensor.stack` contract. All nine numeric storage
dtypes and bool are preserved exactly on CPU/GPU, including qualified int64/uint64
extremes. Inputs and targets are not converted to share a dtype. Float32 input
gradients remain attached to the source computation, without implicit detachment.

Shapes, dtypes, and placements of both columns are checked before collation
allocations. GPU batches remain resident, without index uploads or hidden
downloads. Custom callbacks explicitly choose computations and transfers.
Reading GPU values requires `cpu()` as for any Tensor. The cursor retains its
source and callback; snapshots do not retain them. Traversal ends without a
worker or deferred work to clean up. Prefetch and complete training checkpoints
are not provided by this API.
