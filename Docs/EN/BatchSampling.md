# Iterate and resume index batches

`use Tensor.Data` exposes `Data.BatchSampler(count, batch_size,
shuffle:false, seed:0, drop_last:false)`. `next()` returns an independent
`int32[]?` list: `null` ends the epoch. The final partial batch is kept by
default; `drop_last:true` skips it and moves the cursor to `count`. Empty batches
are never returned. An empty source ends immediately.

`count` ranges from 0 to 2147483648, allowing indices from 0 to 2147483647.
`batch_size` must be positive. Indices follow ascending order, or a deterministic
permutation with `shuffle:true`. The same configuration and epoch reproduce
the same batches. `seed` accepts any `int`; epoch accepts a nonnegative `int`.
`set_epoch(epoch)` rebuilds the order and resets position to zero. Exhaustion
never advances the epoch. Sampler aliases share their cursor.

## Resume the next batch

`state()` returns an immutable `BatchSamplerState`: `count`, `batch_size`,
`shuffle`, `seed`, `drop_last`, `epoch`, `position`. Position counts traversed
indices, rather than the batch number. A snapshot does not change as the sampler
advances. `restore(state)` requires matching configuration, a nonnegative epoch,
and a position between zero and count, either a multiple of batch_size or equal
to count. Validation precedes cursor mutation. The permutation is reconstructed
from metadata rather than stored in the snapshot.

This complete example consumes the first list, saves its position, then
reconstructs the following batches in another sampler:

```sx
use Tensor
use Tensor.Data

func main() {
    var values:float[] = [10.0,20.0,30.0,40.0,50.0]
    let source = Tensor.vector(values)
    var sampler = Data.BatchSampler(5,2)
    if let indices = sampler.next() {
        let batch = source.gather(0,Tensor.vector(indices))
        assert(batch.values()[0] == 10.0 && batch.values()[1] == 20.0)
    } else { panic("missing first batch") }
    let saved = sampler.state()
    var resumed = Data.BatchSampler(5,2)
    resumed.restore(saved)
    if let indices = resumed.next() {
        let batch = source.gather(0,Tensor.vector(indices))
        assert(batch.values()[0] == 30.0 && batch.values()[1] == 40.0)
    } else { panic("missing resumed batch") }
    if let final = resumed.next() { assert(final.count() == 1 && final[0] == 4 as int32) }
    else { panic("missing final partial batch") }
    assert(resumed.next() == null && resumed.state().position == 5)
}
```

## Cost and placement

Ordered sampling creates only the indices of the requested batch, without a
count-sized list. Shuffling rebuilds a complete CPU permutation, with linear
memory and work in count. Generation uses explicitly seeded `STD.Randomizer`
and Fisher–Yates; seed and epoch bit patterns are combined with unsigned XOR.
There is no global RNG or system seed.

The sampler handles CPU metadata: no Tensor, GPU buffer, upload, download or
dispatch is created. The consumer turns indices into a Tensor and explicitly
chooses placement. For a GPU source, use `Tensor.vector(indices).to(device)`
before a gather compatible with its shape. Returned lists can be modified
without affecting later batches.

This snapshot describes index traversal only. It does not save a source,
collation, model, optimizer or dropout RNG; the existing parameter checkpoint
retains its own contract. [Supervised sources and collation](SupervisedBatches.md) use these same guarantees in BatchCursor.
