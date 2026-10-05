# Build sequence visibility

Declare your batch’s valid lengths with `Neural.sequence_mask`, then compose them
with `Neural.causal_mask`. Both functions return CPU boolean masks; `true` allows
a position. They extract no data from a GPU Tensor. Lengths come from metadata
already known by the application.

The example composes three sequences of lengths 0, 2 and 4 with three query
positions. Each sequence’s keys form a valid prefix; causality prevents a query
from seeing a future key.

```sx
use Tensor
use Tensor.Neural

func main() {
    let lengths:int[]=[0,2,4]
    let keys=Neural.sequence_mask(lengths,maximum_length:4)
    let causal=Neural.causal_mask(query_length:3,key_length:4)
    let visible=keys.unsqueeze(1).logical_and(causal.unsqueeze(0))
    let scores=Tensor.ones([3,3,4])
    let selected=Tensor.where(visible,scores,Tensor.zeros([3,3,4]))
    assert(selected.shape()[0]==3 && selected.shape()[1]==3 && selected.shape()[2]==4)
    assert(selected.sum().item()==11.0)
    let continuation=Neural.causal_mask(query_length:2,key_length:5,query_offset:2)
    assert(continuation.bool_at([0,2]) && !continuation.bool_at([0,3]))
    print("Sequence visibility: lengths and causal positions composed")
}
```

The resulting visibility has shape `[batch,queries,keys]`. Its selection contains
11 values equal to 1; all others are 0. Offset 2 allows the first query of the
next fragment to see keys at positions 0, 1 and 2.

## Explicit lengths and positions

| Call | CPU result shape | Allowed position |
|---|---|---|
| `Neural.sequence_mask(lengths, maximum_length)` | `[lengths.count(),maximum_length]` | `position < lengths[batch]` |
| `Neural.causal_mask(query_length, key_length, query_offset:0)` | `[query_length,key_length]` | `key <= query + query_offset` |

Dimensions and offset must be nonnegative. Each length must be between 0 and
`maximum_length`, inclusive. The dimension product and last query position must
fit in `int`; both are checked before construction. The position check also
applies with zero keys. A single query at the maximum offset remains valid.

A zero length produces an entirely `false` row. An empty batch, zero queries or
zero keys preserves both declared axes with zero elements. Rectangular causality
always uses query positions `query_offset + query`; alignment is not inferred
from the difference between query and key counts.

A length mask describes a contiguous prefix. If your padding contains holes,
supply an explicit boolean mask instead of a length. An [Embedding](Embeddings.md)
and its padding index do not construct this mask automatically. Valid query
positions can be composed separately using a `[batch,queries,1]` mask.

## Compose and place the mask

`unsqueeze` adds the axes required for broadcasting: keys become `[batch,1,keys]`,
causality becomes `[1,queries,keys]`. Add a head axis to obtain
`[batch,1,queries,keys]` when your scores have multiple heads.
[Boolean logic and where](Masks.md) preserve the `true` convention.

For GPU computation, explicitly place each mask with `mask.to(device)` before
composing it with resident scores. No readback or index cast is required. The
usual Tensor shape and device constraints apply to GPU operations. Both factories
remain CPU operations and choose no device for the application.

Masks are detached. `where` routes gradients to selected branches and reduces
their broadcast axes; an entirely `false` row selects the second branch throughout.
These functions build visibility and do not constitute a masked softmax.
