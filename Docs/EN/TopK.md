# Select the best elements

`Tensor.topk(k, axis, largest:true)` selects the k best values on an explicit
axis. The `TopKResult` contains `values` and `indices`: values retain their dtype,
indices are detached int32 data. Use `largest:false` to select the smallest
values.

This program selects two maximum observations and computes their gradient:

```sx
use Tensor
use Tensor.Autograd

func main() {
    var observations:float[] = [2.0,7.0,7.0,1.0]
    var input = Autograd.Variable(Tensor.vector(observations))
    let selected = input.value().topk(2,axis:0)
    let indices = selected.indices.int32_values()
    assert(indices[0] == 1 && indices[1] == 2)
    let loss = selected.values.sum()
    assert(loss.detach().item() == 14.0)
    var gradients = Autograd.grad(loss,[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 0.0 && actual[1] == 1.0 && actual[2] == 1.0 && actual[3] == 0.0)
    assert(input.gradient() == null)
}
```

The indices are [1,2], the values [7,7], and the sum is 14. The two selected
observations receive a gradient of 1; the others receive zero. `Autograd.grad`
preserves the Variable slot. To annotate a result, import its declaration with
`use Tensor.TopKResult`.

## Choose the axis and number of elements

The axis is required, nonnegative and less than the rank. Output shape retains
all dimensions and replaces the selected axis with k. For input [2,3],
topk(2,axis:1) returns values and indices of shape [2,2]. Each index is a
coordinate on that input axis, rather than a position in the output.

k must lie between zero and the axis size. k equal to zero returns an empty
output, including when that axis is empty. An empty outer dimension remains
empty. k equal to the axis size selects all values in the requested order.
Noncontiguous views, offsets and expanded dimensions are read in logical order;
their storage remains unchanged.

## Predict order and gradients

Values run from best to worst: descending for largest, ascending otherwise.
Equal values retain input index order, including +0 and −0, whose original
signs remain in the returned values. NaNs come first in both directions, by
increasing index, followed by numbers and infinities in the requested order.
This convention follows [indexed extrema](IndexedReductions.md).

Indices do not differentiate. Float32 values differentiate through selection,
including on GPU: each gradient returns to the selected position; unselected
positions receive zero. An expanded view accumulates gradients back to its
source. Selection is discontinuous when ordering changes; run
[gradcheck](Gradients.md) away from ties or ranking changes.

## Choose placement and limits

All nine CPU numeric dtypes are accepted, with exact integer comparison through
int64/uint64. Booleans are rejected. GPU input must be float32; values and indices
remain on the same device without implicit uploads or readbacks. Observation
requires an explicit `cpu()`.

The last possible index must fit int32, limiting the axis to at most 2^31
elements. Shapes and value/index sizes are checked before allocation. GPU
retains rank 5 and uint32 metadata limits. Selection exposes no general sorting
operation; no speed advantage is promised for large domains or large k.
The [GPU profile](Profiling.md) lets you measure your complete workload.
