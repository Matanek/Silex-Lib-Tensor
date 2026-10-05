# Adapt a shape to batches

Add or remove a unit dimension with `unsqueeze` and `squeeze`. `expand` exposes
multiple occurrences through a view; `repeat` materializes tiles. These
operations preserve dtype, placement, and float32 gradients.

This program prepares four occurrences of a vector and checks their gradient:

```sx
use Tensor
use Tensor.Autograd

func main() {
    var values:float[] = [1.0,2.0,3.0]
    var input = Autograd.Variable(Tensor.vector(values))
    let batch = input.value().unsqueeze(0).expand([2,3])
    let tiled = batch.repeat([2,1])
    assert(batch.strides()[0] == 0 && tiled.is_contiguous())
    var gradients = Autograd.grad(tiled.multiply(tiled).sum(),[input.value()])
    let actual = gradients[0].values()
    assert(actual[0] == 8.0 && actual[1] == 16.0 && actual[2] == 24.0)
    assert(input.gradient() == null)
    assert(input.value().unsqueeze(0).squeeze(0).rank() == 1)
}
```

Each value occurs four times in the sum of squares, so its derivative is eight
times that value. Returned gradients leave the slot absent.

## Add and remove unit dimensions

`unsqueeze(axis)` inserts a dimension of size one at an axis from zero through
the rank inclusive. `squeeze(axis)` removes that axis only when its size is one;
an invalid axis or another size produces a diagnostic. `squeeze()` removes all
unit dimensions; a scalar stays scalar and an empty axis stays empty. Negative
axes are rejected.

These views preserve values, storage, and offset, including after transpose or
narrow. The inserted axis has a zero stride; other strides stay identical. They
require no copy for a noncontiguous source. If a later operation requires a
contiguous Tensor, request `contiguous()` explicitly.

## Expand a view

`expand(shape)` takes a nonnegative shape with at least the source rank. Axes
align on the right. An equal dimension is preserved; a source dimension of size
one can take another size through a zero stride. New leading axes are implicitly
unit dimensions. Source dimensions other than one cannot change. No `-1` is accepted.

A unit dimension can become empty. A zero source dimension stays zero and cannot
become nonempty. Cardinality and logical byte size are checked before creating
the view. Views create neither GPU buffers nor commands and perform no transfers.
Expansion occurrences are reduced to source dimensions during backward or
`Autograd.grad`, including an empty domain that returns zeros.

## Repeat tiles

`repeat(repeats)` takes at least one nonnegative factor per source axis.
Additional factors add leading unit axes. Each factor multiplies its corresponding
dimension; values repeat as tiles in logical order, rather than duplicating each
element adjacently.

The result is materialized and contiguous, including when every factor is one.
A zero factor produces an empty shape whose other dimensions remain multiplied.
A scalar accepts an empty list or new axes. All shapes, factors, and sizes are
validated before allocation. Gradients sum occurrence contributions. Source values stay immutable;
`Autograd.grad` preserves their slots.

Views and repetition preserve all nine numeric dtypes and booleans exactly on
CPU/GPU. Gradients apply to float32 and retain the rank limits of the GPU
operations used (up to five for computation). No implicit readback or upload:
use `cpu()` to observe a GPU result, and `detach().cpu()` for a still-tracked
value. Repetition incurs materialization costs; no speed gain is promised.

See the [gradient guide](Gradients.md) and [reference](Reference.md) to compose
these shapes with differentiable computation.
