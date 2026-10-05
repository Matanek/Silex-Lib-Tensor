# Learn vectors from indices

`Neural.Embedding(name, vocabulary_size, embedding_features, seed,
padding_index:null)` maps each `int32` index to a row of a trainable `float32`
table. An input `[tokens]` produces `[tokens, features]`; a batch-first input
`[batch, tokens]` produces `[batch, tokens, features]`. Repeated indices add
their contributions to the dense weight gradient.

```sx
use Tensor
use Tensor.Neural
use Tensor.Optim

func main() {
    var embedding=Neural.Embedding("tokens",4,2,seed:7,padding_index:0)
    var ids:int32[]=[0,1,2,3]
    let input=Tensor.vector(ids)
    var targets:float[]=[0.0,0.0,0.2,-0.3,0.4,-0.5,0.6,-0.7]
    let target=Tensor(targets,[4,2])
    var optimizer=Optim.SGD(embedding.parameters(),learning_rate:0.2)
    var step=0
    while step<40 {
        optimizer.zero_grad()
        let difference=embedding.forward(input).subtract(target)
        difference.multiply(difference).mean().backward()
        optimizer.step()
        step++
    }
    let weights=embedding.parameters()[0].value().detach().values()
    assert(weights[0]==0.0 && weights[1]==0.0)
    assert(embedding.forward(input).shape()[1]==2)
    print("Embedding training: padding preserved")
}
```

The [executable example](../../Tests/Consumer/Examples/EmbeddingTraining.sx)
uses public APIs only. The weight is named `name + ".weight"` and has shape
`[vocabulary_size, embedding_features]`. The seed uniformly initializes the
table in `[-1, 1]`; the padding row, when present, is then zeroed. The vocabulary
has between 1 and 2147483648 rows and features is positive; ordinary Tensor
cardinality and allocation checks apply.

## Padding and model state

`padding_index` is optional and must belong to the vocabulary. Forward reads
that row normally, including after explicit weight replacement. Its **lookup
gradient is stopped**. A selection oracle checks this contract; finite
differences on the stopped row do not. The complete Jacobian without padding
is qualified through finite differences.

The initially zero row remains zero under SGD, Adam and AdamW, including
momentum or weight decay, provided no other operation contributes a gradient
and no mutation or restoration replaces it. Decay may change a nonzero row
replaced by the application; Embedding does not reset optimizer moments or
restore a row after a step.

Embedding follows `Layer`: `Sequential` composition, `parameters`, freezing,
modes, explicit transfers and weight checkpoints. `TrainingCheckpoint` keeps
its configuration, including padding, and rejects an incompatible configuration
before mutation. Parameter names remain unique in composed models. A frozen
layer returns a computation without provenance from its weight.

## Placement and invalid indices

Indices and weight must share placement and device. Use `embedding.to(device)`
with `input.to(device)`, or `embedding.cpu()` with `input.cpu()`; reading GPU
values requires `detach().cpu().values()`. Lookup and derivative stay on GPU
without implicit uploads/downloads. Strided views and empty axes are accepted.

On CPU, a negative or out-of-vocabulary index fails before lookup. On GPU,
the existing `gather` contract produces NaN for that vector and ignores its
weight contribution, with checked memory access. Validate CPU indices before
upload if your application requires immediate rejection. No dtype conversion,
GPU extraction or index normalization is implicit.
