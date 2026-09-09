# Tensor 0.1.0 direction

Tensor provides explicit multidimensional computation for Silex. One immutable
value moves between CPU and GPU, while eager training uses named mutable
leaves. This direction borrows proven ideas without trying to reproduce
another framework's surface.

## Three influences, one Silex contract

PyTorch inspires the natural `forward -> loss -> backward -> step` order,
eager graphs consumed after backpropagation, composable layers, and the numeric
conventions of SGD and Adam. Tensor does not adopt general tensor mutation,
retained graphs, or higher-order derivatives.

TensorFlow makes the distinction between a computation value and a trainable
variable useful, together with explicit runtime or device placement. Tensor
does not adopt deferred graphs, sessions, function compilation, or implicit
placement.

JAX demonstrates the value of immutable array values and predictable
functional transformations. Tensor nevertheless provides neither `jit`,
`vmap`, nor a general functional gradient transformation: autodifferentiation
remains eager and attached to executed operations.

## What version 0.1.0 guarantees

- nine dense CPU dtypes with views, broadcasting, reductions, and checked
  linear algebra;
- bit-exact transfer of all nine dtypes between CPU and GPU;
- resident `float32` GPU computation without fallback or implicit readback;
- eager reverse-mode `float32` autodifferentiation, named parameters, SGD,
  Adam, and global-norm clipping;
- sequential dense, simple convolutional, and simple recurrent models;
- deterministic parameter checkpoints validated before mutation.

The NumPy, PyTorch, TensorFlow, and JAX oracles verify this contract. They do
not extend it automatically when their own APIs evolve.

## Deliberate limits

Version 0.1.0 does not include integer GPU computation, sparse tensors,
quantization, deferred graphs, multi-device distribution, higher-order
derivatives, AdamW, BatchNorm, grouped or transposed convolutions, LSTM/GRU,
embeddings, or attention.

A checkpoint contains parameters only: no code, autograd graph, device, or
optimizer state. Observing a GPU value and saving a GPU model remain explicit
synchronization and readback points.

Use the [compatibility reference](Reference.md) to choose a dtype and placement,
then the [training recipes](Recipes/Training.md) to build an MLP, CNN, or RNN.
