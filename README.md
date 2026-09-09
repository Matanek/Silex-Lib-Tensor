# Tensor

`Tensor` provides Silex's multidimensional numerical foundation. One immutable
public value covers `float32` and eight integer dtypes, explicit CPU/GPU
placement, lossless transfers, shared shape views, typed indexing, broadcasting,
checked numerical computation, seeded initialization, and eager neural
primitives for dense, convolutional, and recurrent models. Its eager
reverse-mode autodifferentiation keeps immutable Tensor values and GPU-resident
gradients behind explicit `Tensor.Autograd.Variable` leaves. Named
`Tensor.NN.Parameter` values, composable dense, convolutional and recurrent
layers, deterministic checkpoints, and `Tensor.Optim` SGD/Adam optimizers
complete an explicit eager training workflow without hidden transfers.

```text
silex install Tensor
```

## Documentation

- [French documentation](Docs/FR/README.md)
- [English documentation](Docs/EN/README.md)
- [French API reference](Docs/FR/Reference.md) · [English](Docs/EN/Reference.md)
- [French design direction](Docs/FR/Direction.md) · [English](Docs/EN/Direction.md)
- [French training recipes](Docs/FR/Recipes/Training.md) · [English](Docs/EN/Recipes/Training.md)
- [Differential oracle maintenance](Tools/Oracle/README.md)
- [Public memory and compute campaign](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)

The package requires Silex 0.44.1 or newer.
