# Tensor

`Tensor` provides Silex's multidimensional numerical foundation. One immutable
public value covers `float32`, eight integer dtypes, and boolean masks, explicit CPU/GPU
placement, lossless transfers, shared shape views, typed indexing, broadcasting,
checked numerical computation, seeded initialization, and eager neural
primitives for dense, convolutional, and recurrent models. Its eager
reverse-mode autodifferentiation keeps immutable Tensor values and GPU-resident
gradients behind explicit `Tensor.Autograd.Variable` leaves. Named
`Tensor.Neural.Parameter` values, composable dense, convolutional and recurrent
layers, deterministic checkpoints, and `Tensor.Optim` SGD/Adam optimizers
complete an explicit eager training workflow without hidden transfers.

```text
silex install Tensor
```

## Documentation

- [Release notes](CHANGELOG.md) · [French](CHANGELOG.fr.md)

- [French documentation](Docs/FR/README.md)
- [English documentation](Docs/EN/README.md)
- [French API reference](Docs/FR/Reference.md) · [English](Docs/EN/Reference.md)
- [French design direction](Docs/FR/Direction.md) · [English](Docs/EN/Direction.md)
- [French training recipes](Docs/FR/Recipes/Training.md) · [English](Docs/EN/Recipes/Training.md)
- [French functional gradient guide](Docs/FR/Gradients.md) · [English](Docs/EN/Gradients.md)
- [French shape guide](Docs/FR/Shapes.md) · [English](Docs/EN/Shapes.md)
- [French statistics guide](Docs/FR/Statistics.md) · [English](Docs/EN/Statistics.md)
- [French top-k guide](Docs/FR/TopK.md) · [English](Docs/EN/TopK.md)
- [French extremum index guide](Docs/FR/IndexedReductions.md) · [English](Docs/EN/IndexedReductions.md)
- [French sequence guide](Docs/FR/Sequences.md) · [English](Docs/EN/Sequences.md)
- [French mask guide](Docs/FR/Masks.md) · [English](Docs/EN/Masks.md)
- [Differential oracle maintenance](Tools/Oracle/README.md)
- [Public memory and compute campaign](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)

The package requires Silex 0.44.1 or newer.
