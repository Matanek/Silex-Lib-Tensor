# Tensor

`Tensor` provides Silex's multidimensional numerical foundation. One immutable
public value covers `float32` and eight integer dtypes, explicit CPU/GPU
placement, lossless transfers, shared shape views, typed indexing, broadcasting,
checked numerical computation, seeded initialization, and eager neural
primitives for dense, convolutional, and recurrent models. Automatic
differentiation can build on this immutable contract without exposing its
graph in the core API.

```text
silex install Tensor
```

## Documentation

- [French documentation](Docs/FR/README.md)
- [English documentation](Docs/EN/README.md)
- [Differential oracle maintenance](Tools/Oracle/README.md)
- [Public memory and compute campaign](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)

The package requires Silex 0.43.0 or newer.
