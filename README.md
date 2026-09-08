# Tensor

`Tensor` provides Silex's multidimensional numerical foundation. One immutable
public value covers `float32` and eight integer dtypes, explicit CPU/GPU
placement, lossless transfers, shared shape views, typed indexing, broadcasting,
and checked elementwise computation. Automatic differentiation and neural-network packages
can build on this contract without belonging to the core.

```text
silex install Tensor
```

## Documentation

- [French documentation](Docs/FR/README.md)
- [English documentation](Docs/EN/README.md)
- [Differential oracle maintenance](Tools/Oracle/README.md)
- [Public memory and compute campaign](https://github.com/Matanek/Silex-Benchmarks/tree/main/Sources/TensorStableCompute)

The package requires Silex 0.43.0 or newer.
