# Differential oracle generator

This development-only tool regenerates the reviewable Silex fixtures, fixed
neural checkpoints, and reference reports. It is not part of the Tensor
package manifest or runtime.

Use CPython 3.12.2 and create a disposable environment outside the repository:

```sh
python3.12 -m venv /private/tmp/tensor-oracle
/private/tmp/tensor-oracle/bin/python -m pip install -r Tools/Oracle/requirements.lock
/private/tmp/tensor-oracle/bin/python Tools/Oracle/generate.py
```

Run the command from the Tensor repository root. It uses seed `1592639215` and
row-major (`C`) input arrays. `--check` computes both outputs without writing
and fails if either differs from the committed copy:

```sh
/private/tmp/tensor-oracle/bin/python Tools/Oracle/generate.py --check
PIP_NO_INDEX=1 /private/tmp/tensor-oracle/bin/python Tools/Oracle/generate.py --check
```

The generator rejects a common numeric fixture unless at least two applicable
oracles agree. It deliberately records framework disagreements instead of
selecting whichever result matches Tensor. NumPy and PyTorch establish shapes,
broadcasting, and ordinary eager results; TensorFlow contributes immutable
tensor and placement observations; JAX contributes functional view and
transformation results. The generated report states the exact contributors for
every case.

The Python packages retain their upstream licenses. NumPy and PyTorch include
components under several permissive licenses; TensorFlow and JAX are published
under Apache-2.0. Review the exact notices in the pinned distributions and the
upstream license files for [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt),
[PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE),
[TensorFlow](https://github.com/tensorflow/tensorflow/blob/master/LICENSE), and
[JAX](https://github.com/jax-ml/jax/blob/main/LICENSE). No framework source,
wheel, Python environment, or generated cache is committed.

After regeneration, inspect every generated diff before accepting a contract
change:

```sh
git diff -- Tests/Consumer/Tests/OracleDifferential.sx \
  Tests/Consumer/Tests/NeuralOracleDifferential.sx \
  Tests/Consumer/Tests/Fixtures/NeuralOracle*.sxtc \
  Tools/Oracle/REPORT.md Tools/Oracle/NEURAL_REPORT.md
```

The normal Tensor tests consume only the generated `.sx` fixtures and text
checkpoints and remain fully offline. The general fixture also exercises its
comparison primitives with deliberately altered shape, stride, value, and
tolerance inputs. The neural fixture fixes identical MLP, CNN, and SimpleRNN
weights and data for PyTorch and TensorFlow, then checks each layer output,
named gradient, SGD/Adam update, and training checkpoints. A local finite-
difference test independently covers a smooth derivative.
