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

## Mathematical AdamW oracle

`adamw_decimal.py` uses Python’s standard-library Decimal at 50-digit precision
for six explicit updates, including zero and tiny gradients. It records model
values and both moments in `ADAMW_REPORT.json`; `AdamWOracle.sx` pins the rounded
constants with an absolute tolerance of 2e-5 for float32 CPU/GPU. This independent
mathematical oracle neither imports Tensor nor reuses its integer-power helper.
Ordinary Silex tests remain offline and do not invoke Python. Recheck from the
Tensor repository with `python3 Tools/Oracle/adamw_decimal.py --check`; review
the report and Silex literal changes together after any deliberate regeneration.

## Mathematical BatchNorm oracle

`batchnorm_decimal.py` computes dense and NCHW results using standard-library
Decimal at 50-digit precision. Its closed-form channel derivative is independent
of composite autograd: the report pins outputs, input/weight/bias gradients,
population variance, and unbiased running updates. `BatchNormOracle.sx` uses
these literals with absolute tolerance 3e-5 on float32 CPU/GPU, including strided
NCHW views. Normal tests do not run Python. Recheck the independent report with
`python3 Tools/Oracle/batchnorm_decimal.py --check` and inspect report and fixture
changes together after regeneration. Smooth central-difference tests provide a
second derivative control outside constant regions.

## Dense embedding oracle

`python3 Tools/Oracle/embedding_decimal.py --check` checks the pinned
`EMBEDDING_REPORT.json` and `Tests/Consumer/Tests/EmbeddingOracle.sx` without
external frameworks. It uses Python Decimal at 50 digits to evaluate a fixed
row selection and the transpose of that selection matrix, with and without
stopped padding derivatives. Ordinary Silex tests use the generated fixture
without Python. Weight conversion is direct row-major `[vocabulary,features]`
float32; indices are exact int32 and no bias, transpose or gate reorder applies.
Padding rows are read normally; their declared lookup derivative is zero.
Finite-difference qualification excludes that deliberately stopped row.

## Masked softmax Decimal fixture

`masked_softmax_decimal.py` uses Python’s standard Decimal at 60 decimal digits.
Its small finite inputs use direct exp normalization without Tensor or the
implementation’s max-shift stabilization. The independent VJP is
`probability * (seed - sum(probability * seed))`; excluded positions and empty
rows are zero. No external framework, weights, transpose or gate ordering.
`MASKED_SOFTMAX_REPORT.json` pins inputs, visibility, seed, outputs, gradients
and absolute tolerance 0.000002. The generated CPU/Metal consumer is hermetic.
Regenerate voluntarily with `python3 Tools/Oracle/masked_softmax_decimal.py`,
then check with the same command plus `--check` and review both diffs.

## Scaled attention Decimal fixture

`attention_decimal.py` uses standard-library Decimal at 60 digits to compute
small finite Q/K dot products, direct exp normalization, value aggregation and
closed-form Q/K/V derivatives with a nonuniform output seed. Explicit and default
scales, partial visibility and an empty row are pinned in `ATTENTION_REPORT.json`.
Conversion is direct row-major, with no parameter transpose, bias or gate ordering.
`AttentionOracle.sx` is a hermetic CPU/Metal consumer with absolute tolerance
0.000002. Check with `python3 Tools/Oracle/attention_decimal.py --check`; review
report and generated source together after deliberate regeneration.

## Projected multi-head Decimal fixture

`multihead_decimal.py` uses standard-library Decimal at 60 digits to compute
four asymmetric affine projections, direct exp normalization per head and
analytic gradients of Q/K/V and every weight/bias. Head-specific masks include
an empty query and a query with only one visible head. `MULTIHEAD_REPORT.json`
pins inputs, row-major `[input,output]` weights, biases and output feature order;
there is no framework packing or transpose conversion. Hermetic CPU/Metal tests
use absolute tolerance 0.000003. Recheck with
`python3 Tools/Oracle/multihead_decimal.py --check`, and review fixture/report
together after deliberate regeneration.

## Pre/post-norm Transformer Decimal fixture

`transformer_decimal.py` uses standard-library Decimal at 60 digits, direct
exp-normalized self-attention, population LayerNorm equations and analytic VJPs
through residuals and ReLU. `TRANSFORMER_REPORT.json` pins pre/post cases, feature
order, separate row-major `[input,output]` matrices and feature vectors; no packing
or transpose conversion. The fixture compares output, attention weights, input
and all sixteen parameter gradients on CPU/Metal with absolute tolerance8e-6.
It stays away from ReLU kinks and needs no Python at runtime. Recheck with
`python3 Tools/Oracle/transformer_decimal.py --check`; review source and report
together after deliberate regeneration.

## GRU reset-after

`python3 Tools/Oracle/gru_decimal.py --check` validates the committed GRU fixture.
Decimal precision60, packed r/z/n columns in row-major [input,output] matrices,
separate input and recurrent biases. The analytic VJP includes sequence and final
state seeds, input, initial state and four parameters; all52 derivatives also
match central Decimal differences at epsilon1e-12. Absolute runtime tolerance2e-6.
Nonzero recurrent bias distinguishes reset-after from reset-before. Ordinary
Silex tests consume the committed fixture without Python or network access.
