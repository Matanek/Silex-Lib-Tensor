#!/usr/bin/env python3
"""Generate Tensor's deterministic multi-framework oracle fixture."""

from __future__ import annotations

import argparse
import importlib.metadata
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

EXPECTED_PYTHON = (3, 12, 2)
EXPECTED_VERSIONS = {
    "numpy": "2.5.3",
    "torch": "2.14.0",
    "tensorflow": "2.21.0",
    "jax": "0.11.1",
    "jaxlib": "0.11.1",
}
SEED = 1_592_639_215
ORDER = "C"
ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "Tests/Consumer/Tests/OracleDifferential.sx"
REPORT_PATH = ROOT / "Tools/Oracle/REPORT.md"


@dataclass(frozen=True)
class Result:
    shape: tuple[int, ...]
    values: tuple[float | int, ...]


@dataclass(frozen=True)
class Case:
    name: str
    family: str
    dtype: str
    result: Result
    oracles: tuple[str, ...]
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 0.0


def require_environment() -> dict[str, str]:
    actual_python = sys.version_info[:3]
    if actual_python != EXPECTED_PYTHON:
        expected = ".".join(str(value) for value in EXPECTED_PYTHON)
        actual = ".".join(str(value) for value in actual_python)
        raise SystemExit(f"oracle generator requires Python {expected}, got {actual}")
    actual: dict[str, str] = {}
    for package, expected in EXPECTED_VERSIONS.items():
        version = importlib.metadata.version(package)
        actual[package] = version
        if version != expected:
            raise SystemExit(f"oracle generator requires {package}=={expected}, got {version}")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("reference fixtures require macOS arm64")
    return actual


def normalize(value: object) -> Result:
    import numpy as np

    array = np.asarray(value)
    flattened: list[float | int] = []
    for item in array.reshape(-1, order=ORDER):
        if np.issubdtype(array.dtype, np.floating):
            flattened.append(float(np.float32(item)))
        else:
            flattened.append(int(item))
    return Result(tuple(int(size) for size in array.shape), tuple(flattened))


def same_float(left: float, right: float, absolute: float, relative: float) -> bool:
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    if left == 0.0 and right == 0.0:
        return math.copysign(1.0, left) == math.copysign(1.0, right)
    return abs(left - right) <= max(absolute, relative * abs(right))


def consensus(
    name: str,
    family: str,
    dtype: str,
    producers: dict[str, Callable[[], object]],
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> Case:
    results = {oracle: normalize(producer()) for oracle, producer in producers.items()}
    if len(results) < 2:
        raise RuntimeError(f"{name}: a common fixture requires at least two oracles")
    authority_name = next(iter(results))
    authority = results[authority_name]
    for oracle, result in tuple(results.items())[1:]:
        if authority.shape != result.shape or len(authority.values) != len(result.values):
            raise RuntimeError(f"{name}: {authority_name} and {oracle} disagree on shape")
        for left, right in zip(authority.values, result.values, strict=True):
            if dtype == "float32":
                if not same_float(float(left), float(right), absolute_tolerance, relative_tolerance):
                    raise RuntimeError(f"{name}: {authority_name}={left!r}, {oracle}={right!r}")
            elif left != right:
                raise RuntimeError(f"{name}: {authority_name}={left!r}, {oracle}={right!r}")
    return Case(
        name,
        family,
        dtype,
        authority,
        tuple(results),
        absolute_tolerance,
        relative_tolerance,
    )


def reference(
    name: str,
    family: str,
    dtype: str,
    oracle: str,
    producer: Callable[[], object],
    absolute_tolerance: float = 0.0,
    relative_tolerance: float = 0.0,
) -> Case:
    return Case(
        name,
        family,
        dtype,
        normalize(producer()),
        (oracle,),
        absolute_tolerance,
        relative_tolerance,
    )


def build_cases() -> tuple[dict[str, Case], dict[str, str]]:
    import jax
    import jax.numpy as jnp
    import numpy as np
    import tensorflow as tf
    import torch

    jax.config.update("jax_enable_x64", True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    tf.random.set_seed(SEED)

    generator = np.random.Generator(np.random.PCG64(SEED))
    left = (generator.integers(-12, 13, size=(2, 1, 3)) / 4.0).astype(np.float32)
    right = np.asarray([0.5, -1.5, 2.0, 4.0], dtype=np.float32).reshape(1, 4, 1)
    unary = np.asarray([0.25, 1.0, 4.0, 9.0], dtype=np.float32)
    extremes = np.asarray([1e-30, -1e-30, 1e30, -1e30], dtype=np.float32)
    reduction = np.asarray([-4.0, 1.0, 2.0, 8.0, 3.0, 6.0], dtype=np.float32).reshape(2, 3)
    dot_left = np.asarray([1.0, -2.0, 3.0, 0.5], dtype=np.float32)
    dot_right = np.asarray([4.0, 5.0, -1.0, 8.0], dtype=np.float32)
    matrix_left = np.asarray([1.0, 2.0, 3.0, -1.0, 0.5, 4.0], dtype=np.float32).reshape(2, 3)
    matrix_right = np.asarray([2.0, -1.0, 0.0, 3.0, 1.0, 2.0], dtype=np.float32).reshape(3, 2)
    view_base = np.arange(12, dtype=np.int32).reshape(2, 2, 3)

    cases: dict[str, Case] = {}

    def add_case(case: Case) -> None:
        cases[case.name] = case

    float_frameworks = {
        "numpy": lambda operation: operation(np.asarray(left), np.asarray(right)),
        "pytorch": lambda operation: operation(torch.from_numpy(left), torch.from_numpy(right)).numpy(),
        "tensorflow": lambda operation: operation(tf.constant(left), tf.constant(right)).numpy(),
        "jax": lambda operation: np.asarray(operation(jnp.asarray(left), jnp.asarray(right))),
    }
    for name, operation, absolute, relative in (
        ("broadcast_add", lambda a, b: a + b, 0.0, 0.0),
        ("broadcast_subtract", lambda a, b: a - b, 0.0, 0.0),
        ("broadcast_multiply", lambda a, b: a * b, 0.0, 0.0),
        ("broadcast_divide", lambda a, b: a / b, 1e-6, 1e-6),
    ):
        add_case(consensus(
            name,
            "elementwise",
            "float32",
            {oracle: (lambda run=run, op=operation: run(op)) for oracle, run in float_frameworks.items()},
            absolute,
            relative,
        ))

    unary_operations = (
        ("unary_negate", 0.0, 0.0),
        ("unary_abs", 0.0, 0.0),
        ("unary_exp", 1e-6, 1e-6),
        ("unary_log", 1e-6, 1e-6),
        ("unary_sqrt", 1e-6, 1e-6),
    )
    # Framework namespaces disagree on method availability, so use explicit adapters.
    unary_adapters = {
        "unary_negate": {
            "numpy": lambda: -unary,
            "pytorch": lambda: (-torch.from_numpy(unary)).numpy(),
            "tensorflow": lambda: tf.negative(tf.constant(unary)).numpy(),
            "jax": lambda: np.asarray(-jnp.asarray(unary)),
        },
        "unary_abs": {
            "numpy": lambda: np.abs(unary),
            "pytorch": lambda: torch.abs(torch.from_numpy(unary)).numpy(),
            "tensorflow": lambda: tf.abs(tf.constant(unary)).numpy(),
            "jax": lambda: np.asarray(jnp.abs(jnp.asarray(unary))),
        },
        "unary_exp": {
            "numpy": lambda: np.exp(unary),
            "pytorch": lambda: torch.exp(torch.from_numpy(unary)).numpy(),
            "tensorflow": lambda: tf.exp(tf.constant(unary)).numpy(),
            "jax": lambda: np.asarray(jnp.exp(jnp.asarray(unary))),
        },
        "unary_log": {
            "numpy": lambda: np.log(unary),
            "pytorch": lambda: torch.log(torch.from_numpy(unary)).numpy(),
            "tensorflow": lambda: tf.math.log(tf.constant(unary)).numpy(),
            "jax": lambda: np.asarray(jnp.log(jnp.asarray(unary))),
        },
        "unary_sqrt": {
            "numpy": lambda: np.sqrt(unary),
            "pytorch": lambda: torch.sqrt(torch.from_numpy(unary)).numpy(),
            "tensorflow": lambda: tf.sqrt(tf.constant(unary)).numpy(),
            "jax": lambda: np.asarray(jnp.sqrt(jnp.asarray(unary))),
        },
    }
    for name, absolute, relative in unary_operations:
        add_case(consensus(name, "elementwise", "float32", unary_adapters[name], absolute, relative))

    add_case(consensus("scale_extremes", "elementwise", "float32", {
        "numpy": lambda: extremes * np.float32(2.0),
        "pytorch": lambda: (torch.from_numpy(extremes) * 2.0).numpy(),
        "tensorflow": lambda: (tf.constant(extremes) * tf.constant(2.0, dtype=tf.float32)).numpy(),
        "jax": lambda: np.asarray(jnp.asarray(extremes) * jnp.float32(2.0)),
    }))

    reduction_cases = {
        "sum_axis0": {
            "numpy": lambda: np.sum(reduction, axis=0, dtype=np.float32),
            "pytorch": lambda: torch.sum(torch.from_numpy(reduction), dim=0).numpy(),
            "tensorflow": lambda: tf.reduce_sum(tf.constant(reduction), axis=0).numpy(),
            "jax": lambda: np.asarray(jnp.sum(jnp.asarray(reduction), axis=0)),
        },
        "mean_axis1_keep": {
            "numpy": lambda: np.mean(reduction, axis=1, keepdims=True, dtype=np.float32),
            "pytorch": lambda: torch.mean(torch.from_numpy(reduction), dim=1, keepdim=True).numpy(),
            "tensorflow": lambda: tf.reduce_mean(tf.constant(reduction), axis=1, keepdims=True).numpy(),
            "jax": lambda: np.asarray(jnp.mean(jnp.asarray(reduction), axis=1, keepdims=True)),
        },
        "minimum_axis0": {
            "numpy": lambda: np.min(reduction, axis=0),
            "pytorch": lambda: torch.amin(torch.from_numpy(reduction), dim=0).numpy(),
            "tensorflow": lambda: tf.reduce_min(tf.constant(reduction), axis=0).numpy(),
            "jax": lambda: np.asarray(jnp.min(jnp.asarray(reduction), axis=0)),
        },
        "maximum_axis1": {
            "numpy": lambda: np.max(reduction, axis=1),
            "pytorch": lambda: torch.amax(torch.from_numpy(reduction), dim=1).numpy(),
            "tensorflow": lambda: tf.reduce_max(tf.constant(reduction), axis=1).numpy(),
            "jax": lambda: np.asarray(jnp.max(jnp.asarray(reduction), axis=1)),
        },
    }
    for name, producers in reduction_cases.items():
        add_case(consensus(name, "reduction", "float32", producers, 3e-5, 1e-5))

    add_case(consensus("dot", "linear", "float32", {
        "numpy": lambda: np.dot(dot_left, dot_right),
        "pytorch": lambda: torch.dot(torch.from_numpy(dot_left), torch.from_numpy(dot_right)).numpy(),
        "tensorflow": lambda: tf.tensordot(tf.constant(dot_left), tf.constant(dot_right), axes=1).numpy(),
        "jax": lambda: np.asarray(jnp.dot(jnp.asarray(dot_left), jnp.asarray(dot_right))),
    }, 4e-5, 1e-5))
    add_case(consensus("matmul", "linear", "float32", {
        "numpy": lambda: np.matmul(matrix_left, matrix_right),
        "pytorch": lambda: torch.matmul(torch.from_numpy(matrix_left), torch.from_numpy(matrix_right)).numpy(),
        "tensorflow": lambda: tf.matmul(tf.constant(matrix_left), tf.constant(matrix_right)).numpy(),
        "jax": lambda: np.asarray(jnp.matmul(jnp.asarray(matrix_left), jnp.asarray(matrix_right))),
    }, 3e-5, 1e-5))

    neural_values = np.asarray([-100.0, -1.0, 0.0, 1.0, 100.0], dtype=np.float32)
    nonfinite_values = np.asarray([np.nan, np.inf, -np.inf], dtype=np.float32)
    neural_logits = np.asarray([
        [1000.0, 1001.0, 1002.0],
        [-1000.0, -1001.0, -1002.0],
    ], dtype=np.float32)
    for name, torch_operation, tensorflow_operation in (
        ("relu", torch.relu, tf.nn.relu),
        ("sigmoid", torch.sigmoid, tf.math.sigmoid),
        ("tanh", torch.tanh, tf.math.tanh),
    ):
        add_case(consensus(f"neural_{name}", "neural", "float32", {
            "pytorch": lambda operation=torch_operation: operation(torch.from_numpy(neural_values)).numpy(),
            "tensorflow": lambda operation=tensorflow_operation: operation(tf.constant(neural_values)).numpy(),
        }, 1e-6, 1e-6))
        add_case(consensus(f"neural_{name}_nonfinite", "neural", "float32", {
            "pytorch": lambda operation=torch_operation: operation(torch.from_numpy(nonfinite_values)).numpy(),
            "tensorflow": lambda operation=tensorflow_operation: operation(tf.constant(nonfinite_values)).numpy(),
        }, 1e-6, 1e-6))

    add_case(consensus("neural_softmax", "neural", "float32", {
        "pytorch": lambda: torch.softmax(torch.from_numpy(neural_logits), dim=1).numpy(),
        "tensorflow": lambda: tf.nn.softmax(tf.constant(neural_logits), axis=1).numpy(),
    }, 2e-5, 1e-5))
    add_case(consensus("neural_log_softmax", "neural", "float32", {
        "pytorch": lambda: torch.log_softmax(torch.from_numpy(neural_logits), dim=1).numpy(),
        "tensorflow": lambda: tf.nn.log_softmax(tf.constant(neural_logits), axis=1).numpy(),
    }, 2e-5, 1e-5))

    layer_values = np.asarray([[1.0, 2.0, 3.0], [-4.0, 0.0, 4.0]], dtype=np.float32)
    add_case(consensus("neural_layer_norm", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.layer_norm(torch.from_numpy(layer_values), (3,), eps=1e-5).numpy(),
        "tensorflow": lambda: (
            (tf.constant(layer_values) - tf.reduce_mean(tf.constant(layer_values), axis=1, keepdims=True)) /
            tf.sqrt(
                tf.reduce_mean(
                    tf.square(tf.constant(layer_values) - tf.reduce_mean(tf.constant(layer_values), axis=1, keepdims=True)),
                    axis=1,
                    keepdims=True,
                ) + tf.constant(1e-5, dtype=tf.float32)
            )
        ).numpy(),
    }, 2e-5, 1e-5))

    batched_left = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    batched_right = np.asarray([1.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(1, 3, 2)
    add_case(consensus("neural_batched_matmul", "neural", "float32", {
        "pytorch": lambda: torch.matmul(torch.from_numpy(batched_left), torch.from_numpy(batched_right)).numpy(),
        "tensorflow": lambda: tf.matmul(tf.constant(batched_left), tf.constant(batched_right)).numpy(),
    }, 3e-5, 1e-5))

    convolution_input = np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    convolution_kernel = np.ones((1, 1, 2, 2), dtype=np.float32)
    add_case(consensus("neural_conv2d", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.conv2d(
            torch.from_numpy(convolution_input), torch.from_numpy(convolution_kernel)
        ).numpy(),
        "tensorflow": lambda: np.transpose(tf.nn.conv2d(
            tf.constant(np.transpose(convolution_input, (0, 2, 3, 1))),
            tf.constant(np.transpose(convolution_kernel, (2, 3, 1, 0))),
            strides=1,
            padding="VALID",
        ).numpy(), (0, 3, 1, 2)),
    }, 4e-5, 1e-5))
    add_case(consensus("neural_max_pool2d", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.max_pool2d(torch.from_numpy(convolution_input), 2, 1).numpy(),
        "tensorflow": lambda: np.transpose(tf.nn.max_pool2d(
            tf.constant(np.transpose(convolution_input, (0, 2, 3, 1))), 2, 1, "VALID"
        ).numpy(), (0, 3, 1, 2)),
    }))
    add_case(consensus("neural_average_pool2d", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.avg_pool2d(torch.from_numpy(convolution_input), 2, 1).numpy(),
        "tensorflow": lambda: np.transpose(tf.nn.avg_pool2d(
            tf.constant(np.transpose(convolution_input, (0, 2, 3, 1))), 2, 1, "VALID"
        ).numpy(), (0, 3, 1, 2)),
    }))

    dense_targets = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    class_targets = np.asarray([2, 0], dtype=np.int32)
    mse_predictions = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    mse_targets = np.asarray([0.0, 2.0, 4.0], dtype=np.float32)
    add_case(consensus("neural_mse", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.mse_loss(
            torch.from_numpy(mse_predictions), torch.from_numpy(mse_targets)
        ).numpy(),
        "tensorflow": lambda: tf.reduce_mean(tf.square(
            tf.constant(mse_predictions) - tf.constant(mse_targets)
        )).numpy(),
    }, 1e-6, 1e-6))
    add_case(consensus("neural_cross_entropy_dense", "neural", "float32", {
        "pytorch": lambda: -torch.sum(
            torch.from_numpy(dense_targets) * torch.log_softmax(torch.from_numpy(neural_logits), dim=1), dim=1
        ).mean().numpy(),
        "tensorflow": lambda: tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.constant(dense_targets), logits=tf.constant(neural_logits)
        )).numpy(),
    }, 2e-5, 1e-5))
    add_case(consensus("neural_cross_entropy_classes", "neural", "float32", {
        "pytorch": lambda: torch.nn.functional.cross_entropy(
            torch.from_numpy(neural_logits), torch.from_numpy(class_targets).to(torch.int64)
        ).numpy(),
        "tensorflow": lambda: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.constant(class_targets), logits=tf.constant(neural_logits)
        )).numpy(),
    }, 2e-5, 1e-5))

    autograd_input = np.asarray([
        [-1.0, 2.0, 0.5],
        [3.0, -2.0, 1.0],
    ], dtype=np.float32)
    autograd_bias = np.asarray([0.25, -0.5, 1.0], dtype=np.float32)

    def pytorch_broadcast_gradient(operand: str) -> object:
        input_value = torch.from_numpy(autograd_input).clone().requires_grad_(True)
        bias_value = torch.from_numpy(autograd_bias).clone().requires_grad_(True)
        torch.relu(input_value + bias_value).mean().backward()
        return input_value.grad.numpy() if operand == "input" else bias_value.grad.numpy()

    def tensorflow_broadcast_gradient(operand: str) -> object:
        input_value = tf.Variable(autograd_input)
        bias_value = tf.Variable(autograd_bias)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.relu(input_value + bias_value))
        input_gradient, bias_gradient = tape.gradient(loss, [input_value, bias_value])
        return input_gradient.numpy() if operand == "input" else bias_gradient.numpy()

    for operand in ("input", "bias"):
        add_case(consensus(f"autograd_broadcast_{operand}", "autograd", "float32", {
            "pytorch": lambda operand=operand: pytorch_broadcast_gradient(operand),
            "tensorflow": lambda operand=operand: tensorflow_broadcast_gradient(operand),
        }, 5e-5, 2e-5))

    layer_weights = np.asarray([
        [1.0, -2.0, 3.0],
        [0.5, -1.0, 2.0],
    ], dtype=np.float32)

    def pytorch_layer_norm_gradient() -> object:
        input_value = torch.from_numpy(layer_values).clone().requires_grad_(True)
        normalized = torch.nn.functional.layer_norm(input_value, (3,), eps=1e-5)
        torch.sum(normalized * torch.from_numpy(layer_weights)).backward()
        return input_value.grad.numpy()

    def tensorflow_layer_norm_gradient() -> object:
        input_value = tf.Variable(layer_values)
        weights = tf.constant(layer_weights)
        with tf.GradientTape() as tape:
            mean = tf.reduce_mean(input_value, axis=1, keepdims=True)
            centered = input_value - mean
            normalized = centered / tf.sqrt(
                tf.reduce_mean(tf.square(centered), axis=1, keepdims=True) +
                tf.constant(1e-5, dtype=tf.float32)
            )
            loss = tf.reduce_sum(normalized * weights)
        return tape.gradient(loss, input_value).numpy()

    add_case(consensus("autograd_layer_norm_input", "autograd", "float32", {
        "pytorch": pytorch_layer_norm_gradient,
        "tensorflow": tensorflow_layer_norm_gradient,
    }, 8e-5, 3e-5))

    def pytorch_batched_matmul_gradient(operand: str) -> object:
        left_value = torch.from_numpy(batched_left).clone().requires_grad_(True)
        right_value = torch.from_numpy(batched_right).clone().requires_grad_(True)
        torch.matmul(left_value, right_value).sum().backward()
        return left_value.grad.numpy() if operand == "left" else right_value.grad.numpy()

    def tensorflow_batched_matmul_gradient(operand: str) -> object:
        left_value = tf.Variable(batched_left)
        right_value = tf.Variable(batched_right)
        with tf.GradientTape() as tape:
            loss = tf.reduce_sum(tf.matmul(left_value, right_value))
        left_gradient, right_gradient = tape.gradient(loss, [left_value, right_value])
        return left_gradient.numpy() if operand == "left" else right_gradient.numpy()

    for operand in ("left", "right"):
        add_case(consensus(f"autograd_batched_matmul_{operand}", "autograd", "float32", {
            "pytorch": lambda operand=operand: pytorch_batched_matmul_gradient(operand),
            "tensorflow": lambda operand=operand: tensorflow_batched_matmul_gradient(operand),
        }, 6e-5, 2e-5))

    def pytorch_convolution_gradient(operand: str) -> object:
        input_value = torch.from_numpy(convolution_input).clone().requires_grad_(True)
        kernel_value = torch.from_numpy(convolution_kernel).clone().requires_grad_(True)
        torch.nn.functional.conv2d(input_value, kernel_value).sum().backward()
        return input_value.grad.numpy() if operand == "input" else kernel_value.grad.numpy()

    def tensorflow_convolution_gradient(operand: str) -> object:
        input_value = tf.Variable(convolution_input)
        kernel_value = tf.Variable(convolution_kernel)
        with tf.GradientTape() as tape:
            output = tf.nn.conv2d(
                tf.transpose(input_value, (0, 2, 3, 1)),
                tf.transpose(kernel_value, (2, 3, 1, 0)),
                strides=1,
                padding="VALID",
            )
            loss = tf.reduce_sum(output)
        input_gradient, kernel_gradient = tape.gradient(loss, [input_value, kernel_value])
        return input_gradient.numpy() if operand == "input" else kernel_gradient.numpy()

    for operand in ("input", "kernel"):
        add_case(consensus(f"autograd_conv2d_{operand}", "autograd", "float32", {
            "pytorch": lambda operand=operand: pytorch_convolution_gradient(operand),
            "tensorflow": lambda operand=operand: tensorflow_convolution_gradient(operand),
        }, 7e-5, 2e-5))

    def pytorch_pool_gradient(mode: str) -> object:
        input_value = torch.from_numpy(convolution_input).clone().requires_grad_(True)
        if mode == "max":
            output = torch.nn.functional.max_pool2d(input_value, 2, 1)
        else:
            output = torch.nn.functional.avg_pool2d(input_value, 2, 1)
        output.sum().backward()
        return input_value.grad.numpy()

    def tensorflow_pool_gradient(mode: str) -> object:
        input_value = tf.Variable(convolution_input)
        with tf.GradientTape() as tape:
            channels_last = tf.transpose(input_value, (0, 2, 3, 1))
            if mode == "max":
                output = tf.nn.max_pool2d(channels_last, 2, 1, "VALID")
            else:
                output = tf.nn.avg_pool2d(channels_last, 2, 1, "VALID")
            loss = tf.reduce_sum(output)
        return tape.gradient(loss, input_value).numpy()

    for mode in ("max", "average"):
        add_case(consensus(f"autograd_{mode}_pool2d_input", "autograd", "float32", {
            "pytorch": lambda mode=mode: pytorch_pool_gradient(mode),
            "tensorflow": lambda mode=mode: tensorflow_pool_gradient(mode),
        }, 6e-5, 2e-5))

    def pytorch_cross_entropy_gradient() -> object:
        logits_value = torch.from_numpy(neural_logits).clone().requires_grad_(True)
        torch.nn.functional.cross_entropy(
            logits_value, torch.from_numpy(class_targets).to(torch.int64)
        ).backward()
        return logits_value.grad.numpy()

    def tensorflow_cross_entropy_gradient() -> object:
        logits_value = tf.Variable(neural_logits)
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.constant(class_targets), logits=logits_value
            ))
        return tape.gradient(loss, logits_value).numpy()

    add_case(consensus("autograd_cross_entropy_logits", "autograd", "float32", {
        "pytorch": pytorch_cross_entropy_gradient,
        "tensorflow": tensorflow_cross_entropy_gradient,
    }, 7e-5, 3e-5))

    recurrent_inputs = np.asarray([[0.5, -1.0], [1.5, 0.25]], dtype=np.float32)
    recurrent_initial = np.asarray([[0.1, -0.2]], dtype=np.float32)
    recurrent_input_weights = np.asarray([[0.3, -0.4], [0.2, 0.5]], dtype=np.float32)
    recurrent_hidden_weights = np.asarray([[0.25, 0.1], [-0.3, 0.4]], dtype=np.float32)

    def pytorch_recurrent_gradient() -> object:
        input_weights = torch.from_numpy(recurrent_input_weights).clone().requires_grad_(True)
        hidden_weights = torch.from_numpy(recurrent_hidden_weights)
        hidden = torch.from_numpy(recurrent_initial)
        for step in recurrent_inputs:
            hidden = torch.tanh(torch.from_numpy(step.reshape(1, 2)) @ input_weights + hidden @ hidden_weights)
        hidden.sum().backward()
        return input_weights.grad.numpy()

    def tensorflow_recurrent_gradient() -> object:
        input_weights = tf.Variable(recurrent_input_weights)
        hidden_weights = tf.constant(recurrent_hidden_weights)
        with tf.GradientTape() as tape:
            hidden = tf.constant(recurrent_initial)
            for step in recurrent_inputs:
                hidden = tf.math.tanh(
                    tf.matmul(tf.constant(step.reshape(1, 2)), input_weights) +
                    tf.matmul(hidden, hidden_weights)
                )
            loss = tf.reduce_sum(hidden)
        return tape.gradient(loss, input_weights).numpy()

    add_case(consensus("autograd_recurrent_input_weights", "autograd", "float32", {
        "pytorch": pytorch_recurrent_gradient,
        "tensorflow": tensorflow_recurrent_gradient,
    }, 1e-4, 4e-5))

    def pytorch_sgd_state() -> object:
        parameter = torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.SGD(
            [parameter], lr=0.1, momentum=0.9, weight_decay=0.01
        )
        for gradient in ([0.3, -0.2], [0.3, -0.2]):
            parameter.grad = torch.tensor(gradient, dtype=torch.float32)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        state = optimizer.state[parameter]
        return torch.cat((parameter.detach(), state["momentum_buffer"]))

    def pytorch_adam_state() -> object:
        parameter = torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam(
            [parameter], lr=0.01, betas=(0.8, 0.9), eps=1e-8, weight_decay=0.05
        )
        for gradient in ([0.3, -0.2], [-0.1, 0.4]):
            parameter.grad = torch.tensor(gradient, dtype=torch.float32)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        state = optimizer.state[parameter]
        return torch.cat((
            parameter.detach(), state["exp_avg"], state["exp_avg_sq"],
            state["step"].reshape(1),
        ))

    def pytorch_clipped_gradients() -> object:
        first = torch.tensor([1.0, 1.0], dtype=torch.float32, requires_grad=True)
        second = torch.tensor([1.0], dtype=torch.float32, requires_grad=True)
        first.grad = torch.tensor([3.0, 4.0], dtype=torch.float32)
        second.grad = torch.tensor([12.0], dtype=torch.float32)
        norm = torch.nn.utils.clip_grad_norm_([first, second], 6.5)
        return torch.cat((norm.reshape(1), first.grad, second.grad))

    add_case(reference("optimizer_sgd_state", "optimizer", "float32", "pytorch",
                       pytorch_sgd_state, 2e-6, 2e-6))
    add_case(reference("optimizer_adam_state", "optimizer", "float32", "pytorch",
                       pytorch_adam_state, 3e-6, 3e-6))
    add_case(reference("optimizer_clip_global_norm", "optimizer", "float32", "pytorch",
                       pytorch_clipped_gradients, 2e-6, 2e-6))

    add_case(consensus("permuted_view", "transform", "int32", {
        "numpy": lambda: np.transpose(view_base, (2, 0, 1)),
        "pytorch": lambda: torch.from_numpy(view_base).permute(2, 0, 1).numpy(),
        "tensorflow": lambda: tf.transpose(tf.constant(view_base), perm=(2, 0, 1)).numpy(),
        "jax": lambda: np.asarray(jnp.transpose(jnp.asarray(view_base), (2, 0, 1))),
    }))
    add_case(consensus("selected_view", "transform", "int32", {
        "numpy": lambda: view_base[:, 1, :],
        "pytorch": lambda: torch.from_numpy(view_base).select(1, 1).numpy(),
        "tensorflow": lambda: tf.gather(tf.constant(view_base), 1, axis=1).numpy(),
        "jax": lambda: np.asarray(jnp.take(jnp.asarray(view_base), 1, axis=1)),
    }))
    add_case(consensus("narrowed_view", "transform", "int32", {
        "numpy": lambda: view_base[:, :, 1:3],
        "pytorch": lambda: torch.narrow(torch.from_numpy(view_base), 2, 1, 2).numpy(),
        "tensorflow": lambda: tf.slice(tf.constant(view_base), (0, 0, 1), (2, 2, 2)).numpy(),
        "jax": lambda: np.asarray(jax.lax.slice(jnp.asarray(view_base), (0, 0, 1), (2, 2, 3))),
    }))

    integer_inputs = {
        "int8": np.asarray([6, 8], dtype=np.int8),
        "uint8": np.asarray([6, 8], dtype=np.uint8),
        "int16": np.asarray([6, 8], dtype=np.int16),
        "uint16": np.asarray([6, 8], dtype=np.uint16),
        "int32": np.asarray([6, 8], dtype=np.int32),
        "uint32": np.asarray([6, 8], dtype=np.uint32),
        "int64": np.asarray([6, 8], dtype=np.int64),
        "uint64": np.asarray([6, 8], dtype=np.uint64),
    }
    tf_dtypes = {name: getattr(tf, name) for name in integer_inputs}
    for dtype, values in integer_inputs.items():
        add_case(consensus(f"integer_{dtype}", "integer", dtype, {
            "numpy": lambda values=values: ((values + np.asarray(2, dtype=values.dtype)) * np.asarray(2, dtype=values.dtype)) // np.asarray(4, dtype=values.dtype),
            "tensorflow": lambda values=values, dtype=dtype: tf.math.floordiv(
                (tf.constant(values, dtype=tf_dtypes[dtype]) + tf.cast(2, tf_dtypes[dtype])) * tf.cast(2, tf_dtypes[dtype]),
                tf.cast(4, tf_dtypes[dtype]),
            ).numpy(),
        }))

    signed_zero = np.asarray([0.0, -0.0], dtype=np.float32)
    zero_signs = {
        "numpy": f"min={math.copysign(1.0, float(np.min(signed_zero))):+.0f}, max={math.copysign(1.0, float(np.max(signed_zero))):+.0f}",
        "pytorch": f"min={math.copysign(1.0, float(torch.min(torch.from_numpy(signed_zero)))):+.0f}, max={math.copysign(1.0, float(torch.max(torch.from_numpy(signed_zero)))):+.0f}",
        "tensorflow": f"min={math.copysign(1.0, float(tf.reduce_min(tf.constant(signed_zero)))):+.0f}, max={math.copysign(1.0, float(tf.reduce_max(tf.constant(signed_zero)))):+.0f}",
        "jax": f"min={math.copysign(1.0, float(jnp.min(jnp.asarray(signed_zero)))):+.0f}, max={math.copysign(1.0, float(jnp.max(jnp.asarray(signed_zero)))):+.0f}",
    }
    placement = {
        "tensorflow_devices": ", ".join(device.device_type for device in tf.config.list_physical_devices()) or "none",
        "tensorflow_tensor_device": tf.constant([1.0], dtype=tf.float32).device or "runtime default",
        "signed_zero": "; ".join(f"{name} {value}" for name, value in zero_signs.items()),
    }
    return cases, placement


def sx_float(value: float) -> str:
    if math.isnan(value):
        return "0.0 / 0.0"
    if math.isinf(value):
        return "1.0 / 0.0" if value > 0.0 else "-1.0 / 0.0"
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        return "-0.0"
    rendered = format(value, ".9g")
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered


def sx_values(case: Case) -> str:
    if case.dtype == "float32":
        return ", ".join(sx_float(float(value)) for value in case.result.values)
    return ", ".join(f"{value} as {case.dtype}" for value in case.result.values)


def render_fixture(cases: dict[str, Case], versions: dict[str, str]) -> str:
    oracle_header = ", ".join(f"{name} {version}" for name, version in versions.items())
    values = {name: sx_values(case) for name, case in cases.items()}
    return f'''// Generated by Tools/Oracle/generate.py. Do not edit by hand.
// Command: python Tools/Oracle/generate.py
// Python 3.12.2; seed {SEED}; memory order {ORDER}; reference macOS arm64.
// Oracles: {oracle_header}.

use GFX.GPU
use STD.Math
use Tensor
use Tensor.Autograd
use Tensor.DType
use Tensor.NN
use Tensor.Optim

local func exact_values<T>(actual:T[], expected:T[]) bool {{
    if actual.count() != expected.count() {{ return false }}
    var index = 0
    while index < actual.count() {{
        if actual[index] != expected[index] {{ return false }}
        index++
    }}
    return true
}}

local func same_layout(actual:Tensor, dtype:DType, shape:int[], strides:int[], offset:int, contiguous:bool) bool {{
    return actual.dtype() == dtype && exact_values<int>(actual.shape(), shape) &&
        exact_values<int>(actual.strides(), strides) &&
        actual.offset() == offset && actual.is_contiguous() == contiguous
}}

local func float_value_matches(actual:float, expected:float, absolute:float, relative:float) bool {{
    if Math.is_nan(expected) {{ return Math.is_nan(actual) }}
    if Math.is_infinite(expected) {{ return actual == expected }}
    if expected == 0.0 && actual == 0.0 {{ return Math.sign_bit(actual) == Math.sign_bit(expected) }}
    var tolerance = absolute
    let scaled = relative * Math.abs(expected)
    if scaled > tolerance {{ tolerance = scaled }}
    return Math.abs(actual - expected) <= tolerance
}}

local func float_values_match(actual:float[], expected:float[], absolute:float, relative:float) bool {{
    if actual.count() != expected.count() {{ return false }}
    var index = 0
    while index < actual.count() {{
        if !float_value_matches(actual[index], expected[index], absolute, relative) {{ return false }}
        index++
    }}
    return true
}}

local func required_gradient(variable:Autograd.Variable) Tensor {{
    if let gradient = variable.gradient() {{ return gradient }}
    panic("expected a connected oracle gradient")
}}

local func required_parameter_gradient(parameter:NN.Parameter) Tensor {{
    if let gradient = parameter.gradient() {{ return gradient }}
    panic("expected a connected parameter gradient")
}}

test "oracle comparison harness rejects deliberate mutations" {{
    var shape:int[] = [2, 3]
    var wrong_shape:int[] = [3, 2]
    var strides:int[] = [3, 1]
    var wrong_strides:int[] = [1, 3]
    var exact:int32[] = [1 as int32, 2 as int32]
    var wrong_value:int32[] = [1 as int32, 3 as int32]
    var expected:float[] = [1.0]
    var close:float[] = [1.0001]
    assert(!exact_values<int>(shape, wrong_shape))
    assert(!exact_values<int>(strides, wrong_strides))
    assert(!exact_values<int32>(exact, wrong_value))
    assert(float_values_match(close, expected, 0.001, 0.0))
    assert(!float_values_match(close, expected, 0.00001, 0.0))
}}

test "match multi-oracle float32 elementwise fixtures" {{
    // exact: NumPy, PyTorch, TensorFlow, and JAX agree after float32 normalization.
    var left_values:float[] = {sx_array([float(value) for value in left_values_from_cases(cases)])}
    var right_values:float[] = [0.5, -1.5, 2.0, 4.0]
    let left = Tensor(left_values, [2, 1, 3])
    let right = Tensor(right_values, [1, 4, 1])
    var expected_add:float[] = [{values['broadcast_add']}]
    var expected_subtract:float[] = [{values['broadcast_subtract']}]
    var expected_multiply:float[] = [{values['broadcast_multiply']}]
    var expected_divide:float[] = [{values['broadcast_divide']}]
    assert(same_layout(left.add(right), DType.float32(), [2, 4, 3], [12, 3, 1], 0, true))
    assert(float_values_match(left.add(right).values(), expected_add, 0.0, 0.0))
    assert(float_values_match(left.subtract(right).values(), expected_subtract, 0.0, 0.0))
    assert(float_values_match(left.multiply(right).values(), expected_multiply, 0.0, 0.0))
    assert(float_values_match(left.divide(right).values(), expected_divide, 0.000001, 0.000001))
    assert(float_values_match(left.add(2.0).subtract(2.0).values(), left_values, 0.0, 0.0))

    var unary_values:float[] = [0.25, 1.0, 4.0, 9.0]
    let unary = Tensor.vector(unary_values)
    var expected_negate:float[] = [{values['unary_negate']}]
    var expected_abs:float[] = [{values['unary_abs']}]
    var expected_exp:float[] = [{values['unary_exp']}]
    var expected_log:float[] = [{values['unary_log']}]
    var expected_sqrt:float[] = [{values['unary_sqrt']}]
    assert(float_values_match(unary.negate().values(), expected_negate, 0.0, 0.0))
    assert(float_values_match(unary.abs().values(), expected_abs, 0.0, 0.0))
    assert(float_values_match(unary.exp().values(), expected_exp, 0.000001, 0.000001))
    assert(float_values_match(unary.log().values(), expected_log, 0.000001, 0.000001))
    assert(float_values_match(unary.sqrt().values(), expected_sqrt, 0.000001, 0.000001))

    var extreme_values:float[] = [1e-30, -1e-30, 1e30, -1e30]
    var expected_extremes:float[] = [{values['scale_extremes']}]
    assert(float_values_match(Tensor.vector(extreme_values).multiply(2.0).values(), expected_extremes, 0.0, 0.0))

    let infinity = 1.0 / 0.0
    let negative_zero = -1.0 / infinity
    var special:float[] = [1.0, -1.0, 0.0, negative_zero]
    let divided = Tensor.vector(special).divide(0.0).values()
    assert(Math.is_infinite(divided[0]) && Math.is_infinite(divided[1]))
    assert(Math.is_nan(divided[2]) && Math.is_nan(divided[3]))
}}

test "match functional shape and view fixtures" {{
    // exact: all four oracles agree on logical C-order values.
    var values:int32[] = [
        0 as int32, 1 as int32, 2 as int32, 3 as int32, 4 as int32, 5 as int32,
        6 as int32, 7 as int32, 8 as int32, 9 as int32, 10 as int32, 11 as int32,
    ]
    let source = Tensor(values, [2, 2, 3])
    let reshaped = source.reshape([3, -1])
    let flattened = reshaped.flatten()
    let permuted = source.permute([2, 0, 1])
    let transposed = Tensor.matrix(values, 4, 3).transpose()
    let selected = source.select(1, 1)
    let narrowed = source.narrow(2, 1, 2)
    let compact = narrowed.contiguous()
    var expected_permuted:int32[] = [{values['permuted_view']}]
    var expected_selected:int32[] = [{values['selected_view']}]
    var expected_narrowed:int32[] = [{values['narrowed_view']}]
    assert(same_layout(source, DType.int32(), [2, 2, 3], [6, 3, 1], 0, true))
    assert(same_layout(reshaped, DType.int32(), [3, 4], [4, 1], 0, true))
    assert(flattened.rank() == 1 && flattened.count() == 12 && !flattened.is_empty())
    assert(exact_values<int32>(permuted.int32_values(), expected_permuted))
    assert(same_layout(permuted, DType.int32(), [3, 2, 2], [1, 6, 3], 0, false))
    assert(transposed.int32_at([2, 3]) == 11 as int32)
    assert(exact_values<int32>(selected.int32_values(), expected_selected))
    assert(exact_values<int32>(narrowed.int32_values(), expected_narrowed))
    assert(same_layout(compact, DType.int32(), [2, 2, 2], [4, 2, 1], 0, true))
    assert(source.select(0, 1).select(0, 1).select(0, 2).int32_item() == 11 as int32)
    assert(Tensor.zeros([0, 3]).is_empty())
}}

test "match construction extraction and exact casts for all dtypes" {{
    var f:float[] = [1.0, -2.0]
    var i8:int8[] = [-127 as int8, 127 as int8]
    var u8:uint8[] = [0 as uint8, 255 as uint8]
    var i16:int16[] = [-32767 as int16, 32767 as int16]
    var u16:uint16[] = [0 as uint16, 65535 as uint16]
    var i32:int32[] = [-2147483647 as int32, 2147483647 as int32]
    var u32:uint32[] = [0 as uint32, 0xffff_ffff]
    var i64:int64[] = [-9223372036854775807 as int64, 9223372036854775807 as int64]
    var u64:uint64[] = [0 as uint64, 0xffff_ffff_ffff_ffff]
    assert(Tensor.vector(f).dtype() == DType.float32() && Tensor.vector(f).at([1]) == -2.0)
    assert(Tensor.vector(i8).dtype() == DType.int8() && exact_values<int8>(Tensor.vector(i8).int8_values(), i8))
    assert(Tensor.vector(u8).dtype() == DType.uint8() && exact_values<uint8>(Tensor.vector(u8).uint8_values(), u8))
    assert(Tensor.vector(i16).dtype() == DType.int16() && Tensor.vector(i16).int16_at([1]) == 32767 as int16)
    assert(Tensor.vector(u16).dtype() == DType.uint16() && Tensor.vector(u16).uint16_at([1]) == 65535 as uint16)
    assert(Tensor.vector(i32).dtype() == DType.int32() && Tensor.vector(i32).int32_at([1]) == 2147483647 as int32)
    assert(Tensor.vector(u32).dtype() == DType.uint32() && Tensor.vector(u32).uint32_at([1]) == 0xffff_ffff)
    assert(Tensor.vector(i64).dtype() == DType.int64() && Tensor.vector(i64).int64_at([1]) == 9223372036854775807 as int64)
    assert(Tensor.vector(u64).dtype() == DType.uint64() && Tensor.vector(u64).uint64_at([1]) == 0xffff_ffff_ffff_ffff)
    assert(Tensor.scalar(1 as int8).cast(DType.float32()).item() == 1.0)
    assert(Tensor.scalar(1 as int8).cast(DType.uint8()).uint8_item() == 1 as uint8)
    assert(Tensor.scalar(1 as int8).cast(DType.int16()).int16_item() == 1 as int16)
    assert(Tensor.scalar(1 as int8).cast(DType.uint16()).uint16_item() == 1 as uint16)
    assert(Tensor.scalar(1 as int8).cast(DType.int32()).int32_item() == 1 as int32)
    assert(Tensor.scalar(1 as int8).cast(DType.uint32()).uint32_item() == 1 as uint32)
    assert(Tensor.scalar(1 as int8).cast(DType.int64()).int64_item() == 1 as int64)
    assert(Tensor.scalar(1 as int8).cast(DType.uint64()).uint64_item() == 1 as uint64)
    assert(Tensor.ones([1], DType.int8()).int8_item() == 1 as int8)
    assert(Tensor.full([1], 7 as uint16).uint16_item() == 7 as uint16)
}}

test "match exact integer arithmetic fixtures for every integer dtype" {{
    var i8:int8[] = [6 as int8, 8 as int8]
    var u8:uint8[] = [6 as uint8, 8 as uint8]
    var i16:int16[] = [6 as int16, 8 as int16]
    var u16:uint16[] = [6 as uint16, 8 as uint16]
    var i32:int32[] = [6 as int32, 8 as int32]
    var u32:uint32[] = [6 as uint32, 8 as uint32]
    var i64:int64[] = [6 as int64, 8 as int64]
    var u64:uint64[] = [6 as uint64, 8 as uint64]
    var expected_i8:int8[] = [{values['integer_int8']}]
    var expected_u8:uint8[] = [{values['integer_uint8']}]
    var expected_i16:int16[] = [{values['integer_int16']}]
    var expected_u16:uint16[] = [{values['integer_uint16']}]
    var expected_i32:int32[] = [{values['integer_int32']}]
    var expected_u32:uint32[] = [{values['integer_uint32']}]
    var expected_i64:int64[] = [{values['integer_int64']}]
    var expected_u64:uint64[] = [{values['integer_uint64']}]
    assert(exact_values<int8>(Tensor.vector(i8).add(2 as int8).multiply(2 as int8).divide(4 as int8).int8_values(), expected_i8))
    assert(exact_values<uint8>(Tensor.vector(u8).add(2 as uint8).multiply(2 as uint8).divide(4 as uint8).uint8_values(), expected_u8))
    assert(exact_values<int16>(Tensor.vector(i16).add(2 as int16).multiply(2 as int16).divide(4 as int16).int16_values(), expected_i16))
    assert(exact_values<uint16>(Tensor.vector(u16).add(2 as uint16).multiply(2 as uint16).divide(4 as uint16).uint16_values(), expected_u16))
    assert(exact_values<int32>(Tensor.vector(i32).add(2 as int32).multiply(2 as int32).divide(4 as int32).int32_values(), expected_i32))
    assert(exact_values<uint32>(Tensor.vector(u32).add(2 as uint32).multiply(2 as uint32).divide(4 as uint32).uint32_values(), expected_u32))
    assert(exact_values<int64>(Tensor.vector(i64).add(2 as int64).multiply(2 as int64).divide(4 as int64).int64_values(), expected_i64))
    assert(exact_values<uint64>(Tensor.vector(u64).add(2 as uint64).multiply(2 as uint64).divide(4 as uint64).uint64_values(), expected_u64))
    assert(Tensor.scalar(-8 as int32).negate().abs().int32_item() == 8 as int32)
    assert(Tensor.scalar(8 as uint64).abs().uint64_item() == 8 as uint64)
}}

test "match multi-oracle reductions and linear algebra" {{
    var source_values:float[] = [-4.0, 1.0, 2.0, 8.0, 3.0, 6.0]
    let source = Tensor(source_values, [2, 3])
    var expected_sum:float[] = [{values['sum_axis0']}]
    var expected_mean:float[] = [{values['mean_axis1_keep']}]
    var expected_min:float[] = [{values['minimum_axis0']}]
    var expected_max:float[] = [{values['maximum_axis1']}]
    assert(float_values_match(source.sum([0]).values(), expected_sum, 0.00003, 0.00001))
    assert(float_values_match(source.mean([1], true).values(), expected_mean, 0.00003, 0.00001))
    assert(float_values_match(source.min([0]).values(), expected_min, 0.00003, 0.00001))
    assert(float_values_match(source.max([1]).values(), expected_max, 0.00003, 0.00001))
    assert(source.sum([]).values()[0] == -4.0)

    var dot_left:float[] = [1.0, -2.0, 3.0, 0.5]
    var dot_right:float[] = [4.0, 5.0, -1.0, 8.0]
    var expected_dot:float[] = [{values['dot']}]
    assert(float_values_match(Tensor.vector(dot_left).dot(Tensor.vector(dot_right)).values(), expected_dot, 0.00004, 0.00001))
    var matrix_left:float[] = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0]
    var matrix_right:float[] = [2.0, -1.0, 0.0, 3.0, 1.0, 2.0]
    var expected_matmul:float[] = [{values['matmul']}]
    let product = Tensor(matrix_left, [2, 3]).matmul(Tensor(matrix_right, [3, 2]))
    assert(float_values_match(product.values(), expected_matmul, 0.00003, 0.00001))

    var zero_values:float[] = [0.0, -0.0]
    assert(Math.sign_bit(Tensor.vector(zero_values).min().item()))
    assert(!Math.sign_bit(Tensor.vector(zero_values).max().item()))
    var nan_values:float[] = [1.0, 0.0 / 0.0]
    assert(Math.is_nan(Tensor.vector(nan_values).sum().item()))
}}

test "match PyTorch and TensorFlow neural fixtures" {{
    var activation_values:float[] = [-100.0, -1.0, 0.0, 1.0, 100.0]
    let activations = Tensor.vector(activation_values)
    var expected_relu:float[] = [{values['neural_relu']}]
    var expected_sigmoid:float[] = [{values['neural_sigmoid']}]
    var expected_tanh:float[] = [{values['neural_tanh']}]
    assert(float_values_match(activations.relu().values(), expected_relu, 0.000001, 0.000001))
    assert(float_values_match(activations.sigmoid().values(), expected_sigmoid, 0.000001, 0.000001))
    assert(float_values_match(activations.tanh().values(), expected_tanh, 0.000001, 0.000001))

    var nonfinite_values:float[] = [0.0 / 0.0, 1.0 / 0.0, -1.0 / 0.0]
    let nonfinite = Tensor.vector(nonfinite_values)
    var expected_relu_nonfinite:float[] = [{values['neural_relu_nonfinite']}]
    var expected_sigmoid_nonfinite:float[] = [{values['neural_sigmoid_nonfinite']}]
    var expected_tanh_nonfinite:float[] = [{values['neural_tanh_nonfinite']}]
    assert(float_values_match(nonfinite.relu().values(), expected_relu_nonfinite, 0.000001, 0.000001))
    assert(float_values_match(nonfinite.sigmoid().values(), expected_sigmoid_nonfinite, 0.000001, 0.000001))
    assert(float_values_match(nonfinite.tanh().values(), expected_tanh_nonfinite, 0.000001, 0.000001))

    var logits_values:float[] = [1000.0, 1001.0, 1002.0, -1000.0, -1001.0, -1002.0]
    let logits = Tensor(logits_values, [2, 3])
    var expected_softmax:float[] = [{values['neural_softmax']}]
    var expected_log_softmax:float[] = [{values['neural_log_softmax']}]
    assert(float_values_match(logits.softmax(1).values(), expected_softmax, 0.00002, 0.00001))
    assert(float_values_match(logits.log_softmax(1).values(), expected_log_softmax, 0.00002, 0.00001))

    var layer_values:float[] = [1.0, 2.0, 3.0, -4.0, 0.0, 4.0]
    var expected_layer_norm:float[] = [{values['neural_layer_norm']}]
    assert(float_values_match(Tensor(layer_values, [2, 3]).layer_norm().values(), expected_layer_norm, 0.00002, 0.00001))

    var batch_left:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    var batch_right:float[] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    var expected_batch:float[] = [{values['neural_batched_matmul']}]
    assert(float_values_match(
        Tensor(batch_left, [2, 2, 3]).matmul(Tensor(batch_right, [1, 3, 2])).values(),
        expected_batch, 0.00003, 0.00001
    ))

    var image_values:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    let image = Tensor(image_values, [1, 1, 3, 3])
    var expected_convolution:float[] = [{values['neural_conv2d']}]
    var expected_max_pool:float[] = [{values['neural_max_pool2d']}]
    var expected_average_pool:float[] = [{values['neural_average_pool2d']}]
    assert(float_values_match(image.conv2d(Tensor.ones([1, 1, 2, 2])).values(), expected_convolution, 0.00004, 0.00001))
    assert(float_values_match(image.max_pool2d(2, 1).values(), expected_max_pool, 0.0, 0.0))
    assert(float_values_match(image.average_pool2d(2, 1).values(), expected_average_pool, 0.0, 0.0))

    var dense_targets:float[] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    var class_targets:int32[] = [2 as int32, 0 as int32]
    var mse_predictions:float[] = [1.0, 2.0, 3.0]
    var mse_targets:float[] = [0.0, 2.0, 4.0]
    var expected_mse:float[] = [{values['neural_mse']}]
    var expected_dense_loss:float[] = [{values['neural_cross_entropy_dense']}]
    var expected_class_loss:float[] = [{values['neural_cross_entropy_classes']}]
    assert(float_values_match(Tensor.vector(mse_predictions).mse_loss(Tensor.vector(mse_targets)).values(), expected_mse, 0.000001, 0.000001))
    assert(float_values_match(logits.cross_entropy(Tensor(dense_targets, [2, 3]), 1).values(), expected_dense_loss, 0.00002, 0.00001))
    assert(float_values_match(logits.cross_entropy(Tensor.vector(class_targets), 1).values(), expected_class_loss, 0.00002, 0.00001))
}}

test "match PyTorch and TensorFlow eager gradients" {{
    var input_values:float[] = [-1.0, 2.0, 0.5, 3.0, -2.0, 1.0]
    var bias_values:float[] = [0.25, -0.5, 1.0]
    var input = Autograd.Variable(Tensor(input_values, [2, 3]))
    var bias = Autograd.Variable(Tensor.vector(bias_values))
    input.value().add(bias.value()).relu().mean().backward()
    var expected_input:float[] = [{values['autograd_broadcast_input']}]
    var expected_bias:float[] = [{values['autograd_broadcast_bias']}]
    assert(float_values_match(required_gradient(input).values(), expected_input, 0.00005, 0.00002))
    assert(float_values_match(required_gradient(bias).values(), expected_bias, 0.00005, 0.00002))

    var layer_values:float[] = [1.0, 2.0, 3.0, -4.0, 0.0, 4.0]
    var layer_weights:float[] = [1.0, -2.0, 3.0, 0.5, -1.0, 2.0]
    var layer = Autograd.Variable(Tensor(layer_values, [2, 3]))
    layer.value().layer_norm().multiply(Tensor(layer_weights, [2, 3])).sum().backward()
    var expected_layer:float[] = [{values['autograd_layer_norm_input']}]
    assert(float_values_match(required_gradient(layer).values(), expected_layer, 0.00008, 0.00003))

    var batch_left_values:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    var batch_right_values:float[] = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    var batch_left = Autograd.Variable(Tensor(batch_left_values, [2, 2, 3]))
    var batch_right = Autograd.Variable(Tensor(batch_right_values, [1, 3, 2]))
    batch_left.value().matmul(batch_right.value()).sum().backward()
    var expected_batch_left:float[] = [{values['autograd_batched_matmul_left']}]
    var expected_batch_right:float[] = [{values['autograd_batched_matmul_right']}]
    assert(float_values_match(required_gradient(batch_left).values(), expected_batch_left, 0.00006, 0.00002))
    assert(float_values_match(required_gradient(batch_right).values(), expected_batch_right, 0.00006, 0.00002))

    var image_values:float[] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    var kernel_values:float[] = [1.0, 1.0, 1.0, 1.0]
    var image = Autograd.Variable(Tensor(image_values, [1, 1, 3, 3]))
    var kernel = Autograd.Variable(Tensor(kernel_values, [1, 1, 2, 2]))
    image.value().conv2d(kernel.value()).sum().backward()
    var expected_image:float[] = [{values['autograd_conv2d_input']}]
    var expected_kernel:float[] = [{values['autograd_conv2d_kernel']}]
    assert(float_values_match(required_gradient(image).values(), expected_image, 0.00007, 0.00002))
    assert(float_values_match(required_gradient(kernel).values(), expected_kernel, 0.00007, 0.00002))

    var maximum = Autograd.Variable(Tensor(image_values, [1, 1, 3, 3]))
    maximum.value().max_pool2d(2, 1).sum().backward()
    var average = Autograd.Variable(Tensor(image_values, [1, 1, 3, 3]))
    average.value().average_pool2d(2, 1).sum().backward()
    var expected_maximum:float[] = [{values['autograd_max_pool2d_input']}]
    var expected_average:float[] = [{values['autograd_average_pool2d_input']}]
    assert(float_values_match(required_gradient(maximum).values(), expected_maximum, 0.00006, 0.00002))
    assert(float_values_match(required_gradient(average).values(), expected_average, 0.00006, 0.00002))

    var logits_values:float[] = [1000.0, 1001.0, 1002.0, -1000.0, -1001.0, -1002.0]
    var class_targets:int32[] = [2 as int32, 0 as int32]
    var logits = Autograd.Variable(Tensor(logits_values, [2, 3]))
    logits.value().cross_entropy(Tensor.vector(class_targets), 1).backward()
    var expected_logits:float[] = [{values['autograd_cross_entropy_logits']}]
    assert(float_values_match(required_gradient(logits).values(), expected_logits, 0.00007, 0.00003))

    var recurrent_weight_values:float[] = [0.3, -0.4, 0.2, 0.5]
    var recurrent_hidden_values:float[] = [0.25, 0.1, -0.3, 0.4]
    var recurrent_input_values:float[] = [0.5, -1.0, 1.5, 0.25]
    var recurrent_initial_values:float[] = [0.1, -0.2]
    var recurrent_weights = Autograd.Variable(Tensor(recurrent_weight_values, [2, 2]))
    let hidden_weights = Tensor(recurrent_hidden_values, [2, 2])
    let recurrent_inputs = Tensor(recurrent_input_values, [2, 2])
    var hidden = Tensor(recurrent_initial_values, [1, 2])
    hidden = recurrent_inputs.select(0, 0).reshape([1, 2]).matmul(recurrent_weights.value()).add(hidden.matmul(hidden_weights)).tanh()
    hidden = recurrent_inputs.select(0, 1).reshape([1, 2]).matmul(recurrent_weights.value()).add(hidden.matmul(hidden_weights)).tanh()
    hidden.sum().backward()
    var expected_recurrent:float[] = [{values['autograd_recurrent_input_weights']}]
    assert(float_values_match(required_gradient(recurrent_weights).values(), expected_recurrent, 0.0001, 0.00004))
}}

test "match PyTorch optimizer values state and global clipping" {{
    var initial:float[] = [1.0, -2.0]
    var repeated_gradient:float[] = [0.3, -0.2]
    var sgd_parameters:NN.Parameter[] = [NN.Parameter("weight", Tensor.vector(initial))]
    var sgd = Optim.SGD(sgd_parameters, 0.1, momentum:0.9, weight_decay:0.01)
    let fixed = Tensor.vector(repeated_gradient)
    sgd_parameters[0].value().multiply(fixed).sum().backward()
    sgd.step()
    sgd.zero_grad()
    sgd_parameters[0].value().multiply(fixed).sum().backward()
    sgd.step()
    var expected_sgd:float[] = [{values['optimizer_sgd_state']}]
    var expected_sgd_value:float[] = [expected_sgd[0], expected_sgd[1]]
    assert(float_values_match(sgd_parameters[0].value().detach().values(), expected_sgd_value, 0.000002, 0.000002))

    var first_gradient:float[] = [0.3, -0.2]
    var second_gradient:float[] = [-0.1, 0.4]
    var adam_parameters:NN.Parameter[] = [NN.Parameter("weight", Tensor.vector(initial))]
    var adam = Optim.Adam(adam_parameters, learning_rate:0.01, beta1:0.8, beta2:0.9, weight_decay:0.05)
    adam_parameters[0].value().multiply(Tensor.vector(first_gradient)).sum().backward()
    adam.step()
    adam.zero_grad()
    adam_parameters[0].value().multiply(Tensor.vector(second_gradient)).sum().backward()
    adam.step()
    var expected_adam:float[] = [{values['optimizer_adam_state']}]
    var expected_adam_value:float[] = [expected_adam[0], expected_adam[1]]
    assert(float_values_match(adam_parameters[0].value().detach().values(), expected_adam_value, 0.000003, 0.000003))

    var first_values:float[] = [1.0, 1.0]
    var second_values:float[] = [1.0]
    var clip_first:float[] = [3.0, 4.0]
    var clip_second:float[] = [12.0]
    var clipped:NN.Parameter[] = [
        NN.Parameter("first", Tensor.vector(first_values)),
        NN.Parameter("second", Tensor.vector(second_values))
    ]
    var clipper = Optim.SGD(clipped, 0.1)
    clipped[0].value().multiply(Tensor.vector(clip_first)).sum().backward()
    clipped[1].value().multiply(Tensor.vector(clip_second)).sum().backward()
    let norm = clipper.clip_grad_norm(6.5)
    var expected_clip:float[] = [{values['optimizer_clip_global_norm']}]
    var expected_first:float[] = [expected_clip[1], expected_clip[2]]
    var expected_second:float[] = [expected_clip[3]]
    assert(float_value_matches(norm.item(), expected_clip[0], 0.000002, 0.000002))
    assert(float_values_match(required_parameter_gradient(clipped[0]).values(), expected_first, 0.000002, 0.000002))
    assert(float_values_match(required_parameter_gradient(clipped[1]).values(), expected_second, 0.000002, 0.000002))
}}

test "run the float32 oracle fixture on the available GPU" {{
    if !GPU.Device.is_supported() {{ return }}
    var device = GPU.Device(GPU.DeviceSettings(debug:true))
    device.reset_command_stats()
    var left_values:float[] = {sx_array([float(value) for value in left_values_from_cases(cases)])}
    var right_values:float[] = [0.5, -1.5, 2.0, 4.0]
    var expected:float[] = [{values['broadcast_add']}]
    let left = Tensor(left_values, [2, 1, 3]).to(device)
    let same = left.to(device)
    let right = Tensor(right_values, [1, 4, 1]).to(device)
    let result = same.add(right)
    let before = device.command_stats()
    assert(result.is_gpu() && before.uploads == 2 && before.downloads == 0 && before.compute_passes == 1)
    assert(float_values_match(result.cpu().values(), expected, 0.000001, 0.000001))
}}
'''


def left_values_from_cases(cases: dict[str, Case]) -> tuple[float, ...]:
    # Recover the broadcast input from output differences: right[0] is 0.5.
    result = cases["broadcast_add"].result.values
    return tuple(float(result[index]) - 0.5 for index in (0, 1, 2, 12, 13, 14))


def sx_array(values: list[float]) -> str:
    return "[" + ", ".join(sx_float(value) for value in values) + "]"


def render_report(cases: dict[str, Case], versions: dict[str, str], placement: dict[str, str]) -> str:
    lines = [
        "# Differential oracle report",
        "",
        "This file is generated by `Tools/Oracle/generate.py`; review its diff before accepting a contract change.",
        "",
        "## Provenance",
        "",
        f"- Command: `python Tools/Oracle/generate.py`",
        f"- Seed: `{SEED}`",
        f"- Memory order: `{ORDER}` (row-major)",
        "- Reference platform: macOS arm64",
        f"- Python: `3.12.2`",
    ]
    lines.extend(f"- {name}: `{version}`" for name, version in versions.items())
    lines.extend([
        "",
        "## Generated cases",
        "",
        "| Case | Family | Dtype | Classification | Oracles | Tolerance |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for case in cases.values():
        classification = "tolerated" if case.absolute_tolerance or case.relative_tolerance else "exact"
        tolerance = (
            f"abs {case.absolute_tolerance:g}, rel {case.relative_tolerance:g}"
            if classification == "tolerated" else "exact"
        )
        lines.append(
            f"| `{case.name}` | {case.family} | `{case.dtype}` | {classification} | "
            f"{', '.join(case.oracles)} | {tolerance} |"
        )
    lines.extend([
        "",
        "## Public surface coverage",
        "",
        "| Surface | Successful corpus | Failure corpus |",
        "| --- | --- | --- |",
        "| `DType` values and queries | `Core.sx`, all nine dtypes | closed family, no custom dtype |",
        "| constructors, filled values, typed extraction, `cast` | `Core.sx`, `OracleDifferential.sx` | extraction mismatch, cardinality, byte size, cast range |",
        "| shape, strides, offset, ranks 0-5, empty dimensions | `Views.sx`, `OracleDifferential.sx` | negative/overflowing shapes and indices |",
        "| `reshape`, `flatten`, `permute`, `transpose`, `select`, `narrow`, `contiguous` | `Views.sx`, `GPUViews.sx`, `OracleDifferential.sx` | every axis/rank/range invariant in `Diagnostics/` |",
        "| `add`, `subtract`, `multiply`, `divide`, scalar overloads and broadcasting | `Elementwise.sx`, `GPUCompute.sx`, `OracleDifferential.sx` | shape, dtype, placement, divide-by-zero, overflow/underflow |",
        "| `negate`, `abs`, `exp`, `log`, `sqrt` | `Elementwise.sx`, `GPUCompute.sx`, `OracleDifferential.sx` | unsigned negate, float-only operations, checked signed minimum |",
        "| `sum`, `mean`, `min`, `max`, axes and retained dimensions | `Reductions.sx`, `GPULinearReduction.sx`, `OracleDifferential.sx` | axes, empty domains, integer mean/overflow, GPU integer compute |",
        "| `dot`, `matmul` | `LinearAlgebra.sx`, `GPULinearReduction.sx`, `OracleDifferential.sx` | ranks, shapes, dtype/placement and multiply-accumulate overflow |",
        "| `stack`, `concatenate`, `gather`, batched `matmul` | `Neural.sx`, `NeuralGPU.sx`, `OracleDifferential.sx` | axes, shapes, dtype and placement |",
        "| activations, softmax, layer norm and losses | `Neural.sx`, `NeuralGPU.sx`, `OracleDifferential.sx` | axes, dtype and target shapes |",
        "| NCHW/OIHW convolution and pooling | `Neural.sx`, `NeuralGPU.sx`, `OracleDifferential.sx` | channels, ranks, stride, padding and kernel shape |",
        "| seeded initialization and dropout | `Neural.sx`, `NeuralGPU.sx` | bounds, fan sizes, probability, seed and iteration |",
        "| eager reverse-mode autodifferentiation | `Autograd.sx`, `AutogradGPU.sx`, `OracleDifferential.sx` | dtype, seed, consumed graph, transfer and disconnected graph |",
        "| named parameters, SGD, Adam and global gradient clipping | `Optimizers.sx`, `OptimizerState.sx`, `OptimizersGPU.sx`, `OracleDifferential.sx` | duplicate names, hyperparameters, non-finite gradients and initialized-state placement |",
        "| `to`, `cpu`, CPU/GPU placement and resource lifetime | `GPU.sx`, `GPUViews.sx`, `GPUCompute.sx`, `GPUStress.sx`, `OracleDifferential.sx` | extraction on GPU, cross-device use and integer GPU compute |",
        "",
        "The successful suite includes targeted scalar, singleton, zero-sized, broadcast, strided-view, axis, signed-zero, NaN, infinity, integer-extrema, exact-conversion, large/small-amplitude, and bounded seeded pseudo-random cases. `Tests/Consumer/Diagnostics/README.md` indexes division-by-zero, overflow, out-of-range conversion, and structural errors.",
        "",
        "## Dtype execution matrix",
        "",
        "| Dtype | Exact | Tolerated | Transfer-only | Expected error | Justified skip |",
        "| --- | --- | --- | --- | --- | --- |",
        "| `float32` | construction, views, ordinary elementwise | division, transcendental, reduction, linear CPU/GPU | none | shape, axis, dtype, placement | GPU fixture only when no local provider exists |",
        "| `int8` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, divide by zero, GPU compute | none |",
        "| `uint8` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, underflow, divide by zero, GPU compute | none |",
        "| `int16` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, divide by zero, GPU compute | none |",
        "| `uint16` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, underflow, divide by zero, GPU compute | none |",
        "| `int32` | construction, four-oracle views, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, divide by zero, GPU compute | none |",
        "| `uint32` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, underflow, divide by zero, GPU compute | none |",
        "| `int64` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, divide by zero, GPU compute | none |",
        "| `uint64` | construction, transform, CPU arithmetic, GPU bits | none | GPU residency and roundtrip | overflow, underflow, divide by zero, GPU compute | none |",
        "",
        "High unsigned construction and transformation are independent from arithmetic and backend support. PyTorch is therefore not treated as universal authority for `uint16`, `uint32`, or `uint64`; NumPy and TensorFlow must agree on the bounded arithmetic fixtures, while Tensor GPU remains transfer-only.",
        "",
        "## Deliberate contract decisions",
        "",
        f"- Signed-zero probe: `{placement['signed_zero']}`. Tensor fixes `min(+0,-0)` to `-0` and `max(+0,-0)` to `+0`; this is tested as an explicit contract rather than selected from one convenient oracle.",
        "- Integer overflow, unsigned underflow, integer division by zero, invalid axes/shapes, incompatible dtype/placement, out-of-range casts, invalid neural targets, and random-parameter errors are `expected-error` cases. The executable sources and required diagnostics are indexed in `Tests/Consumer/Diagnostics/README.md`.",
        "- Framework integer promotion and wraparound are not imported into Tensor. Tensor retains its dtype and checks representability before returning a result.",
        "",
        "## TensorFlow placement comparison",
        "",
        f"- Physical devices observed: `{placement['tensorflow_devices']}`.",
        f"- Reference tensor placement: `{placement['tensorflow_tensor_device']}`.",
        "- TensorFlow may place or copy operands automatically. Tensor intentionally requires explicit `to(device)` and `cpu()` boundaries.",
        "- Tensor's nine-dtype GPU roundtrip is checked bit-for-bit by `Tests/Consumer/Tests/GPU.sx`. The resident float32 fixture checks device order, a same-device `to` no-op, no download before observation, and exactly one explicit readback.",
        "",
        "## Hermetic consumption",
        "",
        "The normal suite reads only `Tests/Consumer/Tests/OracleDifferential.sx`. It imports no Python module, performs no download, and requires no network. The Tensor package manifest contains only its runtime Silex dependencies. A missing local GPU is the sole justified runtime skip; CPU fixtures always run.",
        "",
    ])
    return "\n".join(lines)


def write_or_check(path: Path, content: str, check: bool) -> None:
    content = content.rstrip() + "\n"
    if check:
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            raise SystemExit(f"generated output differs: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="compare outputs without writing")
    arguments = parser.parse_args()
    versions = require_environment()
    cases, placement = build_cases()
    write_or_check(FIXTURE_PATH, render_fixture(cases, versions), arguments.check)
    write_or_check(REPORT_PATH, render_report(cases, versions, placement), arguments.check)


if __name__ == "__main__":
    main()
