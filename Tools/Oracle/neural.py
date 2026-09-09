"""Pinned PyTorch/TensorFlow neural training corpus for Tensor's oracle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class NeuralCase:
    name: str
    family: str
    values: tuple[float, ...]
    shape: tuple[int, ...]
    absolute_tolerance: float
    relative_tolerance: float


@dataclass(frozen=True)
class ModelSpec:
    inputs: Any
    targets: Any
    parameters: tuple[tuple[str, Any], ...]


def model_specs(np: Any) -> dict[str, ModelSpec]:
    return {
        "mlp": ModelSpec(
            np.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.asarray([0, 1, 1, 0], dtype=np.int32),
            (
                ("hidden.weight", np.asarray([0.2, -0.3, 0.4, -0.5, 0.6, -0.1], dtype=np.float32).reshape(2, 3)),
                ("hidden.bias", np.asarray([0.1, -0.2, 0.05], dtype=np.float32)),
                ("classifier.weight", np.asarray([0.3, -0.4, -0.2, 0.5, 0.7, -0.1], dtype=np.float32).reshape(3, 2)),
                ("classifier.bias", np.asarray([0.05, -0.05], dtype=np.float32)),
            ),
        ),
        "cnn": ModelSpec(
            np.asarray([
                1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 2.0, 0.0,
                1.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0,
                0.0, 1.0, 2.0, 1.0, 2.0, 0.0, 1.0, 0.0,
                1.0, 2.0, 0.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            ], dtype=np.float32).reshape(2, 1, 4, 4),
            np.asarray([0, 1], dtype=np.int32),
            (
                ("features.weight", np.asarray([
                    0.2, -0.1, 0.3, 0.4,
                    -0.2, 0.5, 0.1, -0.3,
                ], dtype=np.float32).reshape(2, 1, 2, 2)),
                ("features.bias", np.asarray([0.05, -0.02], dtype=np.float32)),
                ("classifier.weight", np.asarray([
                    0.1, -0.2, 0.3, 0.05, -0.1, 0.25, 0.2, -0.15,
                    -0.05, 0.35, 0.4, -0.3, 0.15, 0.1, -0.25, 0.2,
                ], dtype=np.float32).reshape(8, 2)),
                ("classifier.bias", np.asarray([0.02, -0.03], dtype=np.float32)),
            ),
        ),
        "rnn": ModelSpec(
            np.asarray([-1.0, 0.5, 1.0, 1.0, 0.0, -0.5], dtype=np.float32).reshape(2, 3, 1),
            np.asarray([0, 1], dtype=np.int32),
            (
                ("memory.input_weight", np.asarray([0.4, -0.3], dtype=np.float32).reshape(1, 2)),
                ("memory.recurrent_weight", np.asarray([0.2, 0.1, -0.25, 0.35], dtype=np.float32).reshape(2, 2)),
                ("memory.bias", np.asarray([0.05, -0.1], dtype=np.float32)),
                ("classifier.weight", np.asarray([0.3, -0.2, -0.4, 0.5], dtype=np.float32).reshape(2, 2)),
                ("classifier.bias", np.asarray([0.02, -0.03], dtype=np.float32)),
            ),
        ),
    }


def torch_forward(torch: Any, family: str, inputs: Any, parameters: list[Any]) -> tuple[Any, dict[str, Any]]:
    if family == "mlp":
        hidden = inputs @ parameters[0] + parameters[1]
        activation = torch.tanh(hidden)
        logits = activation @ parameters[2] + parameters[3]
        return logits, {"dense": hidden, "activation": activation, "logits": logits}
    if family == "cnn":
        convolution = torch.nn.functional.conv2d(inputs, parameters[0], parameters[1])
        activation = torch.relu(convolution)
        pooling = torch.nn.functional.max_pool2d(activation, 2, 1)
        flatten = pooling.reshape(pooling.shape[0], -1)
        logits = flatten @ parameters[2] + parameters[3]
        return logits, {
            "convolution": convolution,
            "activation": activation,
            "pooling": pooling,
            "flatten": flatten,
            "logits": logits,
        }
    hidden = torch.zeros((inputs.shape[0], 2), dtype=torch.float32)
    for step in range(inputs.shape[1]):
        hidden = torch.tanh(inputs[:, step, :] @ parameters[0] + hidden @ parameters[1] + parameters[2])
    logits = hidden @ parameters[3] + parameters[4]
    return logits, {"state": hidden, "logits": logits}


def tensorflow_forward(tf: Any, family: str, inputs: Any, parameters: list[Any]) -> tuple[Any, dict[str, Any]]:
    if family == "mlp":
        hidden = tf.matmul(inputs, parameters[0]) + parameters[1]
        activation = tf.math.tanh(hidden)
        logits = tf.matmul(activation, parameters[2]) + parameters[3]
        return logits, {"dense": hidden, "activation": activation, "logits": logits}
    if family == "cnn":
        channels_last = tf.transpose(inputs, (0, 2, 3, 1))
        kernel = tf.transpose(parameters[0], (2, 3, 1, 0))
        convolution_nhwc = tf.nn.conv2d(channels_last, kernel, strides=1, padding="VALID") + parameters[1]
        activation_nhwc = tf.nn.relu(convolution_nhwc)
        pooling_nhwc = tf.nn.max_pool2d(activation_nhwc, 2, 1, "VALID")
        pooling = tf.transpose(pooling_nhwc, (0, 3, 1, 2))
        flatten = tf.reshape(pooling, (tf.shape(pooling)[0], -1))
        logits = tf.matmul(flatten, parameters[2]) + parameters[3]
        return logits, {
            "convolution": tf.transpose(convolution_nhwc, (0, 3, 1, 2)),
            "activation": tf.transpose(activation_nhwc, (0, 3, 1, 2)),
            "pooling": pooling,
            "flatten": flatten,
            "logits": logits,
        }
    hidden = tf.zeros((tf.shape(inputs)[0], 2), dtype=tf.float32)
    for step in range(int(inputs.shape[1])):
        hidden = tf.math.tanh(
            tf.matmul(inputs[:, step, :], parameters[0]) +
            tf.matmul(hidden, parameters[1]) + parameters[2]
        )
    logits = tf.matmul(hidden, parameters[3]) + parameters[4]
    return logits, {"state": hidden, "logits": logits}


def torch_results(np: Any, torch: Any, family: str, spec: ModelSpec) -> dict[str, Any]:
    inputs = torch.from_numpy(spec.inputs)
    targets = torch.from_numpy(spec.targets).to(torch.int64)

    def fresh() -> list[Any]:
        return [torch.from_numpy(values).clone().requires_grad_(True) for _, values in spec.parameters]

    parameters = fresh()
    logits, layers = torch_forward(torch, family, inputs, parameters)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()
    result: dict[str, Any] = {f"layer_{name}": value.detach().numpy() for name, value in layers.items()}
    result["loss"] = loss.detach().numpy()
    for (name, _), parameter in zip(spec.parameters, parameters, strict=True):
        result[f"gradient_{name}"] = parameter.grad.detach().numpy()

    for optimizer_name in ("sgd", "adam"):
        current = fresh()
        if optimizer_name == "sgd":
            optimizer = torch.optim.SGD(current, lr=0.05)
        else:
            optimizer = torch.optim.Adam(current, lr=0.01, betas=(0.8, 0.9), eps=1e-8)
        step_logits, _ = torch_forward(torch, family, inputs, current)
        torch.nn.functional.cross_entropy(step_logits, targets).backward()
        optimizer.step()
        result[optimizer_name] = np.concatenate([value.detach().numpy().reshape(-1) for value in current])

    trajectory = fresh()
    optimizer = torch.optim.SGD(trajectory, lr=0.05)
    losses = []
    with torch.no_grad():
        initial, _ = torch_forward(torch, family, inputs, trajectory)
        losses.append(float(torch.nn.functional.cross_entropy(initial, targets)))
    for step in range(1, 6):
        optimizer.zero_grad(set_to_none=True)
        step_logits, _ = torch_forward(torch, family, inputs, trajectory)
        torch.nn.functional.cross_entropy(step_logits, targets).backward()
        optimizer.step()
        if step in (1, 3, 5):
            with torch.no_grad():
                observed, _ = torch_forward(torch, family, inputs, trajectory)
                losses.append(float(torch.nn.functional.cross_entropy(observed, targets)))
    result["trajectory"] = np.asarray(losses, dtype=np.float32)
    return result


def tensorflow_results(np: Any, tf: Any, family: str, spec: ModelSpec) -> dict[str, Any]:
    inputs = tf.constant(spec.inputs)
    targets = tf.constant(spec.targets)

    def fresh() -> list[Any]:
        return [tf.Variable(values) for _, values in spec.parameters]

    parameters = fresh()
    with tf.GradientTape() as tape:
        logits, layers = tensorflow_forward(tf, family, inputs, parameters)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
    gradients = tape.gradient(loss, parameters)
    result: dict[str, Any] = {f"layer_{name}": value.numpy() for name, value in layers.items()}
    result["loss"] = loss.numpy()
    for (name, _), gradient in zip(spec.parameters, gradients, strict=True):
        result[f"gradient_{name}"] = gradient.numpy()

    sgd = fresh()
    with tf.GradientTape() as tape:
        step_logits, _ = tensorflow_forward(tf, family, inputs, sgd)
        step_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=step_logits))
    for parameter, gradient in zip(sgd, tape.gradient(step_loss, sgd), strict=True):
        parameter.assign_sub(tf.constant(0.05, dtype=tf.float32) * gradient)
    result["sgd"] = np.concatenate([value.numpy().reshape(-1) for value in sgd])

    adam = fresh()
    with tf.GradientTape() as tape:
        step_logits, _ = tensorflow_forward(tf, family, inputs, adam)
        step_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=step_logits))
    adam_gradients = tape.gradient(step_loss, adam)
    updated = []
    for parameter, gradient in zip(adam, adam_gradients, strict=True):
        # At step one, bias-corrected m and v reduce to g and g².
        parameter.assign_sub(tf.constant(0.01, dtype=tf.float32) * gradient / (tf.abs(gradient) + 1e-8))
        updated.append(parameter.numpy().reshape(-1))
    result["adam"] = np.concatenate(updated)

    trajectory = fresh()
    initial, _ = tensorflow_forward(tf, family, inputs, trajectory)
    losses = [float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=initial)))]
    for step in range(1, 6):
        with tf.GradientTape() as tape:
            step_logits, _ = tensorflow_forward(tf, family, inputs, trajectory)
            step_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=step_logits))
        for parameter, gradient in zip(trajectory, tape.gradient(step_loss, trajectory), strict=True):
            parameter.assign_sub(tf.constant(0.05, dtype=tf.float32) * gradient)
        if step in (1, 3, 5):
            observed, _ = tensorflow_forward(tf, family, inputs, trajectory)
            losses.append(float(tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=observed))))
    result["trajectory"] = np.asarray(losses, dtype=np.float32)
    return result


def normalized(np: Any, value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    array = np.asarray(value, dtype=np.float32)
    return tuple(int(size) for size in array.shape), tuple(float(item) for item in array.reshape(-1))


def matches(left: tuple[float, ...], right: tuple[float, ...], absolute: float, relative: float) -> bool:
    if len(left) != len(right):
        return False
    return all(abs(a - b) <= max(absolute, relative * abs(b)) for a, b in zip(left, right, strict=True))


def build_neural_cases(np: Any, torch: Any, tf: Any) -> tuple[dict[str, NeuralCase], dict[str, ModelSpec]]:
    specs = model_specs(np)
    cases: dict[str, NeuralCase] = {}
    for family, spec in specs.items():
        pytorch = torch_results(np, torch, family, spec)
        tensorflow = tensorflow_results(np, tf, family, spec)
        if pytorch.keys() != tensorflow.keys():
            raise RuntimeError(f"{family}: neural oracle keys disagree")
        for role, torch_value in pytorch.items():
            torch_shape, torch_values = normalized(np, torch_value)
            tensorflow_shape, tensorflow_values = normalized(np, tensorflow[role])
            if torch_shape != tensorflow_shape:
                raise RuntimeError(f"{family}_{role}: PyTorch and TensorFlow disagree on shape")
            absolute = 2e-5 if role.startswith("layer_") or role == "loss" else 8e-5
            relative = 2e-5 if role.startswith("layer_") or role == "loss" else 4e-5
            if not matches(torch_values, tensorflow_values, absolute, relative):
                raise RuntimeError(f"{family}_{role}: PyTorch and TensorFlow disagree")
            name = f"network_{family}_{role}"
            cases[name] = NeuralCase(name, family, torch_values, torch_shape, absolute, relative)
    return cases, specs


def render_float(value: float) -> str:
    rendered = format(value, ".9g")
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered


def render_values(values: tuple[float, ...]) -> str:
    return ", ".join(render_float(value) for value in values)


def render_checkpoint(spec: ModelSpec) -> str:
    lines = ["SILEX_TENSOR_CHECKPOINT\t1", str(len(spec.parameters))]
    for name, values in spec.parameters:
        shape = ",".join(str(int(size)) for size in values.shape) or "-"
        flattened = ",".join(render_float(float(value)) for value in values.reshape(-1))
        lines.append(f"{name.encode('utf-8').hex()}\tfloat32\t{values.ndim}\t{shape}\t{values.size}\t{flattened}")
    return "\n".join(lines) + "\n"


def render_neural_report(cases: dict[str, NeuralCase]) -> str:
    lines = [
        "# Neural differential oracle",
        "",
        "This generated corpus fixes identical weights, row-major inputs, sparse cross-entropy reduction,",
        "PyTorch/TensorFlow ReLU-at-zero behavior, and Adam `(beta1=0.8, beta2=0.9, epsilon=1e-8)`.",
        "Every case is accepted only when both frameworks agree after float32 normalization.",
        "",
        "| Case | Network | Shape | Tolerance |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases.values():
        shape = "[" + ", ".join(str(value) for value in case.shape) + "]"
        lines.append(
            f"| `{case.name}` | {case.family.upper()} | `{shape}` | "
            f"abs {case.absolute_tolerance:g}, rel {case.relative_tolerance:g} |"
        )
    lines.extend([
        "",
        "The fixture covers every layer output, scalar loss, every named parameter gradient, one SGD step,",
        "one Adam step, and losses before training plus steps 1, 3, and 5 for MLP, CNN, and SimpleRNN models.",
        "A separate hermetic finite-difference test checks a smooth tanh derivative without either framework.",
        "GPU parity and transfer residency use the same public layer families in `TrainingMLPGPU.sx`,",
        "`TrainingCNNGPU.sx`, and `TrainingRNNGPU.sx`.",
        "",
    ])
    return "\n".join(lines)


def render_neural_fixture(cases: dict[str, NeuralCase], specs: dict[str, ModelSpec], versions: dict[str, str]) -> str:
    def values(name: str) -> str:
        return render_values(cases[name].values)

    def gradients(family: str) -> str:
        merged: list[float] = []
        for name, _ in specs[family].parameters:
            merged.extend(cases[f"network_{family}_gradient_{name}"].values)
        return render_values(tuple(merged))

    def parameters(name: str) -> str:
        return values(name)

    header = ", ".join(f"{name} {version}" for name, version in versions.items() if name in ("torch", "tensorflow"))
    return f'''// Generated by Tools/Oracle/generate.py. Do not edit by hand.
// Identical float32 weights and data; PyTorch and TensorFlow: {header}.

use Tensor
use STD.Math
use Tensor.Neural
use Tensor.Optim
use Tensor.Autograd

func neural_values_match(actual:float[], expected:float[], absolute:float = 0.00008, relative:float = 0.00004) bool {{
    if actual.count() != expected.count() {{ return false }}
    var index = 0
    while index < actual.count() {{
        var tolerance = absolute
        let scaled = relative * Math.abs(expected[index])
        if scaled > tolerance {{ tolerance = scaled }}
        if Math.abs(actual[index] - expected[index]) > tolerance {{ return false }}
        index++
    }}
    return true
}}

func neural_gradient(parameter:Neural.Parameter) float[] {{
    if let gradient = parameter.gradient() {{ return gradient.detach().values() }}
    panic("expected generated neural gradient")
}}

func neural_parameters_match(parameters:Neural.Parameter[], expected:float[]) bool {{
    var offset = 0
    for parameter in parameters {{
        let observed = parameter.value().detach().values()
        for value in observed {{
            if !neural_values_match([value], [expected[offset]]) {{ return false }}
            offset++
        }}
    }}
    return offset == expected.count()
}}

func neural_gradients_match(parameters:Neural.Parameter[], expected:float[]) bool {{
    var offset = 0
    for parameter in parameters {{
        let observed = neural_gradient(parameter)
        for value in observed {{
            if !neural_values_match([value], [expected[offset]]) {{ return false }}
            offset++
        }}
    }}
    return offset == expected.count()
}}

func neural_sgd_matches(model:Neural.Layer, path:str, input:Tensor, target:Tensor, expected:float[]) bool {{
    Neural.Checkpoint.load(model, path)
    var parameters = model.parameters()
    var optimizer = Optim.SGD(parameters, 0.05)
    model.forward(input).cross_entropy(target, 1).backward()
    optimizer.step()
    return neural_parameters_match(parameters, expected)
}}

func neural_adam_matches(model:Neural.Layer, path:str, input:Tensor, target:Tensor, expected:float[]) bool {{
    Neural.Checkpoint.load(model, path)
    var parameters = model.parameters()
    var optimizer = Optim.Adam(parameters, learning_rate:0.01, beta1:0.8, beta2:0.9, epsilon:0.00000001)
    model.forward(input).cross_entropy(target, 1).backward()
    optimizer.step()
    return neural_parameters_match(parameters, expected)
}}

func neural_trajectory_matches(model:Neural.Layer, path:str, input:Tensor, target:Tensor, expected:float[]) bool {{
    Neural.Checkpoint.load(model, path)
    var parameters = model.parameters()
    var optimizer = Optim.SGD(parameters, 0.05)
    var observed:float[] = [model.forward(input).detach().cross_entropy(target, 1).item()]
    var step = 1
    while step <= 5 {{
        optimizer.zero_grad()
        model.forward(input).cross_entropy(target, 1).backward()
        optimizer.step()
        if step == 1 || step == 3 || step == 5 {{
            observed.append(model.forward(input).detach().cross_entropy(target, 1).item())
        }}
        step++
    }}
    return neural_values_match(observed, expected)
}}

func neural_mlp_model() Neural.Sequential {{
    var layers:Neural.Layer[] = [Neural.Dense("hidden", 2, 3, 1), Neural.Activation.tanh(), Neural.Dense("classifier", 3, 2, 2)]
    return Neural.Sequential(layers)
}}

func neural_cnn_model() Neural.Sequential {{
    var layers:Neural.Layer[] = [
        Neural.Conv2D("features", 1, 2, 2, 3), Neural.Activation.relu(),
        Neural.MaxPool2D(2, stride:1), Neural.Flatten(), Neural.Dense("classifier", 8, 2, 4)
    ]
    return Neural.Sequential(layers)
}}

func neural_rnn_model() Neural.Sequential {{
    var layers:Neural.Layer[] = [Neural.SimpleRNN("memory", 1, 2, 5), Neural.Dense("classifier", 2, 2, 6)]
    return Neural.Sequential(layers)
}}

test "match PyTorch and TensorFlow MLP layers gradients optimizers and trajectory" {{
    let path = "Packages/Tensor/Tests/Consumer/Tests/Fixtures/NeuralOracleMLP.sxtc"
    var input_values:float[] = [{render_values(tuple(float(v) for v in specs['mlp'].inputs.reshape(-1)))}]
    var target_values:int32[] = [0, 1, 1, 0]
    let input = Tensor(input_values, [4, 2])
    let target = Tensor(target_values, [4])
    var model = neural_mlp_model(); Neural.Checkpoint.load(model, path)
    var weight0:float[] = [{render_values(tuple(float(v) for v in specs['mlp'].parameters[0][1].reshape(-1)))}]
    var bias0:float[] = [{render_values(tuple(float(v) for v in specs['mlp'].parameters[1][1].reshape(-1)))}]
    var weight1:float[] = [{render_values(tuple(float(v) for v in specs['mlp'].parameters[2][1].reshape(-1)))}]
    var bias1:float[] = [{render_values(tuple(float(v) for v in specs['mlp'].parameters[3][1].reshape(-1)))}]
    let dense = input.matmul(Tensor(weight0, [2, 3])).add(Tensor.vector(bias0))
    let activation = dense.tanh()
    let logits = activation.matmul(Tensor(weight1, [3, 2])).add(Tensor.vector(bias1))
    assert(neural_values_match(dense.values(), [{values('network_mlp_layer_dense')}]))
    assert(neural_values_match(activation.values(), [{values('network_mlp_layer_activation')}]))
    assert(neural_values_match(logits.values(), [{values('network_mlp_layer_logits')}]))
    let loss = model.forward(input).cross_entropy(target, 1)
    assert(neural_values_match(loss.detach().values(), [{values('network_mlp_loss')}]))
    loss.backward()
    assert(neural_gradients_match(model.parameters(), [{gradients('mlp')}]))
    assert(neural_sgd_matches(neural_mlp_model(), path, input, target, [{parameters('network_mlp_sgd')}]))
    assert(neural_adam_matches(neural_mlp_model(), path, input, target, [{parameters('network_mlp_adam')}]))
    assert(neural_trajectory_matches(neural_mlp_model(), path, input, target, [{values('network_mlp_trajectory')}]))
}}

test "match PyTorch and TensorFlow CNN layers gradients optimizers and trajectory" {{
    let path = "Packages/Tensor/Tests/Consumer/Tests/Fixtures/NeuralOracleCNN.sxtc"
    var input_values:float[] = [{render_values(tuple(float(v) for v in specs['cnn'].inputs.reshape(-1)))}]
    var target_values:int32[] = [0, 1]
    let input = Tensor(input_values, [2, 1, 4, 4])
    let target = Tensor(target_values, [2])
    var model = neural_cnn_model(); Neural.Checkpoint.load(model, path)
    var kernel:float[] = [{render_values(tuple(float(v) for v in specs['cnn'].parameters[0][1].reshape(-1)))}]
    var convolution_bias:float[] = [{render_values(tuple(float(v) for v in specs['cnn'].parameters[1][1].reshape(-1)))}]
    var head_weight:float[] = [{render_values(tuple(float(v) for v in specs['cnn'].parameters[2][1].reshape(-1)))}]
    var head_bias:float[] = [{render_values(tuple(float(v) for v in specs['cnn'].parameters[3][1].reshape(-1)))}]
    let convolution = input.conv2d(Tensor(kernel, [2, 1, 2, 2]), Tensor.vector(convolution_bias))
    let activation = convolution.relu()
    let pooling = activation.max_pool2d(2, 1)
    let flattened = pooling.reshape([2, 8])
    let logits = flattened.matmul(Tensor(head_weight, [8, 2])).add(Tensor.vector(head_bias))
    assert(neural_values_match(convolution.values(), [{values('network_cnn_layer_convolution')}]))
    assert(neural_values_match(activation.values(), [{values('network_cnn_layer_activation')}]))
    assert(neural_values_match(pooling.values(), [{values('network_cnn_layer_pooling')}]))
    assert(neural_values_match(flattened.values(), [{values('network_cnn_layer_flatten')}]))
    assert(neural_values_match(logits.values(), [{values('network_cnn_layer_logits')}]))
    let loss = model.forward(input).cross_entropy(target, 1)
    assert(neural_values_match(loss.detach().values(), [{values('network_cnn_loss')}]))
    loss.backward()
    assert(neural_gradients_match(model.parameters(), [{gradients('cnn')}]))
    assert(neural_sgd_matches(neural_cnn_model(), path, input, target, [{parameters('network_cnn_sgd')}]))
    assert(neural_adam_matches(neural_cnn_model(), path, input, target, [{parameters('network_cnn_adam')}]))
    assert(neural_trajectory_matches(neural_cnn_model(), path, input, target, [{values('network_cnn_trajectory')}]))
}}

test "match PyTorch and TensorFlow RNN layers gradients optimizers and trajectory" {{
    let path = "Packages/Tensor/Tests/Consumer/Tests/Fixtures/NeuralOracleRNN.sxtc"
    var input_values:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].inputs.reshape(-1)))}]
    var target_values:int32[] = [0, 1]
    let input = Tensor(input_values, [2, 3, 1])
    let target = Tensor(target_values, [2])
    var model = neural_rnn_model(); Neural.Checkpoint.load(model, path)
    var input_weight:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].parameters[0][1].reshape(-1)))}]
    var recurrent_weight:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].parameters[1][1].reshape(-1)))}]
    var recurrent_bias:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].parameters[2][1].reshape(-1)))}]
    var head_weight:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].parameters[3][1].reshape(-1)))}]
    var head_bias:float[] = [{render_values(tuple(float(v) for v in specs['rnn'].parameters[4][1].reshape(-1)))}]
    var state = Tensor.zeros([2, 2])
    var step = 0
    while step < 3 {{
        state = input.select(1, step).matmul(Tensor(input_weight, [1, 2]))
            .add(state.matmul(Tensor(recurrent_weight, [2, 2])))
            .add(Tensor.vector(recurrent_bias)).tanh()
        step++
    }}
    let logits = state.matmul(Tensor(head_weight, [2, 2])).add(Tensor.vector(head_bias))
    assert(neural_values_match(state.values(), [{values('network_rnn_layer_state')}]))
    assert(neural_values_match(logits.values(), [{values('network_rnn_layer_logits')}]))
    let output = model.forward(input)
    assert(neural_values_match(output.detach().values(), [{values('network_rnn_layer_logits')}]))
    let loss = output.cross_entropy(target, 1)
    assert(neural_values_match(loss.detach().values(), [{values('network_rnn_loss')}]))
    loss.backward()
    assert(neural_gradients_match(model.parameters(), [{gradients('rnn')}]))
    assert(neural_sgd_matches(neural_rnn_model(), path, input, target, [{parameters('network_rnn_sgd')}]))
    assert(neural_adam_matches(neural_rnn_model(), path, input, target, [{parameters('network_rnn_adam')}]))
    assert(neural_trajectory_matches(neural_rnn_model(), path, input, target, [{values('network_rnn_trajectory')}]))
}}

test "match a smooth tanh derivative against finite differences" {{
    var input_values:float[] = [0.25, -0.5, 1.0, 0.75]
    var weight_values:float[] = [0.2, -0.3, 0.4, 0.1]
    let input = Tensor(input_values, [2, 2])
    var weight = Autograd.Variable(Tensor(weight_values, [2, 2]))
    input.matmul(weight.value()).tanh().sum().backward()
    if let actual = weight.gradient() {{
        let observed = actual.values()
        let epsilon = 0.001
        var index = 0
        while index < weight_values.count() {{
            var positive = copy weight_values
            var negative = copy weight_values
            positive[index] += epsilon
            negative[index] -= epsilon
            let upper = input.matmul(Tensor(positive, [2, 2])).tanh().sum().item()
            let lower = input.matmul(Tensor(negative, [2, 2])).tanh().sum().item()
            let finite = (upper - lower) / (2.0 * epsilon)
            assert(Math.abs(observed[index] - finite) < 0.002)
            index++
        }}
    }} else {{ panic("expected finite-difference gradient") }}
}}
'''
