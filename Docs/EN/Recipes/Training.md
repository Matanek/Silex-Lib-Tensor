# Train an MLP, CNN, or RNN

These recipes change the architecture and data shape while keeping the same
eager loop. Their complete sources belong to Tensor's public consumer and can
be run from the workspace root.

## Shared loop

A training step first clears gradients, runs the model, constructs a scalar
loss, consumes its graph, then replaces parameter values:

```sx
optimizer.zero_grad()
let loss = model.forward(input).cross_entropy(target, 1)
loss.backward()
optimizer.step()
```

`Tensor` remains an immutable value. `Neural.Parameter` owns the trainable leaf;
`optimizer.step()` assigns it a new detached value. Reading `loss.item()` on
every iteration is correct on the CPU but synchronizes a GPU model. Space that
observation out when the loop should remain resident.

## Dense MLP

The executable [MLP example](../../../Tests/Consumer/Examples/TrainMLP.sx)
learns XOR, saves its parameters, and loads a fresh model. Its architecture is:

```sx
var layers:Neural.Layer[] = [
    Neural.Dense("hidden", 2, 8, 101),
    Neural.Activation.tanh(),
    Neural.Dense("classifier", 8, 2, 102)
]
var model = Neural.Sequential(layers)
```

Layer names determine stable checkpoint names. Reusing one name in the same
collection is rejected before training.

## NCHW CNN

The executable [CNN example](../../../Tests/Consumer/Examples/TrainCNN.sx)
classifies four `4 × 4` patterns. Tensor exposes NCHW inputs and OIHW kernels:

```sx
var layers:Neural.Layer[] = [
    Neural.Conv2D("features", 1, 4, 2, 211),
    Neural.Activation.relu(),
    Neural.MaxPool2D(2, stride:1),
    Neural.Flatten(),
    Neural.Dense("classifier", 16, 2, 212)
]
```

`Flatten` preserves the batch dimension. `Dense` input size must exactly match
the remaining channels and spatial dimensions.

## Simple RNN

The executable [RNN example](../../../Tests/Consumer/Examples/TrainRNN.sx)
receives `[batch, time, features]`. `SimpleRNN` unfolds a unidirectional tanh
recurrence and returns the final state only:

```sx
var layers:Neural.Layer[] = [
    Neural.SimpleRNN("memory", 1, 6, 307),
    Neural.Dense("classifier", 6, 2, 308)
]
```

The recipe clips the global norm before `step()`. This operation composes all
available gradients; it does not impose an independent limit per parameter.

## Keep the step on the GPU

Construct the model and data first, move them to the same device, then create
the optimizer from already-placed parameters:

```sx
var device = GPU.Device()
model.to(device)
let gpu_input = input.to(device)
let gpu_target = target.to(device)
var parameters = model.parameters()
var optimizer = Optim.Adam(parameters, learning_rate:0.01)

optimizer.zero_grad()
let loss = model.forward(gpu_input).cross_entropy(gpu_target, 1)
loss.backward()
optimizer.step()

let observed = loss.detach().cpu().item()
```

Forward, backward, and `step()` remain resident. The final `cpu()` is the
deliberate readback and wait point. `TrainingMLPGPU.sx`, `TrainingCNNGPU.sx`,
and `TrainingRNNGPU.sx` verify all three complete CPU/GPU workflows.

## Save parameters

```sx
Neural.Checkpoint.save(model, path)
var restored = make_model(999)
Neural.Checkpoint.load(restored, path)
```

Load a CPU model with the same architecture and names, then call
`restored.to(device)` if needed. A checkpoint does not restore Adam state;
applications must manage exact optimizer resumption separately.
