# Neural differential oracle

This generated corpus fixes identical weights, row-major inputs, sparse cross-entropy reduction,
PyTorch/TensorFlow ReLU-at-zero behavior, and Adam `(beta1=0.8, beta2=0.9, epsilon=1e-8)`.
Every case is accepted only when both frameworks agree after float32 normalization.

| Case | Network | Shape | Tolerance |
| --- | --- | --- | --- |
| `network_mlp_layer_dense` | MLP | `[4, 3]` | abs 2e-05, rel 2e-05 |
| `network_mlp_layer_activation` | MLP | `[4, 3]` | abs 2e-05, rel 2e-05 |
| `network_mlp_layer_logits` | MLP | `[4, 2]` | abs 2e-05, rel 2e-05 |
| `network_mlp_loss` | MLP | `[]` | abs 2e-05, rel 2e-05 |
| `network_mlp_gradient_hidden.weight` | MLP | `[2, 3]` | abs 8e-05, rel 4e-05 |
| `network_mlp_gradient_hidden.bias` | MLP | `[3]` | abs 8e-05, rel 4e-05 |
| `network_mlp_gradient_classifier.weight` | MLP | `[3, 2]` | abs 8e-05, rel 4e-05 |
| `network_mlp_gradient_classifier.bias` | MLP | `[2]` | abs 8e-05, rel 4e-05 |
| `network_mlp_sgd` | MLP | `[17]` | abs 8e-05, rel 4e-05 |
| `network_mlp_adam` | MLP | `[17]` | abs 8e-05, rel 4e-05 |
| `network_mlp_trajectory` | MLP | `[4]` | abs 8e-05, rel 4e-05 |
| `network_cnn_layer_convolution` | CNN | `[2, 2, 3, 3]` | abs 2e-05, rel 2e-05 |
| `network_cnn_layer_activation` | CNN | `[2, 2, 3, 3]` | abs 2e-05, rel 2e-05 |
| `network_cnn_layer_pooling` | CNN | `[2, 2, 2, 2]` | abs 2e-05, rel 2e-05 |
| `network_cnn_layer_flatten` | CNN | `[2, 8]` | abs 2e-05, rel 2e-05 |
| `network_cnn_layer_logits` | CNN | `[2, 2]` | abs 2e-05, rel 2e-05 |
| `network_cnn_loss` | CNN | `[]` | abs 2e-05, rel 2e-05 |
| `network_cnn_gradient_features.weight` | CNN | `[2, 1, 2, 2]` | abs 8e-05, rel 4e-05 |
| `network_cnn_gradient_features.bias` | CNN | `[2]` | abs 8e-05, rel 4e-05 |
| `network_cnn_gradient_classifier.weight` | CNN | `[8, 2]` | abs 8e-05, rel 4e-05 |
| `network_cnn_gradient_classifier.bias` | CNN | `[2]` | abs 8e-05, rel 4e-05 |
| `network_cnn_sgd` | CNN | `[28]` | abs 8e-05, rel 4e-05 |
| `network_cnn_adam` | CNN | `[28]` | abs 8e-05, rel 4e-05 |
| `network_cnn_trajectory` | CNN | `[4]` | abs 8e-05, rel 4e-05 |
| `network_rnn_layer_state` | RNN | `[2, 2]` | abs 2e-05, rel 2e-05 |
| `network_rnn_layer_logits` | RNN | `[2, 2]` | abs 2e-05, rel 2e-05 |
| `network_rnn_loss` | RNN | `[]` | abs 2e-05, rel 2e-05 |
| `network_rnn_gradient_memory.input_weight` | RNN | `[1, 2]` | abs 8e-05, rel 4e-05 |
| `network_rnn_gradient_memory.recurrent_weight` | RNN | `[2, 2]` | abs 8e-05, rel 4e-05 |
| `network_rnn_gradient_memory.bias` | RNN | `[2]` | abs 8e-05, rel 4e-05 |
| `network_rnn_gradient_classifier.weight` | RNN | `[2, 2]` | abs 8e-05, rel 4e-05 |
| `network_rnn_gradient_classifier.bias` | RNN | `[2]` | abs 8e-05, rel 4e-05 |
| `network_rnn_sgd` | RNN | `[14]` | abs 8e-05, rel 4e-05 |
| `network_rnn_adam` | RNN | `[14]` | abs 8e-05, rel 4e-05 |
| `network_rnn_trajectory` | RNN | `[4]` | abs 8e-05, rel 4e-05 |

The fixture covers every layer output, scalar loss, every named parameter gradient, one SGD step,
one Adam step, and losses before training plus steps 1, 3, and 5 for MLP, CNN, and SimpleRNN models.
A separate hermetic finite-difference test checks a smooth tanh derivative without either framework.
GPU parity and transfer residency use the same public layer families in `TrainingMLPGPU.sx`,
`TrainingCNNGPU.sx`, and `TrainingRNNGPU.sx`.
