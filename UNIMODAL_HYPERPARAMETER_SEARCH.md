# Unimodal Hyperparameter Search

This document summarizes the local hyperparameter search used for the final
unimodal CMU-MOSEI baselines. The search was not intended to be an exhaustive
grid. Instead, we evaluated a small set of nearby configurations for each
modality, varying the most relevant capacity, regularization, and optimization
parameters. Final configurations were selected using validation MAE.

## Text BERT Baseline

The text baseline used `bert-base-uncased` with masked mean pooling over token
representations. The local search varied maximum sequence length, dropout, and
learning rate around the selected configuration.

| Config | Max Length | Dropout | Learning Rate |
|---:|---:|---:|---:|
| 1 | 32 | 0.4 | `1e-5` |
| 2 | 64 | 0.2 | `1e-3` |
| 3 | 64 | 0.3 | `5e-4` |
| 4 | 64 | 0.4 | `3e-4` |
| 5 **selected** | 96 | 0.3 | `5e-4` |
| 6 | 96 | 0.3 | `1e-5` |
| 7 | 128 | 0.4 | `5e-4` |

Final selected text configuration:

```text
max_length = 96
dropout = 0.3
learning_rate = 5e-4
```

## Visual BiLSTM Baseline

The visual baseline used OpenFace2 visual features with a BiLSTM sequence
encoder and mean pooling. Because the visual-only model was comparatively
sensitive to optimization and regularization, the manual search was centered on
small one-layer BiLSTMs with moderate-to-high dropout, low learning rates, and
light weight decay. Selection was based on validation MAE.

| Config | BiLSTM Hidden | Layers | Dropout | Learning Rate | Weight Decay | Loss | Delta |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 64 | 1 | 0.35 | `1e-4` | `1e-4` | SmoothL1 | 0.5 |
| 2 | 64 | 1 | 0.45 | `1e-4` | `1e-4` | SmoothL1 | 0.5 |
| 3 | 64 | 1 | 0.45 | `7e-5` | `1e-4` | SmoothL1 | 0.5 |
| 4 | 64 | 1 | 0.55 | `7e-5` | `3e-4` | SmoothL1 | 0.5 |
| 5 | 96 | 1 | 0.35 | `1e-4` | `1e-4` | SmoothL1 | 0.5 |
| 6 | 96 | 1 | 0.45 | `7e-5` | `1e-4` | SmoothL1 | 0.5 |
| 7 **selected** | 96 | 1 | 0.55 | `7e-5` | `3e-4` | SmoothL1 | 0.5 |
| 8 | 96 | 1 | 0.45 | `5e-5` | `3e-4` | SmoothL1 | 1.0 |
| 9 | 48 | 1 | 0.45 | `7e-5` | `3e-4` | MSE | - |
| 10 | 96 | 1 | 0.50 | `5e-5` | `5e-4` | MSE | - |

Selected visual configuration:

```text
run_name = v_st07_h96_l1_dp55_lr7e5_wd3e4_smooth05_seed10
seed = 10
bilstm_hidden_size = 96
bilstm_layers = 1
hidden_dropout_prob = 0.55
learning_rate = 7e-5
weight_decay = 3e-4
loss_type = smoothl1
huber_delta = 0.5
batch_train = 48
batch_eval = 48
selection_metric = mae
```

## Audio BiLSTM Baseline

The audio baseline used COVAREP acoustic features with a BiLSTM sequence
encoder and mean pooling. The manual search was centered on a compact one-layer
BiLSTM with low dropout and low learning rates, with nearby variants testing
capacity, dropout, weight decay, and regression loss. Selection was based on
validation MAE.

| Config | BiLSTM Hidden | Layers | Dropout | Learning Rate | Weight Decay | Loss | Delta |
|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 128 | 1 | 0.10 | `1e-4` | `0` | MSE | - |
| 2 | 128 | 1 | 0.10 | `7.5e-5` | `0` | MSE | - |
| 3 | 96 | 1 | 0.10 | `7.5e-5` | `0` | MSE | - |
| 4 | 160 | 1 | 0.10 | `7.5e-5` | `0` | MSE | - |
| 5 | 128 | 1 | 0.15 | `7.5e-5` | `0` | MSE | - |
| 6 | 128 | 1 | 0.20 | `7.5e-5` | `0` | MSE | - |
| 7 | 128 | 2 | 0.15 | `7.5e-5` | `0` | MSE | - |
| 8 | 128 | 1 | 0.10 | `7.5e-5` | `1e-5` | MSE | - |
| 9 | 128 | 1 | 0.10 | `7.5e-5` | `1e-4` | MSE | - |
| 10 | 128 | 1 | 0.10 | `7.5e-5` | `0` | SmoothL1 | 1.0 |
| 11 **selected** | 128 | 1 | 0.10 | `7.5e-5` | `0` | SmoothL1 | 0.5 |
| 12 | 128 | 2 | 0.20 | `5e-5` | `0` | SmoothL1 | 1.0 |

Selected audio configuration:

```text
run_name = a_w16_l1_h128_dp10_lr75e6_smooth05_seed10
seed = 10
bilstm_hidden_size = 128
bilstm_layers = 1
hidden_dropout_prob = 0.10
learning_rate = 7.5e-5
weight_decay = 0.0
loss_type = smoothl1
huber_delta = 0.5
batch_train = 48
batch_eval = 48
selection_metric = mae
```
