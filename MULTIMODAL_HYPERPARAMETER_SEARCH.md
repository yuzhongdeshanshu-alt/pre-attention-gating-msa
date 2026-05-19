# Multimodal Hyperparameter Search

This document summarizes the limited manual hyperparameter search used for the
multimodal CMU-MOSEI models. The search was conducted on the UCL Myriad HPC
cluster. It was not designed as an exhaustive grid search; instead, we evaluated
a compact set of nearby configurations around plausible model-capacity,
regularization, and optimization settings.

For ablation fairness, hyperparameter selection was performed within the
pre-gating condition for each architecture family. The corresponding NoGate
condition then used the matched architecture and optimization settings, with the
pre-gating mechanism removed. This keeps the comparison focused on the gating
ablation rather than on independent tuning differences between gated and
ungated models.

The manual search focused on key hyperparameters including learning rate,
batch size, dropout, fusion/attention hidden size, attention heads or
cross-attention dimension, and model depth. The broad candidate ranges were:

```text
learning_rate = {1e-3, 1e-4, 5e-5, 1e-5}
batch_size = {16, 32}
dropout = {0.1, 0.3, 0.5}
fusion_hidden_size = {128, 256, 512}
attention_heads = {4, 8}
```

Validation MAE was used as the model-selection criterion. The tables below
report the PreGate search candidates for each architecture family. The NoGate
conditions were not independently tuned; they use the selected matched
architecture and optimization settings with the gating mechanism removed.

## Selected Multimodal Configurations

| Model | Run Name | LR | Gate LR | Batch | Dropout | Hidden | Heads / Cross Dim | Layers | Fusion Dim | Gate Setting | Selection |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| Self-Attention NoGate | `self_attention_nogate_seed10` | `1e-4` | - | 48 | 0.10 | 768 | 8 heads | 2 | - | removed | MAE |
| Self-Attention PreGate | `self_attention_pregate_seed10` | `1e-4` | `1e-4` | 48 | 0.10 | 768 | 8 heads | 2 | - | `gate_dim=96`, `alpha=1.0` | MAE |
| Cross-Attention NoGate | `cross_attention_nogate_seed10` | `3e-5` | - | 48 | 0.10 | 768 | 128 dim | 2 | 128 | removed | MAE |
| Cross-Attention PreGate | `cross_attention_pregate_seed10` | `3e-5` | - | 48 | 0.10 | 768 | 128 dim | 2 | 128 | pre-softmax gate | MAE |

Shared settings for all four multimodal scripts were `seed = 10`,
`epochs = 40`, `warmup_ratio = 0.10`, `weight_decay = 0.0`,
`av_bilstm_hidden_size = 128`, `av_bilstm_layers = 3`, and
`max_grad_norm = 1.0`.

## Self-Attention PreGate Manual Search Candidates

The self-attention search focused on transformer capacity, number of heads,
dropout, and the token-level gate dimension. The selected setting is marked
below.

| Config | LR | Gate LR | Batch | Dropout | Hidden | Heads | Layers | FFN Size | Gate Dim | Alpha |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `1e-3` | `1e-3` | 16 | 0.10 | 512 | 4 | 1 | 1024 | 64 | 1.0 |
| 2 | `1e-3` | `1e-3` | 32 | 0.30 | 512 | 8 | 1 | 1024 | 64 | 1.0 |
| 3 | `1e-4` | `1e-4` | 16 | 0.10 | 768 | 4 | 1 | 1536 | 64 | 1.0 |
| 4 | `1e-4` | `1e-4` | 32 | 0.10 | 768 | 8 | 1 | 1536 | 64 | 1.0 |
| 5 | `1e-4` | `1e-4` | 32 | 0.30 | 768 | 8 | 1 | 1536 | 96 | 1.0 |
| 6 | `1e-4` | `1e-4` | 48 | 0.10 | 768 | 8 | 2 | 1536 | 64 | 1.0 |
| 7 **selected** | `1e-4` | `1e-4` | 48 | 0.10 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 8 | `1e-4` | `5e-5` | 48 | 0.10 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 9 | `1e-4` | `1e-4` | 48 | 0.30 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 10 | `1e-4` | `1e-4` | 32 | 0.10 | 768 | 8 | 3 | 1536 | 96 | 1.0 |
| 11 | `5e-5` | `5e-5` | 32 | 0.10 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 12 | `5e-5` | `1e-4` | 32 | 0.10 | 768 | 8 | 2 | 1536 | 128 | 1.0 |
| 13 | `5e-5` | `5e-5` | 48 | 0.10 | 768 | 8 | 2 | 1536 | 96 | 0.5 |
| 14 | `5e-5` | `5e-5` | 48 | 0.30 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 15 | `1e-5` | `1e-5` | 32 | 0.10 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 16 | `1e-5` | `5e-5` | 32 | 0.30 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 17 | `1e-4` | `1e-4` | 48 | 0.10 | 768 | 4 | 2 | 1536 | 96 | 1.0 |
| 18 | `1e-4` | `1e-4` | 48 | 0.10 | 768 | 8 | 2 | 3072 | 96 | 1.0 |
| 19 | `5e-5` | `5e-5` | 48 | 0.50 | 768 | 8 | 2 | 1536 | 96 | 1.0 |
| 20 | `1e-4` | `1e-4` | 48 | 0.30 | 768 | 8 | 3 | 3072 | 128 | 1.0 |

The corresponding Self-Attention NoGate condition uses the selected matched
architecture and optimization settings, with the token-level pre-gating
mechanism removed. This keeps the comparison focused on the gating ablation
rather than on independent tuning differences between gated and ungated models.

## Cross-Attention PreGate Manual Search Candidates

The cross-attention search varied the cross-attention projection dimension,
number of cross-attention layers, fusion dimension, dropout, and learning rate.
The selected setting is marked below.

| Config | LR | Batch | Dropout | Hidden | Cross Dim | Layers | FFN Size | Fusion Dim | Gate Setting |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `1e-3` | 16 | 0.10 | 512 | 128 | 1 | 1024 | 128 | pre-gate |
| 2 | `1e-3` | 32 | 0.30 | 512 | 256 | 1 | 1024 | 256 | pre-gate |
| 3 | `1e-4` | 16 | 0.10 | 768 | 128 | 1 | 1536 | 128 | pre-gate |
| 4 | `1e-4` | 32 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 5 | `1e-4` | 32 | 0.30 | 768 | 128 | 2 | 1536 | 256 | pre-gate |
| 6 | `5e-5` | 32 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 7 | `5e-5` | 48 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 8 **selected** | `3e-5` | 48 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 9 | `3e-5` | 48 | 0.30 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 10 | `3e-5` | 32 | 0.10 | 768 | 256 | 2 | 1536 | 128 | pre-gate |
| 11 | `3e-5` | 48 | 0.10 | 768 | 256 | 2 | 1536 | 256 | pre-gate |
| 12 | `1e-5` | 32 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 13 | `1e-5` | 48 | 0.10 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 14 | `1e-5` | 48 | 0.30 | 768 | 128 | 2 | 1536 | 256 | pre-gate |
| 15 | `5e-5` | 48 | 0.50 | 768 | 128 | 2 | 1536 | 128 | pre-gate |
| 16 | `5e-5` | 32 | 0.10 | 768 | 512 | 1 | 1536 | 256 | pre-gate |
| 17 | `1e-4` | 48 | 0.10 | 768 | 128 | 3 | 1536 | 128 | pre-gate |
| 18 | `3e-5` | 48 | 0.10 | 768 | 128 | 3 | 1536 | 128 | pre-gate |
| 19 | `3e-5` | 48 | 0.10 | 768 | 128 | 2 | 3072 | 128 | pre-gate |
| 20 | `3e-5` | 48 | 0.30 | 768 | 256 | 2 | 3072 | 256 | pre-gate |

The corresponding Cross-Attention NoGate condition uses the selected matched
architecture and optimization settings, with the pre-softmax gating mechanism
removed. This keeps the comparison focused on the gating ablation rather than
on independent tuning differences between gated and ungated models.

## Notes on Ablation Interpretation

Within each architecture family, NoGate and PreGate share the selected optimizer,
hidden dimensions, depth, dropout, and A/V BiLSTM encoder settings. The only
intended architectural difference is the removal or inclusion of the gating
mechanism. The search is therefore reported as limited manual tuning for the
gated model family, followed by matched ablation settings for the ungated
comparison.
