# Attention Analysis

This directory contains the scripts and per-sample CSV files used to analyze final-layer attention behavior in the multimodal models. The included CSV files in `final_layer_per_sample_attention/` are already assembled and can be plotted directly.

## Directory Contents

- `analyze_self_attention_mass.py`: extracts self-attention macro allocation metrics from the NoGate and PreGate self-attention checkpoints.
- `analyze_self_inter_modality_entropy.py`: extracts self-attention inter-modality normalized entropy from the NoGate and PreGate self-attention checkpoints.
- `analyze_cross_inter_modality_entropy.py`: extracts cross-attention inter-modality normalized entropy from the NoGate and PreGate cross-attention checkpoints.
- `build_final_layer_per_sample_attention.py`: combines the per-sample extraction outputs into the four CSV files used for plotting.
- `plot_final_layer_attention_figures.py`: reads the four assembled CSV files and creates the two attention summary figures.
- `final_layer_per_sample_attention/`: contains the four assembled per-sample CSV files used by the plotting script.

## Metrics

### Micro-Level Inter-Modality Entropy

Micro-level selectivity is measured with normalized entropy. For an attention distribution over `K` valid attended positions,

```text
NEnt(p) = - sum_k p_k log(p_k) / log(K)
```

The score is bounded between 0 and 1. Lower values indicate more selective attention, while higher values indicate more diffuse attention.

For self-attention, each query token can attend to all modalities. The script keeps only key positions from modalities different from the query modality, renormalizes those inter-modality weights to sum to one, and computes normalized entropy for each valid query row. Query-level values are averaged to obtain `self_inter_norm_entropy` for each sample.

For cross-attention, each target-to-source direction is already a cross-modal attention map. The script computes normalized entropy over valid source positions for each direction and then uses a simple average over the six directions. This measures overall cross-modal selectivity without assigning contribution weights to individual directions.

### Macro-Level Self-Attention Mass

Macro-level allocation is measured only for self-attention. For each valid query row, attention mass is grouped by whether keys belong to the same modality or to a different modality. The script reports:

- `row_weighted_intra`: average mass assigned to same-modality keys.
- `row_weighted_inter`: average mass assigned to other-modality keys.

The row-weighted average gives each valid query token or frame equal influence, rather than weighting modalities equally regardless of sequence length.

## Recomputing the Per-Sample Extraction Outputs

The three extraction scripts require the trained model result JSON files, checkpoint files, and the A/V feature cache used during training. They also import the training/model utilities from the repository. If the scripts are run from this bundled directory, they search upward for the required training files; `MOSEI_CODE_ROOT` can also be set explicitly.

Example command structure:

```bash
python analyze_self_attention_mass.py \
  --none-results /path/to/self_nogate_results.json \
  --none-checkpoint /path/to/self_nogate.pt \
  --pre-results /path/to/self_pregate_results.json \
  --pre-checkpoint /path/to/self_pregate.pt \
  --feature-cache-path /path/to/feature_cache.pkl \
  --output-dir /path/to/self_attention_mass_outputs \
  --layer -1

python analyze_self_inter_modality_entropy.py \
  --none-results /path/to/self_nogate_results.json \
  --none-checkpoint /path/to/self_nogate.pt \
  --pre-results /path/to/self_pregate_results.json \
  --pre-checkpoint /path/to/self_pregate.pt \
  --feature-cache-path /path/to/feature_cache.pkl \
  --output-dir /path/to/self_attention_entropy_outputs \
  --layer -1

python analyze_cross_inter_modality_entropy.py \
  --none-results /path/to/cross_nogate_results.json \
  --none-checkpoint /path/to/cross_nogate.pt \
  --pre-results /path/to/cross_pregate_results.json \
  --pre-checkpoint /path/to/cross_pregate.pt \
  --feature-cache-path /path/to/feature_cache.pkl \
  --output-dir /path/to/cross_attention_entropy_outputs \
  --layer -1
```

The main per-sample outputs are:

- `attention_allocation_per_sample.csv`
- `self_attention_contribution_per_sample.csv`
- `cross_attention_contribution_aggregate_per_sample.csv`

## Building the Four Plot-Input CSVs

Use `build_final_layer_per_sample_attention.py` after the three extraction outputs are available:

```bash
python build_final_layer_per_sample_attention.py \
  --self-allocation-csv /path/to/attention_allocation_per_sample.csv \
  --self-entropy-csv /path/to/self_attention_contribution_per_sample.csv \
  --cross-entropy-csv /path/to/cross_attention_contribution_aggregate_per_sample.csv \
  --final-csv-dir final_layer_per_sample_attention \
  --layer -1 \
  --mask-mode include_special
```

This writes:

- `final_layer_per_sample_attention/self_attention_NoGate_Final Layer.csv`
- `final_layer_per_sample_attention/self_attention_PreGate_Final Layer.csv`
- `final_layer_per_sample_attention/cross_attention_NoGate_Final Layer.csv`
- `final_layer_per_sample_attention/cross_attention_PreGate_Final Layer.csv`

These four CSVs are included in this directory, so plotting does not require checkpoints.

## Plotting the Included CSVs

To recreate the two figures from the included CSV files:

```bash
python plot_final_layer_attention_figures.py
```

By default, the script reads from `final_layer_per_sample_attention/` and writes the following files to this directory:

- `01_entropy_overview.png`
- `02_self_macro_mass.png`

Use `--input-dir` or `--output-dir` to override these paths.
