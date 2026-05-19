# CMU-MOSEI Multimodal Sentiment Regression

This repository contains the code, selected hyperparameter settings, result summaries, statistical analyses, and attention analyses for CMU-MOSEI sentiment regression experiments. The project compares unimodal baselines with self-attention and cross-attention multimodal models, with and without pre-softmax gating.

The repository is organized to support two use cases:

1. Reproduce the training and evaluation pipeline from the provided scripts.
2. Recreate the reported figures and analysis tables from the included CSV/JSON result files.

## Repository Structure

```text
.
├── train_text_bert_unimodal.py
├── train_av_bilstm_unimodal.py
├── train_self_attention_NoGate.py
├── train_self_attention_PreGate.py
├── train_cross_attention_NoGate.py
├── train_cross_attention_PreGate.py
├── mosei_data_eval_utils.py
├── mosei_modeling_utils.py
├── 10 runs Statistical Analysis/
├── Attention Analysis/
├── Seed 10 Training Plot and Results/
├── docs/
├── MULTIMODAL_HYPERPARAMETER_SEARCH.md
├── UNIMODAL_HYPERPARAMETER_SEARCH.md
├── requirements.txt
└── README.md
```

### Core Training Scripts

- `train_text_bert_unimodal.py`: text-only BERT regression baseline.
- `train_av_bilstm_unimodal.py`: audio-only and visual-only BiLSTM baselines.
- `train_self_attention_NoGate.py`: multimodal self-attention model without pre-gating.
- `train_self_attention_PreGate.py`: multimodal self-attention model with pre-gating.
- `train_cross_attention_NoGate.py`: multimodal cross-attention model without pre-gating.
- `train_cross_attention_PreGate.py`: multimodal cross-attention model with pre-gating.

### Shared Utilities

- `mosei_data_eval_utils.py`: dataset loading, CSD feature caching, label handling, evaluation metrics, and output writers.
- `mosei_modeling_utils.py`: shared PyTorch datasets, BERT sequence encoding, A/V BiLSTM sequence encoding, and regression heads.

### Analysis and Result Folders

- `Seed 10 Training Plot and Results/`: training curves from one seed, result JSON files, test prediction CSVs, and per-condition test summaries for demonstration.
- `10 runs Statistical Analysis/`: 10-run aggregate result CSV, statistical analysis script, ANOVA/t-test tables, and summary plots.
- `Attention Analysis/`: final-layer attention extraction scripts, assembled per-sample attention CSVs, and attention visualization figures.

## Environment Setup

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch installation is recommended for training on GPU. The scripts also run on CPU for inspection or small smoke tests, but full training is intended for a compute environment such as UCL Myriad HPC.

## Data

The experiments use CMU-MOSEI sentiment regression labels together with aligned audio and visual computational sequence features.

### Text and Labels

Raw text data and sentiment labels are obtained from the HuggingFace dataset [`vintp/CMU-Mosei-text`](https://huggingface.co/datasets/vintp/CMU-Mosei-text). The dataset provides the train/validation/test utterance splits used by the scripts:

- train: 16,274 utterances
- validation: 1,861 utterances
- test: 4,653 utterances

Each example contains the video id, utterance start/end timestamps, sentiment score, emotion labels, manual transcript, and ASR transcript. The text baseline uses the transcript fields and the continuous `sentiment` label. The audio and visual baselines use the video id and timestamps to align utterance-level labels with frame-level CSD features.

### Audio and Visual Features

The audio and visual baselines use CMU-MOSEI computational sequence files in `.csd` format. These files follow the [CMU Multimodal SDK](https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK) data format. In our experiments:

- audio features are read from `CMU_MOSEI_COVAREP.csd`
- visual features are read from `CMU_MOSEI_OpenFace2.csd`

The scripts expect the following data structure by default:

```text
data/mosei_comp_seq/data/
  CMU_MOSEI_COVAREP.csd
  CMU_MOSEI_OpenFace2.csd
```

Only these two CSD files are required for the audio and visual baselines. Other computational sequence files, such as labels, timestamped words, or word vectors, may be present in the same directory but are not required by these scripts.

If the CSD files are stored elsewhere, pass the path explicitly:

```bash
--mosei_compseq_dir /path/to/mosei_comp_seq/data
```

The A/V feature cache is built automatically on first use unless an existing cache path is supplied.

For a brief comparison of related multimodal affective datasets and the rationale for selecting CMU-MOSEI, see [docs/dataset_comparison.md](docs/dataset_comparison.md).

## Training Examples

### Text Baseline

```bash
python train_text_bert_unimodal.py \
  --output_dir outputs/text_bert
```

### Visual BiLSTM Baseline

```bash
python train_av_bilstm_unimodal.py \
  --modality visual \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/visual_bilstm
```

### Audio BiLSTM Baseline

```bash
python train_av_bilstm_unimodal.py \
  --modality audio \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/audio_bilstm
```

### Self-Attention Models

```bash
python train_self_attention_NoGate.py \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/self_attention/NoGate

python train_self_attention_PreGate.py \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/self_attention/PreGate
```

### Cross-Attention Models

```bash
python train_cross_attention_NoGate.py \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/cross_attention/NoGate

python train_cross_attention_PreGate.py \
  --mosei_compseq_dir data/mosei_comp_seq/data \
  --output_dir outputs/cross_attention/PreGate
```

Each training run writes a checkpoint, results JSON, test predictions CSV, and training-curve PNG to the selected output directory.

## Selected Hyperparameters

The selected hyperparameters are defined in the `MODEL_DEFAULTS`, `TEXT_DEFAULTS`, or `MODALITY_DEFAULTS` dictionaries at the top of the training scripts.

For a compact record of the manual search process, see:

- `UNIMODAL_HYPERPARAMETER_SEARCH.md`
- `MULTIMODAL_HYPERPARAMETER_SEARCH.md`

The multimodal search was a limited manual search conducted on UCL Myriad HPC. For ablation fairness, hyperparameter selection was performed under the pre-gating condition within each architecture family, and the corresponding NoGate model used the matched architecture and optimization settings with the gate removed.

## Recreating Result Tables and Figures

### Training Curves and Test Summaries

The folder `Seed 10 Training Plot and Results/` contains training outputs from one seed for the seven reported models. These files demonstrate the training curves and per-model test summaries used for inspection:

- Text-only BERT
- Audio-only BiLSTM
- Visual-only BiLSTM
- Self-Attention NoGate
- Self-Attention PreGate
- Cross-Attention NoGate
- Cross-Attention PreGate

Each condition folder includes a `test_results_summary.csv` table extracted from the corresponding result JSON.

### 10-Run Statistical Analysis

To recreate the statistical analysis from the included 10-run CSV:

```bash
cd "10 runs Statistical Analysis"
python statistical_analysis_v2.py \
  --input raw_all_seeds.csv \
  --output output_stats
```

This produces descriptive statistics, repeated-measures ANOVA results, planned paired t-tests with Holm-Bonferroni correction, and summary figures.

### Attention Analysis

The folder `Attention Analysis/` contains final-layer attention analysis scripts and the assembled per-sample CSVs used for plotting. To recreate the included attention figures directly from the provided CSVs:

```bash
cd "Attention Analysis"
python plot_final_layer_attention_figures.py
```

This writes:

- `01_entropy_overview.png`
- `02_self_macro_mass.png`

See `Attention Analysis/README.md` for details on how the final-layer per-sample attention CSVs are produced from checkpoint-based extraction scripts.

## Notes on Reproducibility

- Training outputs may vary slightly across hardware, CUDA/cuDNN versions, and PyTorch versions.
- The included result folders allow the reported statistical analyses and figures to be recreated without rerunning model training.
- Model checkpoints and feature caches are not included because they are large intermediate artefacts. They are not required for recreating the reported statistical tables and attention-summary figures from the provided CSV/JSON files, and can be regenerated by rerunning the corresponding training or extraction scripts.
