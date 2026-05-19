#!/usr/bin/env python3
"""Compute self-attention inter-modality entropy on the test split.

The exported per-sample table contains the final-layer metric used by the
entropy overview figure:

    self_inter_norm_entropy

For each valid query token, attention assigned to keys from the other
modalities is renormalized and converted to normalized entropy. Query-level
values are then averaged to obtain one score per sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast


def find_code_root() -> Path:
    # Allow the script to run either from the repository root or from this
    # bundled analysis directory.
    env_root = os.environ.get("MOSEI_CODE_ROOT")
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root).expanduser())

    here = Path(__file__).resolve()
    candidates.extend(
        [
            here.parents[2],
            here.parents[2] / "acl-style-files-master",
            here.parents[1],
            here.parent,
            Path.cwd(),
        ]
    )
    required_files = [
        "mosei_data_eval_utils.py",
        "mosei_modeling_utils.py",
        "train_self_attention_NoGate.py",
        "train_self_attention_PreGate.py",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in required_files):
            return candidate
    return candidates[0]


CODE_ROOT = find_code_root()
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import mosei_data_eval_utils as common
import mosei_modeling_utils as mmc
import train_self_attention_NoGate as self_no
import train_self_attention_PreGate as self_pre


EPS = 1e-12
CONFIG_NAME_KEYS = ("config_name", "final" + "_config_name")


@dataclass
class ConditionSpec:
    key: str
    label: str
    results_path: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export self-attention inter-modality normalized entropy.")
    parser.add_argument("--none-results", type=Path, required=True)
    parser.add_argument("--none-checkpoint", type=Path, required=True)
    parser.add_argument("--pre-results", type=Path, required=True)
    parser.add_argument("--pre-checkpoint", type=Path, required=True)
    parser.add_argument("--feature-cache-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layer", type=int, default=-1, help="Layer index to analyze; -1 means final layer.")
    parser.add_argument(
        "--mask-modes",
        nargs="+",
        default=["include_special"],
        choices=["include_special", "content_only"],
    )
    # Accepted by shared runner scripts; these options do not affect the exported metrics.
    parser.add_argument("--none-predictions", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--pre-predictions", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-iters", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--bootstrap-seed", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def read_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_test_split(
    base_config: Dict[str, object],
    feature_cache_path: Path,
) -> Tuple[Dict[str, object], mmc.MoseiMultimodalDataset, Dict[str, object]]:
    cfg = dict(base_config)
    cfg["feature_cache_path"] = str(feature_cache_path)
    cfg["rebuild_feature_cache"] = False
    cfg.setdefault("dataset_name", common.DEFAULT_DATASET_NAME)
    cfg.setdefault("mosei_compseq_dir", common.DEFAULT_MOSEI_COMPSEQ_DIR)
    cfg.setdefault("audio_csd", common.DEFAULT_AUDIO_CSD)
    cfg.setdefault("visual_csd", common.DEFAULT_VISUAL_CSD)
    cfg.setdefault("apply_av_zscore", True)
    cfg.setdefault("text_field", "text")
    cfg.setdefault("fallback_to_asr_for_char_spaced", True)

    # The saved training configuration defines the tokenization and feature
    # cache settings; only the test split is needed for this analysis.
    raw = load_dataset(str(cfg["dataset_name"]))
    tokenizer = BertTokenizerFast.from_pretrained(str(cfg["text_model_name"]))
    splits, reports = common.load_or_build_av_feature_cache(SimpleNamespace(**cfg))
    test_split = splits["test"]
    test_dataset = mmc.MoseiMultimodalDataset(
        raw["test"],
        test_split,
        tokenizer,
        int(cfg["max_tlen"]),
        str(cfg["text_field"]),
        bool(cfg["fallback_to_asr_for_char_spaced"]),
    )
    return test_split, test_dataset, reports


def self_module_for_condition(condition_key: str, results_data: Dict[str, object]):
    config_name = str(next((results_data.get(key) for key in CONFIG_NAME_KEYS if results_data.get(key)), "")).lower()
    if condition_key == "pregate" or "pregate" in config_name or "token" in config_name:
        return self_pre
    return self_no


def instantiate_model(
    condition_key: str,
    results_data: Dict[str, object],
    checkpoint_path: Path,
    visual_dim: int,
    audio_dim: int,
    device: torch.device,
) -> torch.nn.Module:
    # Recreate the trained architecture before loading checkpoint weights.
    cfg = results_data["config"]
    module = self_module_for_condition(condition_key, results_data)
    cls = module.SelfAttentionTokenGateRegressor if module is self_pre else module.SelfAttentionRegressor
    model = cls(SimpleNamespace(**cfg), visual_dim=visual_dim, audio_dim=audio_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.no_grad()
def run_self_attention_layer_with_attn(
    layer: torch.nn.Module,
    hidden_state: torch.Tensor,
    valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block = layer.self_attn
    q = block._reshape_heads(block.q_proj(hidden_state))
    k = block._reshape_heads(block.k_proj(hidden_state))
    v = block._reshape_heads(block.v_proj(hidden_state))
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(block.head_dim)
    if hasattr(block, "_apply_token_gate"):
        scores = block._apply_token_gate(scores, hidden_state)

    pair_mask = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
    attn = block._masked_softmax(scores, pair_mask.unsqueeze(1))
    context = block._merge_heads(torch.matmul(block.attn_dropout(attn), v))
    context = block.hidden_dropout(block.out_proj(context))
    hidden_state = block.layer_norm(context + hidden_state)
    hidden_state = hidden_state * valid_mask.unsqueeze(-1).to(hidden_state.dtype)

    ff = layer.linear2(layer.activation(layer.linear1(hidden_state)))
    hidden_state = layer.layer_norm(hidden_state + layer.dropout_1(ff))
    hidden_state = layer.dropout_2(hidden_state)
    hidden_state = hidden_state * valid_mask.unsqueeze(-1).to(hidden_state.dtype)
    return hidden_state, attn


@torch.no_grad()
def forward_with_attention(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]:
    input_ids = batch["input_ids"]
    text_mask = batch["text_mask"].bool()
    visual = batch["visual"]
    visual_mask = batch["visual_mask"].bool()
    audio = batch["audio"]
    audio_mask = batch["audio_mask"].bool()
    bsz = input_ids.shape[0]

    text_hidden, text_valid = model.text_encoder(input_ids, text_mask)
    visual_hidden, visual_valid = model.visual_encoder(visual, visual_mask)
    audio_hidden, audio_valid = model.audio_encoder(audio, audio_mask)
    multimodal = torch.cat([text_hidden, visual_hidden, audio_hidden], dim=1)
    valid_mask = torch.cat([text_valid, visual_valid, audio_valid], dim=1).bool()
    hidden_state = multimodal
    all_attn: List[torch.Tensor] = []
    for layer in model.encoder.layers:
        hidden_state, attn = run_self_attention_layer_with_attn(layer, hidden_state, valid_mask)
        all_attn.append(attn)
    return all_attn, {"text": text_valid, "visual": visual_valid, "audio": audio_valid}


def make_modality_masks(
    batch: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
    model: torch.nn.Module,
    mask_mode: str,
) -> List[torch.Tensor]:
    # ``include_special`` keeps the learned sequence-level slot; ``content_only``
    # restricts the analysis to lexical tokens and observed A/V frames.
    text_mask = masks["text"].clone().bool()
    visual_mask = masks["visual"].clone().bool()
    audio_mask = masks["audio"].clone().bool()
    if mask_mode == "content_only":
        text_mask = text_mask & (batch["input_ids"] != 0) & (batch["input_ids"] != 101) & (batch["input_ids"] != 102)
        if visual_mask.shape[1] > 0:
            visual_mask[:, 0] = False
        if audio_mask.shape[1] > 0:
            audio_mask[:, 0] = False

    bsz = text_mask.shape[0]
    total_len = model.max_tlen + model.max_vlen + model.max_alen
    full_masks = [torch.zeros((bsz, total_len), dtype=torch.bool, device=text_mask.device) for _ in range(3)]
    full_masks[0][:, : model.max_tlen] = text_mask
    full_masks[1][:, model.max_tlen : model.max_tlen + model.max_vlen] = visual_mask
    full_masks[2][:, model.max_tlen + model.max_vlen :] = audio_mask
    return full_masks


def safe_float(value: torch.Tensor) -> float:
    out = float(value.detach().cpu().item())
    return out if math.isfinite(out) else float("nan")


def compute_self_inter_entropy(attn_mean: torch.Tensor, modality_masks: List[torch.Tensor]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for b_idx in range(attn_mean.shape[0]):
        entropy_values: List[torch.Tensor] = []
        entropy_counts: List[int] = []
        for q_mod in range(3):
            q_mask = modality_masks[q_mod][b_idx]
            if int(q_mask.sum().item()) == 0:
                continue
            inter_key_mask = torch.zeros_like(q_mask)
            for k_mod in range(3):
                if k_mod != q_mod:
                    inter_key_mask = inter_key_mask | modality_masks[k_mod][b_idx]
            k_count = int(inter_key_mask.sum().item())
            if k_count <= 1:
                continue

            # Self-attention rows are full multimodal distributions; isolate the
            # cross-modal part and renormalize it before computing entropy.
            probs = attn_mean[b_idx][q_mask][:, inter_key_mask]
            row_mass = probs.sum(dim=-1, keepdim=True)
            valid_rows = row_mass.squeeze(-1) > EPS
            if not bool(valid_rows.any()):
                continue
            probs = probs[valid_rows] / row_mass[valid_rows].clamp_min(EPS)
            ent = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1) / math.log(k_count)
            entropy_values.append(ent.sum())
            entropy_counts.append(int(ent.numel()))

        if entropy_counts:
            entropy = safe_float(torch.stack(entropy_values).sum() / float(sum(entropy_counts)))
        else:
            entropy = float("nan")
        rows.append({"batch_index": float(b_idx), "self_inter_norm_entropy": entropy})
    return rows


@torch.no_grad()
def analyze_condition(
    spec: ConditionSpec,
    model: torch.nn.Module,
    test_split: Dict[str, object],
    test_dataset: mmc.MoseiMultimodalDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    mask_modes: Sequence[str],
    layer_idx: int,
) -> List[Dict[str, object]]:
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows: List[Dict[str, object]] = []
    if hasattr(model, "reset_running_gate_stats"):
        model.reset_running_gate_stats()

    for step, batch_cpu in enumerate(loader, start=1):
        batch = batch_to_device(batch_cpu, device)
        all_attn, masks = forward_with_attention(model, batch)
        attn_mean = all_attn[layer_idx].mean(dim=1)
        sample_indices = batch_cpu["sample_idx"].detach().cpu().numpy()
        hf_indices = batch_cpu["hf_index"].detach().cpu().numpy()
        scores = batch_cpu["score"].detach().cpu().numpy()

        for mask_mode in mask_modes:
            modality_masks = make_modality_masks(batch, masks, model, mask_mode)
            for metric_row in compute_self_inter_entropy(attn_mean, modality_masks):
                b_idx = int(metric_row.pop("batch_index"))
                sample_idx = int(sample_indices[b_idx])
                rows.append(
                    {
                        "condition": spec.key,
                        "condition_label": spec.label,
                        "mask_mode": mask_mode,
                        "layer": layer_idx,
                        "sample_idx": sample_idx,
                        "hf_index": int(hf_indices[b_idx]),
                        "seg_key": test_split["seg_keys"][sample_idx] if sample_idx < len(test_split["seg_keys"]) else "",
                        "score_true": float(scores[b_idx]),
                        **metric_row,
                    }
                )
        if step % 25 == 0:
            print(f"[{spec.key}] processed {step}/{len(loader)} batches", flush=True)
    return rows


def write_csv(rows: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def finite_mean_sd(values: Iterable[object]) -> Tuple[int, float, float]:
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return 0, float("nan"), float("nan")
    mean = sum(vals) / len(vals)
    sd = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals)) if len(vals) > 1 else 0.0
    return len(vals), mean, sd


def summarize(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    groups = sorted({(r["condition"], r["condition_label"], r["mask_mode"], r["layer"]) for r in rows})
    for condition, condition_label, mask_mode, layer in groups:
        selected = [
            r
            for r in rows
            if r["condition"] == condition and r["mask_mode"] == mask_mode and int(r["layer"]) == int(layer)
        ]
        n, mean_value, sd_value = finite_mean_sd(r["self_inter_norm_entropy"] for r in selected)
        out.append(
            {
                "condition": condition,
                "condition_label": condition_label,
                "mask_mode": mask_mode,
                "layer": layer,
                "metric": "self_inter_norm_entropy",
                "n": n,
                "mean": mean_value,
                "sd": sd_value,
            }
        )
    return out


def selected_layer(args: argparse.Namespace, num_layers: int) -> int:
    layer = num_layers - 1 if args.layer < 0 else args.layer
    if layer < 0 or layer >= num_layers:
        raise ValueError(f"Layer must be in [0, {num_layers - 1}] or -1, got {args.layer}")
    return layer


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    specs = [
        ConditionSpec("none", "NoGate", args.none_results, args.none_checkpoint),
        ConditionSpec("pregate", "PreGate", args.pre_results, args.pre_checkpoint),
    ]
    for spec in specs:
        for path in [spec.results_path, spec.checkpoint_path]:
            if not path.exists():
                raise FileNotFoundError(f"Missing required file for {spec.key}: {path}")
    if not args.feature_cache_path.exists():
        raise FileNotFoundError(f"Missing feature cache: {args.feature_cache_path}")

    results_data = {spec.key: read_json(spec.results_path) for spec in specs}
    test_split, test_dataset, reports = load_test_split(dict(results_data["none"]["config"]), args.feature_cache_path)
    visual_dim = int(test_split["visual"].shape[-1])
    audio_dim = int(test_split["audio"].shape[-1])
    models = {
        spec.key: instantiate_model(spec.key, results_data[spec.key], spec.checkpoint_path, visual_dim, audio_dim, device)
        for spec in specs
    }
    layer_idx = selected_layer(args, len(models["none"].encoder.layers))
    print(f"Device: {device}")
    print(f"Test samples: {len(test_split['scores'])} | layer={layer_idx}")

    all_rows: List[Dict[str, object]] = []
    for spec in specs:
        print(f"Analyzing {spec.label} ({spec.key})", flush=True)
        all_rows.extend(
            analyze_condition(
                spec=spec,
                model=models[spec.key],
                test_split=test_split,
                test_dataset=test_dataset,
                device=device,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                mask_modes=args.mask_modes,
                layer_idx=layer_idx,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    per_sample_path = args.output_dir / "self_attention_contribution_per_sample.csv"
    summary_path = args.output_dir / "self_attention_contribution_summary.csv"
    write_csv(all_rows, per_sample_path)
    write_csv(summarize(all_rows), summary_path)

    metadata = {
        "metric_purpose": "self-attention inter-modality entropy",
        "selected_layer": layer_idx,
        "mask_modes": list(args.mask_modes),
        "outputs": {"per_sample": str(per_sample_path), "summary": str(summary_path)},
        "notes": {
            "self_inter_norm_entropy": (
                "For each query row, keys from other modalities are retained, "
                "renormalized, and evaluated with entropy/log(K_inter)."
            )
        },
        "test_reports": reports,
    }
    with (args.output_dir / "self_attention_contribution_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved self-attention entropy outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
