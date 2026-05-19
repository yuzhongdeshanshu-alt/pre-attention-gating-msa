#!/usr/bin/env python3
"""Compute cross-attention inter-modality entropy on the test split.

The exported per-sample table contains the final-layer metric used by the
entropy overview figure:

    aggregate_pair_equal_norm_entropy

Each target-to-source cross-attention direction has its own softmax. The script
computes normalized entropy within each direction and reports the simple mean
across the six cross-modal directions for each sample.
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
        "train_cross_attention_NoGate.py",
        "train_cross_attention_PreGate.py",
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
import train_cross_attention_NoGate as cross_no
import train_cross_attention_PreGate as cross_pre


EPS = 1e-12
CONFIG_NAME_KEYS = ("config_name", "final" + "_config_name")
DIRECTIONAL_PAIRS = cross_no.DIRECTIONAL_PAIRS


@dataclass
class ConditionSpec:
    key: str
    label: str
    results_path: Path
    checkpoint_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export simple-average cross-attention normalized entropy.")
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


def cross_module_for_condition(condition_key: str, results_data: Dict[str, object]):
    config_name = str(next((results_data.get(key) for key in CONFIG_NAME_KEYS if results_data.get(key)), "")).lower()
    if condition_key == "pre" or "pregate" in config_name or "pre" in config_name:
        return cross_pre
    return cross_no


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
    module = cross_module_for_condition(condition_key, results_data)
    model = module.CrossAttentionRegressor(SimpleNamespace(**cfg), visual_dim=visual_dim, audio_dim=audio_dim)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def masks_for_mode(
    batch: Dict[str, torch.Tensor],
    modal_masks: Dict[str, torch.Tensor],
    mask_mode: str,
) -> Dict[str, torch.Tensor]:
    # ``include_special`` keeps the learned sequence-level slot; ``content_only``
    # restricts the analysis to lexical tokens and observed A/V frames.
    masks = {name: mask.clone().bool() for name, mask in modal_masks.items()}
    if mask_mode == "content_only":
        masks["text"] = masks["text"] & (batch["input_ids"] != 0) & (batch["input_ids"] != 101) & (batch["input_ids"] != 102)
        if masks["visual"].shape[1] > 0:
            masks["visual"][:, 0] = False
        if masks["audio"].shape[1] > 0:
            masks["audio"][:, 0] = False
    return masks


def safe_float(value: torch.Tensor) -> float:
    out = float(value.detach().cpu().item())
    return out if math.isfinite(out) else float("nan")


@torch.no_grad()
def encode_modalities(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    text_hidden, text_valid = model.text_encoder(batch["input_ids"], batch["text_mask"].bool())
    visual_hidden, visual_valid = model.visual_encoder(batch["visual"], batch["visual_mask"].bool())
    audio_hidden, audio_valid = model.audio_encoder(batch["audio"], batch["audio_mask"].bool())
    return (
        {"text": text_hidden, "visual": visual_hidden, "audio": audio_hidden},
        {"text": text_valid, "visual": visual_valid, "audio": audio_valid},
    )


@torch.no_grad()
def run_cross_layer_with_attn(
    layer: torch.nn.Module,
    modal_hidden: Dict[str, torch.Tensor],
    modal_masks: Dict[str, torch.Tensor],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    updated_hidden: Dict[str, torch.Tensor] = {}
    attn_by_pair: Dict[str, torch.Tensor] = {}

    for modality in cross_no.MODALITIES:
        query_hidden = modal_hidden[modality]
        query_mask = modal_masks[modality]
        cross_sum = None

        for pair_label, query_name, source_name in DIRECTIONAL_PAIRS:
            if query_name != modality:
                continue
            source_hidden = modal_hidden[source_name]
            source_mask = modal_masks[source_name]
            q = layer.q_proj[query_name](query_hidden)
            k = layer.k_proj[source_name](source_hidden)
            v = layer.v_proj[source_name](source_hidden)
            scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(layer.cross_attn_dim)
            pair_mask = query_mask.unsqueeze(-1) & source_mask.unsqueeze(1)
            if hasattr(layer, "_compute_pre_pair_prob"):
                scores = scores * layer._compute_pre_pair_prob(q, k)
            attn = layer._masked_softmax(scores, pair_mask)
            attended = torch.matmul(layer.attn_dropout(attn), v)
            attended = attended * query_mask.unsqueeze(-1).to(attended.dtype)
            attn_by_pair[pair_label] = attn
            cross_sum = attended if cross_sum is None else cross_sum + attended

        if cross_sum is None:
            raise RuntimeError(f"Failed to build cross attention for modality: {modality}")
        attn_update = layer.hidden_dropout(layer.attn_out_proj[modality](cross_sum))
        hidden = layer.attn_norm[modality](query_hidden + attn_update)
        ff = layer.ffn_out[modality](layer.activation(layer.ffn_in[modality](hidden)))
        hidden = layer.ffn_norm[modality](hidden + layer.hidden_dropout(ff))
        hidden = hidden * query_mask.unsqueeze(-1).to(hidden.dtype)
        updated_hidden[modality] = hidden

    return updated_hidden, attn_by_pair


def pair_norm_entropy(attn: torch.Tensor, query_mask: torch.Tensor, source_mask: torch.Tensor) -> float:
    q_mask = query_mask.bool()
    s_mask = source_mask.bool()
    source_count = int(s_mask.sum().item())
    if int(q_mask.sum().item()) == 0 or source_count == 0:
        return float("nan")
    if source_count == 1:
        return 0.0

    probs = attn[q_mask][:, s_mask]
    row_mass = probs.sum(dim=-1, keepdim=True)
    valid_rows = row_mass.squeeze(-1) > EPS
    if not bool(valid_rows.any()):
        return float("nan")
    probs = probs[valid_rows] / row_mass[valid_rows].clamp_min(EPS)
    raw_entropy = -(probs * torch.log(probs.clamp_min(EPS))).sum(dim=-1)
    return safe_float((raw_entropy / math.log(source_count)).mean())


def mean_finite(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


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
        modal_hidden, modal_masks = encode_modalities(model, batch)
        sample_indices = batch_cpu["sample_idx"].detach().cpu().numpy()
        hf_indices = batch_cpu["hf_index"].detach().cpu().numpy()
        scores = batch_cpu["score"].detach().cpu().numpy()

        for current_layer_idx, layer in enumerate(model.fusion.layers):
            updated_hidden, attn_by_pair = run_cross_layer_with_attn(layer, modal_hidden, modal_masks)
            if current_layer_idx == layer_idx:
                # Direction-level entropies are averaged equally because the
                # figure measures overall cross-modal selectivity rather than
                # direction-specific contribution.
                for mask_mode in mask_modes:
                    mode_masks = masks_for_mode(batch, modal_masks, mask_mode)
                    per_sample_entropies: Dict[int, List[float]] = {b_idx: [] for b_idx in range(len(sample_indices))}
                    for pair_label, query_name, source_name in DIRECTIONAL_PAIRS:
                        attn = attn_by_pair[pair_label]
                        q_masks = mode_masks[query_name]
                        s_masks = mode_masks[source_name]
                        for b_idx in range(attn.shape[0]):
                            per_sample_entropies[b_idx].append(pair_norm_entropy(attn[b_idx], q_masks[b_idx], s_masks[b_idx]))

                    for b_idx, entropies in per_sample_entropies.items():
                        sample_idx = int(sample_indices[b_idx])
                        entropy = mean_finite(entropies)
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
                                "aggregate_pair_equal_norm_entropy": entropy,
                                "direction_mean_norm_entropy": entropy,
                            }
                        )
            modal_hidden = updated_hidden

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
        n, mean_value, sd_value = finite_mean_sd(r["aggregate_pair_equal_norm_entropy"] for r in selected)
        out.append(
            {
                "condition": condition,
                "condition_label": condition_label,
                "mask_mode": mask_mode,
                "layer": layer,
                "metric": "aggregate_pair_equal_norm_entropy",
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
        ConditionSpec("pre", "PreGate", args.pre_results, args.pre_checkpoint),
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
    layer_idx = selected_layer(args, len(models["none"].fusion.layers))
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
    per_sample_path = args.output_dir / "cross_attention_contribution_aggregate_per_sample.csv"
    summary_path = args.output_dir / "cross_attention_contribution_aggregate_summary.csv"
    write_csv(all_rows, per_sample_path)
    write_csv(summarize(all_rows), summary_path)

    metadata = {
        "metric_purpose": "cross-attention inter-modality entropy",
        "selected_layer": layer_idx,
        "mask_modes": list(args.mask_modes),
        "outputs": {"aggregate_per_sample": str(per_sample_path), "aggregate_summary": str(summary_path)},
        "notes": {
            "aggregate_pair_equal_norm_entropy": (
                "Simple average of normalized entropy across the six target-to-source "
                "cross-attention directions."
            )
        },
        "test_reports": reports,
    }
    with (args.output_dir / "cross_attention_contribution_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Saved cross-attention entropy outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
