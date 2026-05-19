"""Shared data, metric, and output helpers for CMU-MOSEI experiments.

The utilities here cover text cleaning, evaluation metrics, computational
sequence loading, feature caching, and common result writers used by the
training scripts.
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score


DEFAULT_DATASET_NAME = "vintp/CMU-Mosei-text"
DEFAULT_MOSEI_COMPSEQ_DIR = "data/mosei_comp_seq/data"
DEFAULT_AUDIO_CSD = "CMU_MOSEI_COVAREP.csd"
DEFAULT_VISUAL_CSD = "CMU_MOSEI_OpenFace2.csd"


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_spaces(text: str) -> str:
    """Collapse repeated whitespace and strip non-breaking spaces."""
    return re.sub(r"\s+", " ", str(text).replace("\u00A0", " ")).strip()


def is_char_spaced(text: str, min_tokens: int = 6, ratio: float = 0.7) -> bool:
    """Detect transcripts where words appear as character-spaced tokens."""
    words = normalize_spaces(text).split()
    if len(words) < min_tokens:
        return False
    return sum(len(w) == 1 for w in words) / len(words) >= ratio


def clean_text(text: str) -> str:
    """Normalize a transcript and rebuild simple character-spaced words."""
    text = normalize_spaces(text)
    if not text or not is_char_spaced(text):
        return text

    rebuilt: List[str] = []
    run: List[str] = []

    def flush_run() -> None:
        """Flush the current sequence of one-character tokens."""
        nonlocal run
        if not run:
            return
        if len(run) >= 3:
            rebuilt.append("".join(run))
        else:
            rebuilt.extend(run)
        run = []

    for word in text.split():
        if len(word) == 1 and word.isalnum():
            run.append(word)
        else:
            flush_run()
            rebuilt.append(word)
    flush_run()
    return " ".join(rebuilt).strip()


def choose_text(row: dict, text_field: str = "text", fallback_to_asr_for_char_spaced: bool = False) -> str:
    """Select and clean the text field used by text-based models."""
    primary_raw = str(row.get(text_field, ""))
    primary_clean = clean_text(primary_raw)
    # Some MOSEI transcripts contain character-spaced text; ASR can provide a
    # more usable fallback when requested by the training script.
    if fallback_to_asr_for_char_spaced and text_field != "ASR" and is_char_spaced(primary_raw):
        asr_raw = str(row.get("ASR", "")).strip()
        if asr_raw:
            asr_clean = clean_text(asr_raw)
            if asr_clean:
                return asr_clean
    return primary_clean


def filter_neutral(split, neutral_eps: float):
    """Optionally remove samples with sentiment scores close to zero."""
    if neutral_eps <= 0:
        return split, 0
    keep_idx = [i for i, score in enumerate(split["sentiment"]) if abs(float(score)) > neutral_eps]
    return split.select(keep_idx), len(split) - len(keep_idx)


def load_mosei_text_splits(dataset_name: str, neutral_eps: float = 0.0, smoke_test: bool = False):
    """Load HuggingFace MOSEI train/validation/test splits."""
    raw = load_dataset(dataset_name)
    train_split = raw["train"]
    val_split = raw["validation"]
    test_split = raw["test"]
    if neutral_eps > 0:
        train_split, tr_drop = filter_neutral(train_split, neutral_eps)
        val_split, va_drop = filter_neutral(val_split, neutral_eps)
        test_split, te_drop = filter_neutral(test_split, neutral_eps)
        print(
            f"[FILTER] neutral_eps={neutral_eps:.3f} | "
            f"removed train/validation/test={tr_drop}/{va_drop}/{te_drop}"
        )
    if smoke_test:
        train_split = train_split.select(range(min(256, len(train_split))))
        val_split = val_split.select(range(min(64, len(val_split))))
        test_split = test_split.select(range(min(64, len(test_split))))
        print(f"[SMOKE] train={len(train_split)} validation={len(val_split)} test={len(test_split)}")
    return train_split, val_split, test_split


def multiclass_acc(preds: np.ndarray, truths: np.ndarray) -> float:
    """Compute 7-class accuracy after rounding continuous sentiment scores."""
    if preds.size == 0:
        return 0.0
    return float(np.sum(np.round(preds) == np.round(truths)) / float(len(truths)))


def score_to_7class(score: float) -> int:
    """Map a continuous MOSEI sentiment score to the nearest class in [-3, 3]."""
    return int(np.round(np.clip(float(score), a_min=-3.0, a_max=3.0)))


def error_category_7class(pred_score: float, true_score: float) -> str:
    """Describe the direction and size of a rounded 7-class error."""
    pred_class = score_to_7class(pred_score)
    true_class = score_to_7class(true_score)
    delta = pred_class - true_class
    if delta == 0:
        return "correct"
    if delta > 0:
        return f"pred_more_positive_by_{delta}"
    return f"pred_more_negative_by_{abs(delta)}"


def regression_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute MAE, Pearson correlation, and rounded 7-class accuracy."""
    mae = float(np.mean(np.abs(preds - labels))) if preds.size else 0.0
    if preds.size > 1 and float(np.std(preds)) > 0 and float(np.std(labels)) > 0:
        corr = float(np.corrcoef(preds, labels)[0][1])
    else:
        corr = 0.0
    preds_clip = np.clip(preds, a_min=-3.0, a_max=3.0)
    labels_clip = np.clip(labels, a_min=-3.0, a_max=3.0)
    return {"mae": mae, "corr": corr, "acc7": multiclass_acc(preds_clip, labels_clip)}


def binary_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    classification: bool = False,
    accbi: bool = False,
) -> Dict[str, float]:
    """Compute binary sentiment accuracy/F1 from continuous predictions."""
    if preds.size == 0:
        return {"accuracy": 0.0, "f1_weighted": 0.0, "n": 0}
    if classification:
        mask = labels != 0
    elif accbi:
        mask = np.ones_like(labels, dtype=bool)
    else:
        mask = np.ones_like(labels, dtype=bool)
    preds_sel = preds[mask]
    labels_sel = labels[mask]
    if preds_sel.size == 0:
        return {"accuracy": 0.0, "f1_weighted": 0.0, "n": 0}
    pred_bin = preds_sel >= 0
    label_bin = labels_sel >= 0
    return {
        "accuracy": float(accuracy_score(label_bin, pred_bin)),
        "f1_weighted": float(f1_score(label_bin, pred_bin, average="weighted", zero_division=0)),
        "n": int(preds_sel.size),
    }


def evaluate_predictions(preds: np.ndarray, labels: np.ndarray) -> Dict[str, object]:
    """Bundle the regression metrics used in reports and model selection."""
    return {"regression": regression_metrics(preds, labels)}


def metric_for_selection(
    metrics: Dict[str, object],
    name: str,
    binary: Optional[Dict[str, float]] = None,
) -> float:
    """Convert a named metric into a score where larger means better."""
    if name == "acc2":
        if binary is None:
            raise ValueError("binary metrics are required when selecting by acc2.")
        return float(binary["accuracy"])
    if name == "f1":
        if binary is None:
            raise ValueError("binary metrics are required when selecting by f1.")
        return float(binary["f1_weighted"])
    if name == "acc7":
        return float(metrics["regression"]["acc7"])
    if name == "corr":
        return float(metrics["regression"]["corr"])
    if name == "mae":
        return -float(metrics["regression"]["mae"])
    raise ValueError(f"Unknown selection metric: {name}")


def sidecar_path(base_path: str, suffix: str) -> str:
    """Create a result sidecar path next to a checkpoint path."""
    path = Path(base_path)
    if path.suffix:
        return str(path.with_name(f"{path.stem}{suffix}"))
    return f"{base_path}{suffix}"


def save_json(path: str, payload: Dict[str, object]) -> None:
    """Write a JSON payload, creating the parent directory if needed."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_test_predictions_csv(
    csv_path: str,
    preds: np.ndarray,
    labels: np.ndarray,
    hf_indices: Optional[torch.Tensor] = None,
    seg_keys: Optional[List[str]] = None,
) -> None:
    """Save per-sample regression predictions and derived labels to CSV."""
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    if hf_indices is None:
        hf_idx_arr = np.arange(len(preds), dtype=np.int64)
    else:
        hf_idx_arr = hf_indices.detach().cpu().numpy()
    seg_keys = seg_keys or ["" for _ in range(len(preds))]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_idx",
                "hf_index",
                "seg_key",
                "score_true",
                "score_pred",
                "label_true_binary",
                "label_pred_binary",
                "label_true_7class",
                "label_pred_7class",
                "error_7class",
            ],
        )
        writer.writeheader()
        for i in range(len(preds)):
            # Store both continuous sentiment scores and derived labels so
            # downstream analysis can inspect errors without re-running models.
            true_score = float(labels[i])
            pred_score = float(preds[i])
            writer.writerow(
                {
                    "sample_idx": int(i),
                    "hf_index": int(hf_idx_arr[i]) if i < len(hf_idx_arr) else int(i),
                    "seg_key": seg_keys[i] if i < len(seg_keys) else "",
                    "score_true": true_score,
                    "score_pred": pred_score,
                    "label_true_binary": int(true_score >= 0),
                    "label_pred_binary": int(pred_score >= 0),
                    "label_true_7class": score_to_7class(true_score),
                    "label_pred_7class": score_to_7class(pred_score),
                    "error_7class": error_category_7class(pred_score, true_score),
                }
            )


def plot_training_curves(
    history: Dict[str, List[float]],
    best_epoch: int,
    output_path: str,
    title_prefix: str,
) -> None:
    """Plot the compact two-panel training curve used across experiments."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    epochs_range = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    axes[0].plot(epochs_range, history["train_loss"], marker="o", label="Train")
    axes[0].plot(epochs_range, history["val_loss"], marker="s", label="Val")
    if best_epoch > 0:
        axes[0].axvline(best_epoch, color="gray", linestyle="--", label=f"Best epoch ({best_epoch})")
    axes[0].set_title(f"{title_prefix} - MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend()

    train_acc2 = history.get("train_acc2")
    val_acc2 = history.get("val_acc2")
    axes[1].plot(epochs_range, train_acc2, marker="o", color="orange", label="Train Acc2")
    axes[1].plot(epochs_range, val_acc2, marker="o", linestyle="--", color="orange", label="Val Acc2")
    axes[1].plot(epochs_range, history["train_acc7"], marker="s", color="blue", label="Train Acc7")
    axes[1].plot(epochs_range, history["val_acc7"], marker="s", linestyle="--", color="blue", label="Val Acc7")
    if best_epoch > 0:
        axes[1].axvline(best_epoch, color="gray", linestyle="--")
    axes[1].set_title(f"{title_prefix} - Train/Val Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def torch_load(path: str, map_location="cpu"):
    """Load a PyTorch object while staying compatible across torch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_exists(path: str, name: str) -> None:
    """Raise a clear error if a required data file is missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def nan_sanitize(x: np.ndarray) -> np.ndarray:
    """Replace NaN and infinite feature values with finite zeros."""
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


class CSDReader:
    """Small HDF5 reader for CMU-MOSEI computational sequence files."""

    def __init__(self, path: str):
        """Open a CSD file and locate the segment-level data group."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required for audio/visual CSD feature loading.") from exc
        self._h5py = h5py
        self.path = path
        self.h5 = h5py.File(path, "r")
        self.data_group = self._locate_data_group()
        self.keys = sorted(list(self.data_group.keys()))
        if len(self.keys) == 0:
            raise RuntimeError(f"No segment keys found in CSD data group: {path}")

    def close(self) -> None:
        """Close the underlying HDF5 handle."""
        try:
            self.h5.close()
        except Exception:
            pass

    def _iter_groups(self):
        """Yield all HDF5 groups for data-group discovery."""
        stack = [self.h5["/"]]
        while stack:
            group = stack.pop()
            yield group
            for _, child in group.items():
                if isinstance(child, self._h5py.Group):
                    stack.append(child)

    def _locate_data_group(self):
        """Find the group containing CSD segment records."""
        if "data" in self.h5 and isinstance(self.h5["data"], self._h5py.Group):
            return self.h5["data"]
        for _, obj in self.h5.items():
            if isinstance(obj, self._h5py.Group) and "data" in obj and isinstance(obj["data"], self._h5py.Group):
                return obj["data"]
        for group in self._iter_groups():
            child_groups = [child for _, child in group.items() if isinstance(child, self._h5py.Group)]
            if not child_groups:
                continue
            hits = 0
            for child_group in child_groups[:5]:
                if "features" in child_group and "intervals" in child_group:
                    hits += 1
            if hits >= 1 and "features" not in group and "intervals" not in group:
                return group
        raise RuntimeError(f"Could not locate CSD data group for: {self.path}")

    def read(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Read one segment record as feature and interval arrays."""
        if key not in self.data_group:
            return None
        node = self.data_group[key]
        if "features" not in node:
            return None
        features = np.asarray(node["features"])
        intervals = np.asarray(node["intervals"]) if "intervals" in node else np.zeros((0, 2), dtype=np.float32)
        return {"features": features, "intervals": intervals}

    def feature_dim(self) -> int:
        """Infer the feature dimension from the first non-empty segment."""
        for key in self.keys:
            rec = self.read(key)
            if rec is None:
                continue
            x = rec["features"]
            if x.ndim == 1:
                return int(x.shape[0])
            if x.ndim == 2 and x.shape[1] > 0:
                return int(x.shape[1])
        raise RuntimeError(f"Unable to infer feature dim from {self.path}")


def parse_video_id_from_segment_key(seg_key: str) -> str:
    """Extract a video id from common MOSEI segment-key formats."""
    if "[" in seg_key and seg_key.endswith("]"):
        return seg_key[: seg_key.rfind("[")]
    if "___" in seg_key:
        return seg_key.split("___", 1)[0]
    if "_" in seg_key:
        parts = seg_key.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return "_".join(parts[:-1])
        if (
            len(parts) >= 3
            and re.fullmatch(r"-?\d+(\.\d+)?", parts[-1] or "") is not None
            and re.fullmatch(r"-?\d+(\.\d+)?", parts[-2] or "") is not None
        ):
            return "_".join(parts[:-2])
        base, suffix = seg_key.rsplit("_", 1)
        if suffix.endswith(".0") and suffix[:-2].isdigit():
            return base
    return seg_key


def _try_keys(row: dict, keys: List[str]):
    """Return the first present, non-null value from a row."""
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def get_row_video(row: dict) -> Optional[str]:
    """Read the video identifier from a HuggingFace MOSEI row."""
    video = _try_keys(row, ["video", "video_id", "VideoID", "vid"])
    if video is None:
        return None
    value = str(video).strip()
    return value if value else None


def get_row_start_end(row: dict) -> Tuple[Optional[float], Optional[float]]:
    """Read utterance start/end timestamps from a MOSEI row when available."""
    start_raw = _try_keys(row, ["start_time", "start", "segment_start", "clip_start"])
    end_raw = _try_keys(row, ["end_time", "end", "segment_end", "clip_end"])
    try:
        start = float(start_raw) if start_raw is not None else None
    except Exception:
        start = None
    try:
        end = float(end_raw) if end_raw is not None else None
    except Exception:
        end = None
    return start, end


def resolve_video_key(video_id: Optional[str], available_video_keys: set) -> Tuple[Optional[str], str]:
    """Match a row video id to the corresponding CSD video key."""
    if video_id is None:
        return None, "missing_video"
    value = str(video_id).strip()
    if not value:
        return None, "missing_video"
    if value in available_video_keys:
        return value, "video_exact"
    base = re.sub(r"__\d+$", "", value)
    if base != value and base in available_video_keys:
        return base, "video_strip_dunder_idx"
    return None, "video_unresolved"


def build_video_to_segment_index(reader: CSDReader) -> Dict[str, List[str]]:
    """Group CSD segment keys by their parsed video id."""
    out: Dict[str, List[str]] = {}
    for key in reader.keys:
        video = parse_video_id_from_segment_key(str(key))
        out.setdefault(video, []).append(str(key))
    for video in list(out.keys()):
        out[video].sort()
    return out


def build_video_level_record(
    reader: CSDReader,
    segment_keys: List[str],
    feat_dim_raw: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Merge all CSD segments from a video into one time-ordered stream."""
    feats_list: List[np.ndarray] = []
    ints_list: List[np.ndarray] = []
    for key in segment_keys:
        rec = reader.read(key)
        if rec is None:
            continue
        feats = np.asarray(rec.get("features", np.zeros((0, feat_dim_raw), dtype=np.float32)), dtype=np.float32)
        ints = np.asarray(rec.get("intervals", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if feats.size == 0:
            continue
        if feats.ndim == 1:
            feats = feats.reshape(1, -1)
        feats = nan_sanitize(feats)
        if ints.ndim == 1:
            if ints.shape[0] >= 2:
                ints = ints[:2].reshape(1, 2)
            else:
                ints = np.zeros((0, 2), dtype=np.float32)
        elif ints.ndim >= 2:
            ints = ints[:, :2].astype(np.float32)
        else:
            ints = np.zeros((0, 2), dtype=np.float32)
        if ints.size == 0:
            continue
        n = min(int(feats.shape[0]), int(ints.shape[0]))
        if n <= 0:
            continue
        feats_list.append(feats[:n])
        ints_list.append(ints[:n])
    if not feats_list:
        return None
    feats_cat = np.concatenate(feats_list, axis=0).astype(np.float32)
    ints_cat = np.concatenate(ints_list, axis=0).astype(np.float32)
    order = np.argsort(ints_cat[:, 0], kind="mergesort")
    return {"features": feats_cat[order], "intervals": ints_cat[order]}


def collect_window_sequence(
    rec: Optional[Dict[str, np.ndarray]],
    start: Optional[float],
    end: Optional[float],
    feat_dim_raw: int,
    window_pad_sec: float,
) -> np.ndarray:
    """Select feature frames overlapping a row's utterance time window."""
    if rec is None:
        return np.zeros((0, feat_dim_raw), dtype=np.float32)
    feats = np.asarray(rec.get("features", np.zeros((0, feat_dim_raw), dtype=np.float32)), dtype=np.float32)
    ints = np.asarray(rec.get("intervals", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    if feats.size == 0:
        return np.zeros((0, feat_dim_raw), dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    feats = nan_sanitize(feats)
    if ints.size == 0 or start is None or end is None:
        return feats.astype(np.float32)
    if ints.ndim == 1:
        if ints.shape[0] >= 2:
            ints = ints[:2].reshape(1, 2)
        else:
            return feats.astype(np.float32)
    if ints.shape[1] < 2:
        return feats.astype(np.float32)
    n = min(int(feats.shape[0]), int(ints.shape[0]))
    if n <= 0:
        return np.zeros((0, feat_dim_raw), dtype=np.float32)
    feats = feats[:n]
    ints = ints[:n, :2].astype(np.float32)
    start_padded = float(start) - float(window_pad_sec)
    end_padded = float(end) + float(window_pad_sec)
    mask = (ints[:, 1] >= start_padded) & (ints[:, 0] <= end_padded)
    if not np.any(mask):
        return np.zeros((0, feat_dim_raw), dtype=np.float32)
    return feats[mask].astype(np.float32)


def compress_temporal_sequence(seq: np.ndarray, target_len: int) -> Tuple[np.ndarray, int, int]:
    """Pad or average-pool a variable-length sequence to a fixed length."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim == 1:
        seq = seq.reshape(1, -1)
    feat_dim = int(seq.shape[1]) if seq.ndim == 2 else 0
    out = np.zeros((target_len, feat_dim), dtype=np.float32)
    raw_len = int(seq.shape[0]) if seq.ndim == 2 else 0
    if raw_len <= 0:
        return out, 0, 0
    if raw_len <= target_len:
        out[:raw_len] = seq
        return out, raw_len, raw_len
    boundaries = np.linspace(0, raw_len, target_len + 1)
    for i in range(target_len):
        left = int(math.floor(boundaries[i]))
        right = int(math.floor(boundaries[i + 1]))
        if right <= left:
            right = min(left + 1, raw_len)
        if i == target_len - 1:
            right = raw_len
        out[i] = seq[left:right].mean(axis=0)
    return out, target_len, raw_len


def storage_dtype_to_torch(name: str) -> torch.dtype:
    """Convert a storage dtype name to a PyTorch dtype."""
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported storage dtype: {name}")


def storage_dtype_to_numpy(name: str) -> np.dtype:
    """Convert a storage dtype name to a NumPy dtype."""
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported storage dtype: {name}")


def _count_stats(counts: List[int]) -> Dict[str, Optional[float]]:
    """Summarize a list of sequence-length counts."""
    if len(counts) == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    arr = np.asarray(counts, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def compute_masked_feature_stats(
    feats: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Compute train-split feature mean/std over valid temporal frames."""
    feats_f = feats.float()
    mask_f = valid_mask.unsqueeze(-1).to(feats_f.dtype)
    count = float(mask_f.sum().item())
    feat_dim = int(feats_f.shape[-1])
    if count <= 0:
        mean = torch.zeros((feat_dim,), dtype=torch.float32)
        std = torch.ones((feat_dim,), dtype=torch.float32)
        return mean, std, {"count": 0.0, "min_std": 1.0, "max_std": 1.0, "mean_std": 1.0, "near_zero_std_dims": 0.0}
    # Statistics are computed only from valid frames; padded zeros are excluded
    # so short utterances do not bias the normalization constants.
    mean = (feats_f * mask_f).sum(dim=(0, 1)) / count
    centered = (feats_f - mean.view(1, 1, -1)) * mask_f
    var = centered.square().sum(dim=(0, 1)) / count
    std_raw = torch.sqrt(var)
    std = std_raw.clamp_min(eps)
    return mean, std, {
        "count": count,
        "min_std": float(std_raw.min().item()),
        "max_std": float(std_raw.max().item()),
        "mean_std": float(std_raw.mean().item()),
        "near_zero_std_dims": float((std_raw < eps).sum().item()),
    }


def apply_masked_feature_zscore(
    split_tensors: Dict[str, object],
    feature_key: str,
    mask_key: str,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> None:
    """Apply feature z-score normalization while preserving padded zeros."""
    feats = split_tensors[feature_key].float()
    valid_mask = split_tensors[mask_key][:, 1:].bool()
    mask_f = valid_mask.unsqueeze(-1).to(feats.dtype)
    normed = (feats - mean.view(1, 1, -1)) / std.view(1, 1, -1)
    # Keep padded positions exactly zero after normalization.
    split_tensors[feature_key] = normed * mask_f


def normalize_av_splits_train_stats(splits: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    """Normalize audio and visual features using train-split statistics."""
    reports: Dict[str, object] = {}
    for modality in ["audio", "visual"]:
        feature_key = modality
        mask_key = f"{modality}_mask"
        train_feats = splits["train"][feature_key]
        train_mask = splits["train"][mask_key][:, 1:].bool()
        mean, std, report = compute_masked_feature_stats(train_feats, train_mask)
        for split in splits.values():
            apply_masked_feature_zscore(split, feature_key, mask_key, mean, std)
        reports[modality] = report
    return reports


def build_av_sequence_split(
    split_name: str,
    hf_split,
    hf_indices_src: List[int],
    audio_reader: CSDReader,
    visual_reader: CSDReader,
    audio_video_to_segment_keys: Dict[str, List[str]],
    visual_video_to_segment_keys: Dict[str, List[str]],
    available_video_keys: set,
    max_vlen: int,
    max_alen: int,
    window_pad_sec: float,
    video_cache_size: int,
    storage_dtype: str,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Build fixed-length audio/visual tensors for one MOSEI split."""
    n = len(hf_split)
    audio_dim_raw = int(audio_reader.feature_dim())
    visual_dim_raw = int(visual_reader.feature_dim())
    torch_storage_dtype = storage_dtype_to_torch(storage_dtype)
    np_storage_dtype = storage_dtype_to_numpy(storage_dtype)

    visual = torch.zeros((n, max_vlen - 1, visual_dim_raw), dtype=torch_storage_dtype)
    audio = torch.zeros((n, max_alen - 1, audio_dim_raw), dtype=torch_storage_dtype)
    # Masks include one leading special position, while feature tensors store
    # only observed CSD frames. The encoders prepend the corresponding vector.
    visual_mask = torch.zeros((n, max_vlen), dtype=torch.bool)
    audio_mask = torch.zeros((n, max_alen), dtype=torch.bool)
    labels = torch.zeros((n,), dtype=torch.long)
    scores = torch.zeros((n,), dtype=torch.float32)
    hf_indices = torch.tensor(hf_indices_src, dtype=torch.long)
    seg_keys: List[str] = []

    missing_video_key = 0
    audio_empty = 0
    visual_empty = 0
    both_empty = 0
    audio_raw_lengths: List[int] = []
    visual_raw_lengths: List[int] = []
    audio_cache: "OrderedDict[str, Optional[Dict[str, np.ndarray]]]" = OrderedDict()
    visual_cache: "OrderedDict[str, Optional[Dict[str, np.ndarray]]]" = OrderedDict()

    def cached_video_read(reader, cache, video_key, video_to_segment_keys, feat_dim_raw):
        """Reuse recently merged video streams while building a split."""
        if video_key in cache:
            cache.move_to_end(video_key)
            return cache[video_key]
        rec = build_video_level_record(reader, video_to_segment_keys.get(video_key, []), feat_dim_raw)
        cache[video_key] = rec
        if len(cache) > max(1, int(video_cache_size)):
            cache.popitem(last=False)
        return rec

    for i in range(n):
        row = hf_split[i]
        score = float(row["sentiment"])
        scores[i] = score
        labels[i] = 1 if score > 0 else 0
        audio_mask[i, 0] = True
        visual_mask[i, 0] = True

        video_id = get_row_video(row)
        start, end = get_row_start_end(row)
        video_key, _ = resolve_video_key(video_id, available_video_keys)
        if video_key is None:
            missing_video_key += 1
            seg_keys.append("")
            audio_empty += 1
            visual_empty += 1
            both_empty += 1
            continue

        a_rec = cached_video_read(audio_reader, audio_cache, video_key, audio_video_to_segment_keys, audio_dim_raw)
        v_rec = cached_video_read(visual_reader, visual_cache, video_key, visual_video_to_segment_keys, visual_dim_raw)
        a_seq = collect_window_sequence(a_rec, start, end, audio_dim_raw, window_pad_sec)
        v_seq = collect_window_sequence(v_rec, start, end, visual_dim_raw, window_pad_sec)
        a_comp, kept_a_len, raw_a_len = compress_temporal_sequence(a_seq, max_alen - 1)
        v_comp, kept_v_len, raw_v_len = compress_temporal_sequence(v_seq, max_vlen - 1)

        if kept_a_len > 0:
            audio[i] = torch.from_numpy(a_comp.astype(np_storage_dtype, copy=False))
            audio_mask[i, : kept_a_len + 1] = True
            audio_raw_lengths.append(raw_a_len)
        else:
            audio_empty += 1
        if kept_v_len > 0:
            visual[i] = torch.from_numpy(v_comp.astype(np_storage_dtype, copy=False))
            visual_mask[i, : kept_v_len + 1] = True
            visual_raw_lengths.append(raw_v_len)
        else:
            visual_empty += 1
        if kept_a_len == 0 and kept_v_len == 0:
            both_empty += 1

        st_s = "" if start is None else f"{start:.3f}"
        en_s = "" if end is None else f"{end:.3f}"
        seg_keys.append(f"{video_key}|{st_s}|{en_s}")

    report = {
        "n": int(n),
        "missing_video_key": int(missing_video_key),
        "missing_video_key_ratio": float(missing_video_key / n if n else 0.0),
        "audio_empty": int(audio_empty),
        "visual_empty": int(visual_empty),
        "both_empty": int(both_empty),
        "audio_empty_ratio": float(audio_empty / n if n else 0.0),
        "visual_empty_ratio": float(visual_empty / n if n else 0.0),
        "both_empty_ratio": float(both_empty / n if n else 0.0),
        "audio_raw_frame_lengths": _count_stats(audio_raw_lengths),
        "visual_raw_frame_lengths": _count_stats(visual_raw_lengths),
        "max_vlen": int(max_vlen),
        "max_alen": int(max_alen),
        "window_pad_sec": float(window_pad_sec),
    }
    print(f"[{split_name}] {report}")

    return {
        "visual": visual,
        "audio": audio,
        "visual_mask": visual_mask,
        "audio_mask": audio_mask,
        "labels": labels,
        "scores": scores,
        "hf_indices": hf_indices,
        "seg_keys": seg_keys,
    }, report


def load_or_build_av_feature_cache(args) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Load cached A/V tensors or build them from CSD files."""
    cache_path = Path(args.feature_cache_path)
    if cache_path.exists() and not args.rebuild_feature_cache:
        cached = torch_load(str(cache_path), map_location="cpu")
        print(f"[CACHE] Loaded A/V feature cache: {cache_path}")
        splits = cached["splits"]
        reports = dict(cached.get("reports", {}))
        if args.apply_av_zscore:
            reports["normalization"] = normalize_av_splits_train_stats(splits)
            print("[NORM] Applied train-split A/V z-score")
        return splits, reports

    raw = load_dataset(args.dataset_name)
    train_split = raw["train"]
    val_split = raw["validation"]
    test_split = raw["test"]
    train_indices = list(range(len(train_split)))
    val_indices = list(range(len(val_split)))
    test_indices = list(range(len(test_split)))
    if args.neutral_eps > 0:
        train_keep = [i for i, s in enumerate(train_split["sentiment"]) if abs(float(s)) > args.neutral_eps]
        val_keep = [i for i, s in enumerate(val_split["sentiment"]) if abs(float(s)) > args.neutral_eps]
        test_keep = [i for i, s in enumerate(test_split["sentiment"]) if abs(float(s)) > args.neutral_eps]
        train_split = train_split.select(train_keep)
        val_split = val_split.select(val_keep)
        test_split = test_split.select(test_keep)
        train_indices = [train_indices[i] for i in train_keep]
        val_indices = [val_indices[i] for i in val_keep]
        test_indices = [test_indices[i] for i in test_keep]
        print(
            f"[FILTER] neutral_eps={args.neutral_eps:.3f} | removed train/validation/test="
            f"{len(raw['train']) - len(train_split)}/{len(raw['validation']) - len(val_split)}/"
            f"{len(raw['test']) - len(test_split)}"
        )

    compseq_dir = Path(args.mosei_compseq_dir)
    audio_path = str(compseq_dir / args.audio_csd)
    visual_path = str(compseq_dir / args.visual_csd)
    assert_exists(audio_path, "audio CSD")
    assert_exists(visual_path, "visual CSD")

    print("Opening CSD files ...")
    audio_reader = CSDReader(audio_path)
    visual_reader = CSDReader(visual_path)
    try:
        audio_index = build_video_to_segment_index(audio_reader)
        visual_index = build_video_to_segment_index(visual_reader)
        available_video_keys = set(audio_index.keys()) | set(visual_index.keys())
        print(f"Video key coverage: audio={len(audio_index)} visual={len(visual_index)} union={len(available_video_keys)}")
        split_train, rep_train = build_av_sequence_split(
            "TRAIN",
            train_split,
            train_indices,
            audio_reader,
            visual_reader,
            audio_index,
            visual_index,
            available_video_keys,
            args.max_vlen,
            args.max_alen,
            args.window_pad_sec,
            args.video_cache_size,
            args.storage_dtype,
        )
        split_val, rep_val = build_av_sequence_split(
            "VALIDATION",
            val_split,
            val_indices,
            audio_reader,
            visual_reader,
            audio_index,
            visual_index,
            available_video_keys,
            args.max_vlen,
            args.max_alen,
            args.window_pad_sec,
            args.video_cache_size,
            args.storage_dtype,
        )
        split_test, rep_test = build_av_sequence_split(
            "TEST",
            test_split,
            test_indices,
            audio_reader,
            visual_reader,
            audio_index,
            visual_index,
            available_video_keys,
            args.max_vlen,
            args.max_alen,
            args.window_pad_sec,
            args.video_cache_size,
            args.storage_dtype,
        )
    finally:
        audio_reader.close()
        visual_reader.close()

    splits = {"train": split_train, "validation": split_val, "test": split_test}
    reports = {"train": rep_train, "validation": rep_val, "test": rep_test}
    cache_payload = {
        "splits": splits,
        "reports": reports,
        "config": {
            "dataset_name": args.dataset_name,
            "neutral_eps": float(args.neutral_eps),
            "max_vlen": int(args.max_vlen),
            "max_alen": int(args.max_alen),
            "window_pad_sec": float(args.window_pad_sec),
            "video_cache_size": int(args.video_cache_size),
            "storage_dtype": str(args.storage_dtype),
            "audio_csd": args.audio_csd,
            "visual_csd": args.visual_csd,
        },
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache_payload, cache_path)
    print(f"[CACHE] Saved A/V feature cache: {cache_path}")
    if args.apply_av_zscore:
        reports["normalization"] = normalize_av_splits_train_stats(splits)
        print("[NORM] Applied train-split A/V z-score")
    return splits, reports


def smoke_limit_av_splits(splits: Dict[str, Dict[str, object]]) -> None:
    """Trim cached A/V splits for quick end-to-end smoke tests."""
    for split_name in ["train", "validation", "test"]:
        limit = {"train": 256, "validation": 64, "test": 64}[split_name]
        split = splits[split_name]
        for key, value in list(split.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] > limit:
                split[key] = value[:limit]
            elif isinstance(value, list) and len(value) > limit:
                split[key] = value[:limit]
        print(f"[SMOKE] {split_name} => {limit}")
