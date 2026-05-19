"""Audio-only and visual-only BiLSTM baselines for CMU-MOSEI.

The same sequence encoder is used for both modalities. Modality-specific
defaults provide the training and regularization settings used for each run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

import mosei_data_eval_utils as common


MODALITY_DEFAULTS = {
    # Modality-specific defaults used when a command-line option is not set.
    "audio": {
        "run_name": "audio_bilstm_seed10",
        "batch_train": 48,
        "batch_eval": 48,
        "epochs": 40,
        "lr": 7.5e-5,
        "warmup_ratio": 0.10,
        "weight_decay": 0.0,
        "patience": 8,
        "min_epochs": 10,
        "seed": 10,
        "selection_metric": "mae",
        "hidden_size": 256,
        "bilstm_hidden_size": 128,
        "bilstm_layers": 1,
        "regressor_hidden_size": 128,
        "hidden_dropout_prob": 0.1,
        "pooling": "mean",
        "loss_type": "smoothl1",
        "huber_delta": 0.5,
        "scheduler_type": "linear",
    },
    "visual": {
        "run_name": "visual_bilstm_seed10",
        "batch_train": 48,
        "batch_eval": 48,
        "epochs": 30,
        "lr": 7e-5,
        "warmup_ratio": 0.10,
        "weight_decay": 3e-4,
        "patience": 5,
        "min_epochs": 6,
        "seed": 10,
        "selection_metric": "mae",
        "hidden_size": 256,
        "bilstm_hidden_size": 96,
        "bilstm_layers": 1,
        "regressor_hidden_size": 128,
        "hidden_dropout_prob": 0.55,
        "pooling": "mean",
        "loss_type": "smoothl1",
        "huber_delta": 0.5,
        "scheduler_type": "linear",
    },
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODALITY_CHOICES = ("audio", "visual")


class BiLSTMFeatureSequenceEncoder(nn.Module):
    """Encode one modality's temporal feature sequence with a BiLSTM."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        max_len: int,
        modality_id: int,
        lstm_hidden_size: int,
        num_layers: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        """Create the temporal encoder and modality-aware hidden representation."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("BiLSTM encoder requires num_layers >= 1")
        self.max_len = max_len
        self.modality_id = modality_id
        # A zero-valued prefix slot keeps the sequence length aligned with the
        # cached mask convention used by the shared MOSEI preprocessing code.
        self.register_buffer("special", torch.zeros(1, 1, input_dim), persistent=False)
        self.input_norm = nn.LayerNorm(input_dim, eps=layer_norm_eps)
        self.input_dropout = nn.Dropout(hidden_dropout_prob)
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_size,
            num_layers=num_layers,
            dropout=hidden_dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_proj = nn.Linear(lstm_hidden_size * 2, hidden_size)
        self.pos_embed = nn.Embedding(max_len, hidden_size)
        self.modal_embed = nn.Embedding(3, hidden_size)
        self.output_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode padded sequences while respecting valid-frame masks."""
        batch_size = seq.shape[0]
        special = self.special.expand(batch_size, -1, -1)
        seq_full = torch.cat([special, seq], dim=1)
        seq_full = seq_full.to(self.output_proj.weight.dtype)
        seq_full = self.input_dropout(self.input_norm(seq_full))

        valid_mask = mask.bool()
        lengths = valid_mask.sum(dim=1).clamp_min(1).to(torch.long).cpu()
        # Packed sequences avoid recurrent computation over padded frames while
        # preserving the original batch order.
        packed = pack_padded_sequence(seq_full, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=self.max_len)
        hidden = self.output_proj(hidden)

        # Add position and modality embeddings so the pooled representation can
        # distinguish frame order and feature source after the BiLSTM projection.
        pos_ids = torch.arange(self.max_len, device=seq.device).unsqueeze(0).expand(batch_size, -1)
        modal_ids = torch.full((batch_size, self.max_len), self.modality_id, dtype=torch.long, device=seq.device)
        hidden = hidden + self.pos_embed(pos_ids) + self.modal_embed(modal_ids)
        hidden = self.output_dropout(self.output_norm(hidden))
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, valid_mask


class MoseiAVUnimodalDataset(Dataset):
    """Expose cached audio or visual tensors as unimodal training examples."""

    def __init__(self, split_tensors: Dict[str, object], modality: str):
        """Select the feature and mask tensors for the requested modality."""
        self.data = split_tensors
        self.modality = modality
        self.feature_key = modality
        self.mask_key = f"{modality}_mask"
        self.size = int(self.data[self.feature_key].shape[0])

    def __len__(self) -> int:
        """Return the number of utterances in the split."""
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one feature sequence, mask, metadata index, and score."""
        return {
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "hf_index": self.data["hf_indices"][idx].clone().long(),
            "seq": self.data[self.feature_key][idx].clone().float(),
            "mask": self.data[self.mask_key][idx].clone().bool(),
            "score": self.data["scores"][idx].clone().float(),
        }


class UnimodalAVBiLSTMRegressor(nn.Module):
    """BiLSTM sequence encoder followed by a small MLP regression head."""

    def __init__(
        self,
        modality: str,
        input_dim: int,
        max_len: int,
        hidden_size: int,
        bilstm_hidden_size: int,
        bilstm_layers: int,
        regressor_hidden_size: int,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        """Build the modality encoder, mean pooling, and regression head."""
        super().__init__()
        modality_id = 2 if modality == "audio" else 1
        self.encoder = BiLSTMFeatureSequenceEncoder(
            input_dim=input_dim,
            hidden_size=hidden_size,
            max_len=max_len,
            modality_id=modality_id,
            lstm_hidden_size=bilstm_hidden_size,
            num_layers=bilstm_layers,
            hidden_dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, regressor_hidden_size),
            nn.ReLU(),
            nn.Dropout(hidden_dropout_prob),
            nn.Linear(regressor_hidden_size, 1),
        )

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over valid temporal positions."""
        mask_f = mask.unsqueeze(-1).to(hidden.dtype)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (hidden * mask_f).sum(dim=1) / denom

    def forward(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode, pool, and predict a bounded sentiment score."""
        hidden, valid = self.encoder(seq, mask)
        pooled = self._masked_mean(hidden, valid)
        output = self.regression_head(pooled)
        # CMU-MOSEI sentiment scores are evaluated on the [-3, 3] range.
        return (torch.tanh(output) * 3.0).squeeze(-1)


def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect model predictions and gold scores from a data loader."""
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            scores = model(
                seq=batch["seq"].to(device),
                mask=batch["mask"].to(device),
            )
            preds.append(scores.detach().cpu().numpy())
            labels.append(batch["score"].cpu().numpy())
    if not preds:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return np.concatenate(preds), np.concatenate(labels)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    epoch: int,
) -> float:
    """Run one optimization epoch for the BiLSTM regressor."""
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        preds = model(seq=batch["seq"].to(device), mask=batch["mask"].to(device))
        targets = batch["score"].to(device)
        loss = loss_fn(preds.view(-1), targets.view(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += float(loss.item())
        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            print(
                f"  [STEP] epoch={epoch} step={step}/{len(loader)} "
                f"loss={loss.item():.4f} lr={current_lr:.2e} grad_norm={float(grad_norm):.4f}",
                flush=True,
            )
    return total_loss / max(1, len(loader))


def build_loss_fn(args: argparse.Namespace) -> nn.Module:
    """Construct the requested regression loss."""
    if args.loss_type == "mse":
        return nn.MSELoss()
    if args.loss_type in {"smoothl1", "huber"}:
        return nn.SmoothL1Loss(beta=float(args.huber_delta))
    raise ValueError(f"Unsupported loss_type: {args.loss_type}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments shared by the audio and visual baselines."""
    parser = argparse.ArgumentParser(description="Audio/visual BiLSTM CMU-MOSEI regression baseline")
    parser.add_argument("--modality", type=str, required=True, choices=MODALITY_CHOICES)
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default=common.DEFAULT_DATASET_NAME)
    parser.add_argument("--mosei_compseq_dir", type=str, default=common.DEFAULT_MOSEI_COMPSEQ_DIR)
    parser.add_argument("--audio_csd", type=str, default=common.DEFAULT_AUDIO_CSD)
    parser.add_argument("--visual_csd", type=str, default=common.DEFAULT_VISUAL_CSD)
    parser.add_argument("--feature_cache_path", type=str, default="")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--storage_dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--apply_av_zscore", action="store_true")
    parser.add_argument("--no_apply_av_zscore", dest="apply_av_zscore", action="store_false")
    parser.add_argument("--neutral_eps", type=float, default=0.0)
    parser.add_argument("--max_vlen", type=int, default=70)
    parser.add_argument("--max_alen", type=int, default=80)
    parser.add_argument("--window_pad_sec", type=float, default=0.0)
    parser.add_argument("--video_cache_size", type=int, default=64)
    parser.add_argument("--batch_train", type=int, default=None)
    parser.add_argument("--batch_eval", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=None,
        choices=["acc2", "f1", "acc7", "corr", "mae"],
    )
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min_epochs", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--bilstm_hidden_size", type=int, default=None)
    parser.add_argument("--bilstm_layers", type=int, default=None)
    parser.add_argument("--regressor_hidden_size", type=int, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=None)
    parser.add_argument("--loss_type", type=str, default=None, choices=["mse", "smoothl1", "huber"])
    parser.add_argument("--huber_delta", type=float, default=None)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--no_save_test_predictions", dest="save_test_predictions", action="store_false")
    parser.add_argument("--test_predictions_path", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true")
    parser.set_defaults(apply_av_zscore=True, save_test_predictions=True)
    return parser


def apply_modality_defaults(args: argparse.Namespace) -> None:
    """Fill unset CLI options from the selected modality defaults."""
    defaults = MODALITY_DEFAULTS[args.modality]
    for key, value in defaults.items():
        if getattr(args, key, None) in (None, ""):
            setattr(args, key, value)
    if args.output_dir == "":
        args.output_dir = f"outputs/{args.modality}_bilstm"
    if args.save_path == "":
        args.save_path = str(Path(args.output_dir) / f"{args.run_name}.pt")
    if args.feature_cache_path == "":
        args.feature_cache_path = str(Path(args.output_dir) / f"cache_{args.modality}.pt")


def summarize_metrics(metrics: Dict[str, object], binary: Dict[str, float]) -> Dict[str, float]:
    """Create the metric block used by reports and result tables."""
    return {
        "mae": metrics["regression"]["mae"],
        "corr": metrics["regression"]["corr"],
        "acc7": metrics["regression"]["acc7"],
        "acc2": binary["accuracy"],
        "f1": binary["f1_weighted"],
        "n_acc2": binary["n"],
    }


def main() -> None:
    """Train, evaluate, and save an audio or visual BiLSTM baseline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_modality_defaults(args)
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.min_epochs = min(args.min_epochs, 2)

    print(f"Device : {DEVICE}")
    print(f"Config : {vars(args)}\n")
    common.set_seed(args.seed)

    # Build or load fixed-length audio/visual tensors for all data splits.
    splits, reports = common.load_or_build_av_feature_cache(args)
    if args.smoke_test:
        common.smoke_limit_av_splits(splits)

    train_ds = MoseiAVUnimodalDataset(splits["train"], args.modality)
    val_ds = MoseiAVUnimodalDataset(splits["validation"], args.modality)
    test_ds = MoseiAVUnimodalDataset(splits["test"], args.modality)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)

    split_train = splits["train"]
    # Infer the raw CSD feature dimension from the cached tensors so the same
    # model code can be used for COVAREP audio and OpenFace visual features.
    input_dim = int(split_train[args.modality].shape[-1])
    max_len = int(split_train[f"{args.modality}_mask"].shape[1])
    model = UnimodalAVBiLSTMRegressor(
        modality=args.modality,
        input_dim=input_dim,
        max_len=max_len,
        hidden_size=args.hidden_size,
        bilstm_hidden_size=args.bilstm_hidden_size,
        bilstm_layers=args.bilstm_layers,
        regressor_hidden_size=args.regressor_hidden_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        layer_norm_eps=args.layer_norm_eps,
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = build_loss_fn(args)

    history = {
        "train_optim_loss": [],
        "train_loss": [],
        "train_mae": [],
        "train_corr": [],
        "train_acc7": [],
        "train_acc2": [],
        "train_f1": [],
        "val_loss": [],
        "val_selection_score": [],
        "val_mae": [],
        "val_corr": [],
        "val_acc7": [],
        "val_acc2": [],
        "val_f1": [],
    }
    best_epoch = -1
    best_val_score = -float("inf")
    patience_counter = 0

    # The selection metric is evaluated after each epoch on the validation set;
    # the checkpoint with the best validation score is restored for reporting.
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---", flush=True)
        train_optim_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, DEVICE, epoch)
        # Evaluation uses deterministic loaders and is kept separate from the
        # optimization loss so all runs report the same metric definitions.
        train_preds, train_labels = predict_scores(model, train_eval_loader, DEVICE)
        train_loss = float(np.mean((train_preds - train_labels) ** 2)) if train_preds.size else float("nan")
        train_metrics = common.evaluate_predictions(train_preds, train_labels)
        val_preds, val_labels = predict_scores(model, val_loader, DEVICE)
        val_loss = float(np.mean((val_preds - val_labels) ** 2)) if val_preds.size else float("nan")
        val_metrics = common.evaluate_predictions(val_preds, val_labels)
        train_binary = common.binary_metrics(train_preds, train_labels, accbi=True)
        val_binary = common.binary_metrics(val_preds, val_labels, accbi=True)
        selection_score = common.metric_for_selection(val_metrics, args.selection_metric, val_binary)

        history["train_optim_loss"].append(train_optim_loss)
        history["train_loss"].append(train_loss)
        history["train_mae"].append(train_metrics["regression"]["mae"])
        history["train_corr"].append(train_metrics["regression"]["corr"])
        history["train_acc7"].append(train_metrics["regression"]["acc7"])
        history["train_acc2"].append(train_binary["accuracy"])
        history["train_f1"].append(train_binary["f1_weighted"])
        history["val_loss"].append(val_loss)
        history["val_selection_score"].append(selection_score)
        history["val_mae"].append(val_metrics["regression"]["mae"])
        history["val_corr"].append(val_metrics["regression"]["corr"])
        history["val_acc7"].append(val_metrics["regression"]["acc7"])
        history["val_acc2"].append(val_binary["accuracy"])
        history["val_f1"].append(val_binary["f1_weighted"])

        print(
            f"Epoch {epoch} | TrainOptimLoss={train_optim_loss:.4f} | TrainLoss={train_loss:.4f} | "
            f"Train MAE={train_metrics['regression']['mae']:.4f} | Train Corr={train_metrics['regression']['corr']:.4f}"
        )
        print(
            f"             ValLoss={val_loss:.4f} | Val MAE={val_metrics['regression']['mae']:.4f} | "
            f"Val Corr={val_metrics['regression']['corr']:.4f} | Val Acc7={val_metrics['regression']['acc7']:.4f}"
        )
        print(f"  Selection score ({args.selection_metric}) = {selection_score:.4f}")

        if selection_score > best_val_score:
            best_val_score = selection_score
            best_epoch = epoch
            patience_counter = 0
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            # Store the full argument namespace with the checkpoint so evaluation
            # artifacts can be traced back to the exact training configuration.
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": vars(args),
                    "best_epoch": best_epoch,
                    "best_val_score": best_val_score,
                },
                args.save_path,
            )
            print(f"  --> Saved best checkpoint: {args.save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if epoch >= args.min_epochs and patience_counter >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break
        print()

    if best_epoch < 0:
        raise RuntimeError("Training did not produce a best checkpoint.")

    # Reload the selected checkpoint before computing validation and test
    # summaries so the saved model and reported metrics refer to the same state.
    checkpoint = common.torch_load(args.save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    val_preds, val_labels = predict_scores(model, val_loader, DEVICE)
    test_preds, test_labels = predict_scores(model, test_loader, DEVICE)
    val_metrics = common.evaluate_predictions(val_preds, val_labels)
    test_metrics = common.evaluate_predictions(test_preds, test_labels)
    val_binary = common.binary_metrics(val_preds, val_labels, accbi=True)
    test_binary = common.binary_metrics(test_preds, test_labels, accbi=True)
    val_summary = summarize_metrics(val_metrics, val_binary)
    test_summary = summarize_metrics(test_metrics, test_binary)

    print("\n[VAL] Metrics")
    print(json.dumps(val_metrics, indent=2))
    print("\n[TEST] Metrics")
    print(json.dumps(test_metrics, indent=2))
    print("\n[TEST] Summary")
    print(json.dumps(test_summary, indent=2))

    if args.save_test_predictions:
        pred_path = args.test_predictions_path or common.sidecar_path(args.save_path, "_test_predictions.csv")
        common.save_test_predictions_csv(
            pred_path,
            test_preds,
            test_labels,
            hf_indices=splits["test"]["hf_indices"],
            seg_keys=splits["test"]["seg_keys"],
        )
        print(f"Test predictions saved to {pred_path}")

    result_payload = {
        "config": vars(args),
        "modality_defaults": MODALITY_DEFAULTS[args.modality],
        "config_name": f"{args.modality}_bilstm",
        "feature_reports": reports,
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "best_val_score": checkpoint.get("best_val_score", best_val_score),
        "history": history,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    json_path = common.sidecar_path(args.save_path, "_results.json")
    common.save_json(json_path, result_payload)
    print(f"Results JSON saved to {json_path}")

    plot_path = common.sidecar_path(args.save_path, "_training_curves.png")
    common.plot_training_curves(history, best_epoch, plot_path, f"{args.modality.title()} BiLSTM")
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
