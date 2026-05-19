"""Text-only BERT baseline for CMU-MOSEI sentiment regression.

The model predicts continuous sentiment scores from tokenized utterances using
masked mean pooling over BERT token representations and a linear regression
head bounded to the CMU-MOSEI score range.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertModel, BertTokenizerFast, get_linear_schedule_with_warmup

import mosei_data_eval_utils as common


TEXT_DEFAULTS = {
    # Defaults used when command-line overrides are not supplied.
    "run_name": "text_bert_seed10",
    "model_name": "bert-base-uncased",
    "dataset_name": common.DEFAULT_DATASET_NAME,
    "max_length": 96,
    "batch_train": 64,
    "batch_eval": 64,
    "grad_accum_steps": 1,
    "num_workers": 0,
    "lr": 5e-4,
    "epochs": 25,
    "warmup_ratio": 0.05,
    "weight_decay": 0.01,
    "hidden_dropout_prob": 0.3,
    "attention_dropout_prob": None,
    "head_type": "linear",
    "unfreeze_last_n_layers": 0,
    "loss_type": "smoothl1",
    "huber_delta": 1.0,
    "scheduler_type": "linear",
    "patience": 6,
    "min_epochs": 8,
    "seed": 10,
    "text_field": "text",
    "neutral_eps": 0.0,
    "selection_metric": "mae",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MOSEIRegressionTextDataset(Dataset):
    """Tokenize MOSEI text rows and return continuous sentiment scores."""

    def __init__(self, hf_split, tokenizer: BertTokenizerFast, max_length: int, text_field: str):
        """Store the HuggingFace split and tokenizer settings."""
        self.data = hf_split
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_field = text_field

    def __len__(self) -> int:
        """Return the number of utterances in the split."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Tokenize one utterance and package its regression target."""
        row = self.data[idx]
        text = common.choose_text(row, self.text_field)
        score = float(row["sentiment"])
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float32),
        }


class BertRegressionModel(nn.Module):
    """BERT encoder plus a small regression head for scores in [-3, 3]."""

    def __init__(
        self,
        model_name: str,
        hidden_dropout_prob: float,
        attention_dropout_prob: float | None,
    ):
        """Create the BERT backbone and regression head."""
        super().__init__()
        if attention_dropout_prob is None:
            attention_dropout_prob = hidden_dropout_prob
        config = BertConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_dropout_prob,
        )
        self.encoder = BertModel.from_pretrained(model_name, config=config)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Run BERT, mean-pool valid tokens, and predict a bounded score."""
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask.long(),
            return_dict=True,
        ).last_hidden_state
        # Masked mean pooling keeps padding tokens out of the utterance
        # representation while using all retained subword tokens.
        mask_f = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        output = self.regressor(self.dropout(pooled))
        # CMU-MOSEI sentiment scores are evaluated on the [-3, 3] range.
        return torch.tanh(output).squeeze(-1) * 3.0


def configure_encoder_trainability(model: BertRegressionModel, unfreeze_last_n_layers: int) -> None:
    """Freeze BERT fully or unfreeze only the last N encoder layers."""
    if unfreeze_last_n_layers < 0:
        return
    for param in model.encoder.parameters():
        param.requires_grad = False
    if unfreeze_last_n_layers > 0:
        layers = list(model.encoder.encoder.layer)
        for layer in layers[-unfreeze_last_n_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


def trainable_parameter_report(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters for logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_loss_fn() -> nn.Module:
    """Construct the SmoothL1 regression loss."""
    return nn.SmoothL1Loss(beta=TEXT_DEFAULTS["huber_delta"])


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    epoch: int,
    grad_accum_steps: int,
) -> float:
    """Run one optimization epoch with optional gradient accumulation."""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        preds = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        targets = batch["score"].to(device)
        raw_loss = loss_fn(preds.view(-1), targets.view(-1))
        loss = raw_loss / max(1, grad_accum_steps)
        loss.backward()

        should_step = ((step + 1) % max(1, grad_accum_steps) == 0) or ((step + 1) == len(loader))
        grad_norm = float("nan")
        if should_step:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(raw_loss.item())
        if step % 100 == 0:
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
            print(
                f"  [STEP] epoch={epoch} step={step}/{len(loader)} "
                f"loss={raw_loss.item():.4f} lr={current_lr:.2e} grad_norm={float(grad_norm):.4f}",
                flush=True,
            )
    return total_loss / max(1, len(loader))


def predict_scores(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect model predictions and gold scores from a data loader."""
    model.eval()
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            scores = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            preds.append(scores.detach().cpu().numpy())
            labels.append(batch["score"].cpu().numpy())
    if not preds:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return np.concatenate(preds), np.concatenate(labels)


def build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI arguments while keeping model defaults configurable."""
    parser = argparse.ArgumentParser(description="Text-only BERT CMU-MOSEI regression baseline")
    parser.add_argument("--run_name", type=str, default=TEXT_DEFAULTS["run_name"])
    parser.add_argument("--output_dir", type=str, default="outputs/text_bert")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default=TEXT_DEFAULTS["dataset_name"])
    parser.add_argument("--model_name", type=str, default=TEXT_DEFAULTS["model_name"])
    parser.add_argument("--max_length", type=int, default=TEXT_DEFAULTS["max_length"])
    parser.add_argument("--batch_train", type=int, default=TEXT_DEFAULTS["batch_train"])
    parser.add_argument("--batch_eval", type=int, default=TEXT_DEFAULTS["batch_eval"])
    parser.add_argument("--grad_accum_steps", type=int, default=TEXT_DEFAULTS["grad_accum_steps"])
    parser.add_argument("--num_workers", type=int, default=TEXT_DEFAULTS["num_workers"])
    parser.add_argument("--lr", type=float, default=TEXT_DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=TEXT_DEFAULTS["epochs"])
    parser.add_argument("--warmup_ratio", type=float, default=TEXT_DEFAULTS["warmup_ratio"])
    parser.add_argument("--weight_decay", type=float, default=TEXT_DEFAULTS["weight_decay"])
    parser.add_argument("--hidden_dropout_prob", type=float, default=TEXT_DEFAULTS["hidden_dropout_prob"])
    parser.add_argument("--attention_dropout_prob", type=float, default=TEXT_DEFAULTS["attention_dropout_prob"])
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=TEXT_DEFAULTS["unfreeze_last_n_layers"])
    parser.add_argument("--patience", type=int, default=TEXT_DEFAULTS["patience"])
    parser.add_argument("--min_epochs", type=int, default=TEXT_DEFAULTS["min_epochs"])
    parser.add_argument("--seed", type=int, default=TEXT_DEFAULTS["seed"])
    parser.add_argument("--text_field", type=str, default=TEXT_DEFAULTS["text_field"], choices=["text", "ASR"])
    parser.add_argument("--neutral_eps", type=float, default=TEXT_DEFAULTS["neutral_eps"])
    parser.add_argument(
        "--selection_metric",
        type=str,
        default=TEXT_DEFAULTS["selection_metric"],
        choices=["acc2", "f1", "acc7", "corr", "mae"],
    )
    parser.add_argument("--no_save_test_predictions", dest="save_test_predictions", action="store_false")
    parser.add_argument("--test_predictions_path", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true")
    parser.set_defaults(save_test_predictions=True)
    return parser


def summarize_metrics(metrics: Dict[str, object], binary: Dict[str, float]) -> Dict[str, float]:
    """Create the compact metric block used by result tables."""
    return {
        "mae": metrics["regression"]["mae"],
        "corr": metrics["regression"]["corr"],
        "acc7": metrics["regression"]["acc7"],
        "acc2": binary["accuracy"],
        "f1": binary["f1_weighted"],
        "n_acc2": binary["n"],
    }


def main() -> None:
    """Train, evaluate, and save the text BERT baseline."""
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.save_path == "":
        args.save_path = str(Path(args.output_dir) / f"{args.run_name}.pt")
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 2)
        args.min_epochs = min(args.min_epochs, 2)

    print(f"Device : {DEVICE}")
    print(f"Config : {vars(args)}\n")
    common.set_seed(args.seed)

    # Load text splits before constructing tokenized datasets so optional
    # neutral-sample filtering is applied consistently across splits.
    train_split, val_split, test_split = common.load_mosei_text_splits(
        dataset_name=args.dataset_name,
        neutral_eps=args.neutral_eps,
        smoke_test=args.smoke_test,
    )
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    model = BertRegressionModel(
        model_name=args.model_name,
        hidden_dropout_prob=args.hidden_dropout_prob,
        attention_dropout_prob=args.attention_dropout_prob,
    ).to(DEVICE)
    configure_encoder_trainability(model, args.unfreeze_last_n_layers)
    param_report = trainable_parameter_report(model)
    print(
        "[PARAMS] "
        f"total={param_report['total']:,} | trainable={param_report['trainable']:,} | frozen={param_report['frozen']:,}",
        flush=True,
    )

    train_ds = MOSEIRegressionTextDataset(train_split, tokenizer, args.max_length, args.text_field)
    val_ds = MOSEIRegressionTextDataset(val_split, tokenizer, args.max_length, args.text_field)
    test_ds = MOSEIRegressionTextDataset(test_split, tokenizer, args.max_length, args.text_field)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)

    no_decay = ["bias", "LayerNorm.weight"]
    # Use AdamW parameter groups so LayerNorm and bias terms are not decayed.
    optimizer = AdamW(
        [
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ],
        lr=args.lr,
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = build_loss_fn()

    history = {
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
        train_optim_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            DEVICE,
            epoch,
            args.grad_accum_steps,
        )
        train_preds, train_labels = predict_scores(model, train_eval_loader, DEVICE)
        train_loss = float(np.mean((train_preds - train_labels) ** 2)) if train_preds.size else float("nan")
        train_metrics = common.evaluate_predictions(train_preds, train_labels)
        val_preds, val_labels = predict_scores(model, val_loader, DEVICE)
        val_loss = float(np.mean((val_preds - val_labels) ** 2)) if val_preds.size else float("nan")
        val_metrics = common.evaluate_predictions(val_preds, val_labels)
        train_binary = common.binary_metrics(train_preds, train_labels, accbi=True)
        val_binary = common.binary_metrics(val_preds, val_labels, accbi=True)
        selection_score = common.metric_for_selection(val_metrics, args.selection_metric, val_binary)

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
        common.save_test_predictions_csv(pred_path, test_preds, test_labels)
        print(f"Test predictions saved to {pred_path}")

    results = {
        "config": vars(args),
        "model_defaults": TEXT_DEFAULTS,
        "config_name": "text_bert",
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "best_val_score": checkpoint.get("best_val_score", best_val_score),
        "history": history,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    json_path = common.sidecar_path(args.save_path, "_results.json")
    common.save_json(json_path, results)
    print(f"Results JSON saved to {json_path}")

    plot_path = common.sidecar_path(args.save_path, "_training_curves.png")
    common.plot_training_curves(history, best_epoch, plot_path, "Text BERT")
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
