"""Cross-attention model without gating for CMU-MOSEI."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, get_linear_schedule_with_warmup

import mosei_data_eval_utils as common
import mosei_modeling_utils as mmc


MODALITIES = mmc.MODALITIES
# Six directed routes: each modality attends to each of the other two modalities.
DIRECTIONAL_PAIRS = (
    ("t_from_v", "text", "visual"),
    ("t_from_a", "text", "audio"),
    ("v_from_t", "visual", "text"),
    ("v_from_a", "visual", "audio"),
    ("a_from_t", "audio", "text"),
    ("a_from_v", "audio", "visual"),
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Default hyperparameters used when command-line overrides are not supplied.
MODEL_DEFAULTS = {
    "run_name": "cross_attention_nogate_seed10",
    "base_lr": 3e-5,
    "weight_decay": 0.0,
    "batch_train": 48,
    "batch_eval": 48,
    "epochs": 40,
    "warmup_ratio": 0.10,
    "early_stopping_min_epochs": 7,
    "early_stopping_patience": 5,
    "early_stopping_min_delta": 0.001,
    "seed": 10,
    "selection_metric": "mae",
    "hidden_size": 768,
    "av_bilstm_hidden_size": 128,
    "av_bilstm_layers": 3,
    "cross_attn_dim": 128,
    "cross_num_layers": 2,
    "cross_intermediate_size": 1536,
    "fusion_dim": 128,
    "hidden_dropout_prob": 0.10,
    "attention_dropout_prob": 0.10,
    "max_grad_norm": 1.0,
}


class CrossAttentionLayer(nn.Module):
    """One cross-attention layer without gating."""

    def __init__(
        self,
        hidden_size: int,
        cross_attn_dim: int,
        intermediate_size: int,
        attention_dropout_prob: float,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.cross_attn_dim = cross_attn_dim

        # Each modality has its own Q/K/V projections so text, visual, and audio
        # can keep separate parameterizations while sharing the same layer logic.
        self.q_proj = nn.ModuleDict({name: nn.Linear(hidden_size, cross_attn_dim) for name in MODALITIES})
        self.k_proj = nn.ModuleDict({name: nn.Linear(hidden_size, cross_attn_dim) for name in MODALITIES})
        self.v_proj = nn.ModuleDict({name: nn.Linear(hidden_size, cross_attn_dim) for name in MODALITIES})

        # The attended source features are projected back to the common hidden
        # size before the residual/normalization block.
        self.attn_out_proj = nn.ModuleDict({name: nn.Linear(cross_attn_dim, hidden_size) for name in MODALITIES})
        self.ffn_in = nn.ModuleDict({name: nn.Linear(hidden_size, intermediate_size) for name in MODALITIES})
        self.ffn_out = nn.ModuleDict({name: nn.Linear(intermediate_size, hidden_size) for name in MODALITIES})
        self.attn_norm = nn.ModuleDict({name: nn.LayerNorm(hidden_size, eps=layer_norm_eps) for name in MODALITIES})
        self.ffn_norm = nn.ModuleDict({name: nn.LayerNorm(hidden_size, eps=layer_norm_eps) for name in MODALITIES})
        self.attn_dropout = nn.Dropout(attention_dropout_prob)
        self.hidden_dropout = nn.Dropout(hidden_dropout_prob)
        self.activation = nn.GELU()

    @staticmethod
    def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply softmax over valid source positions only."""
        # Invalid query/source pairs are assigned a very small value before
        # softmax, preventing padded frames or tokens from receiving attention.
        masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(masked_scores, dim=-1)
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs * mask.to(probs.dtype)
        # Renormalize after masking so rows with at least one valid source sum
        # to one, while fully masked rows safely remain zero.
        denom = probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return probs / denom

    def _cross_attention(
        self,
        query_name: str,
        source_name: str,
        query_hidden: torch.Tensor,
        source_hidden: torch.Tensor,
        query_mask: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project one query/source modality pair and return attended source features."""
        # Shapes: q=[B, query_len, D], k/v=[B, source_len, D].
        q = self.q_proj[query_name](query_hidden)
        k = self.k_proj[source_name](source_hidden)
        v = self.v_proj[source_name](source_hidden)

        # Dot-product scores have shape [B, query_len, source_len].
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.cross_attn_dim)
        # A pair is valid only when both the query position and source position
        # correspond to real sequence elements.
        pair_mask = query_mask.unsqueeze(-1) & source_mask.unsqueeze(1)

        attn = self._masked_softmax(scores, pair_mask)
        attn = self.attn_dropout(attn)
        attended = torch.matmul(attn, v)
        # Keep padded query positions at zero before they are added back through
        # the residual path.
        attended = attended * query_mask.unsqueeze(-1).to(attended.dtype)
        return attended, q

    def forward(
        self,
        modal_hidden: Dict[str, torch.Tensor],
        modal_masks: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Update each modality using cross-attention from the other modalities."""
        updated_hidden: Dict[str, torch.Tensor] = {}
        cross_outputs: Dict[str, torch.Tensor] = {}
        queries: Dict[str, torch.Tensor] = {}

        for modality in MODALITIES:
            query_hidden = modal_hidden[modality]
            query_mask = modal_masks[modality]
            query_repr: Optional[torch.Tensor] = None
            cross_sum: Optional[torch.Tensor] = None

            # For each target modality, collect the two incoming source
            # modalities and sum their attended representations.
            for _, query_name, source_name in DIRECTIONAL_PAIRS:
                if query_name != modality:
                    continue
                attended, q = self._cross_attention(
                    query_name=query_name,
                    source_name=source_name,
                    query_hidden=query_hidden,
                    source_hidden=modal_hidden[source_name],
                    query_mask=query_mask,
                    source_mask=modal_masks[source_name],
                )
                query_repr = q if query_repr is None else query_repr
                cross_sum = attended if cross_sum is None else cross_sum + attended

            if query_repr is None or cross_sum is None:
                raise RuntimeError(f"Failed to build cross attention for modality: {modality}")

            # Transformer-style update: cross-attention residual block followed
            # by a feed-forward residual block.
            attn_update = self.hidden_dropout(self.attn_out_proj[modality](cross_sum))
            hidden = self.attn_norm[modality](query_hidden + attn_update)
            ff = self.ffn_out[modality](self.activation(self.ffn_in[modality](hidden)))
            hidden = self.ffn_norm[modality](hidden + self.hidden_dropout(ff))
            hidden = hidden * query_mask.unsqueeze(-1).to(hidden.dtype)

            updated_hidden[modality] = hidden
            cross_outputs[modality] = cross_sum
            queries[modality] = query_repr

        return updated_hidden, cross_outputs, queries


class CrossAttentionEncoder(nn.Module):
    """Stacked cross-attention encoder."""

    def __init__(
        self,
        hidden_size: int,
        cross_attn_dim: int,
        num_layers: int,
        intermediate_size: int,
        attention_dropout_prob: float,
        hidden_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(
                    hidden_size=hidden_size,
                    cross_attn_dim=cross_attn_dim,
                    intermediate_size=intermediate_size,
                    attention_dropout_prob=attention_dropout_prob,
                    hidden_dropout_prob=hidden_dropout_prob,
                    layer_norm_eps=layer_norm_eps,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        modal_hidden: Dict[str, torch.Tensor],
        modal_masks: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run the stacked cross-attention layers and return the last layer views."""
        last_cross_outputs: Optional[Dict[str, torch.Tensor]] = None
        last_queries: Optional[Dict[str, torch.Tensor]] = None
        for layer in self.layers:
            # Intermediate layers update the modality streams; the last layer's
            # cross outputs and query projections are pooled downstream.
            modal_hidden, last_cross_outputs, last_queries = layer(modal_hidden, modal_masks)
        if last_cross_outputs is None or last_queries is None:
            raise RuntimeError("Cross encoder did not produce outputs")
        return modal_hidden, last_cross_outputs, last_queries


class CrossAttentionRegressor(nn.Module):
    """Frozen-BERT + A/V BiLSTM encoders followed by cross attention."""

    def __init__(self, args: argparse.Namespace, visual_dim: int, audio_dim: int):
        super().__init__()
        # Text uses token-level BERT representations; visual/audio use BiLSTMs
        # over aligned CSD feature windows, all projected to the same size.
        self.text_encoder = mmc.BertTokenSequenceEncoder(
            model_name=args.text_model_name,
            hidden_size=args.hidden_size,
            max_tlen=args.max_tlen,
            hidden_dropout_prob=args.hidden_dropout_prob,
            attention_dropout_prob=args.attention_dropout_prob,
            layer_norm_eps=args.layer_norm_eps,
            freeze_text_encoder=args.freeze_text_encoder,
        )
        self.visual_encoder = mmc.AVBiLSTMSequenceEncoder(
            input_dim=visual_dim,
            hidden_size=args.hidden_size,
            max_len=args.max_vlen,
            modality_id=1,
            lstm_hidden_size=args.av_bilstm_hidden_size,
            num_layers=args.av_bilstm_layers,
            hidden_dropout_prob=args.hidden_dropout_prob,
            layer_norm_eps=args.layer_norm_eps,
        )
        self.audio_encoder = mmc.AVBiLSTMSequenceEncoder(
            input_dim=audio_dim,
            hidden_size=args.hidden_size,
            max_len=args.max_alen,
            modality_id=2,
            lstm_hidden_size=args.av_bilstm_hidden_size,
            num_layers=args.av_bilstm_layers,
            hidden_dropout_prob=args.hidden_dropout_prob,
            layer_norm_eps=args.layer_norm_eps,
        )
        self.fusion = CrossAttentionEncoder(
            hidden_size=args.hidden_size,
            cross_attn_dim=args.cross_attn_dim,
            num_layers=args.cross_num_layers,
            intermediate_size=args.cross_intermediate_size,
            attention_dropout_prob=args.attention_dropout_prob,
            hidden_dropout_prob=args.hidden_dropout_prob,
            layer_norm_eps=args.layer_norm_eps,
        )
        # The representation keeps two views per modality: cross-modal
        # information from attended sources and modality-local query features.
        self.inter_proj = nn.ModuleDict({name: nn.Linear(args.cross_attn_dim, args.fusion_dim) for name in MODALITIES})
        self.intra_proj = nn.ModuleDict({name: nn.Linear(args.cross_attn_dim, args.fusion_dim) for name in MODALITIES})
        self.output_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.regressor = mmc.MLPRegressionHead(
            input_dim=args.fusion_dim,
            hidden_dim=args.fusion_dim,
            hidden_dropout_prob=args.hidden_dropout_prob,
        )

    def _encode_modalities(
        self,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Encode text, visual, and audio sequences into a common hidden space."""
        text_hidden, text_valid = self.text_encoder(input_ids, text_mask)
        visual_hidden, visual_valid = self.visual_encoder(visual, visual_mask)
        audio_hidden, audio_valid = self.audio_encoder(audio, audio_mask)
        return (
            {"text": text_hidden, "visual": visual_hidden, "audio": audio_hidden},
            {"text": text_valid, "visual": visual_valid, "audio": audio_valid},
        )

    def _predict_from_fused(
        self,
        cross_outputs: Dict[str, torch.Tensor],
        queries: Dict[str, torch.Tensor],
        modal_masks: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Pool cross-modal and modality-local representations into one prediction."""
        joint: Optional[torch.Tensor] = None
        for modality in MODALITIES:
            # inter_vec summarizes information imported from other modalities;
            # intra_vec summarizes the modality's own query-side representation.
            inter_vec = torch.relu(self.inter_proj[modality](mmc.masked_mean(cross_outputs[modality], modal_masks[modality])))
            intra_vec = torch.relu(self.intra_proj[modality](mmc.masked_mean(queries[modality], modal_masks[modality])))
            current = inter_vec + intra_vec
            joint = current if joint is None else joint + current
        if joint is None:
            raise RuntimeError("Joint representation was not constructed")
        return self.regressor(self.output_dropout(joint))

    def forward(
        self,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        audio: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the complete multimodal forward pass and return a regression score."""
        modal_hidden, modal_masks = self._encode_modalities(input_ids, text_mask, visual, visual_mask, audio, audio_mask)
        _, cross_outputs, queries = self.fusion(modal_hidden, modal_masks)
        return self._predict_from_fused(cross_outputs, queries, modal_masks)


def build_arg_parser() -> argparse.ArgumentParser:
    """Define CLI overrides for data paths, model size, and training settings."""
    parser = argparse.ArgumentParser(description="Cross-attention CMU-MOSEI regressor")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default=common.DEFAULT_DATASET_NAME)
    parser.add_argument("--mosei_compseq_dir", type=str, default=common.DEFAULT_MOSEI_COMPSEQ_DIR)
    parser.add_argument("--audio_csd", type=str, default=common.DEFAULT_AUDIO_CSD)
    parser.add_argument("--visual_csd", type=str, default=common.DEFAULT_VISUAL_CSD)
    parser.add_argument("--feature_cache_path", type=str, default="")
    parser.add_argument("--rebuild_feature_cache", action="store_true")
    parser.add_argument("--storage_dtype", type=str, default="float32", choices=("float16", "float32"))
    parser.add_argument("--apply_av_zscore", action="store_true")
    parser.add_argument("--no_apply_av_zscore", dest="apply_av_zscore", action="store_false")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--fallback_to_asr_for_char_spaced", action="store_true")
    parser.add_argument("--neutral_eps", type=float, default=0.0)
    parser.add_argument("--max_tlen", type=int, default=50)
    parser.add_argument("--max_vlen", type=int, default=70)
    parser.add_argument("--max_alen", type=int, default=80)
    parser.add_argument("--window_pad_sec", type=float, default=0.0)
    parser.add_argument("--video_cache_size", type=int, default=64)
    parser.add_argument("--batch_train", type=int, default=None)
    parser.add_argument("--batch_eval", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--base_lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selection_metric", type=str, default=None, choices=("acc2", "f1", "acc7", "corr", "mae"))
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_epochs", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--av_bilstm_hidden_size", type=int, default=None)
    parser.add_argument("--av_bilstm_layers", type=int, default=None)
    parser.add_argument("--cross_attn_dim", type=int, default=None)
    parser.add_argument("--cross_num_layers", type=int, default=None)
    parser.add_argument("--cross_intermediate_size", type=int, default=None)
    parser.add_argument("--fusion_dim", type=int, default=None)
    parser.add_argument("--hidden_dropout_prob", type=float, default=None)
    parser.add_argument("--attention_dropout_prob", type=float, default=None)
    parser.add_argument("--layer_norm_eps", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--freeze_text_encoder", action="store_true")
    parser.add_argument("--unfreeze_text_encoder", dest="freeze_text_encoder", action="store_false")
    parser.add_argument("--no_save_test_predictions", dest="save_test_predictions", action="store_false")
    parser.add_argument("--test_predictions_path", type=str, default="")
    parser.add_argument("--smoke_test", action="store_true")
    parser.set_defaults(apply_av_zscore=True, fallback_to_asr_for_char_spaced=True, freeze_text_encoder=True, save_test_predictions=True)
    return parser


def apply_model_defaults(args: argparse.Namespace) -> None:
    """Fill unspecified CLI arguments with the model defaults."""
    for key, value in MODEL_DEFAULTS.items():
        if getattr(args, key, None) in (None, ""):
            setattr(args, key, value)
    # Output paths are derived after max lengths are known so the cache file
    # name records the sequence truncation used for the run.
    if args.output_dir == "":
        args.output_dir = "outputs/cross_attention"
    if args.save_path == "":
        args.save_path = str(Path(args.output_dir) / f"{args.run_name}.pt")
    if args.feature_cache_path == "":
        args.feature_cache_path = f"outputs/cache_av_t{args.max_tlen}_v{args.max_vlen}_a{args.max_alen}.pt"


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
) -> Tuple[float, Dict[str, object], Dict[str, float], np.ndarray, np.ndarray]:
    """Run prediction on a loader and compute loss plus report metrics."""
    preds, labels = mmc.predict_multimodal_scores(model, loader, DEVICE)
    loss = float(np.mean((preds - labels) ** 2)) if preds.size else float("nan")
    metrics = common.evaluate_predictions(preds, labels)
    binary = common.binary_metrics(preds, labels, accbi=True)
    return loss, metrics, binary, preds, labels


def append_metrics(
    history: Dict[str, List[float]],
    prefix: str,
    loss: float,
    metrics: Dict[str, object],
    binary: Dict[str, float],
) -> None:
    """Append one train/validation metric snapshot to the history dictionary."""
    history[f"{prefix}_loss"].append(loss)
    history[f"{prefix}_mae"].append(metrics["regression"]["mae"])
    history[f"{prefix}_corr"].append(metrics["regression"]["corr"])
    history[f"{prefix}_acc7"].append(metrics["regression"]["acc7"])
    history[f"{prefix}_acc2"].append(binary["accuracy"])
    history[f"{prefix}_f1"].append(binary["f1_weighted"])


def make_history() -> Dict[str, List[float]]:
    """Create the metric history structure used for JSON output and curve plotting."""
    keys = ["loss", "mae", "corr", "acc7", "acc2", "f1"]
    history = {"train_optim_loss": [], "val_selection_score": []}
    for prefix in ("train", "val"):
        for key in keys:
            history[f"{prefix}_{key}"] = []
    return history


def summarize_metrics(metrics: Dict[str, object], binary: Dict[str, float]) -> Dict[str, float]:
    """Flatten nested metric dictionaries into the compact result summary."""
    return {
        "mae": metrics["regression"]["mae"],
        "corr": metrics["regression"]["corr"],
        "acc7": metrics["regression"]["acc7"],
        "acc2": binary["accuracy"],
        "f1": binary["f1_weighted"],
        "n_acc2": binary["n"],
    }


def main() -> None:
    """Train, evaluate, and save a no-gating cross-attention model."""
    parser = build_arg_parser()
    args = parser.parse_args()
    apply_model_defaults(args)

    # Smoke tests keep the full pipeline intact while reducing runtime.
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.early_stopping_patience = min(args.early_stopping_patience, 2)
        args.early_stopping_min_epochs = min(args.early_stopping_min_epochs, 2)

    print(f"Device : {DEVICE}")
    print(f"Config : {vars(args)}\n")
    common.set_seed(args.seed)

    # Load text splits and aligned A/V feature tensors.
    raw = load_dataset(args.dataset_name)
    tokenizer = BertTokenizerFast.from_pretrained(args.text_model_name)
    splits, reports = common.load_or_build_av_feature_cache(args)
    if args.smoke_test:
        common.smoke_limit_av_splits(splits)

    # Build train/eval dataloaders over the shared multimodal dataset wrapper.
    train_ds = mmc.MoseiMultimodalDataset(raw["train"], splits["train"], tokenizer, args.max_tlen, args.text_field, args.fallback_to_asr_for_char_spaced)
    val_ds = mmc.MoseiMultimodalDataset(raw["validation"], splits["validation"], tokenizer, args.max_tlen, args.text_field, args.fallback_to_asr_for_char_spaced)
    test_ds = mmc.MoseiMultimodalDataset(raw["test"], splits["test"], tokenizer, args.max_tlen, args.text_field, args.fallback_to_asr_for_char_spaced)
    train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    visual_dim = int(splits["train"]["visual"].shape[-1])
    audio_dim = int(splits["train"]["audio"].shape[-1])

    # Infer audio/visual feature dimensions from the cached tensors before
    # constructing the modality encoders.
    model = CrossAttentionRegressor(args, visual_dim=visual_dim, audio_dim=audio_dim).to(DEVICE)
    print(f"Parameters : {mmc.trainable_parameter_report(model)}")

    optimizer = AdamW(model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    loss_fn = nn.MSELoss()

    history = make_history()
    best_epoch = -1
    best_val_score = -float("inf")
    patience_counter = 0

    # Train until early stopping, keeping the best validation checkpoint.
    for epoch in range(1, args.epochs + 1):
        print(f"--- Epoch {epoch}/{args.epochs} ---", flush=True)
        train_optim_loss = mmc.train_multimodal_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, DEVICE, epoch, args.max_grad_norm
        )
        train_loss, train_metrics, train_binary, _, _ = evaluate_loader(model, train_eval_loader)
        val_loss, val_metrics, val_binary, _, _ = evaluate_loader(model, val_loader)
        selection_score = common.metric_for_selection(val_metrics, args.selection_metric, val_binary)

        history["train_optim_loss"].append(train_optim_loss)
        append_metrics(history, "train", train_loss, train_metrics, train_binary)
        append_metrics(history, "val", val_loss, val_metrics, val_binary)
        history["val_selection_score"].append(selection_score)

        print(
            f"Epoch {epoch} | TrainOptimLoss={train_optim_loss:.4f} | TrainLoss={train_loss:.4f} | "
            f"Train MAE={train_metrics['regression']['mae']:.4f} | Train Corr={train_metrics['regression']['corr']:.4f}"
        )
        print(
            f"             ValLoss={val_loss:.4f} | Val MAE={val_metrics['regression']['mae']:.4f} | "
            f"Val Corr={val_metrics['regression']['corr']:.4f} | Val Acc7={val_metrics['regression']['acc7']:.4f}"
        )
        print(f"  Selection score ({args.selection_metric}) = {selection_score:.4f}")

        if selection_score > best_val_score + args.early_stopping_min_delta:
            # Store the best validation checkpoint, not necessarily the last
            # epoch, so test evaluation uses the selected model state.
            best_val_score = selection_score
            best_epoch = epoch
            patience_counter = 0
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": vars(args), "best_epoch": best_epoch, "best_val_score": best_val_score}, args.save_path)
            print(f"  --> Saved best checkpoint: {args.save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.early_stopping_patience})")
            if epoch >= args.early_stopping_min_epochs and patience_counter >= args.early_stopping_patience:
                print(f"  Early stopping triggered at epoch {epoch}.")
                break
        print()

    if best_epoch < 0:
        raise RuntimeError("Training did not produce a best checkpoint.")

    checkpoint = common.torch_load(args.save_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])

    # Re-evaluate the best checkpoint and write reproducibility artifacts.
    val_loss, val_metrics, val_binary, _, _ = evaluate_loader(model, val_loader)
    test_loss, test_metrics, test_binary, test_preds, test_labels = evaluate_loader(model, test_loader)
    val_summary = summarize_metrics(val_metrics, val_binary)
    test_summary = summarize_metrics(test_metrics, test_binary)

    print("\n[VAL] Metrics")
    print(json.dumps(val_metrics, indent=2))
    print("\n[TEST] Metrics")
    print(json.dumps(test_metrics, indent=2))
    print("\n[TEST] Summary")
    print(json.dumps(test_summary, indent=2))

    pred_path = ""
    if args.save_test_predictions:
        # Prediction sidecars make later error analysis possible without
        # re-running the model.
        pred_path = args.test_predictions_path or common.sidecar_path(args.save_path, "_test_predictions.csv")
        common.save_test_predictions_csv(pred_path, test_preds, test_labels, hf_indices=splits["test"]["hf_indices"], seg_keys=splits["test"]["seg_keys"])
        print(f"Test predictions saved to {pred_path}")

    # Keep the full history alongside compact summaries so tables and training
    # curves can be recreated from the JSON artifact.
    result_payload = {
        "config": vars(args),
        "model_defaults": MODEL_DEFAULTS,
        "config_name": "cross_attention_nogate",
        "parameter_report": mmc.trainable_parameter_report(model),
        "feature_reports": reports,
        "best_epoch": checkpoint.get("best_epoch", best_epoch),
        "best_val_score": checkpoint.get("best_val_score", best_val_score),
        "history": history,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_summary": val_summary,
        "test_summary": test_summary,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "checkpoint_path": args.save_path,
        "test_predictions_path": pred_path,
    }
    json_path = common.sidecar_path(args.save_path, "_results.json")
    common.save_json(json_path, result_payload)
    print(f"Results JSON saved to {json_path}")

    # Use the shared curve style for unimodal and multimodal runs.
    plot_path = common.sidecar_path(args.save_path, "_training_curves.png")
    common.plot_training_curves(history, int(checkpoint.get("best_epoch", best_epoch)), plot_path, "Cross-Attention")
    print(f"Training curves saved to {plot_path}")


if __name__ == "__main__":
    main()
