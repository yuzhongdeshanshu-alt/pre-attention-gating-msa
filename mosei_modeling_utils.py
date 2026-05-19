"""Reusable model and dataset components for CMU-MOSEI experiments."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from transformers import BertConfig, BertModel, BertTokenizerFast

import mosei_data_eval_utils as common


MODALITIES = ("text", "visual", "audio")


def masked_mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool hidden states over valid positions."""
    mask_f = mask.unsqueeze(-1).to(hidden.dtype)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    return (hidden * mask_f).sum(dim=1) / denom


class MoseiMultimodalDataset(Dataset):
    """Tokenized text plus cached audio/visual tensors for one MOSEI split."""

    def __init__(
        self,
        hf_split,
        split_tensors: Dict[str, object],
        tokenizer: BertTokenizerFast,
        max_tlen: int,
        text_field: str,
        fallback_to_asr_for_char_spaced: bool,
    ):
        """Store aligned split tensors and text-tokenization settings."""
        self.hf_split = hf_split
        self.data = split_tensors
        self.tokenizer = tokenizer
        self.max_tlen = max_tlen
        self.text_field = text_field
        self.fallback_to_asr_for_char_spaced = fallback_to_asr_for_char_spaced
        self.size = int(self.data["scores"].shape[0])

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return one synchronized text/audio/visual training example."""
        hf_index = int(self.data["hf_indices"][idx])
        row = self.hf_split[hf_index]
        text = common.choose_text(
            row,
            text_field=self.text_field,
            fallback_to_asr_for_char_spaced=self.fallback_to_asr_for_char_spaced,
        )
        enc = self.tokenizer(
            text,
            max_length=self.max_tlen,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "sample_idx": torch.tensor(idx, dtype=torch.long),
            "hf_index": self.data["hf_indices"][idx].clone().long(),
            "input_ids": enc["input_ids"].squeeze(0),
            "text_mask": enc["attention_mask"].squeeze(0).bool(),
            "visual": self.data["visual"][idx].clone().float(),
            "visual_mask": self.data["visual_mask"][idx].clone().bool(),
            "audio": self.data["audio"][idx].clone().float(),
            "audio_mask": self.data["audio_mask"][idx].clone().bool(),
            "score": self.data["scores"][idx].clone().float(),
        }


class BertTokenSequenceEncoder(nn.Module):
    """BERT token encoder used by multimodal sequence models."""

    def __init__(
        self,
        model_name: str,
        hidden_size: int,
        max_tlen: int,
        hidden_dropout_prob: float,
        attention_dropout_prob: float,
        layer_norm_eps: float,
        freeze_text_encoder: bool,
    ):
        """Create a token-level BERT encoder projected to the model hidden size."""
        super().__init__()
        config = BertConfig.from_pretrained(
            model_name,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_dropout_prob,
        )
        self.max_tlen = max_tlen
        self.encoder = BertModel.from_pretrained(model_name, config=config)
        if freeze_text_encoder:
            # Freezing keeps the pretrained text representation fixed while the
            # projection and multimodal layers learn task-specific mappings.
            for param in self.encoder.parameters():
                param.requires_grad = False

        text_hidden_size = self.encoder.config.hidden_size
        self.proj: nn.Module
        if text_hidden_size == hidden_size:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(text_hidden_size, hidden_size)
        self.modal_embed = nn.Embedding(3, hidden_size)
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor, text_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode padded token ids and return masked hidden states."""
        bsz = input_ids.shape[0]
        hidden = self.encoder(
            input_ids=input_ids,
            attention_mask=text_mask.long(),
            return_dict=True,
        ).last_hidden_state
        hidden = self.proj(hidden)
        # Modality embeddings mark the text stream before it is concatenated
        # with visual and audio frame sequences.
        modal_ids = torch.zeros((bsz, self.max_tlen), dtype=torch.long, device=input_ids.device)
        hidden = hidden + self.modal_embed(modal_ids)
        hidden = self.dropout(self.norm(hidden))
        hidden = hidden * text_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, text_mask.bool()


class AVBiLSTMSequenceEncoder(nn.Module):
    """Encode cached audio or visual sequences with a BiLSTM."""

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
        """Create a packed BiLSTM encoder for one non-text modality."""
        super().__init__()
        if num_layers < 1:
            raise ValueError("BiLSTM encoder requires num_layers >= 1")
        self.max_len = max_len
        self.modality_id = modality_id
        # A zero-valued prefix slot mirrors the sequence layout used by the
        # downstream pooling code.
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
        """Encode padded frame sequences while preserving valid-frame masks."""
        bsz = seq.shape[0]
        special = self.special.expand(bsz, -1, -1)
        seq_full = torch.cat([special, seq], dim=1)
        seq_full = seq_full.to(self.output_proj.weight.dtype)
        seq_full = self.input_dropout(self.input_norm(seq_full))

        valid_mask = mask.bool()
        lengths = valid_mask.sum(dim=1).clamp_min(1).to(torch.long).cpu()
        # Packed sequences skip padded timesteps inside the recurrent encoder.
        packed = pack_padded_sequence(seq_full, lengths, batch_first=True, enforce_sorted=False)
        packed_hidden, _ = self.encoder(packed)
        hidden, _ = pad_packed_sequence(packed_hidden, batch_first=True, total_length=self.max_len)
        hidden = self.output_proj(hidden)

        pos_ids = torch.arange(self.max_len, device=seq.device).unsqueeze(0).expand(bsz, -1)
        modal_ids = torch.full((bsz, self.max_len), self.modality_id, dtype=torch.long, device=seq.device)
        hidden = hidden + self.pos_embed(pos_ids) + self.modal_embed(modal_ids)
        hidden = self.output_dropout(self.output_norm(hidden))
        hidden = hidden * valid_mask.unsqueeze(-1).to(hidden.dtype)
        return hidden, valid_mask


class MLPRegressionHead(nn.Module):
    """Two-layer bounded regression head for sentiment scores in [-3, 3]."""

    def __init__(self, input_dim: int, hidden_dim: int, hidden_dropout_prob: float):
        super().__init__()
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return (torch.tanh(x) * 3.0).squeeze(-1)


def trainable_parameter_report(model: nn.Module) -> Dict[str, int]:
    """Count total, trainable, and frozen parameters for logging."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    """Move tensor batch fields to the selected device."""
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def train_multimodal_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    loss_fn,
    device: torch.device,
    epoch: int,
    max_grad_norm: float,
) -> float:
    """Run one optimization epoch for a multimodal regression model."""
    model.train()
    total_loss = 0.0
    for step, batch in enumerate(loader):
        batch = batch_to_device(batch, device)
        preds = model(
            input_ids=batch["input_ids"],
            text_mask=batch["text_mask"],
            visual=batch["visual"],
            visual_mask=batch["visual_mask"],
            audio=batch["audio"],
            audio_mask=batch["audio_mask"],
        )
        targets = batch["score"]
        loss = loss_fn(preds.view(-1), targets.view(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch={epoch} step={step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
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


def predict_multimodal_scores(model: nn.Module, loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Collect predictions and gold scores from a multimodal data loader."""
    model.eval()
    preds: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            scores = model(
                input_ids=batch["input_ids"],
                text_mask=batch["text_mask"],
                visual=batch["visual"],
                visual_mask=batch["visual_mask"],
                audio=batch["audio"],
                audio_mask=batch["audio_mask"],
            )
            preds.append(scores.detach().cpu().numpy())
            labels.append(batch["score"].detach().cpu().numpy())
    if not preds:
        return np.asarray([], dtype=np.float32), np.asarray([], dtype=np.float32)
    return np.concatenate(preds), np.concatenate(labels)
