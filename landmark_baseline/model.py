"""PyTorch landmark baseline models."""

from __future__ import annotations

import torch
from torch import nn


class LandmarkTransformerClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        max_frames: int = 64,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_frames, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * mlp_ratio,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F], valid_mask: [B, T] with True for valid frames
        x = self.input_proj(x)
        x = x + self.pos_embed[:, : x.size(1)]
        key_padding_mask = ~valid_mask.bool()
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        valid = valid_mask.unsqueeze(-1).to(dtype=x.dtype)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return self.head(pooled)


class LandmarkGRUClassifier(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        hidden_size: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        lengths = valid_mask.sum(dim=1).to(dtype=torch.int64).clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        out = self.norm(out)
        valid = valid_mask.unsqueeze(-1).to(dtype=out.dtype)
        pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        pooled = self.dropout(pooled)
        return self.head(pooled)


def build_landmark_model(
    model_type: str,
    feature_dim: int,
    num_classes: int,
    max_frames: int,
    d_model: int,
    num_heads: int,
    num_layers: int,
    dropout: float,
    gru_hidden_size: int,
    gru_bidirectional: bool,
) -> nn.Module:
    if model_type == 'transformer':
        return LandmarkTransformerClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            max_frames=max_frames,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
    if model_type == 'gru':
        return LandmarkGRUClassifier(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_size=gru_hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=gru_bidirectional,
        )
    raise ValueError(f'Unsupported model_type: {model_type}')
