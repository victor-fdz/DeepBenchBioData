#!/usr/bin/env python3
"""Siamese CNN with attention for promoter sequence embedding."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseCNN(nn.Module):
    """Multi-scale Siamese CNN with self-attention.

    Per branch:
        DNA one-hot input
        -> multi-scale convolutions
        -> secondary convolution
        -> self-attention over promoter positions
        -> global average pooling
        -> compact normalized embedding
    """

    def __init__(
        self,
        dropout_rate: float = 0.3,
        small_kernel_size: int = 6,
        medium_kernel_size: int = 12,
        large_kernel_size: int = 20,
        attention_heads: int = 4,
        embedding_dim: int = 16,
        transformer_layers: int = 1,
    ) -> None:
        super().__init__()

        if attention_heads <= 0:
            raise ValueError("attention_heads must be a positive integer.")

        if 32 % attention_heads != 0:
            raise ValueError("attention_heads must divide the attention embedding size 32.")

        for kernel_name, kernel_size in {
            "small_kernel_size": small_kernel_size,
            "medium_kernel_size": medium_kernel_size,
            "large_kernel_size": large_kernel_size,
        }.items():
            if kernel_size <= 0:
                raise ValueError(f"{kernel_name} must be a positive integer.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be a positive integer.")

        self.model_config = {
            "dropout_rate": float(dropout_rate),
            "small_kernel_size": int(small_kernel_size),
            "medium_kernel_size": int(medium_kernel_size),
            "large_kernel_size": int(large_kernel_size),
            "attention_heads": int(attention_heads),
            "embedding_dim": int(embedding_dim),
            "transformer_layers": int(transformer_layers),
        }

        self.conv_s = nn.Conv1d(
            in_channels=4,
            out_channels=8,
            kernel_size=small_kernel_size,
            padding=small_kernel_size // 2,
        )

        self.conv_m = nn.Conv1d(
            in_channels=4,
            out_channels=8,
            kernel_size=medium_kernel_size,
            padding=medium_kernel_size // 2,
        )

        self.conv_l = nn.Conv1d(
            in_channels=4,
            out_channels=8,
            kernel_size=large_kernel_size,
            padding=large_kernel_size // 2,
        )

        self.bn1 = nn.BatchNorm1d(24)
        self.drop1 = nn.Dropout(p=dropout_rate)

        self.conv2 = nn.Conv1d(
            in_channels=24,
            out_channels=32,
            kernel_size=8,
            padding=4,
        )

        self.bn2 = nn.BatchNorm1d(32)
        self.drop2 = nn.Dropout(p=dropout_rate)

        self.max_transformer_length = 256

        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.max_transformer_length, 32)
            )

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=attention_heads,
            dim_feedforward=32,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=transformer_layers,
            )        

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop3 = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(32, embedding_dim)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode one promoter sequence batch."""

        xs = F.relu(self.conv_s(x))
        xm = F.relu(self.conv_m(x))
        xl = F.relu(self.conv_l(x))

        min_len = min(
            xs.shape[2],
            xm.shape[2],
            xl.shape[2],
        )

        x = torch.cat(
            [
                xs[:, :, :min_len],
                xm[:, :, :min_len],
                xl[:, :, :min_len],
            ],
            dim=1,
        )

        x = self.drop1(self.bn1(x))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.conv2(x))
        x = self.drop2(self.bn2(x))
        x = F.max_pool1d(x, kernel_size=2)


        # ================ TRANSFORMER ================
        transformer_input = x.transpose(1, 2)

        sequence_length = transformer_input.shape[1]

        if sequence_length > self.max_transformer_length:
            raise ValueError(
                f"Transformer input length {sequence_length} exceeds "
                f"max_transformer_length={self.max_transformer_length}."
            )

        transformer_input = (
            transformer_input
            + self.position_embedding[:, :sequence_length, :]
        )

        transformer_output = self.transformer_encoder(transformer_input)

        x = transformer_output.transpose(1, 2)
        # ===========================================


        x = self.gap(x).squeeze(-1)
        x = self.fc(self.drop3(x))

        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode two promoter sequence batches with shared weights."""

        return (
            self.forward_one(input1),
            self.forward_one(input2),
        )
