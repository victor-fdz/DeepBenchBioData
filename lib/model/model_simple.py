#!/usr/bin/env python3
"""Minimal Siamese CNN without dropout or batch normalisation.

This is an experimental variant of the main model used for quick
architecture trials. It accepts the same constructor arguments as
``model.py`` (e.g. ``dropout_rate``, ``embedding_dim``) but ignores
them, so it can be swapped in without changing any CLI calls or
training scripts.

To use this variant instead of the default model, change the import
in ``train_model.py`` and ``evaluate_model.py``:

    # from lib.model.model import SiameseCNN
    from lib.model.model_simple import SiameseCNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseCNN(nn.Module):
    """Minimal Siamese CNN with two conv layers and two FC layers.

    Architecture per branch:
        Conv1d(4→16, k=5) → MaxPool → Conv1d(16→32, k=5) → MaxPool
        → Flatten → Linear(4000→128) → Linear(128→64)

    Args:
        **kwargs: Accepted but ignored (e.g. ``dropout_rate``,
            ``embedding_dim``) for CLI compatibility with other model
            variants.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()

        if kwargs:
            ignored = ", ".join(kwargs)
            import logging
            logging.getLogger(__name__).warning(
                "model_simple.SiameseCNN ignores constructor args: %s", ignored
            )

        self.conv1 = nn.Conv1d(in_channels=4,  out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool  = nn.MaxPool1d(kernel_size=2, stride=2)
        #self.dropout = nn.Dropout(p=0.5) ################
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # After two pooling steps: 500 → 250 → 125
        self.fc1 = nn.Linear(32 * 125, 128)
        self.fc2 = nn.Linear(128, 64)

    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a single sequence batch.

        Args:
            x: Input tensor of shape ``(N, 4, 500)``.

        Returns:
            Embedding tensor of shape ``(N, 64)``.
        """
        x = self.pool(F.relu(self.conv1(x)))  # (N, 16, 250)
        x = self.pool(F.relu(self.conv2(x)))  # (N, 32, 125)
        x = torch.flatten(x, start_dim=1)     # (N, 4000)
        x = F.relu(self.fc1(x))               # (N, 128)
        # x = self.dropout(x) ##########################
        x = self.fc2(x)                        # (N, 64)
        return x
   
    def forward(
        self,
        input1: torch.Tensor,
        input2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for a pair of sequences.

        Args:
            input1: Human sequence batch ``(N, 4, 500)``.
            input2: Mouse sequence batch ``(N, 4, 500)``.

        Returns:
            Tuple of embedding tensors ``(emb1, emb2)``, each ``(N, 64)``.
        """
        return self.forward_one(input1), self.forward_one(input2)
