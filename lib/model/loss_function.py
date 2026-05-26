#!/usr/bin/env python3
"""Contrastive loss functions for Siamese training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLossCosine(nn.Module):
    """
    Contrastive loss using cosine distance.

    Positive pairs:
        minimize embedding distance

    Negative pairs:
        maximize distance above margin

    Args:
        - margin (float): Separation margin
    """

    def __init__(
        self,
        margin: float = 1.0,
    ) -> None:

        super().__init__()

        self.margin = margin

    def forward(
        self,
        output1: torch.Tensor,
        output2: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            - output1 (torch.Tensor): First embedding batch
            - output2 (torch.Tensor): Second embedding batch
            - label (torch.Tensor):
                1 = positive pair
                0 = negative pair

        Returns:
            torch.Tensor: Mean contrastive loss
        """

        cosine_sim = F.cosine_similarity(
            output1,
            output2,
        )

        distance = 1 - cosine_sim

        loss = (
            label * distance.pow(2)
            + (1 - label)
            * F.relu(self.margin - distance).pow(2)
        )

        return loss.mean()