"""Encoder + pooling + classification head: one interaction logit per pair.

Works with any encoder that maps (input_ids, segment_ids, attention_mask) to hidden
states (B, L, d_model) and exposes `d_model`, so the four architectures share it.
"""

import torch

from ppi_lm.models.shared.heads import (
    ClassificationHead,
    cls_pool,
    masked_max_pool,
    masked_mean_pool,
    pair_pool,
)

POOLINGS = ("pair", "mean", "max", "cls")


def pooled_dim(pooling: str, d_model: int) -> int:
    """Size of the pooled vector: 2 * d_model for pair ([a * b, |a - b|]), d_model otherwise."""
    if pooling not in POOLINGS:
        raise ValueError(f"Unknown pooling '{pooling}', expected one of {list(POOLINGS)}")
    return 2 * d_model if pooling == "pair" else d_model


class PairClassifier(torch.nn.Module):
    """(input_ids, segment_ids, attention_mask) -> interaction logits (B,)."""

    def __init__(self, encoder: torch.nn.Module, head: ClassificationHead, pooling: str = "pair"):
        """
        encoder: any encoder with a `d_model` attribute
        head: ClassificationHead whose input_dim matches the pooling
        pooling: pair | mean | max | cls

        raises ValueError if the pooling is unknown or does not match head.input_dim
        """
        super().__init__()
        expected = pooled_dim(pooling, encoder.d_model)
        if head.input_dim != expected:
            raise ValueError(
                f"Pooling '{pooling}' gives vectors of size {expected}, "
                f"but the head expects {head.input_dim}"
            )
        self.encoder = encoder
        self.head = head
        self.pooling = pooling

    def pool(
        self, hidden: torch.Tensor, attention_mask: torch.Tensor, segment_ids: torch.Tensor
    ) -> torch.Tensor:
        """hidden (B, L, d_model) -> pooled (B, pooled_dim)."""
        if self.pooling == "pair":
            return pair_pool(hidden, attention_mask, segment_ids)
        if self.pooling == "mean":
            return masked_mean_pool(hidden, attention_mask)
        if self.pooling == "max":
            return masked_max_pool(hidden, attention_mask)
        return cls_pool(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns raw logits (B,): apply a sigmoid for probabilities."""
        hidden = self.encoder(input_ids, segment_ids, attention_mask)
        return self.head(self.pool(hidden, attention_mask, segment_ids))
