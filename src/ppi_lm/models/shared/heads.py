import torch

"""Pooling functions and task heads shared by every encoder.

Every encoder returns hidden states of shape (B, L, d_model). The pooling functions
summarize a pair into one vector, which is the input of the classification head.
"""


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    """Only return the first token of the hidden states, which is the <CLS> token.
    hidden: (B, L, d_model)
    returns: (B, d_model)
    """
    return hidden[:, 0, :]


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Averages token vectors over positions where mask is True.

    hidden: (B, L, d_model)
    mask: (B, L) bool
    returns: (B, d_model) — zeros for a row with no True position (no NaN)
    """
    # (B, L) -> (B, L, 1), 1.0 on real tokens, 0.0 on padding
    mask = mask.unsqueeze(-1).to(hidden.dtype)

    # zero out padding, then sum over tokens: (B, d_model)
    summed = (hidden * mask).sum(dim=1)

    # number of real tokens per row: (B, 1), at least 1 to avoid 0 / 0
    counts = mask.sum(dim=1).clamp(min=1.0)

    return summed / counts


def masked_max_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Max-pools token vectors over positions where mask is True.

    hidden: (B, L, d_model)
    mask: (B, L) bool
    returns: (B, d_model) — zeros for a row with no True position (no NaN)
    """
    # (B, L) -> (B, L, 1), stays boolean
    mask = mask.unsqueeze(-1)

    # replace padding by -inf so it can never win the max
    masked_hidden = hidden.masked_fill(~mask, float("-inf"))
    maxed = masked_hidden.max(dim=1).values  # (B, d_model)

    # rows with no real token are all -inf: return zeros instead
    return torch.where(torch.isinf(maxed), torch.zeros_like(maxed), maxed)


def pair_pool(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Pools protein A and protein B separately, then combines them symmetrically.

    a = mean over real tokens of segment 0, b = mean over real tokens of segment 1,
    returns concat([a * b, |a - b|]) — identical if A and B are swapped.

    hidden: (B, L, d)
    attention_mask: (B, L) bool
    segment_ids: (B, L) long, 0 for A, 1 for B
    returns: (B, 2 * d)
    """
    a = masked_mean_pool(hidden, attention_mask & (segment_ids == 0))  # (B, d)
    b = masked_mean_pool(hidden, attention_mask & (segment_ids == 1))  # (B, d)

    return torch.cat([a * b, (a - b).abs()], dim=-1)  # (B, 2 * d)


class ClassificationHead(torch.nn.Module):
    pass


class MLMHead(torch.nn.Module):
    pass
