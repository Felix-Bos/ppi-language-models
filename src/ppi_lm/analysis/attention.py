"""Attention analyses (Transformer only): maps, cross-protein mass, sinks, entropy.

For one pair, the encoder returns n_layers tensors of attention weights (n_heads, L, L):
row i is how token i distributes its attention over every token j (rows sum to 1).
"""

from dataclasses import dataclass

import torch

from ppi_lm.analysis.inference import make_loader


@dataclass
class PairAttention:
    pair: tuple[str, str, int]
    tokens: list[str]  # (L,) e.g. ["<CLS>", "M", "K", ..., "<SEP>", ..., "<EOS>"]
    segment_ids: torch.Tensor  # (L,)
    weights: list[torch.Tensor]  # n_layers x (n_heads, L, L)
    logit: float

    @property
    def residues_a(self) -> torch.Tensor:
        """Positions of the residues of A (special tokens excluded)."""
        return self._residues(0)

    @property
    def residues_b(self) -> torch.Tensor:
        return self._residues(1)

    def _residues(self, segment: int) -> torch.Tensor:
        special = torch.tensor([t.startswith("<") for t in self.tokens])
        return ((self.segment_ids == segment) & ~special).nonzero().squeeze(1)

    def stacked(self) -> torch.Tensor:
        """(n_layers, n_heads, L, L)."""
        return torch.stack(self.weights)


@torch.no_grad()
def pair_attention(model, data, pair: tuple[str, str, int]) -> PairAttention:
    """Attention weights of every layer and head for one pair (explicit attention path)."""
    if not hasattr(model.encoder, "forward_with_attention"):
        raise TypeError("This encoder has no attention maps (Transformer only)")
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(make_loader(model, data, [pair], batch_size=1)))
    batch = {key: value.to(device) for key, value in batch.items()}
    mask, seg = batch["attention_mask"], batch["segment_ids"]

    hidden, weights = model.encoder.forward_with_attention(batch["input_ids"], seg, mask)
    logit = model.head(model.pool(hidden, mask, seg)).item()
    tokens = [data.tokenizer.id2token[i] for i in batch["input_ids"][0].tolist()]
    return PairAttention(
        pair=pair,
        tokens=tokens,
        segment_ids=seg[0].cpu(),
        weights=[w[0].float().cpu() for w in weights],
        logit=logit,
    )


def cross_protein_mass(att: PairAttention) -> torch.Tensor:
    """Share of the attention of A's residues that goes to B's residues, and B -> A.

    returns: (n_layers, n_heads, 2), [..., 0] = A -> B, [..., 1] = B -> A, in [0, 1]
    A value close to 0 means the head stays inside each protein; the partner is ignored.
    """
    w = att.stacked()  # (layers, heads, L, L)
    a, b = att.residues_a, att.residues_b
    a_to_b = w[:, :, a][:, :, :, b].sum(-1).mean(-1)
    b_to_a = w[:, :, b][:, :, :, a].sum(-1).mean(-1)
    return torch.stack([a_to_b, b_to_a], dim=-1)


def sink_mass(att: PairAttention) -> torch.Tensor:
    """Share of the attention (of every token) that goes to special tokens (<CLS>, <SEP>, <EOS>).

    Heads that send most of their attention to a few tokens ("attention sinks") carry
    little information about residue interactions: exclude them before interpreting.
    returns: (n_layers, n_heads)
    """
    special = torch.tensor([t.startswith("<") for t in att.tokens])
    return att.stacked()[..., special].sum(-1).mean(-1)


def attention_focus(att: PairAttention) -> torch.Tensor:
    """1 - normalized entropy of each row, averaged: 0 = uniform attention, 1 = one token.

    returns: (n_layers, n_heads)
    """
    w = att.stacked().clamp_min(1e-12)
    entropy = -(w * w.log()).sum(-1)  # (layers, heads, L)
    return (1 - entropy / torch.log(torch.tensor(float(w.shape[-1])))).mean(-1)


def a_to_b_block(att: PairAttention, layer: int | None = None) -> torch.Tensor:
    """Attention from A's residues (rows) to B's residues (columns), mean over heads.

    layer: None = mean over layers too
    returns: (n_residues_a, n_residues_b)
    """
    w = att.stacked().mean(1)  # (layers, L, L)
    w = w.mean(0) if layer is None else w[layer]
    return w[att.residues_a][:, att.residues_b]
