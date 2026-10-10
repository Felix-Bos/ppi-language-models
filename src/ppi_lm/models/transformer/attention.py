"""Bidirectional multi-head self-attention with rotary position embeddings.

No causal mask: every token attends to every real token of the pair, on both sides,
as required by masked language modeling. Padding is never attended to (key mask).

Two paths compute the same thing:
- fast path: torch's fused scaled_dot_product_attention, used for training;
- explicit path: softmax(q k^T / sqrt(d_head)) written by hand, which also returns the
  attention weights (need_weights=True), used for interpretability.
"""

import math

import torch
import torch.nn.functional as F

from ppi_lm.models.transformer.rope import RotaryEmbedding


class MultiHeadSelfAttention(torch.nn.Module):
    """Self-attention over a padded batch, RoPE applied to queries and keys."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_len: int = 2048,
    ):
        """
        d_model: dimension of the hidden states
        n_heads: number of heads, must divide d_model (d_head = d_model // n_heads, even)
        dropout: dropout on the attention weights
        rope_base, max_len: see RotaryEmbedding

        raises ValueError if n_heads does not divide d_model, or if d_head is odd
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.qkv = torch.nn.Linear(d_model, 3 * d_model)  # q, k, v in one matmul
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.rope = RotaryEmbedding(self.d_head, max_len=max_len, base=rope_base)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: (B, L, d_model)
        attention_mask: (B, L) bool, True on real tokens
        need_weights: also return the attention weights (explicit path)
        returns: output (B, L, d_model), weights (B, n_heads, L, L) or None
        """
        batch, seq_len, _ = x.shape

        # (B, L, 3 * d_model) -> 3 x (B, n_heads, L, d_head)
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        q, k = self.rope(q, k)

        # keys on padding are masked for every query: (B, 1, 1, L), True = attend
        key_mask = attention_mask[:, None, None, :]
        p = self.dropout if self.training else 0.0

        if need_weights:
            scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)  # (B, H, L, L)
            scores = scores.masked_fill(~key_mask, float("-inf"))
            weights = scores.softmax(dim=-1)
            context = F.dropout(weights, p=p) @ v  # (B, H, L, d_head)
        else:
            context = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask, dropout_p=p)
            weights = None

        # (B, H, L, d_head) -> (B, L, d_model)
        context = context.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(context), weights

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n_heads={self.n_heads}, dropout={self.dropout}"
