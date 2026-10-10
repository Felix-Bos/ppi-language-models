"""One Transformer encoder block, pre-LayerNorm as in ESM-2.

    x = x + Dropout(Attention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))

Normalizing before each sub-layer (instead of after, as in the original BERT) leaves the
residual path untouched, which makes training more stable and less sensitive to the
learning-rate warmup.
"""

import torch

from ppi_lm.models.transformer.attention import MultiHeadSelfAttention


class FeedForward(torch.nn.Module):
    """Position-wise MLP: Linear(d_model, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = torch.nn.Linear(d_model, d_ff)
        self.act = torch.nn.GELU()
        self.dropout = torch.nn.Dropout(dropout)
        self.fc2 = torch.nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) -> (B, L, d_model)."""
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class TransformerBlock(torch.nn.Module):
    """Pre-LN block: self-attention then feed-forward, each with a residual connection."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        rope_base: float = 10000.0,
        max_len: int = 2048,
    ):
        """
        d_model, n_heads: see MultiHeadSelfAttention
        d_ff: hidden size of the feed-forward (usually 4 * d_model)
        dropout: dropout on each residual branch and inside the feed-forward
        attention_dropout: dropout on the attention weights
        rope_base, max_len: see RotaryEmbedding
        """
        super().__init__()
        self.norm_attn = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(
            d_model, n_heads, dropout=attention_dropout, rope_base=rope_base, max_len=max_len
        )
        self.norm_ffn = torch.nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x: (B, L, d_model)
        attention_mask: (B, L) bool, True on real tokens
        returns: output (B, L, d_model), attention weights (B, n_heads, L, L) or None
        """
        attn_out, weights = self.attn(self.norm_attn(x), attention_mask, need_weights)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))
        return x, weights
