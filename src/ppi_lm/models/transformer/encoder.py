"""Bidirectional Transformer encoder (BERT / ESM-2 style) for protein pairs.

    input_ids, segment_ids  ->  token embedding + segment embedding  ->  dropout
                            ->  n_layers x TransformerBlock (pre-LN, RoPE)
                            ->  final LayerNorm  ->  hidden states (B, L, d_model)

Positions are encoded by RoPE inside the attention, so there is no learned position
embedding. The segment embedding tells the model which protein (A or B) each token
belongs to.

Configurable from the YAML (model.transformer section), e.g.:
    d_model: 256
    n_layers: 4
    n_heads: 8
    d_ff: null            # null -> 4 * d_model
    dropout: 0.1
    attention_dropout: 0.0
    rope_base: 10000.0
    max_len: 2048         # >= 2 * max_protein_length + 3
"""

from dataclasses import dataclass

import torch

from ppi_lm.models.transformer.block import TransformerBlock


@dataclass
class TransformerConfig:
    """Hyper-parameters of the encoder. Unknown keys (e.g. a typo in the YAML) raise."""

    vocab_size: int  # tokenizer.vocab_size, not from the YAML
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int | None = None  # None -> 4 * d_model
    dropout: float = 0.1
    attention_dropout: float = 0.0
    n_segments: int = 2  # protein A, protein B
    rope_base: float = 10000.0
    max_len: int = 2048
    pad_id: int = 0  # tokenizer.pad_id

    def __post_init__(self):
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"n_heads={self.n_heads} must divide d_model={self.d_model}")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError(
                f"d_head = d_model // n_heads = {self.d_model // self.n_heads} must be even (RoPE)"
            )


class TransformerEncoder(torch.nn.Module):
    """(input_ids, segment_ids, attention_mask) -> one vector per token (B, L, d_model)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.token_embedding = torch.nn.Embedding(config.vocab_size, d, padding_idx=config.pad_id)
        self.segment_embedding = torch.nn.Embedding(config.n_segments, d)
        self.dropout = torch.nn.Dropout(config.dropout)
        self.blocks = torch.nn.ModuleList(
            TransformerBlock(
                d,
                config.n_heads,
                config.d_ff,
                dropout=config.dropout,
                attention_dropout=config.attention_dropout,
                rope_base=config.rope_base,
                max_len=config.max_len,
            )
            for _ in range(config.n_layers)
        )
        self.final_norm = torch.nn.LayerNorm(d)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: torch.nn.Module) -> None:
        """BERT initialization: N(0, 0.02) for weights, zeros for biases and padding."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    @property
    def d_model(self) -> int:
        return self.config.d_model

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _encode(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.token_embedding(input_ids) + self.segment_embedding(segment_ids)  # (B, L, d)
        x = self.dropout(x)

        attentions = []
        for block in self.blocks:
            x, weights = block(x, attention_mask, need_weights)
            if need_weights:
                attentions.append(weights)
        return self.final_norm(x), attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids: (B, L) long
        segment_ids: (B, L) long, 0 for A, 1 for B
        attention_mask: (B, L) bool, True on real tokens
        returns: hidden states (B, L, d_model)
        """
        hidden, _ = self._encode(input_ids, segment_ids, attention_mask, need_weights=False)
        return hidden

    def forward_with_attention(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Same as forward, plus the attention weights of every layer (for analysis).

        returns: hidden (B, L, d_model), n_layers x weights (B, n_heads, L, L)
        """
        return self._encode(input_ids, segment_ids, attention_mask, need_weights=True)
