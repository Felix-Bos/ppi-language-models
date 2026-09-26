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
    """MLP that turns a pooled pair vector into interaction logits."""

    ACTIVATIONS = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dims: list[int] | tuple[int, ...] = (),
        dropout: float = 0.1,
        activations: str | list[str] | tuple[str, ...] = "gelu",
        norm: bool = False,
    ):
        """MLP and classification head.

        input_dim: size of the pooled vector (d_model, or 2 * d_model with pair_pool)
        output_dim: size of the output logits (1 for interaction / no interaction)
        hidden_dims: sizes of the hidden layers, () for a single linear layer (linear probe)
        dropout: dropout probability after each hidden layer
        activations: one activation per hidden layer (same length as hidden_dims),
                     or a single name applied to every hidden layer
        norm: whether to add a LayerNorm in each hidden layer
        """
        super().__init__()
        hidden_dims = tuple(hidden_dims)

        # a single name -> the same activation for every hidden layer
        if isinstance(activations, str):
            activations = [activations] * len(hidden_dims)
        activations = tuple(activations)

        if len(activations) != len(hidden_dims):
            raise ValueError(
                f"Got {len(activations)} activations for {len(hidden_dims)} hidden layers"
            )
        for act in activations:
            if act not in self.ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{act}', expected one of {list(self.ACTIVATIONS)}"
                )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activations = activations
        self.norm = norm

        layers: list[torch.nn.Module] = []
        in_dim = input_dim
        for h, act in zip(hidden_dims, activations, strict=True):
            layers.append(torch.nn.Linear(in_dim, h))
            if norm:
                layers.append(torch.nn.LayerNorm(h))
            layers.append(self.ACTIVATIONS[act]())
            layers.append(torch.nn.Dropout(dropout))
            in_dim = h
        layers.append(torch.nn.Linear(in_dim, output_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, input_dim) -> logits (B,) if output_dim == 1, else (B, output_dim)."""
        logits = self.mlp(x)
        if self.output_dim == 1:
            logits = logits.squeeze(-1)
        return logits


class MLMHead(torch.nn.Module):
    """MLM head: per-token MLP from hidden states to vocabulary logits."""

    ACTIVATIONS = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, "silu": torch.nn.SiLU}

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        hidden_dims: list[int] | tuple[int, ...] | None = None,
        dropout: float = 0.0,
        activations: str | list[str] | tuple[str, ...] = "gelu",
        norm: bool = True,
        tied_embedding: torch.nn.Embedding | None = None,
    ):
        """MLM head, applied independently to every token.

        d_model: dimension of the encoder's hidden states
        vocab_size: number of tokens in the vocabulary (tokenizer.vocab_size)
        hidden_dims: sizes of the hidden layers; None = (d_model,) as in ESM/BERT,
                     () for a single linear layer
        dropout: dropout probability after each hidden layer
        activations: one activation per hidden layer, or a single name for all of them
        norm: whether to add a LayerNorm in each hidden layer (ESM/BERT: True)
        tied_embedding: the encoder's token embedding; if given, the output projection
                        shares its weights (requires the last hidden size == embedding dim)
        """
        super().__init__()
        hidden_dims = (d_model,) if hidden_dims is None else tuple(hidden_dims)

        if isinstance(activations, str):
            activations = [activations] * len(hidden_dims)
        activations = tuple(activations)

        if len(activations) != len(hidden_dims):
            raise ValueError(
                f"Got {len(activations)} activations for {len(hidden_dims)} hidden layers"
            )
        for act in activations:
            if act not in self.ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{act}', expected one of {list(self.ACTIVATIONS)}"
                )

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activations = activations
        self.norm = norm

        layers: list[torch.nn.Module] = []
        in_dim = d_model
        for h, act in zip(hidden_dims, activations, strict=True):
            layers.append(torch.nn.Linear(in_dim, h))
            layers.append(self.ACTIVATIONS[act]())
            if norm:
                layers.append(torch.nn.LayerNorm(h))  # after the activation, as in ESM/BERT
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            in_dim = h
        self.transform = torch.nn.Sequential(*layers)  # empty Sequential = identity
        self.decoder = torch.nn.Linear(in_dim, vocab_size)

        if tied_embedding is not None:
            if tied_embedding.weight.shape != (vocab_size, in_dim):
                raise ValueError(
                    f"Embedding of shape {tuple(tied_embedding.weight.shape)} cannot be tied "
                    f"to an output layer of shape ({vocab_size}, {in_dim})"
                )
            self.decoder.weight = tied_embedding.weight  # same Parameter, not a copy

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden: (B, L, d_model) -> logits (B, L, vocab_size)."""
        return self.decoder(self.transform(hidden))
