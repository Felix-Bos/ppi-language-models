"""Rotary position embeddings (RoPE), as used in ESM-2.

RoPE encodes positions by rotating queries and keys, instead of adding a position
vector to the input. Dimension j is paired with dimension j + d_head/2, and each pair
is rotated by an angle p * theta_i, where p is the token position and
theta_i = base^(-2i / d_head). The dot product of a rotated query and a rotated key
then only depends on the distance between their positions.

Configurable from the YAML (model.transformer section):
    rope_base: 10000.0   # base of the frequencies
    max_len: 2048        # longest sequence (in tokens) the cache covers
"""

import torch


def rope_frequencies(d_head: int, base: float = 10000.0) -> torch.Tensor:
    """Rotation frequencies theta_i = base^(-2i / d_head), for i = 0 .. d_head/2 - 1.

    The first frequencies are close to 1 (fast rotation, short-range positions),
    the last ones close to 0 (slow rotation, long-range positions).

    d_head: dimension of one attention head (must be even) d_head = d_model // n_heads
    base: controls the range of frequencies
    returns: (d_head // 2,)
    """
    if d_head % 2 != 0:
        raise ValueError(f"d_head must be even for RoPE, got {d_head}")
    theta = base ** (-2 * torch.arange(d_head // 2, dtype=torch.float32) / d_head)
    return theta


def rope_angles(max_len: int, d_head: int, base: float = 10000.0) -> torch.Tensor:
    """Angle p * theta_i for every position p and every frequency theta_i.

    Each angle is repeated for both halves of the vector (dimension j and j + d_head/2
    are rotated together).

    max_len: number of positions
    d_head: dimension of one attention head (must be even)
    base: controls the range of frequencies
    returns: (max_len, d_head)
    """
    theta = rope_frequencies(d_head, base)  # (d_head // 2,)
    positions = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
    angles = positions * theta  # (max_len, d_head // 2)

    # same angle for dimension j and j + d_head/2: (max_len, d_head)
    return torch.cat([angles, angles], dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the last dimension in two halves [x_a, x_b] and returns [-x_b, x_a].

    Combined with cos and sin, this applies a 2D rotation to every pair
    (x[..., j], x[..., j + d/2]) without any loop.

    x: (..., d), d even
    returns: (..., d)
    """
    x_a, x_b = x.chunk(2, dim=-1)  # (..., d/2) each
    return torch.cat([-x_b, x_a], dim=-1)


class RotaryEmbedding(torch.nn.Module):
    """Applies RoPE to queries and keys.

    No learned parameters: cos and sin are precomputed once up to max_len and stored
    as non-persistent buffers, so they follow the module with .to(device) and are not
    saved in checkpoints.
    """

    def __init__(self, d_head: int, max_len: int = 2048, base: float = 10000.0):
        """
        d_head: dimension of one attention head (d_model // n_heads), must be even
        max_len: longest sequence the cache covers; a pair of two proteins of
                 max_protein_length residues needs 2 * max_protein_length + 3
        base: controls the range of frequencies (10000 as in the original paper)

        raises ValueError if d_head is odd
        """
        super().__init__()
        self.d_head = d_head
        self.max_len = max_len
        self.base = base

        angles = rope_angles(max_len, d_head, base)  # raises if d_head is odd
        # non-persistent: moved by .to(device), not stored in state_dict
        self.register_buffer("cos", angles.cos(), persistent=False)  # (max_len, d_head)
        self.register_buffer("sin", angles.sin(), persistent=False)  # (max_len, d_head)

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates one tensor (queries or keys) according to the position of each token.

        x: (B, n_heads, L, d_head)
        returns: (B, n_heads, L, d_head)

        raises ValueError if L > max_len
        """
        seq_len = x.size(-2)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence of {seq_len} tokens is longer than the RoPE cache "
                f"(max_len={self.max_len})"
            )
        # (L, d_head), broadcast over batch and heads; cast for float16 / bfloat16
        cos = self.cos[:seq_len].to(x.dtype)
        sin = self.sin[:seq_len].to(x.dtype)

        # 2D rotation of every pair (x_j, x_{j + d/2}) by its angle:
        # x_j' = x_j cos - x_{j+d/2} sin,   x_{j+d/2}' = x_{j+d/2} cos + x_j sin
        return x * cos + rotate_half(x) * sin

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotates queries and keys (values are never rotated).

        q, k: (B, n_heads, L, d_head)
        returns: rotated q and k, same shapes
        """
        return self.rotate(q), self.rotate(k)

    def extra_repr(self) -> str:
        """Shown by print(module), e.g. 'd_head=32, max_len=2048, base=10000.0'."""
        return f"d_head={self.d_head}, max_len={self.max_len}, base={self.base}"
