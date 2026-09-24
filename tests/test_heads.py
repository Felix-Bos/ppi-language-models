import pytest
import torch

from ppi_lm.models.shared.heads import cls_pool, masked_max_pool, masked_mean_pool, pair_pool

B, L, D = 3, 7, 4


@pytest.fixture
def hidden():
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, L, D, generator=g)


@pytest.fixture
def mask():
    """Row 0: full, row 1: 4 real tokens, row 2: 1 real token."""
    lengths = torch.tensor([L, 4, 1])
    return torch.arange(L)[None, :] < lengths[:, None]


def make_pair(len_a: int, len_b: int, pad: int, d: int = D, seed: int = 0):
    """One pair in the PairDataset layout, with its own hidden states.

    Segment 0 = <CLS> A <SEP>, segment 1 = B <EOS>, then `pad` padding positions.
    Returns hidden (1, L, d), attention_mask (1, L), segment_ids (1, L).
    """
    g = torch.Generator().manual_seed(seed)
    n_a, n_b = len_a + 2, len_b + 1
    hidden = torch.randn(1, n_a + n_b + pad, d, generator=g)
    attention_mask = torch.tensor([[True] * (n_a + n_b) + [False] * pad])
    segment_ids = torch.tensor([[0] * n_a + [1] * (n_b + pad)])  # padding is segment 1
    return hidden, attention_mask, segment_ids


# --- cls_pool -------------------------------------------------------------------


def test_cls_pool_shape(hidden):
    assert cls_pool(hidden).shape == (B, D)


def test_cls_pool_returns_first_token(hidden):
    assert torch.equal(cls_pool(hidden), hidden[:, 0, :])


# --- masked_mean_pool -----------------------------------------------------------


def test_mean_pool_shape(hidden, mask):
    assert masked_mean_pool(hidden, mask).shape == (B, D)


def test_mean_pool_full_mask_is_plain_mean(hidden):
    full = torch.ones(B, L, dtype=torch.bool)
    torch.testing.assert_close(masked_mean_pool(hidden, full), hidden.mean(dim=1))


def test_mean_pool_matches_manual_mean(hidden, mask):
    out = masked_mean_pool(hidden, mask)
    for i in range(B):
        torch.testing.assert_close(out[i], hidden[i, mask[i]].mean(dim=0))


def test_mean_pool_ignores_padding_values(hidden, mask):
    polluted = hidden.masked_fill(~mask.unsqueeze(-1), 1e6)
    torch.testing.assert_close(masked_mean_pool(polluted, mask), masked_mean_pool(hidden, mask))


def test_mean_pool_empty_row_gives_zeros(hidden, mask):
    mask[1] = False
    out = masked_mean_pool(hidden, mask)
    assert torch.equal(out[1], torch.zeros(D))
    assert not out.isnan().any()


def test_mean_pool_gradient_only_on_real_tokens(hidden, mask):
    hidden.requires_grad_(True)
    masked_mean_pool(hidden, mask).sum().backward()
    assert (hidden.grad[~mask] == 0).all()
    assert (hidden.grad[mask] != 0).all()


def test_mean_pool_keeps_dtype(hidden, mask):
    assert masked_mean_pool(hidden.double(), mask).dtype == torch.float64


# --- masked_max_pool ------------------------------------------------------------


def test_max_pool_shape(hidden, mask):
    assert masked_max_pool(hidden, mask).shape == (B, D)


def test_max_pool_full_mask_is_plain_max(hidden):
    full = torch.ones(B, L, dtype=torch.bool)
    assert torch.equal(masked_max_pool(hidden, full), hidden.max(dim=1).values)


def test_max_pool_matches_manual_max(hidden, mask):
    out = masked_max_pool(hidden, mask)
    for i in range(B):
        assert torch.equal(out[i], hidden[i, mask[i]].max(dim=0).values)


def test_max_pool_ignores_padding_values(hidden, mask):
    polluted = hidden.masked_fill(~mask.unsqueeze(-1), 1e6)
    assert torch.equal(masked_max_pool(polluted, mask), masked_max_pool(hidden, mask))


def test_max_pool_keeps_negative_values(mask):
    """All real tokens negative: the max is negative, not replaced by 0 or by padding."""
    hidden = -torch.rand(B, L, D) - 1.0
    out = masked_max_pool(hidden, mask)
    assert (out < 0).all()


def test_max_pool_empty_row_gives_zeros(hidden, mask):
    mask[1] = False
    out = masked_max_pool(hidden, mask)
    assert torch.equal(out[1], torch.zeros(D))
    assert torch.isfinite(out).all()


def test_max_pool_gradient_is_finite_and_only_on_real_tokens(hidden, mask):
    mask[1] = False  # an empty row must not produce NaN gradients either
    hidden.requires_grad_(True)
    masked_max_pool(hidden, mask).sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert (hidden.grad[~mask] == 0).all()


# --- pair_pool ------------------------------------------------------------------


def test_pair_pool_shape():
    hidden, attention_mask, segment_ids = make_pair(5, 3, pad=2)
    assert pair_pool(hidden, attention_mask, segment_ids).shape == (1, 2 * D)


def test_pair_pool_matches_definition():
    hidden, attention_mask, segment_ids = make_pair(5, 3, pad=2)
    a = hidden[0, attention_mask[0] & (segment_ids[0] == 0)].mean(dim=0)
    b = hidden[0, attention_mask[0] & (segment_ids[0] == 1)].mean(dim=0)
    expected = torch.cat([a * b, (a - b).abs()])
    torch.testing.assert_close(pair_pool(hidden, attention_mask, segment_ids)[0], expected)


def test_pair_pool_is_symmetric():
    """Swapping the two proteins (with their hidden states) gives the same vector."""
    len_a, len_b = 5, 3
    hidden, attention_mask, segment_ids = make_pair(len_a, len_b, pad=0)
    n_a = len_a + 2
    # swapped pair: B's states become segment 0, A's states segment 1
    hidden_swapped = torch.cat([hidden[:, n_a:], hidden[:, :n_a]], dim=1)
    segment_swapped = torch.tensor([[0] * (len_b + 1) + [1] * n_a])
    torch.testing.assert_close(
        pair_pool(hidden, attention_mask, segment_ids),
        pair_pool(hidden_swapped, attention_mask, segment_swapped),
    )


def test_pair_pool_ignores_padding():
    hidden, attention_mask, segment_ids = make_pair(5, 3, pad=4)
    polluted = hidden.masked_fill(~attention_mask.unsqueeze(-1), 1e6)
    torch.testing.assert_close(
        pair_pool(polluted, attention_mask, segment_ids),
        pair_pool(hidden, attention_mask, segment_ids),
    )


def test_pair_pool_identical_proteins_have_zero_difference():
    hidden, attention_mask, _ = make_pair(4, 5, pad=0)  # 6 + 6 tokens
    hidden[:, 6:] = hidden[:, :6]  # same states in both segments
    segment_ids = torch.tensor([[0] * 6 + [1] * 6])
    out = pair_pool(hidden, attention_mask, segment_ids)
    torch.testing.assert_close(out[0, D:], torch.zeros(D))  # up to float rounding


def test_pair_pool_batches_rows_independently():
    rows = [make_pair(5, 3, pad=0, seed=0), make_pair(5, 3, pad=0, seed=1)]
    batched = pair_pool(*(torch.cat(t, dim=0) for t in zip(*rows, strict=True)))
    for i, row in enumerate(rows):
        torch.testing.assert_close(batched[i], pair_pool(*row)[0])
