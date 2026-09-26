import pytest
import torch

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.shared.heads import (
    ClassificationHead,
    MLMHead,
    cls_pool,
    masked_max_pool,
    masked_mean_pool,
    pair_pool,
)

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


# --- ClassificationHead ---------------------------------------------------------


def linear_layers(module: torch.nn.Module) -> list[torch.nn.Linear]:
    return [m for m in module.modules() if isinstance(m, torch.nn.Linear)]


def count(module: torch.nn.Module, kind: type) -> int:
    return sum(isinstance(m, kind) for m in module.modules())


def test_classif_output_is_one_logit_per_pair():
    head = ClassificationHead(input_dim=8, hidden_dims=(16,))
    assert head(torch.randn(5, 8)).shape == (5,)


def test_classif_multi_output_keeps_last_dim():
    head = ClassificationHead(input_dim=8, output_dim=3, hidden_dims=(16,))
    assert head(torch.randn(5, 8)).shape == (5, 3)


def test_classif_no_hidden_layer_is_a_linear_probe():
    head = ClassificationHead(input_dim=8, hidden_dims=())
    assert [(m.in_features, m.out_features) for m in linear_layers(head)] == [(8, 1)]


def test_classif_layer_sizes_follow_hidden_dims():
    head = ClassificationHead(input_dim=8, hidden_dims=[32, 16])  # a list, as read from yaml
    sizes = [(m.in_features, m.out_features) for m in linear_layers(head)]
    assert sizes == [(8, 32), (32, 16), (16, 1)]


def test_classif_single_activation_applies_to_every_layer():
    head = ClassificationHead(input_dim=8, hidden_dims=(16, 16), activations="relu")
    assert count(head, torch.nn.ReLU) == 2
    assert head.activations == ("relu", "relu")


def test_classif_one_activation_per_layer():
    head = ClassificationHead(input_dim=8, hidden_dims=(16, 16), activations=["gelu", "silu"])
    assert count(head, torch.nn.GELU) == 1
    assert count(head, torch.nn.SiLU) == 1


def test_classif_activation_count_mismatch_fails():
    with pytest.raises(ValueError, match="activations"):
        ClassificationHead(input_dim=8, hidden_dims=(16, 16), activations=["gelu"])


def test_classif_unknown_activation_fails():
    with pytest.raises(ValueError, match="Unknown activation"):
        ClassificationHead(input_dim=8, hidden_dims=(16,), activations="tanh")


@pytest.mark.parametrize("norm, expected", [(False, 0), (True, 2)])
def test_classif_norm_adds_one_layernorm_per_hidden_layer(norm, expected):
    head = ClassificationHead(input_dim=8, hidden_dims=(16, 16), norm=norm)
    assert count(head, torch.nn.LayerNorm) == expected


def test_classif_outputs_raw_logits():
    """No sigmoid inside: the loss is BCEWithLogitsLoss."""
    torch.manual_seed(0)
    head = ClassificationHead(input_dim=8, hidden_dims=())
    out = head(10 * torch.randn(100, 8))
    assert (out < 0).any() and (out > 1).any()


def test_classif_eval_is_deterministic_train_uses_dropout():
    torch.manual_seed(0)
    head = ClassificationHead(input_dim=8, hidden_dims=(64,), dropout=0.5)
    x = torch.randn(16, 8)
    head.eval()
    assert torch.equal(head(x), head(x))
    head.train()
    assert not torch.equal(head(x), head(x))


def test_classif_gradients_reach_every_parameter():
    head = ClassificationHead(input_dim=8, hidden_dims=(16, 16), norm=True)
    loss = torch.nn.BCEWithLogitsLoss()(head(torch.randn(4, 8)), torch.tensor([0.0, 1, 1, 0]))
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_classif_on_pair_pool_is_symmetric():
    """pair_pool is symmetric, so the whole pooling + head is too."""
    len_a, len_b = 5, 3
    hidden, attention_mask, segment_ids = make_pair(len_a, len_b, pad=0)
    n_a = len_a + 2
    hidden_swapped = torch.cat([hidden[:, n_a:], hidden[:, :n_a]], dim=1)
    segment_swapped = torch.tensor([[0] * (len_b + 1) + [1] * n_a])

    head = ClassificationHead(input_dim=2 * D, hidden_dims=(16,)).eval()
    torch.testing.assert_close(
        head(pair_pool(hidden, attention_mask, segment_ids)),
        head(pair_pool(hidden_swapped, attention_mask, segment_swapped)),
    )


# --- MLMHead --------------------------------------------------------------------

VOCAB = ProteinTokenizer().vocab_size


def test_mlm_output_shape(hidden):
    head = MLMHead(d_model=D, vocab_size=VOCAB)
    assert head(hidden).shape == (B, L, VOCAB)


def test_mlm_default_is_esm_bert_transform():
    """hidden_dims=None -> Linear(d, d) -> activation -> LayerNorm, then decoder."""
    head = MLMHead(d_model=D, vocab_size=VOCAB)
    kinds = [type(m) for m in head.transform]
    assert kinds == [torch.nn.Linear, torch.nn.GELU, torch.nn.LayerNorm]
    assert (head.transform[0].in_features, head.transform[0].out_features) == (D, D)
    assert (head.decoder.in_features, head.decoder.out_features) == (D, VOCAB)


def test_mlm_no_hidden_layer_is_a_single_linear(hidden):
    head = MLMHead(d_model=D, vocab_size=VOCAB, hidden_dims=())
    assert len(head.transform) == 0
    torch.testing.assert_close(head(hidden), head.decoder(hidden))


def test_mlm_layer_sizes_follow_hidden_dims():
    head = MLMHead(d_model=D, vocab_size=VOCAB, hidden_dims=[32, 16])
    sizes = [(m.in_features, m.out_features) for m in linear_layers(head)]
    assert sizes == [(D, 32), (32, 16), (16, VOCAB)]


def test_mlm_norm_false_has_no_layernorm():
    head = MLMHead(d_model=D, vocab_size=VOCAB, norm=False)
    assert count(head, torch.nn.LayerNorm) == 0


@pytest.mark.parametrize("dropout, expected", [(0.0, 0), (0.1, 1)])
def test_mlm_dropout_layer_only_when_positive(dropout, expected):
    head = MLMHead(d_model=D, vocab_size=VOCAB, dropout=dropout)
    assert count(head, torch.nn.Dropout) == expected


def test_mlm_activation_count_mismatch_fails():
    with pytest.raises(ValueError, match="activations"):
        MLMHead(d_model=D, vocab_size=VOCAB, hidden_dims=(8, 8), activations=["gelu"])


def test_mlm_unknown_activation_fails():
    with pytest.raises(ValueError, match="Unknown activation"):
        MLMHead(d_model=D, vocab_size=VOCAB, activations="tanh")


def test_mlm_tokens_are_independent(hidden):
    """Changing one token's hidden state only changes that token's logits."""
    head = MLMHead(d_model=D, vocab_size=VOCAB).eval()
    changed = hidden.clone()
    changed[:, 3] += 1.0
    out, out_changed = head(hidden), head(changed)
    others = [i for i in range(L) if i != 3]
    torch.testing.assert_close(out[:, others], out_changed[:, others])
    assert not torch.allclose(out[:, 3], out_changed[:, 3])


def test_mlm_tied_embedding_shares_the_parameter():
    emb = torch.nn.Embedding(VOCAB, D)
    head = MLMHead(d_model=D, vocab_size=VOCAB, tied_embedding=emb)
    assert head.decoder.weight is emb.weight
    # counted once in a model holding both: embedding + head without its own decoder weight
    n_total = sum(p.numel() for p in torch.nn.ModuleList([emb, head]).parameters())
    n_untied_head = sum(p.numel() for p in MLMHead(d_model=D, vocab_size=VOCAB).parameters())
    assert n_total == n_untied_head  # the embedding matrix replaces the decoder matrix


def test_mlm_tied_embedding_follows_updates(hidden):
    emb = torch.nn.Embedding(VOCAB, D)
    head = MLMHead(d_model=D, vocab_size=VOCAB, tied_embedding=emb).eval()
    before = head(hidden)
    with torch.no_grad():
        emb.weight.add_(1.0)
    assert not torch.allclose(before, head(hidden))


def test_mlm_tied_embedding_wrong_shape_fails():
    emb = torch.nn.Embedding(VOCAB, D + 1)
    with pytest.raises(ValueError, match="cannot be tied"):
        MLMHead(d_model=D, vocab_size=VOCAB, tied_embedding=emb)


def test_mlm_tied_embedding_uses_last_hidden_size():
    """With hidden_dims, the embedding must match the last hidden size, not d_model."""
    MLMHead(D, VOCAB, hidden_dims=(8,), tied_embedding=torch.nn.Embedding(VOCAB, 8))
    with pytest.raises(ValueError, match="cannot be tied"):
        MLMHead(D, VOCAB, hidden_dims=(8,), tied_embedding=torch.nn.Embedding(VOCAB, D))


def test_mlm_loss_ignores_unmasked_positions(hidden):
    """End to end with the MLMCollator convention: labels = -100 outside masked positions."""
    head = MLMHead(d_model=D, vocab_size=VOCAB)
    hidden.requires_grad_(True)
    labels = torch.full((B, L), -100)
    labels[:, 2] = 7
    logits = head(hidden)
    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(logits.view(-1, VOCAB), labels.view(-1))
    loss.backward()
    assert torch.isfinite(loss)
    assert (hidden.grad[:, 2] != 0).any()
    others = [i for i in range(L) if i != 2]
    assert (hidden.grad[:, others] == 0).all()
