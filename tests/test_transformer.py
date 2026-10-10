import dataclasses

import pytest
import torch

from ppi_lm.data_scripts.collators import ClassificationCollator, MLMCollator
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.shared.heads import ClassificationHead, MLMHead, pair_pool
from ppi_lm.models.transformer.attention import MultiHeadSelfAttention
from ppi_lm.models.transformer.block import FeedForward, TransformerBlock
from ppi_lm.models.transformer.encoder import TransformerConfig, TransformerEncoder

D_MODEL, N_HEADS = 32, 4


@pytest.fixture
def tok():
    return ProteinTokenizer()


@pytest.fixture
def config(tok):
    """Tiny encoder, no dropout, so that eval and train outputs can be compared."""
    return TransformerConfig(
        vocab_size=tok.vocab_size, d_model=D_MODEL, n_layers=2, n_heads=N_HEADS, dropout=0.0
    )


@pytest.fixture
def encoder(config):
    torch.manual_seed(0)
    return TransformerEncoder(config).eval()


def make_item(tok, seq_a: str, seq_b: str, label: int) -> dict:
    """Same format as PairDataset.__getitem__."""
    a, b = tok.tokenize(seq_a), tok.tokenize(seq_b)
    return {
        "input_ids": torch.tensor([tok.cls_id] + a + [tok.sep_id] + b + [tok.eos_id]),
        "segment_ids": torch.tensor([0] * (len(a) + 2) + [1] * (len(b) + 1)),
        "label": label,
    }


@pytest.fixture
def batch(tok):
    """3 pairs of different lengths, padded: 15, 7 and 11 tokens."""
    items = [
        make_item(tok, "MKAYLVHG", "CDEFG", 1),
        make_item(tok, "HI", "KL", 0),
        make_item(tok, "MNPQRS", "TVW", 1),
    ]
    return ClassificationCollator(tok)(items)


def encode(encoder, batch):
    return encoder(batch["input_ids"], batch["segment_ids"], batch["attention_mask"])


# --- MultiHeadSelfAttention -----------------------------------------------------


@pytest.fixture
def attn():
    torch.manual_seed(0)
    return MultiHeadSelfAttention(D_MODEL, N_HEADS).eval()


@pytest.fixture
def x_and_mask():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 9, D_MODEL, generator=g)
    mask = torch.arange(9)[None, :] < torch.tensor([9, 5, 2])[:, None]
    return x, mask


def test_attention_keeps_shape(attn, x_and_mask):
    x, mask = x_and_mask
    out, weights = attn(x, mask)
    assert out.shape == x.shape
    assert weights is None


def test_attention_heads_must_divide_d_model():
    with pytest.raises(ValueError, match="divide"):
        MultiHeadSelfAttention(30, 4)


def test_attention_odd_d_head_fails():
    with pytest.raises(ValueError, match="even"):
        MultiHeadSelfAttention(12, 4)  # d_head = 3


def test_attention_fast_and_explicit_paths_agree(attn, x_and_mask):
    x, mask = x_and_mask
    fast, _ = attn(x, mask)
    explicit, _ = attn(x, mask, need_weights=True)
    torch.testing.assert_close(fast, explicit, atol=1e-5, rtol=1e-5)


def test_attention_weights_are_distributions_over_real_tokens(attn, x_and_mask):
    x, mask = x_and_mask
    _, weights = attn(x, mask, need_weights=True)
    assert weights.shape == (3, N_HEADS, 9, 9)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(3, N_HEADS, 9))
    # no weight on padded keys, for any query and any head
    assert (weights.masked_select(~mask[:, None, None, :]) == 0).all()


def test_attention_ignores_padding_values(attn, x_and_mask):
    x, mask = x_and_mask
    polluted = x.masked_fill(~mask.unsqueeze(-1), 1e3)
    out, _ = attn(x, mask)
    out_polluted, _ = attn(polluted, mask)
    torch.testing.assert_close(out[mask], out_polluted[mask])


def test_attention_is_bidirectional(attn, x_and_mask):
    """No causal mask: changing the last token changes the first token's output."""
    x, mask = x_and_mask
    changed = x.clone()
    changed[0, -1] += 1.0
    assert not torch.allclose(attn(x, mask)[0][0, 0], attn(changed, mask)[0][0, 0])


def test_attention_dropout_only_in_train(x_and_mask):
    x, mask = x_and_mask
    attn = MultiHeadSelfAttention(D_MODEL, N_HEADS, dropout=0.5)
    attn.eval()
    assert torch.equal(attn(x, mask)[0], attn(x, mask)[0])
    attn.train()
    assert not torch.equal(attn(x, mask)[0], attn(x, mask)[0])


# --- FeedForward / TransformerBlock ---------------------------------------------


def test_feedforward_shape():
    ffn = FeedForward(D_MODEL, 4 * D_MODEL)
    assert ffn(torch.randn(2, 5, D_MODEL)).shape == (2, 5, D_MODEL)


def test_block_keeps_shape(x_and_mask):
    x, mask = x_and_mask
    block = TransformerBlock(D_MODEL, N_HEADS, 4 * D_MODEL).eval()
    out, weights = block(x, mask, need_weights=True)
    assert out.shape == x.shape
    assert weights.shape == (3, N_HEADS, 9, 9)


def test_block_is_identity_when_branches_output_zero(x_and_mask):
    """Residual connections: with both sub-layers silenced, the block returns its input."""
    x, mask = x_and_mask
    block = TransformerBlock(D_MODEL, N_HEADS, 4 * D_MODEL).eval()
    for layer in (block.attn.out_proj, block.ffn.fc2):
        torch.nn.init.zeros_(layer.weight)
        torch.nn.init.zeros_(layer.bias)
    torch.testing.assert_close(block(x, mask)[0], x)


def test_block_ignores_padding_values(x_and_mask):
    x, mask = x_and_mask
    block = TransformerBlock(D_MODEL, N_HEADS, 4 * D_MODEL).eval()
    polluted = x.masked_fill(~mask.unsqueeze(-1), 1e3)
    torch.testing.assert_close(block(x, mask)[0][mask], block(polluted, mask)[0][mask])


# --- TransformerConfig ----------------------------------------------------------


def test_config_default_d_ff_is_four_times_d_model(tok):
    assert TransformerConfig(vocab_size=tok.vocab_size, d_model=64).d_ff == 256


def test_config_explicit_d_ff_is_kept(tok):
    assert TransformerConfig(vocab_size=tok.vocab_size, d_model=64, d_ff=100).d_ff == 100


def test_config_heads_must_divide_d_model(tok):
    with pytest.raises(ValueError, match="divide"):
        TransformerConfig(vocab_size=tok.vocab_size, d_model=30, n_heads=4)


def test_config_odd_d_head_fails(tok):
    with pytest.raises(ValueError, match="even"):
        TransformerConfig(vocab_size=tok.vocab_size, d_model=24, n_heads=8)


def test_config_unknown_key_fails(tok):
    """A typo in the YAML section must not be silently ignored."""
    with pytest.raises(TypeError):
        TransformerConfig(vocab_size=tok.vocab_size, n_layer=4)


def test_config_from_yaml_like_dict(tok):
    section = {"d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": None, "dropout": 0.0}
    config = TransformerConfig(vocab_size=tok.vocab_size, **section)
    assert dataclasses.asdict(config)["d_ff"] == 256


# --- TransformerEncoder ---------------------------------------------------------


def test_encoder_output_shape(encoder, batch):
    assert encode(encoder, batch).shape == (*batch["input_ids"].shape, D_MODEL)


def test_encoder_default_size_is_a_few_million_parameters(tok):
    n = TransformerEncoder(TransformerConfig(vocab_size=tok.vocab_size)).num_parameters()
    assert 1e6 < n < 10e6


def test_encoder_number_of_blocks(encoder, config):
    assert len(encoder.blocks) == config.n_layers


def test_encoder_padding_embedding_is_zero(encoder, tok):
    assert (encoder.token_embedding.weight[tok.pad_id] == 0).all()


def test_encoder_result_does_not_depend_on_batch_padding(encoder, tok, batch):
    """A pair encoded alone gives the same vectors as inside a padded batch."""
    short = make_item(tok, "HI", "KL", 0)  # row 1 of `batch`, padded from 7 to 15 tokens
    alone = ClassificationCollator(tok)([short])
    n = len(short["input_ids"])
    torch.testing.assert_close(
        encode(encoder, alone)[0], encode(encoder, batch)[1, :n], atol=1e-5, rtol=1e-5
    )


def test_encoder_uses_segment_ids(encoder, batch):
    swapped = dict(batch, segment_ids=1 - batch["segment_ids"])
    assert not torch.allclose(encode(encoder, batch), encode(encoder, swapped))


def test_encoder_uses_positions(encoder, tok):
    """Same tokens in another order: not just a permutation of the outputs (RoPE)."""
    item = make_item(tok, "MKAYLV", "CDEFG", 1)
    b = ClassificationCollator(tok)([item])
    perm = torch.tensor([0, 3, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])  # inside protein A
    permuted = dict(b, input_ids=b["input_ids"][:, perm])
    out, out_permuted = encode(encoder, b), encode(encoder, permuted)
    assert not torch.allclose(out[:, perm], out_permuted, atol=1e-4)


def test_encoder_is_bidirectional(encoder, batch):
    """Protein B influences the vectors of protein A (and <CLS>)."""
    changed = dict(batch, input_ids=batch["input_ids"].clone())
    changed["input_ids"][0, -2] = 5 if batch["input_ids"][0, -2] != 5 else 6  # last residue of B
    assert not torch.allclose(encode(encoder, batch)[0, 0], encode(encoder, changed)[0, 0])


def test_encoder_forward_with_attention(encoder, batch, config):
    hidden, attentions = encoder.forward_with_attention(
        batch["input_ids"], batch["segment_ids"], batch["attention_mask"]
    )
    torch.testing.assert_close(hidden, encode(encoder, batch), atol=1e-5, rtol=1e-5)
    B, L = batch["input_ids"].shape
    assert len(attentions) == config.n_layers
    assert all(w.shape == (B, config.n_heads, L, L) for w in attentions)


def test_encoder_dropout_only_in_train(tok, batch):
    torch.manual_seed(0)
    encoder = TransformerEncoder(
        TransformerConfig(vocab_size=tok.vocab_size, d_model=D_MODEL, n_heads=N_HEADS, dropout=0.3)
    )
    encoder.eval()
    assert torch.equal(encode(encoder, batch), encode(encoder, batch))
    encoder.train()
    assert not torch.equal(encode(encoder, batch), encode(encoder, batch))


def test_encoder_sequence_longer_than_max_len_fails(tok, batch):
    config = TransformerConfig(vocab_size=tok.vocab_size, d_model=D_MODEL, n_heads=N_HEADS)
    encoder = TransformerEncoder(dataclasses.replace(config, max_len=8))
    with pytest.raises(ValueError, match="max_len"):
        encode(encoder, batch)


def test_encoder_gradients_reach_every_parameter(encoder, batch):
    encoder.train()
    hidden = encode(encoder, batch)
    target = torch.randn_like(hidden)  # a plain sum would cancel out through the final LayerNorm
    (hidden * target).sum().backward()
    for name, p in encoder.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


def test_encoder_padding_embedding_gets_no_gradient(encoder, batch, tok):
    encoder.train()
    hidden = encode(encoder, batch)
    (hidden * torch.randn_like(hidden)).sum().backward()
    assert (encoder.token_embedding.weight.grad[tok.pad_id] == 0).all()


# --- encoder + heads, end to end ------------------------------------------------


def test_mlm_end_to_end_with_tied_embedding(tok, config):
    torch.manual_seed(0)
    encoder = TransformerEncoder(config)
    head = MLMHead(D_MODEL, tok.vocab_size, tied_embedding=encoder.token_embedding)
    items = [make_item(tok, "MKAYLVHGCDEF" * 3, "PQRSTVW" * 3, 1) for _ in range(4)]
    batch = MLMCollator(tok)(items)

    logits = head(encode(encoder, batch))
    assert logits.shape == (*batch["input_ids"].shape, tok.vocab_size)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, tok.vocab_size), batch["labels"].view(-1), ignore_index=-100
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert encoder.token_embedding.weight.grad is not None


def test_classification_can_overfit_a_tiny_batch(tok, config, batch):
    """Smoke test of the whole stack: a few steps of Adam drive the loss towards 0."""
    torch.manual_seed(0)
    encoder = TransformerEncoder(config)
    head = ClassificationHead(2 * D_MODEL, hidden_dims=(32,), dropout=0.0)
    params = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def step():
        hidden = encode(encoder, batch)
        logits = head(pair_pool(hidden, batch["attention_mask"], batch["segment_ids"]))
        return loss_fn(logits, batch["labels"])

    first = step().item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = step()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1 < first
