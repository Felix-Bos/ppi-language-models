import pytest
import torch

from ppi_lm.models.transformer.rope import (
    RotaryEmbedding,
    rope_angles,
    rope_frequencies,
    rotate_half,
)

B, H, L, D_HEAD = 2, 3, 10, 8


@pytest.fixture
def rope():
    return RotaryEmbedding(d_head=D_HEAD, max_len=64)


@pytest.fixture
def qk():
    g = torch.Generator().manual_seed(0)
    return torch.randn(B, H, L, D_HEAD, generator=g), torch.randn(B, H, L, D_HEAD, generator=g)


# --- rope_frequencies -----------------------------------------------------------


def test_frequencies_shape():
    assert rope_frequencies(D_HEAD).shape == (D_HEAD // 2,)


def test_frequencies_start_at_one_and_decrease():
    theta = rope_frequencies(64)
    assert theta[0] == 1.0
    assert (theta[1:] < theta[:-1]).all()
    assert (theta > 0).all()


def test_frequencies_formula():
    theta = rope_frequencies(D_HEAD, base=100.0)
    expected = torch.tensor([100.0 ** (-2 * i / D_HEAD) for i in range(D_HEAD // 2)])
    torch.testing.assert_close(theta, expected)


def test_frequencies_odd_d_head_fails():
    with pytest.raises(ValueError, match="even"):
        rope_frequencies(7)


# --- rope_angles ----------------------------------------------------------------


def test_angles_shape():
    assert rope_angles(L, D_HEAD).shape == (L, D_HEAD)


def test_angles_are_position_times_frequency():
    angles, theta = rope_angles(L, D_HEAD), rope_frequencies(D_HEAD)
    for p in range(L):
        torch.testing.assert_close(angles[p, : D_HEAD // 2], p * theta)


def test_angles_both_halves_are_equal():
    angles = rope_angles(L, D_HEAD)
    assert torch.equal(angles[:, : D_HEAD // 2], angles[:, D_HEAD // 2 :])


def test_angles_position_zero_is_zero():
    assert (rope_angles(L, D_HEAD)[0] == 0).all()


# --- rotate_half ----------------------------------------------------------------


def test_rotate_half_example():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    assert torch.equal(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_rotate_half_twice_is_minus_identity(qk):
    """A rotation by 90° applied twice is a rotation by 180°."""
    q, _ = qk
    assert torch.equal(rotate_half(rotate_half(q)), -q)


# --- RotaryEmbedding ------------------------------------------------------------


def test_no_learned_parameters(rope):
    assert list(rope.parameters()) == []


def test_cache_is_not_saved_in_checkpoints(rope):
    assert rope.state_dict() == {}
    assert rope.cos.shape == rope.sin.shape == (64, D_HEAD)


def test_odd_d_head_fails():
    with pytest.raises(ValueError, match="even"):
        RotaryEmbedding(d_head=7)


def test_rotate_keeps_shape(rope, qk):
    q, _ = qk
    assert rope.rotate(q).shape == q.shape


def test_forward_rotates_q_and_k_the_same_way(rope, qk):
    q, k = qk
    q_rot, k_rot = rope(q, k)
    assert torch.equal(q_rot, rope.rotate(q))
    assert torch.equal(k_rot, rope.rotate(k))


def test_position_zero_is_unchanged(rope, qk):
    q, _ = qk
    torch.testing.assert_close(rope.rotate(q)[:, :, 0], q[:, :, 0])


def test_rotation_keeps_the_norm_of_every_pair(rope, qk):
    """A rotation never changes a vector's length, pair by pair."""
    q, _ = qk
    q_rot = rope.rotate(q)
    half = D_HEAD // 2

    def pair_norms(x):
        return x[..., :half] ** 2 + x[..., half:] ** 2

    torch.testing.assert_close(pair_norms(q_rot), pair_norms(q))


def test_matches_complex_multiplication(rope, qk):
    """Independent check: pair (x_j, x_{j+d/2}) as a complex number times e^{i p theta_j}."""
    q, _ = qk
    half = D_HEAD // 2
    z = torch.complex(q[..., :half], q[..., half:])  # (B, H, L, d/2)
    angles = torch.arange(L)[:, None] * rope_frequencies(D_HEAD)  # (L, d/2)
    z_rot = z * torch.polar(torch.ones_like(angles), angles)
    expected = torch.cat([z_rot.real, z_rot.imag], dim=-1)
    torch.testing.assert_close(rope.rotate(q), expected)


def test_scores_depend_only_on_relative_position(rope):
    """Same q and k vectors at positions (i, j) and (i + s, j + s): same dot product."""
    g = torch.Generator().manual_seed(1)
    q_vec, k_vec = torch.randn(D_HEAD, generator=g), torch.randn(D_HEAD, generator=g)
    x_q = q_vec.expand(1, 1, 40, D_HEAD)
    x_k = k_vec.expand(1, 1, 40, D_HEAD)
    q_rot, k_rot = rope(x_q, x_k)

    def score(i, j):
        return (q_rot[0, 0, i] * k_rot[0, 0, j]).sum()

    for i, j, shift in [(2, 5, 10), (7, 3, 20), (0, 0, 30), (12, 30, 5)]:
        torch.testing.assert_close(score(i, j), score(i + shift, j + shift))


def test_scores_change_with_distance(rope):
    g = torch.Generator().manual_seed(2)
    q_vec, k_vec = torch.randn(D_HEAD, generator=g), torch.randn(D_HEAD, generator=g)
    q_rot, k_rot = rope(q_vec.expand(1, 1, L, D_HEAD), k_vec.expand(1, 1, L, D_HEAD))
    s1 = (q_rot[0, 0, 0] * k_rot[0, 0, 1]).sum()
    s5 = (q_rot[0, 0, 0] * k_rot[0, 0, 5]).sum()
    assert not torch.allclose(s1, s5)


def test_shorter_sequences_use_the_start_of_the_cache(rope, qk):
    """A prefix of a sequence is rotated exactly as in the full sequence."""
    q, _ = qk
    torch.testing.assert_close(rope.rotate(q[:, :, :4]), rope.rotate(q)[:, :, :4])


def test_sequence_longer_than_cache_fails():
    rope = RotaryEmbedding(d_head=D_HEAD, max_len=8)
    with pytest.raises(ValueError, match="max_len"):
        rope.rotate(torch.randn(1, 1, 9, D_HEAD))


def test_sequence_of_exactly_max_len_works():
    rope = RotaryEmbedding(d_head=D_HEAD, max_len=8)
    assert rope.rotate(torch.randn(1, 1, 8, D_HEAD)).shape == (1, 1, 8, D_HEAD)


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_keeps_input_dtype(rope, dtype):
    x = torch.randn(B, H, L, D_HEAD, dtype=dtype)
    assert rope.rotate(x).dtype == dtype


def test_gradients_flow(rope, qk):
    q, k = (t.clone().requires_grad_(True) for t in qk)
    q_rot, k_rot = rope(q, k)
    (q_rot * k_rot).sum().backward()
    assert torch.isfinite(q.grad).all() and (q.grad != 0).any()
    assert torch.isfinite(k.grad).all() and (k.grad != 0).any()


def test_extra_repr(rope):
    assert repr(rope) == "RotaryEmbedding(d_head=8, max_len=64, base=10000.0)"
