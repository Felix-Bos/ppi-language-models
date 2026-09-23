import pytest
import torch

from ppi_lm.data_scripts.collators import ClassificationCollator, MLMCollator, pad_batch
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer

AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def tok():
    return ProteinTokenizer()


def make_item(tok, seq_a: str, seq_b: str, label: int) -> dict:
    """Same format as PairDataset.__getitem__."""
    a, b = tok.tokenize(seq_a), tok.tokenize(seq_b)
    return {
        "input_ids": torch.tensor([tok.cls_id] + a + [tok.sep_id] + b + [tok.eos_id]),
        "segment_ids": torch.tensor([0] * (len(a) + 2) + [1] * (len(b) + 1)),
        "label": label,
    }


def random_seq(n: int, g: torch.Generator) -> str:
    return "".join(AA[i] for i in torch.randint(len(AA), (n,), generator=g))


@pytest.fixture
def small_batch(tok):
    return [
        make_item(tok, "MKAY", "CDEFG", 1),  # 12 tokens
        make_item(tok, "HI", "KL", 0),  # 7 tokens
        make_item(tok, "MNPQRS", "TV", 1),  # 11 tokens
    ]


@pytest.fixture
def big_batch(tok):
    """64 paires aléatoires de 50 à 300 résidus, pour les tests statistiques."""
    g = torch.Generator().manual_seed(0)
    lengths = torch.randint(50, 300, (64, 2), generator=g).tolist()
    return [
        make_item(tok, random_seq(la, g), random_seq(lb, g), i % 2)
        for i, (la, lb) in enumerate(lengths)
    ]


# --- pad_batch ------------------------------------------------------------------


def test_pad_batch_shape_and_values():
    seqs = [torch.tensor([5, 6, 7]), torch.tensor([8]), torch.tensor([9, 10])]
    padded, mask = pad_batch(seqs, pad_value=0)
    assert padded.tolist() == [[5, 6, 7], [8, 0, 0], [9, 10, 0]]
    assert mask.tolist() == [[True, True, True], [True, False, False], [True, True, False]]
    assert mask.dtype == torch.bool


def test_pad_batch_uses_given_pad_value():
    padded, _ = pad_batch([torch.tensor([1, 2]), torch.tensor([3])], pad_value=-7)
    assert padded[1, 1] == -7


def test_pad_batch_equal_lengths_no_padding():
    padded, mask = pad_batch([torch.tensor([1, 2]), torch.tensor([3, 4])], pad_value=0)
    assert padded.tolist() == [[1, 2], [3, 4]]
    assert mask.all()


# --- ClassificationCollator --------------------------------------------------------


def test_classif_keys_and_shapes(tok, small_batch):
    out = ClassificationCollator(tok)(small_batch)
    assert set(out) == {"input_ids", "segment_ids", "attention_mask", "labels"}
    assert out["input_ids"].shape == (3, 12)
    assert out["segment_ids"].shape == out["attention_mask"].shape == (3, 12)
    assert out["labels"].shape == (3,)


def test_classif_labels_are_float(tok, small_batch):
    labels = ClassificationCollator(tok)(small_batch)["labels"]
    assert labels.dtype == torch.float  # attendu par BCEWithLogitsLoss
    assert labels.tolist() == [1.0, 0.0, 1.0]


def test_classif_keeps_tokens_and_pads_with_pad_id(tok, small_batch):
    out = ClassificationCollator(tok)(small_batch)
    for i, item in enumerate(small_batch):
        n = len(item["input_ids"])
        assert torch.equal(out["input_ids"][i, :n], item["input_ids"])  # aucun token modifié
        assert (out["input_ids"][i, n:] == tok.pad_id).all()
        assert out["attention_mask"][i].sum() == n


def test_classif_pads_segments(tok, small_batch):
    out = ClassificationCollator(tok)(small_batch)
    item = small_batch[1]
    n = len(item["segment_ids"])
    assert torch.equal(out["segment_ids"][1, :n], item["segment_ids"])
    assert (out["segment_ids"][1, n:] == 0).all()


# --- MLMCollator : structure -----------------------------------------------------


def test_mlm_keys_and_shapes(tok, small_batch):
    out = MLMCollator(tok)(small_batch)
    assert set(out) == {"input_ids", "segment_ids", "attention_mask", "labels"}
    for key in ("input_ids", "segment_ids", "attention_mask", "labels"):
        assert out[key].shape == (3, 12)


def test_mlm_does_not_modify_input_batch(tok, small_batch):
    before = [item["input_ids"].clone() for item in small_batch]
    MLMCollator(tok)(small_batch)
    for item, original in zip(small_batch, before, strict=True):
        assert torch.equal(item["input_ids"], original)


# --- MLMCollator : quelles positions sont masquées ----------------------------------


def test_mlm_never_targets_special_tokens_or_padding(tok, big_batch):
    torch.manual_seed(0)
    original, _ = pad_batch([item["input_ids"] for item in big_batch], tok.pad_id)
    out = MLMCollator(tok)(big_batch)
    selected = out["labels"] != -100
    assert not torch.isin(original[selected], torch.tensor(tok.special_ids)).any()
    assert not (selected & ~out["attention_mask"]).any()


def test_mlm_special_tokens_and_padding_unchanged(tok, big_batch):
    torch.manual_seed(0)
    original, _ = pad_batch([item["input_ids"] for item in big_batch], tok.pad_id)
    out = MLMCollator(tok)(big_batch)
    special = torch.isin(original, torch.tensor(tok.special_ids))
    assert torch.equal(out["input_ids"][special], original[special])


def test_mlm_labels_hold_original_tokens(tok, big_batch):
    torch.manual_seed(0)
    original, _ = pad_batch([item["input_ids"] for item in big_batch], tok.pad_id)
    out = MLMCollator(tok)(big_batch)
    selected = out["labels"] != -100
    assert torch.equal(out["labels"][selected], original[selected])
    assert (out["labels"][~selected] == -100).all()


def test_mlm_unselected_positions_are_unchanged(tok, big_batch):
    torch.manual_seed(0)
    original, _ = pad_batch([item["input_ids"] for item in big_batch], tok.pad_id)
    out = MLMCollator(tok)(big_batch)
    selected = out["labels"] != -100
    assert torch.equal(out["input_ids"][~selected], original[~selected])


@pytest.mark.parametrize("mask_prob", [0.15, 0.3])
def test_mlm_selects_about_mask_prob(tok, big_batch, mask_prob):
    torch.manual_seed(0)
    out = MLMCollator(tok, mask_prob=mask_prob)(big_batch)
    for i in range(len(big_batch)):
        n_residues = out["attention_mask"][i].sum().item() - 3  # sans CLS, SEP, EOS
        n_selected = (out["labels"][i] != -100).sum().item()
        # au moins la cible ; un span peut la dépasser de max_span - 1 positions
        assert round(mask_prob * n_residues) <= n_selected <= round(mask_prob * n_residues) + 2


def test_mlm_at_least_one_target_per_sequence(tok):
    torch.manual_seed(0)
    batch = [make_item(tok, "M", "K", 1), make_item(tok, "A", "C", 0)]
    out = MLMCollator(tok, mask_prob=0.01)(batch)
    assert ((out["labels"] != -100).sum(dim=1) >= 1).all()


def test_mlm_uses_spans(tok, big_batch):
    """Avec max_span=3, on doit trouver des positions masquées adjacentes."""
    torch.manual_seed(0)
    out = MLMCollator(tok, max_span=3)(big_batch)
    selected = out["labels"] != -100
    adjacent = (selected[:, 1:] & selected[:, :-1]).sum()
    assert adjacent > 0


def test_mlm_is_dynamic(tok, big_batch):
    """Deux appels sur le même batch ne masquent pas les mêmes positions."""
    collator = MLMCollator(tok)
    first = collator(big_batch)["labels"] != -100
    second = collator(big_batch)["labels"] != -100
    assert not torch.equal(first, second)


# --- MLMCollator : règle 80 / 10 / 10 ----------------------------------------------


def test_mlm_80_10_10_rule(tok, big_batch):
    torch.manual_seed(0)
    original, _ = pad_batch([item["input_ids"] for item in big_batch], tok.pad_id)
    out = MLMCollator(tok)(big_batch)
    selected = out["labels"] != -100
    new, old = out["input_ids"][selected], original[selected]
    n = selected.sum().item()

    frac_mask = (new == tok.mask_id).sum().item() / n
    frac_same = (new == old).sum().item() / n  # inclut les tirages aléatoires tombés sur le même AA
    frac_random = 1 - frac_mask - frac_same

    assert 0.75 < frac_mask < 0.85
    assert 0.05 < frac_random < 0.15
    assert 0.05 < frac_same < 0.15


def test_mlm_random_replacements_are_standard_amino_acids(tok, big_batch):
    torch.manual_seed(0)
    out = MLMCollator(tok)(big_batch)
    selected = out["labels"] != -100
    replaced = out["input_ids"][selected]
    replaced = replaced[replaced != tok.mask_id]
    assert torch.isin(replaced, torch.tensor(tok.aa_ids)).all()  # jamais X ni un token spécial
