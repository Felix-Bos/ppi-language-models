import pytest

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def tok():
    return ProteinTokenizer()


# --- Vocabulaire -----------------------------------------------------------


def test_vocab_size(tok):
    assert tok.vocab_size == 5 + 21


def test_pad_is_zero(tok):
    assert tok.pad_id == 0


def test_special_ids_are_distinct(tok):
    ids = [tok.pad_id, tok.mask_id, tok.cls_id, tok.sep_id, tok.eos_id]
    assert len(set(ids)) == 5


def test_special_and_amino_acid_ids_do_not_overlap(tok):
    assert not set(tok.special_ids) & set(tok.aa_ids)


def test_aa_ids_cover_only_the_20_standard(tok):
    # utilisés pour le remplacement aléatoire du MLM : ni X, ni tokens spéciaux
    assert len(tok.aa_ids) == 20
    assert tok.tokenize("X")[0] not in tok.aa_ids


def test_all_ids_are_valid_embedding_indices(tok):
    ids = tok.tokenize(STANDARD_AA + "X") + list(tok.special_ids)
    assert all(0 <= i < tok.vocab_size for i in ids)


# --- Tokenisation ------------------------------------------------------------


def test_tokenize_returns_one_id_per_residue(tok):
    assert len(tok.tokenize("MKAYCDE")) == 7


def test_tokenize_adds_no_special_tokens(tok):
    # <CLS>, <SEP>, <EOS> sont ajoutés par le dataset, pas par tokenize
    ids = tok.tokenize("MKAY")
    assert not set(ids) & set(tok.special_ids)


def test_tokenize_empty_sequence(tok):
    assert tok.tokenize("") == []


def test_each_standard_aa_has_a_unique_id(tok):
    ids = tok.tokenize(STANDARD_AA)
    assert len(set(ids)) == 20


@pytest.mark.parametrize("rare", list("UOBZ"))
def test_rare_amino_acids_map_to_x(tok, rare):
    assert tok.tokenize(rare) == tok.tokenize("X")


def test_lowercase_is_accepted(tok):
    assert tok.tokenize("mkay") == tok.tokenize("MKAY")


@pytest.mark.parametrize("weird", ["*", "-", "1", "J"])
def test_unknown_characters_do_not_crash(tok, weird):
    ids = tok.tokenize(weird)
    assert len(ids) == 1
    assert 0 <= ids[0] < tok.vocab_size


# --- Détokenisation ----------------------------------------------------------


def test_tokenize_detokenize_roundtrip(tok):
    seq = STANDARD_AA + "X"
    assert tok.detokenize(tok.tokenize(seq)) == seq


def test_detokenize_special_tokens(tok):
    ids = [tok.cls_id] + tok.tokenize("MK") + [tok.sep_id] + tok.tokenize("AY") + [tok.eos_id]
    decoded = tok.detokenize(ids)
    assert decoded.startswith("<CLS>")
    assert "<SEP>" in decoded
    assert decoded.endswith("<EOS>")
