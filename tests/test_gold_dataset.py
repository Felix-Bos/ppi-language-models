import pytest
import torch

from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset, PairDataset
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer

# Mini jeu de données : 8 protéines, dont une longue (LONG, 20 résidus) pour tester le crop.
# Chaque split a ses propres protéines, comme dans le vrai Bernett.
SEQS = {
    "P1": "MKAY",
    "P2": "CDEFG",
    "P3": "HIKL",
    "P4": "MNPQR",
    "P5": "STVW",
    "P6": "YACDE",
    "P7": "FGHIK",
    "LONG": "ACDEFGHIKLMNPQRSTVWY",
}
PAIR_FILES = {
    "Intra1_pos_rr.txt": "P1 P2\nP1 LONG\nP2 LONG\n",  # train : 3 positives
    "Intra1_neg_rr.txt": "P1 P2\nP2 LONG\nP1 LONG\n",  # train : 3 négatives
    "Intra0_pos_rr.txt": "P3 P4\n",  # val
    "Intra0_neg_rr.txt": "P3 P4\n",
    "Intra2_pos_rr.txt": "P5 P6\n\n",  # test (avec une ligne vide à ignorer)
    "Intra2_neg_rr.txt": "P6 P7\n",
}
MAX_LEN = 8  # petit, pour que LONG soit découpée


@pytest.fixture
def data_dir(tmp_path):
    fasta = "".join(f">{pid} some description\n{seq}\n" for pid, seq in SEQS.items())
    (tmp_path / "human_swissprot_oneliner.fasta").write_text(fasta)
    for name, content in PAIR_FILES.items():
        (tmp_path / name).write_text(content)
    return tmp_path


@pytest.fixture
def data(data_dir):
    return GoldStandardDataset(data_dir)


@pytest.fixture
def tok():
    return ProteinTokenizer()


def make_ds(data, split="train", **kwargs):
    kwargs.setdefault("max_protein_length", MAX_LEN)
    return PairDataset(data.pairs[split], data.seqs, data.tokenizer, **kwargs)


# --- GoldStandardDataset : lecture des fichiers ------------------------------


def test_reads_all_sequences(data):
    assert data.seqs == SEQS  # l'id est bien le premier mot de l'en-tête


def test_split_sizes(data):
    assert len(data.pairs["train"]) == 6
    assert len(data.pairs["val"]) == 2
    assert len(data.pairs["test"]) == 2  # la ligne vide est ignorée


def test_labels_come_from_file_names(data):
    for pairs in data.pairs.values():
        n = len(pairs) // 2
        assert all(label == 1 for _, _, label in pairs[:n])  # pos d'abord
        assert all(label == 0 for _, _, label in pairs[n:])  # puis neg


def test_fasta_with_odd_number_of_lines_fails(data_dir):
    (data_dir / "human_swissprot_oneliner.fasta").write_text(">P1\nMKAY\n>P2\n")
    with pytest.raises(ValueError):  # levé par zip(strict=True)
        GoldStandardDataset(data_dir)


def test_fasta_with_multiline_sequence_fails(data_dir):
    (data_dir / "human_swissprot_oneliner.fasta").write_text(">P1\nMK\nAY\n>P2\nCD\n")
    with pytest.raises(AssertionError):  # une ligne de séquence tombe à la place d'un en-tête
        GoldStandardDataset(data_dir)


def test_no_protein_shared_between_splits(data):
    prots = {s: {p for a, b, _ in pairs for p in (a, b)} for s, pairs in data.pairs.items()}
    assert not prots["train"] & prots["val"]
    assert not prots["train"] & prots["test"]
    assert not prots["val"] & prots["test"]


# --- PairDataset : structure d'un exemple ------------------------------------


def test_len(data):
    assert len(make_ds(data)) == 6


def test_item_structure(data, tok):
    item = make_ds(data)[0]  # (P1, P2) : MKAY + CDEFG, aucun crop
    ids, seg = item["input_ids"], item["segment_ids"]

    assert ids.dtype == torch.long and seg.dtype == torch.long
    assert ids.shape == seg.shape
    assert tok.detokenize(ids.tolist()) == "<CLS>MKAY<SEP>CDEFG<EOS>"
    # segment 0 : <CLS> + A + <SEP> ; segment 1 : B + <EOS>
    assert seg.tolist() == [0] * (4 + 2) + [1] * (5 + 1)
    assert item["label"] == 1


def test_every_item_is_well_formed(data, tok):
    ds = make_ds(data, augmentation_percent=1.0)
    for i in range(len(ds)):
        ids, seg = ds[i]["input_ids"], ds[i]["segment_ids"]
        assert ids[0] == tok.cls_id and ids[-1] == tok.eos_id
        assert (ids == tok.sep_id).sum() == 1
        sep = (ids == tok.sep_id).nonzero().item()
        assert (seg[: sep + 1] == 0).all() and (seg[sep + 1 :] == 1).all()
        assert len(ids) <= 2 * MAX_LEN + 3
        assert 0 <= ids.min() and ids.max() < tok.vocab_size


# --- PairDataset : crop -------------------------------------------------------


def test_short_proteins_are_not_cropped(data):
    ds = make_ds(data, max_protein_length=100)
    item = ds[1]  # (P1, LONG)
    assert len(item["input_ids"]) == 4 + 20 + 3


def test_crop_keeps_a_contiguous_window(data, tok):
    ds = make_ds(data)
    torch.manual_seed(0)
    for _ in range(20):
        decoded = tok.detokenize(ds[1]["input_ids"].tolist())  # (P1, LONG)
        window = decoded.split("<SEP>")[1].removesuffix("<EOS>")
        assert len(window) == MAX_LEN
        assert window in SEQS["LONG"]  # un morceau d'un seul tenant


def test_crop_is_fixed_in_eval(data, tok):
    ds = make_ds(data, train=False)
    runs = {tuple(ds[1]["input_ids"].tolist()) for _ in range(10)}
    assert len(runs) == 1
    window = tok.detokenize(ds[1]["input_ids"].tolist()).split("<SEP>")[1]
    assert window.startswith(SEQS["LONG"][:MAX_LEN])  # on garde le début


def test_crop_is_random_in_train(data):
    ds = make_ds(data, train=True)
    torch.manual_seed(0)
    runs = {tuple(ds[1]["input_ids"].tolist()) for _ in range(30)}
    assert len(runs) > 1


# --- PairDataset : augmentation ----------------------------------------------


def test_no_augmentation_by_default(data):
    assert make_ds(data).pairs == data.pairs["train"]


@pytest.mark.parametrize("percent, expected_len", [(0.0, 6), (0.5, 9), (1.0, 12)])
def test_augmentation_size(data, percent, expected_len):
    assert len(make_ds(data, augmentation_percent=percent)) == expected_len


def test_augmented_pairs_are_reversed_originals(data):
    original = data.pairs["train"]
    ds = make_ds(data, augmentation_percent=0.5)
    assert ds.pairs[: len(original)] == original  # les originales ne bougent pas
    for a, b, label in ds.pairs[len(original) :]:
        assert (b, a, label) in original  # label conservé


def test_augmentation_picks_distinct_pairs(data):
    extra = make_ds(data, augmentation_percent=1.0).pairs[6:]
    assert sorted(extra) == sorted((b, a, lab) for a, b, lab in data.pairs["train"])


def test_augmentation_is_reproducible_with_seed(data):
    ds1 = make_ds(data, augmentation_percent=0.5, seed=42)
    ds2 = make_ds(data, augmentation_percent=0.5, seed=42)
    assert ds1.pairs == ds2.pairs


def test_no_augmentation_in_eval(data):
    ds = make_ds(data, split="val", train=False, augmentation_percent=1.0)
    assert ds.pairs == data.pairs["val"]


def test_original_pairs_list_is_not_modified(data):
    before = list(data.pairs["train"])
    make_ds(data, augmentation_percent=1.0)
    assert data.pairs["train"] == before


# --- GoldStandardDataset : loaders --------------------------------------------


def make_data(data_dir, **kwargs):
    kwargs.setdefault("max_protein_length", MAX_LEN)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("num_workers", 0)
    return GoldStandardDataset(data_dir, **kwargs)


def test_mlm_loader_uses_only_positive_pairs(data_dir):
    loader = make_data(data_dir).mlm_loader("train")
    assert len(loader.dataset) == 3
    assert all(label == 1 for _, _, label in loader.dataset.pairs)


def test_classif_loader_uses_all_pairs(data_dir):
    loader = make_data(data_dir).classif_loader("train")
    assert len(loader.dataset) == 6
    assert {label for _, _, label in loader.dataset.pairs} == {0, 1}


def test_loader_batch_size(data_dir):
    batch = next(iter(make_data(data_dir, batch_size=4).classif_loader("train")))
    assert batch["input_ids"].shape[0] == 4


def test_mlm_loader_batch_is_masked(data_dir, tok):
    batch = next(iter(make_data(data_dir).mlm_loader("train")))
    assert set(batch) == {"input_ids", "segment_ids", "attention_mask", "labels"}
    assert batch["labels"].shape == batch["input_ids"].shape  # labels par position
    assert (batch["labels"] != -100).any()


def test_classif_loader_batch_has_pair_labels(data_dir):
    batch = next(iter(make_data(data_dir).classif_loader("train")))
    assert set(batch) == {"input_ids", "segment_ids", "attention_mask", "labels"}
    assert batch["labels"].shape == (2,)  # un label par paire
    assert batch["labels"].dtype == torch.float


def test_train_is_shuffled_eval_is_not(data_dir):
    d = make_data(data_dir)
    assert isinstance(d.classif_loader("train").sampler, torch.utils.data.RandomSampler)
    assert isinstance(d.classif_loader("val").sampler, torch.utils.data.SequentialSampler)
    assert isinstance(d.classif_loader("test").sampler, torch.utils.data.SequentialSampler)


def test_train_and_eval_modes(data_dir):
    d = make_data(data_dir)
    assert d.classif_loader("train").dataset.train is True
    assert d.classif_loader("val").dataset.train is False
    assert d.classif_loader("test").dataset.train is False


def test_augmentation_only_on_train_loader(data_dir):
    d = make_data(data_dir, augmentation_percent=1.0)
    assert len(d.classif_loader("train").dataset) == 12
    assert len(d.mlm_loader("train").dataset) == 6
    assert len(d.classif_loader("val").dataset) == 2
    assert len(d.classif_loader("test").dataset) == 2


def test_loaders_pass_max_protein_length(data_dir):
    batch = next(iter(make_data(data_dir, batch_size=6).classif_loader("train")))
    assert batch["input_ids"].shape[1] <= 2 * MAX_LEN + 3


def test_eval_loader_covers_every_pair_once(data_dir):
    loader = make_data(data_dir, batch_size=1).classif_loader("test")
    labels = [batch["labels"].item() for batch in loader]
    assert sorted(labels) == [0.0, 1.0]
