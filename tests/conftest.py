"""Shared fixtures: a synthetic mini-dataset and a tiny CPU config built from the template."""

from pathlib import Path

import pytest
import torch
import yaml

import ppi_lm

TEMPLATE = Path(ppi_lm.__file__).parent / "config" / "config_template.yaml"
AA = "ACDEFGHIKLMNPQRSTVWY"


def make_tiny_data(path: Path) -> Path:
    """3 splits with their own proteins (5-14 residues), both classes in each split.

    train (Intra1): 10 proteins, 8 + 8 pairs; val (Intra0) and test (Intra2): 6 proteins,
    4 + 4 pairs. Files follow the Bernett naming, positives listed before negatives.
    """
    g = torch.Generator().manual_seed(0)
    fasta, files = [], {}
    for prefix, split, n_prot, n_pairs in [
        ("T", "Intra1", 10, 8),
        ("V", "Intra0", 6, 4),
        ("S", "Intra2", 6, 4),
    ]:
        ids = [f"{prefix}{i}" for i in range(n_prot)]
        for pid in ids:
            length = torch.randint(5, 15, (1,), generator=g).item()
            fasta.append(
                f">{pid}\n{''.join(AA[i] for i in torch.randint(20, (length,), generator=g))}\n"
            )
        for label in ("pos", "neg"):
            pairs = torch.randint(n_prot, (n_pairs, 2), generator=g).tolist()
            files[f"{split}_{label}_rr.txt"] = "".join(f"{ids[a]} {ids[b]}\n" for a, b in pairs)
    path.mkdir(parents=True, exist_ok=True)
    (path / "human_swissprot_oneliner.fasta").write_text("".join(fasta))
    for name, content in files.items():
        (path / name).write_text(content)
    return path


def make_tiny_cfg(output_dir: Path, data_dir: Path) -> dict:
    """The template with a tiny model, tiny batches, CPU, nothing launched."""
    with open(TEMPLATE) as f:
        cfg = yaml.safe_load(f)
    cfg["experiment"].update(name="test", output_dir=str(output_dir), seed=0)
    cfg["data"].update(data_dir=str(data_dir), max_protein_length=8, batch_size=4, num_workers=0)
    cfg["model"]["transformer"].update(d_model=16, n_layers=1, n_heads=2, max_len=64, dropout=0.0)
    cfg["heads"]["classification"].update(hidden_dims=[8], dropout=0.0)
    cfg["training"]["device"] = "cpu"
    cfg["training"]["classification"].update(epochs=2, early_stopping_patience=2)
    cfg["logging"].update(tensorboard=False, launch_tensorboard=False, log_every_steps=1)
    return cfg


@pytest.fixture
def tiny_data_dir(tmp_path):
    return make_tiny_data(tmp_path / "data")


@pytest.fixture
def tiny_cfg(tmp_path, tiny_data_dir):
    return make_tiny_cfg(tmp_path / "runs", tiny_data_dir)
