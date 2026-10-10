"""The config template must stay in sync with the constructors it feeds."""

import inspect
import re
from pathlib import Path

import pytest
import torch
import yaml

import ppi_lm
from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.shared.heads import ClassificationHead, MLMHead
from ppi_lm.models.transformer.encoder import TransformerConfig, TransformerEncoder

TEMPLATE = Path(ppi_lm.__file__).parent / "config" / "config_template.yaml"
POOLINGS = {"pair", "mean", "max", "cls"}


@pytest.fixture(scope="module")
def cfg():
    with open(TEMPLATE) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tok():
    return ProteinTokenizer()


def leaves(node, path=""):
    """Every (path, value) of the nested config."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}.{key}" if path else key)
    else:
        yield path, node


def test_template_has_every_section(cfg):
    assert set(cfg) >= {"experiment", "data", "model", "heads", "training", "logging"}


def test_no_number_was_read_as_a_string(cfg):
    """YAML 1.1 reads `3e-4` as a string: floats need a dot (`3.0e-4`)."""
    number = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
    for path, value in leaves(cfg):
        assert not (isinstance(value, str) and number.match(value)), path


def test_experiment_values(cfg):
    assert cfg["experiment"]["strategy"] in {"A", "B", "C"}
    assert cfg["model"]["architecture"] in {"transformer", "cnn", "bilstm", "bimamba"}


def test_data_keys_are_gold_standard_dataset_arguments(cfg):
    params = set(inspect.signature(GoldStandardDataset.__init__).parameters) - {"self"}
    assert set(cfg["data"]) <= params - {"tokenizer"}


def test_transformer_section_builds_the_encoder(cfg, tok):
    config = TransformerConfig(
        vocab_size=tok.vocab_size, pad_id=tok.pad_id, **cfg["model"]["transformer"]
    )
    encoder = TransformerEncoder(config)
    assert encoder.d_model == cfg["model"]["transformer"]["d_model"]


def test_rope_cache_covers_the_longest_pair(cfg):
    longest_pair = 2 * cfg["data"]["max_protein_length"] + 3
    assert cfg["model"]["transformer"]["max_len"] >= longest_pair


def test_classification_section_builds_the_head(cfg):
    section = dict(cfg["heads"]["classification"])
    pooling = section.pop("pooling")
    assert pooling in POOLINGS
    d_model = cfg["model"]["transformer"]["d_model"]
    input_dim = 2 * d_model if pooling == "pair" else d_model

    head = ClassificationHead(input_dim, **section)
    assert head(torch.randn(2, input_dim)).shape == (2,)


def test_mlm_section_builds_the_head(cfg, tok):
    section = dict(cfg["heads"]["mlm"])
    tie = section.pop("tie_embedding")
    assert isinstance(tie, bool)
    config = TransformerConfig(vocab_size=tok.vocab_size, **cfg["model"]["transformer"])
    encoder = TransformerEncoder(config)

    head = MLMHead(
        encoder.d_model,
        tok.vocab_size,
        tied_embedding=encoder.token_embedding if tie else None,
        **section,
    )
    assert head(torch.randn(1, 3, encoder.d_model)).shape == (1, 3, tok.vocab_size)


@pytest.mark.parametrize("stage", ["classification", "mlm"])
def test_training_stage_values(cfg, stage):
    section = cfg["training"][stage]
    assert section["epochs"] > 0 and section["early_stopping_patience"] > 0
    assert 0 < section["lr"] < 1
    assert 0 <= section["warmup_ratio"] < 1
    assert isinstance(section["grad_accumulation"], int) and section["grad_accumulation"] >= 1


def test_training_options(cfg):
    assert cfg["training"]["device"] in {"auto", "cuda", "mps", "cpu"}
    assert cfg["training"]["precision"] in {"fp32", "bf16"}
    assert cfg["training"]["classification"]["encoder_lr_scale"] > 0
