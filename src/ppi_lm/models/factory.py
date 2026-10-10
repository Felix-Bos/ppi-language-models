"""Builds models from a config (see config/config_template.yaml).

Values that depend on the tokenizer or on the encoder (vocab_size, pad_id, input_dim)
are filled in here, never read from the YAML.
"""

import torch

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.shared.classifier import PairClassifier, pooled_dim
from ppi_lm.models.shared.heads import ClassificationHead
from ppi_lm.models.transformer.encoder import TransformerConfig, TransformerEncoder


def build_encoder(model_cfg: dict, tokenizer: ProteinTokenizer) -> torch.nn.Module:
    """model_cfg: the `model` section. Returns an encoder with a `d_model` attribute."""
    architecture = model_cfg["architecture"]
    if architecture == "transformer":
        config = TransformerConfig(
            vocab_size=tokenizer.vocab_size, pad_id=tokenizer.pad_id, **model_cfg["transformer"]
        )
        return TransformerEncoder(config)
    raise NotImplementedError(f"Architecture '{architecture}' is not implemented yet")


def build_pair_classifier(cfg: dict, tokenizer: ProteinTokenizer) -> PairClassifier:
    """cfg: the whole config. Encoder + pooling + ClassificationHead."""
    encoder = build_encoder(cfg["model"], tokenizer)
    head_cfg = dict(cfg["heads"]["classification"])
    pooling = head_cfg.pop("pooling")
    head = ClassificationHead(pooled_dim(pooling, encoder.d_model), **head_cfg)
    return PairClassifier(encoder, head, pooling)
