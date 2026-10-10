"""AdamW with separate encoder / head learning rates, and a warmup + cosine schedule."""

import math

import torch


def _decay_groups(module: torch.nn.Module, name: str, lr: float, weight_decay: float) -> list[dict]:
    """Weight decay on matrices only: never on biases nor LayerNorm weights (1D).

    name: stored in each group (encoder / head), used to log one lr per group
    """
    params = [p for p in module.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = [
        {"params": decay, "name": name, "lr": lr, "weight_decay": weight_decay},
        {"params": no_decay, "name": name, "lr": lr, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def build_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.01,
    encoder_lr_scale: float = 1.0,
    freeze_encoder: bool = False,
) -> torch.optim.AdamW:
    """AdamW for a model with `encoder` and `head` sub-modules.

    freeze_encoder: the encoder gets requires_grad=False and is left out (head only)
    encoder_lr_scale: encoder lr = lr * encoder_lr_scale (fine-tuning a pre-trained encoder)
    """
    if freeze_encoder:
        model.encoder.requires_grad_(False)
        groups = []
    else:
        groups = _decay_groups(model.encoder, "encoder", lr * encoder_lr_scale, weight_decay)
    groups += _decay_groups(model.head, "head", lr, weight_decay)
    return torch.optim.AdamW(groups)


def warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float = 0.05
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup from ~0 to the group's lr, then cosine decay to 0 at total_steps.

    The factor multiplies each group's own lr, so the encoder / head ratio is kept.
    """
    warmup_steps = round(warmup_ratio * total_steps)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
