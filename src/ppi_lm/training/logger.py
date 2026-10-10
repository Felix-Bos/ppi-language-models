"""Run logger: one JSON line per event in metrics.jsonl, mirrored to TensorBoard.

metrics.jsonl is the source of truth (read by notebooks and, later, the web platform);
TensorBoard is the live viewer. Scalars are logged as <event>/<key>, e.g. val/auroc.

TensorBoard tabs filled during a run:
    Scalars     train/*, system/*, train_epoch/*, val/*, test/*
    Histograms  weights/*, val/prob_*, test/prob_*
    PR curves   val/pr_curve, test/pr_curve
    Images      val/roc_curve, test/roc_curve
    Text        config, model
    HParams     hyper-parameters of every run against their final metrics
"""

import json
import time
import warnings
from pathlib import Path

import torch


def flatten(cfg: dict, prefix: str = "") -> dict:
    """Nested config -> {"model.transformer.d_model": 256, ...} with TensorBoard-safe values."""
    flat = {}
    for key, value in cfg.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{name}."))
        elif isinstance(value, bool | int | float | str):
            flat[name] = value
        else:  # None, lists
            flat[name] = str(value)
    return flat


class RunLogger:
    def __init__(self, run_dir: str | Path, tensorboard: bool = True):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.writer = None
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                warnings.warn(
                    "tensorboard is not installed: logging to metrics.jsonl only", stacklevel=2
                )
            else:
                self.writer = SummaryWriter(str(Path(run_dir) / "tensorboard"))

    def log(self, event: str, step: int, **values) -> None:
        """event: e.g. train, val, test. Numeric values also go to TensorBoard as event/key."""
        record = {"event": event, "step": step, "time": time.time(), **values}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")
        if self.writer is not None:
            for key, value in values.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    self.writer.add_scalar(f"{event}/{key}", value, step)

    # --- TensorBoard only (no-ops without TensorBoard) -----------------------------

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        if self.writer is not None and values.numel() > 0:
            self.writer.add_histogram(tag, values.detach().float().cpu(), step)

    def log_weights(self, model: torch.nn.Module, step: int) -> None:
        """One histogram per parameter tensor, e.g. weights/encoder.blocks.0.attn.qkv.weight."""
        if self.writer is not None:
            for name, param in model.named_parameters():
                self.log_histogram(f"weights/{name}", param, step)

    def log_pr_curve(self, tag: str, labels: torch.Tensor, probs: torch.Tensor, step: int) -> None:
        if self.writer is not None:
            self.writer.add_pr_curve(tag, labels.detach().cpu(), probs.detach().cpu(), step)

    def log_figure(self, tag: str, figure, step: int) -> None:
        if self.writer is not None and figure is not None:
            self.writer.add_figure(tag, figure, step)

    def log_hparams(self, hparams: dict, metrics: dict) -> None:
        """HParams tab: compares runs by hyper-parameters. metrics keys: e.g. hparam/test_auroc."""
        if self.writer is not None:
            self.writer.add_hparams(hparams, metrics, run_name=".")  # "." = this run's directory

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_metrics(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
