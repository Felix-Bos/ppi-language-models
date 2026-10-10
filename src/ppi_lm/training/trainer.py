"""Training and evaluation loops for pair classification.

One optimizer step every `grad_accumulation` batches, warmup + cosine schedule,
gradient clipping, early stopping on the validation AUROC, best weights reloaded at
the end. The same loop serves experiment A and the classification stage of B
(frozen or fine-tuned encoder).

Logged (x-axis: optimizer step, so that train and validation curves line up):
    every log_every_steps   train/loss, train/grad_norm, train/lr_<group>
                            system/pairs_per_s, tokens_per_s, step_time_s, memory
    every epoch             train_epoch/* (metrics on the training predictions, with dropout)
                            val/* (every metric of binary_metrics), val/pr_curve,
                            val/roc_curve, val/prob_positives|negatives, weights/*
                            val_best/auroc, val_best/epoch
"""

import math
import time
from itertools import islice
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ppi_lm.training.logger import RunLogger
from ppi_lm.training.metrics import binary_metrics, roc_figure
from ppi_lm.training.optim import build_optimizer, warmup_cosine_schedule
from ppi_lm.training.utils import memory_stats

INPUT_KEYS = ("input_ids", "segment_ids", "attention_mask")


class _Window:
    """Training statistics accumulated between two log steps."""

    def __init__(self):
        self.start = time.perf_counter()
        self.loss = 0.0  # becomes a device tensor: summed without synchronizing
        self.batches = self.steps = self.pairs = self.tokens = 0

    def add(self, loss: torch.Tensor, n_pairs: int, n_tokens: int) -> None:
        self.loss = self.loss + loss.detach()
        self.batches += 1
        self.pairs += n_pairs
        self.tokens += n_tokens


class Trainer:
    def __init__(
        self,
        device: torch.device,
        logger: RunLogger,
        run_dir: str | Path,
        precision: str = "fp32",
        log_every_steps: int = 50,
        limit_train_batches: int | None = None,
        limit_eval_batches: int | None = None,
    ):
        """
        precision: fp32 | bf16 (autocast, the loss stays in fp32)
        limit_*_batches: only use the first batches of each epoch (quick checks)
        """
        if precision not in ("fp32", "bf16"):
            raise ValueError(f"Unknown precision '{precision}', expected fp32 or bf16")
        self.device = device
        self.logger = logger
        self.run_dir = Path(run_dir)
        self.precision = precision
        self.log_every_steps = log_every_steps
        self.limit_train_batches = limit_train_batches
        self.limit_eval_batches = limit_eval_batches
        self.global_step = 0  # optimizer steps since the start of the run
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

    # --- helpers ----------------------------------------------------------------

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.precision == "bf16",
        )

    def _to_device(self, batch: dict) -> dict:
        return {key: value.to(self.device) for key, value in batch.items()}

    @staticmethod
    def _n_batches(loader, limit: int | None) -> int:
        return len(loader) if limit is None else min(len(loader), limit)

    def _logits(self, model: torch.nn.Module, batch: dict) -> torch.Tensor:
        with self._autocast():
            logits = model(*(batch[key] for key in INPUT_KEYS))
        return logits.float()

    def _metrics(self, labels: torch.Tensor, logits: torch.Tensor) -> dict:
        return {
            "loss": self.loss_fn(logits, labels.float()).item(),
            **binary_metrics(labels, logits),
            "n_pairs": len(labels),
        }

    # --- evaluation -------------------------------------------------------------

    def _eval_subset(self, loader):
        """With limit_eval_batches: a fixed subset spread evenly over the whole split.

        The evaluation loaders list every positive pair, then every negative one, so
        their first batches would hold a single class (AUROC undefined).
        """
        if self.limit_eval_batches is None or self.limit_eval_batches >= len(loader):
            return loader
        n_pairs = self.limit_eval_batches * loader.batch_size
        indices = torch.linspace(0, len(loader.dataset) - 1, n_pairs).round().long()
        return DataLoader(
            Subset(loader.dataset, indices.unique().tolist()),
            batch_size=loader.batch_size,
            shuffle=False,
            num_workers=loader.num_workers,
            collate_fn=loader.collate_fn,
        )

    @torch.no_grad()
    def predict(self, model: torch.nn.Module, loader) -> tuple[torch.Tensor, torch.Tensor]:
        """labels (N,) and logits (N,) on CPU, over the loader (or a spread-out subset)."""
        model.eval()
        all_logits, all_labels = [], []
        loader = self._eval_subset(loader)
        for batch in tqdm(loader, desc="eval", leave=False):
            batch = self._to_device(batch)
            all_logits.append(self._logits(model, batch).cpu())
            all_labels.append(batch["labels"].cpu())
        return torch.cat(all_labels), torch.cat(all_logits)

    def evaluate(self, model: torch.nn.Module, loader) -> dict:
        """Every metric (loss, AUROC, AUPRC, accuracy, F1, MCC...) over the loader."""
        return self._metrics(*self.predict(model, loader))

    def log_evaluation(
        self, split: str, labels: torch.Tensor, logits: torch.Tensor, step: int, **extra
    ) -> dict:
        """Metrics of one split, logged as scalars, PR curve, ROC curve and histograms."""
        metrics = self._metrics(labels, logits)
        self.logger.log(split, step, **extra, **metrics)
        if self.logger.writer is not None:
            positives, probs = labels.bool(), torch.sigmoid(logits)
            self.logger.log_pr_curve(f"{split}/pr_curve", positives, probs, step)
            self.logger.log_figure(f"{split}/roc_curve", roc_figure(labels, logits, split), step)
            self.logger.log_histogram(f"{split}/prob_positives", probs[positives], step)
            self.logger.log_histogram(f"{split}/prob_negatives", probs[~positives], step)
        return metrics

    # --- training ---------------------------------------------------------------

    def _optimizer_step(self, model, optimizer, scheduler, grad_clip: float | None):
        """Clips, steps, returns the gradient norm before clipping (a tensor)."""
        trainable = [p for p in model.parameters() if p.requires_grad]
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip or math.inf)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        self.global_step += 1
        return grad_norm

    def _log_train_step(self, optimizer, grad_norm: torch.Tensor, window: _Window) -> None:
        elapsed = time.perf_counter() - window.start
        lrs = {f"lr_{group['name']}": group["lr"] for group in optimizer.param_groups}
        self.logger.log(
            "train",
            self.global_step,
            loss=(window.loss / window.batches).item(),
            grad_norm=grad_norm.item(),
            **lrs,
        )
        self.logger.log(
            "system",
            self.global_step,
            pairs_per_s=window.pairs / elapsed,
            tokens_per_s=window.tokens / elapsed,
            step_time_s=elapsed / window.steps,
            **memory_stats(self.device),
        )

    def _train_epoch(self, model, loader, optimizer, scheduler, stage_cfg: dict, epoch: int):
        """One pass over the training loader. Returns metrics on the training predictions."""
        model.train()
        if stage_cfg["freeze_encoder"]:
            model.encoder.eval()  # a frozen encoder must not apply dropout either
        accumulation = stage_cfg["grad_accumulation"]
        n = self._n_batches(loader, self.limit_train_batches)

        start, window = time.perf_counter(), _Window()
        all_logits, all_labels = [], []
        optimizer.zero_grad(set_to_none=True)
        for i, batch in enumerate(tqdm(islice(loader, n), total=n, desc=f"epoch {epoch}")):
            n_tokens = int(batch["attention_mask"].sum())  # on CPU: no device sync
            batch = self._to_device(batch)
            logits = self._logits(model, batch)
            loss = self.loss_fn(logits, batch["labels"])
            (loss / accumulation).backward()
            window.add(loss, len(logits), n_tokens)
            all_logits.append(logits.detach())
            all_labels.append(batch["labels"])

            if (i + 1) % accumulation == 0 or i + 1 == n:
                grad_norm = self._optimizer_step(
                    model, optimizer, scheduler, stage_cfg["grad_clip"]
                )
                window.steps += 1
                if self.global_step % self.log_every_steps == 0:
                    self._log_train_step(optimizer, grad_norm, window)
                    window = _Window()

        metrics = self._metrics(torch.cat(all_labels).cpu(), torch.cat(all_logits).cpu())
        return {**metrics, "time_s": time.perf_counter() - start}

    def fit_classifier(self, model: torch.nn.Module, train_loader, val_loader, stage_cfg: dict):
        """Trains a PairClassifier, early stopping on the validation AUROC.

        stage_cfg: the training.classification section of the config
        returns: best_epoch, best_val (metrics), epochs_run; the model holds the best weights
        """
        n_batches = self._n_batches(train_loader, self.limit_train_batches)
        steps_per_epoch = math.ceil(n_batches / stage_cfg["grad_accumulation"])
        optimizer = build_optimizer(
            model,
            lr=stage_cfg["lr"],
            weight_decay=stage_cfg["weight_decay"],
            encoder_lr_scale=stage_cfg["encoder_lr_scale"],
            freeze_encoder=stage_cfg["freeze_encoder"],
        )
        scheduler = warmup_cosine_schedule(
            optimizer, stage_cfg["epochs"] * steps_per_epoch, stage_cfg["warmup_ratio"]
        )
        self.logger.log_weights(model, self.global_step)  # initial weights

        checkpoint = self.run_dir / "best.pt"
        best_score, best_epoch, best_val, bad_epochs = -math.inf, 0, None, 0
        for epoch in range(1, stage_cfg["epochs"] + 1):
            train = self._train_epoch(model, train_loader, optimizer, scheduler, stage_cfg, epoch)
            self.logger.log("train_epoch", self.global_step, epoch=epoch, **train)
            val = self.log_evaluation(
                "val", *self.predict(model, val_loader), self.global_step, epoch=epoch
            )
            self.logger.log_weights(model, self.global_step)

            score = -math.inf if math.isnan(val["auroc"]) else val["auroc"]
            if epoch == 1 or score > best_score:
                best_score, best_epoch, best_val, bad_epochs = score, epoch, val, 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val}, checkpoint)
            else:
                bad_epochs += 1
            self.logger.log("val_best", self.global_step, auroc=best_val["auroc"], epoch=best_epoch)
            if bad_epochs >= stage_cfg["early_stopping_patience"]:
                break

        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        model.load_state_dict(state["model"])
        return {"best_epoch": best_epoch, "best_val": best_val, "epochs_run": epoch}
