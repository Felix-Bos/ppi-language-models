"""Loads a run and collects everything the analyses need from its model.

run = load_run(find_latest_run("runs"), data_dir="data/gold_standard_data")
pairs = sample_pairs(run.data, "test", n=2000)
preds = predict_pairs(run.model, run.data, pairs)

A run interrupted after its first epoch (Ctrl+C) can be analysed too: best.pt holds the
best epoch reached so far, only results.json (test metrics) is missing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ppi_lm.data_scripts.collators import ClassificationCollator
from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset, PairDataset
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.factory import build_pair_classifier
from ppi_lm.models.shared.classifier import PairClassifier
from ppi_lm.models.shared.heads import masked_mean_pool
from ppi_lm.training.logger import read_metrics
from ppi_lm.training.utils import load_config, resolve_device

SMOKE_COMMAND = (
    "python -m ppi_lm.training.run --config configs/transformer_A.yaml "
    "--limit-train-batches 50 --limit-eval-batches 10 --name smoke"
)


def find_latest_run(runs_dir: str | Path = "runs", pattern: str = "*") -> Path:
    """Most recent run (by the time of its best.pt) whose directory name matches pattern.

    raises FileNotFoundError with the command to train one if there is none
    """
    runs = [checkpoint.parent for checkpoint in Path(runs_dir).glob(f"{pattern}/best.pt")]
    if not runs:
        raise FileNotFoundError(
            f"No trained run matching '{pattern}' in {Path(runs_dir).resolve()} "
            "(a run has a best.pt after its first epoch). Train one first, from the "
            f"repository root, e.g. a 5-minute check:\n    {SMOKE_COMMAND}"
        )
    return max(runs, key=lambda run_dir: (run_dir / "best.pt").stat().st_mtime)


@dataclass
class LoadedRun:
    run_dir: Path
    cfg: dict
    model: PairClassifier  # best weights, eval mode
    tokenizer: ProteinTokenizer
    data: GoldStandardDataset
    results: dict  # results.json, empty if the run was interrupted
    metrics: list[dict]  # metrics.jsonl
    device: torch.device
    best_epoch: int  # epoch of the weights in best.pt

    @property
    def finished(self) -> bool:
        return bool(self.results)

    def validation_history(self) -> list[dict]:
        """One record per finished epoch (validation metrics), also for interrupted runs."""
        return [record for record in self.metrics if record["event"] == "val"]


def load_run(
    run_dir: str | Path, data_dir: str | Path | None = None, device: str = "cpu"
) -> LoadedRun:
    """Rebuilds the model of a run from its config.yaml and loads best.pt.

    data_dir: overrides data.data_dir (needed when the notebook runs from another folder)
    """
    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.yaml")
    data_cfg = {**cfg["data"], "num_workers": 0}  # no worker processes inside a notebook
    if data_dir is not None:
        data_cfg["data_dir"] = str(data_dir)

    device = resolve_device(device)
    tokenizer = ProteinTokenizer()
    model = build_pair_classifier(cfg, tokenizer)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device).eval()

    results_path = run_dir / "results.json"
    return LoadedRun(
        run_dir=run_dir,
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        data=GoldStandardDataset(tokenizer=tokenizer, **data_cfg),
        results=json.loads(results_path.read_text()) if results_path.exists() else {},
        metrics=read_metrics(run_dir / "metrics.jsonl"),
        device=device,
        best_epoch=state["epoch"],
    )


def sample_pairs(
    data: GoldStandardDataset, split: str, n: int | None = None, seed: int = 0
) -> list[tuple[str, str, int]]:
    """n pairs of a split drawn at random (fixed seed), all of them if n is None."""
    pairs = data.pairs[split]
    if n is None or n >= len(pairs):
        return list(pairs)
    g = torch.Generator().manual_seed(seed)
    return [pairs[i] for i in torch.randperm(len(pairs), generator=g)[:n].tolist()]


@dataclass
class PairPredictions:
    """Model outputs for a list of pairs, aligned with `pairs`."""

    pairs: list[tuple[str, str, int]]
    labels: torch.Tensor  # (N,) 0 / 1
    logits: torch.Tensor  # (N,)
    pair_vectors: torch.Tensor  # (N, pooled_dim): what the classification head sees
    protein_a: torch.Tensor  # (N, d_model): mean of the hidden states of protein A
    protein_b: torch.Tensor  # (N, d_model): same for protein B
    length_a: torch.Tensor  # (N,) residues before cropping
    length_b: torch.Tensor  # (N,)
    extra: dict = field(default_factory=dict)

    @property
    def probs(self) -> torch.Tensor:
        return torch.sigmoid(self.logits)

    def __len__(self) -> int:
        return len(self.pairs)


def make_loader(model, data, pairs, batch_size: int = 32) -> DataLoader:
    """Deterministic loader (fixed crop at the start, no shuffle) over a list of pairs."""
    dataset = PairDataset(
        pairs, data.seqs, data.tokenizer, max_protein_length=data.max_protein_length, train=False
    )
    return DataLoader(
        dataset, batch_size=batch_size, collate_fn=ClassificationCollator(data.tokenizer)
    )


@torch.no_grad()
def predict_pairs(
    model: PairClassifier,
    data: GoldStandardDataset,
    pairs: list[tuple[str, str, int]],
    batch_size: int = 32,
) -> PairPredictions:
    """Runs the encoder once per pair and keeps logits, pooled vectors and protein vectors."""
    model.eval()
    device = next(model.parameters()).device
    outputs = {key: [] for key in ("labels", "logits", "pair", "a", "b")}
    for batch in make_loader(model, data, pairs, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        mask, seg = batch["attention_mask"], batch["segment_ids"]
        hidden = model.encoder(batch["input_ids"], seg, mask)
        pooled = model.pool(hidden, mask, seg)
        outputs["labels"].append(batch["labels"].cpu())
        outputs["logits"].append(model.head(pooled).float().cpu())
        outputs["pair"].append(pooled.float().cpu())
        outputs["a"].append(masked_mean_pool(hidden, mask & (seg == 0)).float().cpu())
        outputs["b"].append(masked_mean_pool(hidden, mask & (seg == 1)).float().cpu())

    cat = {key: torch.cat(values) for key, values in outputs.items()}
    return PairPredictions(
        pairs=list(pairs),
        labels=cat["labels"],
        logits=cat["logits"],
        pair_vectors=cat["pair"],
        protein_a=cat["a"],
        protein_b=cat["b"],
        length_a=torch.tensor([len(data.seqs[a]) for a, _, _ in pairs]),
        length_b=torch.tensor([len(data.seqs[b]) for _, b, _ in pairs]),
    )


def swapped(pairs: list[tuple[str, str, int]]) -> list[tuple[str, str, int]]:
    """(A, B) -> (B, A): to measure how much the prediction depends on the order."""
    return [(b, a, label) for a, b, label in pairs]


def protein_embeddings(preds: PairPredictions) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """One vector per protein: mean of its vectors over every pair it appears in.

    The encoder only ever sees pairs, so a protein's vector depends on its partner;
    averaging over its partners gives a context-independent summary.

    returns: protein ids, vectors (P, d_model), number of pairs per protein (P,)
    """
    sums, counts = {}, {}
    for i, (a, b, _) in enumerate(preds.pairs):
        for pid, vector in ((a, preds.protein_a[i]), (b, preds.protein_b[i])):
            sums[pid] = sums.get(pid, 0) + vector
            counts[pid] = counts.get(pid, 0) + 1
    ids = sorted(sums)
    vectors = torch.stack([sums[pid] / counts[pid] for pid in ids])
    return ids, vectors, torch.tensor([counts[pid] for pid in ids])
