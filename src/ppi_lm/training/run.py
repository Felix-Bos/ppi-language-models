"""Runs one experiment from a YAML config.

    python -m ppi_lm.training.run --config configs/transformer_A.yaml --caffeinate

    # quick check of speed and memory before a full run
    python -m ppi_lm.training.run --config configs/transformer_A.yaml \
        --limit-train-batches 50 --limit-eval-batches 10 --name smoke

--caffeinate keeps the Mac awake until the run ends (macOS `caffeinate`).
With logging.launch_tensorboard, TensorBoard is started on experiment.output_dir (or
reused if already running) and opened in the browser.

Creates runs/<name>_<timestamp>/ with:
    config.yaml     the exact config of the run
    metrics.jsonl   one JSON line per event (train steps, system, epochs, validation, test)
    best.pt         weights of the best epoch (validation AUROC)
    results.json    best epoch, validation and test metrics, training time
    tensorboard/    scalars, curves, histograms, config and hyper-parameters

Only experiment A (direct classification) is implemented for now.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.factory import build_pair_classifier
from ppi_lm.training.logger import RunLogger, flatten
from ppi_lm.training.trainer import Trainer
from ppi_lm.training.utils import (
    count_parameters,
    keep_awake,
    launch_tensorboard,
    load_config,
    make_run_dir,
    resolve_device,
    save_config,
    set_seed,
)


def _as_code(text: str) -> str:
    """Markdown code block (4-space indent), as rendered by the TensorBoard Text tab."""
    return "\n".join(f"    {line}" for line in text.splitlines())


def _log_setup(logger: RunLogger, cfg: dict, model: torch.nn.Module, device) -> None:
    sizes = {
        "n_parameters": count_parameters(model),
        "n_parameters_encoder": count_parameters(model.encoder),
        "n_parameters_head": count_parameters(model.head),
    }
    logger.log("setup", 0, device=str(device), **sizes)
    logger.log_text("config", _as_code(yaml.safe_dump(cfg, sort_keys=False)))
    summary = "\n".join(f"{key}: {value:,}" for key, value in sizes.items())
    logger.log_text("model", _as_code(f"{summary}\ndevice: {device}\n\n{model}"))


def _log_hparams(logger: RunLogger, cfg: dict, results: dict) -> None:
    """HParams tab: this run's hyper-parameters against its final metrics."""
    test, val = results["test"], results["best_val"]
    logger.log_hparams(
        flatten(cfg),
        {
            "hparam/best_val_auroc": val["auroc"],
            "hparam/test_auroc": test["auroc"],
            "hparam/test_auprc": test["auprc"],
            "hparam/test_accuracy": test["accuracy"],
            "hparam/test_f1": test["f1"],
            "hparam/test_mcc": test["mcc"],
            "hparam/best_epoch": results["best_epoch"],
            "hparam/train_time_min": results["train_time_s"] / 60,
            "hparam/n_parameters": results["n_parameters"],
        },
    )


def run_experiment(
    cfg: dict,
    limit_train_batches: int | None = None,
    limit_eval_batches: int | None = None,
) -> Path:
    """Trains, selects the best epoch on validation, evaluates once on test.

    returns: the run directory
    """
    experiment, logging_cfg = cfg["experiment"], cfg["logging"]
    if experiment["strategy"] != "A":
        raise NotImplementedError(
            f"Strategy '{experiment['strategy']}' is not implemented yet (only A)"
        )

    set_seed(experiment["seed"])
    device = resolve_device(cfg["training"]["device"])
    run_dir = make_run_dir(experiment["output_dir"], experiment["name"])
    save_config(cfg, run_dir / "config.yaml")
    if logging_cfg["tensorboard"] and logging_cfg["launch_tensorboard"]:
        launch_tensorboard(experiment["output_dir"], port=logging_cfg["tensorboard_port"])

    tokenizer = ProteinTokenizer()
    data = GoldStandardDataset(tokenizer=tokenizer, **cfg["data"])
    model = build_pair_classifier(cfg, tokenizer).to(device)

    with RunLogger(run_dir, tensorboard=logging_cfg["tensorboard"]) as logger:
        _log_setup(logger, cfg, model, device)
        trainer = Trainer(
            device,
            logger,
            run_dir,
            precision=cfg["training"]["precision"],
            log_every_steps=logging_cfg["log_every_steps"],
            limit_train_batches=limit_train_batches,
            limit_eval_batches=limit_eval_batches,
        )
        start = time.time()
        summary = trainer.fit_classifier(
            model,
            data.classif_loader("train"),
            data.classif_loader("val"),
            cfg["training"]["classification"],
        )
        train_time = time.time() - start

        labels, logits = trainer.predict(model, data.classif_loader("test"))
        test = trainer.log_evaluation("test", labels, logits, trainer.global_step)

        results = {
            "run_dir": str(run_dir),
            "device": str(device),
            "n_parameters": count_parameters(model),
            "train_time_s": train_time,
            "limit_train_batches": limit_train_batches,
            "limit_eval_batches": limit_eval_batches,
            **summary,
            "test": test,
        }
        _log_hparams(logger, cfg, results)

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    return run_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PPI experiment from a YAML config.")
    parser.add_argument("--config", required=True, help="path to the YAML config")
    parser.add_argument("--device", help="override training.device (cpu, mps, cuda)")
    parser.add_argument("--name", help="override experiment.name")
    parser.add_argument("--limit-train-batches", type=int, help="batches per training epoch")
    parser.add_argument("--limit-eval-batches", type=int, help="batches per evaluation")
    parser.add_argument(
        "--caffeinate", action="store_true", help="keep the Mac awake until the run ends"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.device:
        cfg["training"]["device"] = args.device
    if args.name:
        cfg["experiment"]["name"] = args.name

    with keep_awake(args.caffeinate):
        run_dir = run_experiment(cfg, args.limit_train_batches, args.limit_eval_batches)
    results = json.loads((run_dir / "results.json").read_text())
    print(
        f"best epoch {results['best_epoch']}/{results['epochs_run']} | "
        f"val AUROC {results['best_val']['auroc']:.4f} | "
        f"test AUROC {results['test']['auroc']:.4f} | "
        f"{results['train_time_s'] / 60:.1f} min | {run_dir}"
    )


if __name__ == "__main__":
    main()
