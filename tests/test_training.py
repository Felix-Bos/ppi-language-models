"""Experiment A end to end on a synthetic mini-dataset, tiny model, CPU only."""

import copy
import json
import math
import os
import socket
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import yaml

import ppi_lm
from ppi_lm.data_scripts.collators import ClassificationCollator
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer
from ppi_lm.models.factory import build_encoder, build_pair_classifier
from ppi_lm.models.shared.classifier import PairClassifier, pooled_dim
from ppi_lm.models.shared.heads import ClassificationHead
from ppi_lm.training import run as run_module
from ppi_lm.training import utils
from ppi_lm.training.logger import RunLogger, flatten, read_metrics
from ppi_lm.training.metrics import binary_metrics, roc_figure
from ppi_lm.training.optim import build_optimizer, warmup_cosine_schedule
from ppi_lm.training.run import main, run_experiment
from ppi_lm.training.trainer import Trainer
from ppi_lm.training.utils import (
    keep_awake,
    launch_tensorboard,
    make_run_dir,
    memory_stats,
    port_in_use,
    resolve_device,
)

TEMPLATE = Path(ppi_lm.__file__).parent / "config" / "config_template.yaml"


@pytest.fixture
def tok():
    return ProteinTokenizer()


@pytest.fixture
def data_dir(tiny_data_dir):
    return tiny_data_dir


@pytest.fixture
def cfg(tiny_cfg):
    return tiny_cfg


def make_item(tok, seq_a: str, seq_b: str, label: int) -> dict:
    a, b = tok.tokenize(seq_a), tok.tokenize(seq_b)
    return {
        "input_ids": torch.tensor([tok.cls_id] + a + [tok.sep_id] + b + [tok.eos_id]),
        "segment_ids": torch.tensor([0] * (len(a) + 2) + [1] * (len(b) + 1)),
        "label": label,
    }


@pytest.fixture
def batch(tok):
    items = [make_item(tok, "MKAYL", "CDEFG", 1), make_item(tok, "HI", "KLMN", 0)]
    return ClassificationCollator(tok)(items)


# --- PairClassifier / factory ---------------------------------------------------


@pytest.mark.parametrize(
    "pooling, expected", [("pair", 32), ("mean", 16), ("max", 16), ("cls", 16)]
)
def test_pooled_dim(pooling, expected):
    assert pooled_dim(pooling, 16) == expected


def test_pooled_dim_unknown_pooling_fails():
    with pytest.raises(ValueError, match="Unknown pooling"):
        pooled_dim("sum", 16)


@pytest.mark.parametrize("pooling", ["pair", "mean", "max", "cls"])
def test_classifier_gives_one_logit_per_pair(cfg, tok, batch, pooling):
    cfg["heads"]["classification"]["pooling"] = pooling
    model = build_pair_classifier(cfg, tok)
    logits = model(batch["input_ids"], batch["segment_ids"], batch["attention_mask"])
    assert logits.shape == (2,)


def test_classifier_rejects_a_head_of_the_wrong_size(cfg, tok):
    encoder = build_encoder(cfg["model"], tok)
    with pytest.raises(ValueError, match="expects"):
        PairClassifier(encoder, ClassificationHead(16), pooling="pair")  # pair needs 32


def test_classifier_from_the_real_template(tok):
    with open(TEMPLATE) as f:
        model = build_pair_classifier(yaml.safe_load(f), tok)
    assert model.head.input_dim == 2 * model.encoder.d_model


def test_unknown_architecture_fails(cfg, tok):
    cfg["model"]["architecture"] = "bimamba"
    with pytest.raises(NotImplementedError, match="bimamba"):
        build_encoder(cfg["model"], tok)


# --- metrics --------------------------------------------------------------------


def test_metrics_perfect_separation():
    m = binary_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([-3.0, -1.0, 1.0, 2.0]))
    for key in ("auroc", "auprc", "accuracy", "balanced_accuracy", "precision", "recall"):
        assert m[key] == 1.0, key
    assert m["specificity"] == m["f1"] == m["mcc"] == 1.0
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (2, 0, 2, 0)
    assert m["predicted_positive_rate"] == 0.5
    assert m["mean_prob_positives"] > 0.5 > m["mean_prob_negatives"]


def test_metrics_inverted_scores():
    m = binary_metrics(torch.tensor([0, 0, 1, 1]), torch.tensor([3.0, 1.0, -1.0, -2.0]))
    assert m["auroc"] == 0.0 and m["accuracy"] == 0.0 and m["mcc"] == -1.0


def test_metrics_match_sklearn():
    from sklearn import metrics as skm

    g = torch.Generator().manual_seed(0)
    labels = torch.randint(2, (200,), generator=g)
    logits = torch.randn(200, generator=g) + labels  # informative but noisy
    probs = torch.sigmoid(logits).numpy()
    y, pred = labels.numpy(), (probs >= 0.5).astype(int)

    m = binary_metrics(labels, logits)
    expected = {
        "auroc": skm.roc_auc_score(y, probs),
        "auprc": skm.average_precision_score(y, probs),
        "accuracy": skm.accuracy_score(y, pred),
        "balanced_accuracy": skm.balanced_accuracy_score(y, pred),
        "precision": skm.precision_score(y, pred),
        "recall": skm.recall_score(y, pred),
        "f1": skm.f1_score(y, pred),
        "mcc": skm.matthews_corrcoef(y, pred),
        "brier": skm.brier_score_loss(y, probs),
    }
    for key, value in expected.items():
        assert m[key] == pytest.approx(value, abs=1e-6), key


def test_metrics_no_predicted_positive_does_not_divide_by_zero():
    m = binary_metrics(torch.tensor([0, 1, 1]), torch.tensor([-1.0, -2.0, -3.0]))
    assert m["precision"] == m["recall"] == m["f1"] == m["mcc"] == 0.0


def test_metrics_single_class_gives_nan_ranking_metrics():
    m = binary_metrics(torch.tensor([1, 1]), torch.tensor([0.5, -0.5]))
    assert math.isnan(m["auroc"]) and math.isnan(m["auprc"])
    assert math.isnan(m["mean_prob_negatives"]) and m["accuracy"] == 0.5


def test_roc_figure():
    from matplotlib.figure import Figure

    fig = roc_figure(torch.tensor([0, 1, 0, 1]), torch.tensor([-1.0, 1.0, 0.5, 2.0]))
    assert isinstance(fig, Figure)
    assert roc_figure(torch.tensor([1, 1]), torch.tensor([0.0, 1.0])) is None


# --- optimizer and schedule -----------------------------------------------------


def test_optimizer_no_weight_decay_on_biases_and_norms(cfg, tok):
    optimizer = build_optimizer(build_pair_classifier(cfg, tok), lr=1e-3, weight_decay=0.1)
    for group in optimizer.param_groups:
        dims = {p.ndim for p in group["params"]}
        assert group["weight_decay"] == (0.1 if dims == {2} else 0.0)


def test_optimizer_scales_the_encoder_lr(cfg, tok):
    model = build_pair_classifier(cfg, tok)
    optimizer = build_optimizer(model, lr=1e-3, encoder_lr_scale=0.1)
    encoder_params = {id(p) for p in model.encoder.parameters()}
    for group in optimizer.param_groups:
        is_encoder = id(group["params"][0]) in encoder_params
        assert group["name"] == ("encoder" if is_encoder else "head")
        assert group["lr"] == pytest.approx(1e-4 if is_encoder else 1e-3)


def test_optimizer_frozen_encoder_is_left_out(cfg, tok):
    model = build_pair_classifier(cfg, tok)
    optimizer = build_optimizer(model, lr=1e-3, freeze_encoder=True)
    in_optimizer = {id(p) for g in optimizer.param_groups for p in g["params"]}
    assert in_optimizer == {id(p) for p in model.head.parameters()}
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert {g["name"] for g in optimizer.param_groups} == {"head"}


def test_schedule_warmup_then_cosine_to_zero():
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    scheduler = warmup_cosine_schedule(optimizer, total_steps=100, warmup_ratio=0.1)
    lrs = []
    for _ in range(100):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1)  # (0 + 1) / 10
    assert lrs[9] == pytest.approx(1.0)  # end of warmup
    assert all(a < b for a, b in zip(lrs[:9], lrs[1:10], strict=True))  # increasing
    assert all(a >= b for a, b in zip(lrs[10:], lrs[11:], strict=False))  # decreasing
    assert lrs[-1] < 0.01


# --- utils ----------------------------------------------------------------------


def test_resolve_device():
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("auto").type in {"cuda", "mps", "cpu"}


def test_run_dirs_are_never_reused(tmp_path):
    first, second = make_run_dir(tmp_path, "run"), make_run_dir(tmp_path, "run")
    assert first != second and first.is_dir() and second.is_dir()


def test_memory_stats_on_cpu():
    stats = memory_stats(torch.device("cpu"))
    assert stats["process_rss_gb"] > 0
    assert 0 < stats["system_memory_used_percent"] <= 100


class FakePopen:
    """Records the commands instead of starting processes."""

    calls: list = []

    def __init__(self, command, **kwargs):
        FakePopen.calls.append((command, kwargs))
        self.pid, self.terminated = 12345, False

    def terminate(self):
        self.terminated = True

    def wait(self):
        return 0


@pytest.fixture
def fake_popen(monkeypatch):
    FakePopen.calls = []
    monkeypatch.setattr(utils.subprocess, "Popen", FakePopen)
    return FakePopen


def test_keep_awake_disabled_starts_nothing(fake_popen):
    with keep_awake(False):
        pass
    assert fake_popen.calls == []


def test_keep_awake_runs_caffeinate_bound_to_this_process(fake_popen, monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/usr/bin/caffeinate")
    with keep_awake(True):
        assert len(fake_popen.calls) == 1
    command, _ = fake_popen.calls[0]
    assert command == ["caffeinate", "-ims", "-w", str(os.getpid())]


def test_keep_awake_stops_caffeinate_even_after_an_error(monkeypatch):
    started = []

    class Recording(FakePopen):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            started.append(self)

    monkeypatch.setattr(utils.subprocess, "Popen", Recording)
    monkeypatch.setattr(utils.shutil, "which", lambda name: "/usr/bin/caffeinate")
    with pytest.raises(RuntimeError), keep_awake(True):
        raise RuntimeError("training crashed")
    assert started[0].terminated


def test_keep_awake_without_caffeinate_warns(fake_popen, monkeypatch):
    monkeypatch.setattr(utils.shutil, "which", lambda name: None)
    with pytest.warns(UserWarning, match="caffeinate"), keep_awake(True):
        pass
    assert fake_popen.calls == []


def test_port_in_use():
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        assert port_in_use(port)
    assert not port_in_use(port)


def test_launch_tensorboard_reuses_one_watching_the_same_directory(
    fake_popen, monkeypatch, tmp_path
):
    logdir = tmp_path / "runs"
    monkeypatch.setattr(utils, "port_in_use", lambda port: True)
    monkeypatch.setattr(utils, "_tensorboard_logdir", lambda port: str(logdir.resolve()))
    assert launch_tensorboard(logdir, port=6123) == "http://localhost:6123/"
    assert fake_popen.calls == []


def test_launch_tensorboard_skips_a_port_used_by_another_project(fake_popen, monkeypatch, tmp_path):
    """e.g. a TensorBoard of another project already on 6006: ours goes on the next port."""
    busy = {6123}
    monkeypatch.setattr(utils, "port_in_use", lambda port: port in busy)
    monkeypatch.setattr(utils, "_tensorboard_logdir", lambda port: "/elsewhere/runs")

    def fake_start(command, **kwargs):
        FakePopen(command, **kwargs)
        busy.add(int(command[-1]))  # our TensorBoard now listens

    monkeypatch.setattr(utils.subprocess, "Popen", fake_start)
    monkeypatch.setattr(utils.webbrowser, "open", lambda url: None)

    assert launch_tensorboard(tmp_path / "runs", port=6123) == "http://localhost:6124/"
    assert fake_popen.calls[0][0][-2:] == ["--port", "6124"]


def test_launch_tensorboard_starts_it_and_opens_the_browser(fake_popen, monkeypatch, tmp_path):
    answers = iter([False, True])  # free before the launch, listening right after
    monkeypatch.setattr(utils, "port_in_use", lambda port: next(answers))
    opened = []
    monkeypatch.setattr(utils.webbrowser, "open", opened.append)

    url = launch_tensorboard(tmp_path / "runs", port=6123)
    command, kwargs = fake_popen.calls[0]
    assert command[-4:] == ["--logdir", str((tmp_path / "runs").resolve()), "--port", "6123"]
    assert kwargs["start_new_session"]  # survives the end of the run
    assert url == "http://localhost:6123/" and opened == [url]


# --- logger ---------------------------------------------------------------------


def test_flatten():
    flat = flatten({"a": {"b": 1, "c": [2, 3]}, "d": None, "e": "x", "f": True})
    assert flat == {"a.b": 1, "a.c": "[2, 3]", "d": "None", "e": "x", "f": True}


def test_logger_writes_one_json_line_per_event(tmp_path):
    with RunLogger(tmp_path, tensorboard=False) as logger:
        logger.log("train", 1, loss=0.7)
        logger.log("val", 1, auroc=0.6)
    records = read_metrics(tmp_path / "metrics.jsonl")
    assert [(r["event"], r["step"]) for r in records] == [("train", 1), ("val", 1)]
    assert records[0]["loss"] == 0.7


def test_logger_tensorboard(tmp_path):
    with RunLogger(tmp_path, tensorboard=True) as logger:
        logger.log("train", 1, loss=0.7, device="cpu")  # strings are not sent to TensorBoard
    assert any((tmp_path / "tensorboard").iterdir())


# --- Trainer --------------------------------------------------------------------


def test_trainer_unknown_precision_fails(tmp_path):
    with pytest.raises(ValueError, match="precision"):
        Trainer(torch.device("cpu"), RunLogger(tmp_path, tensorboard=False), tmp_path, "fp16")


def test_frozen_encoder_is_not_modified(cfg, tok, tmp_path, data_dir):
    from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset

    cfg["training"]["classification"].update(freeze_encoder=True, lr=1e-2)
    data = GoldStandardDataset(tokenizer=tok, **cfg["data"])
    model = build_pair_classifier(cfg, tok)
    encoder_before = copy.deepcopy(model.encoder.state_dict())
    head_before = copy.deepcopy(model.head.state_dict())

    with RunLogger(tmp_path, tensorboard=False) as logger:
        trainer = Trainer(torch.device("cpu"), logger, tmp_path)
        trainer.fit_classifier(
            model,
            data.classif_loader("train"),
            data.classif_loader("val"),
            cfg["training"]["classification"],
        )
    for key, value in model.encoder.state_dict().items():
        assert torch.equal(value, encoder_before[key]), key
    assert any(not torch.equal(v, head_before[k]) for k, v in model.head.state_dict().items())


# --- run_experiment -------------------------------------------------------------


def test_run_creates_every_output(cfg):
    run_dir = run_experiment(cfg)
    for name in ("config.yaml", "metrics.jsonl", "best.pt", "results.json"):
        assert (run_dir / name).is_file(), name

    results = json.loads((run_dir / "results.json").read_text())
    assert 0.0 <= results["test"]["auroc"] <= 1.0
    assert results["test"]["n_pairs"] == 8
    assert {"f1", "mcc", "brier", "tp"} <= set(results["test"])
    assert results["best_epoch"] in (1, 2)
    assert results["n_parameters"] > 0 and results["train_time_s"] > 0


def test_run_saves_the_exact_config(cfg):
    run_dir = run_experiment(cfg)
    with open(run_dir / "config.yaml") as f:
        assert yaml.safe_load(f) == cfg


def test_run_logs_every_event(cfg):
    records = read_metrics(run_experiment(cfg) / "metrics.jsonl")
    events = [r["event"] for r in records]
    assert events[0] == "setup" and events[-1] == "test"
    assert events.count("train_epoch") == events.count("val") == events.count("val_best") == 2
    # 16 training pairs / batch 4 = 4 steps per epoch, one log per step
    assert [r["step"] for r in records if r["event"] == "train"] == list(range(1, 9))
    assert [r["step"] for r in records if r["event"] == "system"] == list(range(1, 9))
    # validation is logged at the optimizer step of the end of each epoch
    assert [(r["step"], r["epoch"]) for r in records if r["event"] == "val"] == [(4, 1), (8, 2)]

    train = next(r for r in records if r["event"] == "train")
    assert {"loss", "grad_norm", "lr_encoder", "lr_head"} <= set(train)
    system = next(r for r in records if r["event"] == "system")
    assert {"pairs_per_s", "tokens_per_s", "step_time_s", "process_rss_gb"} <= set(system)
    val = next(r for r in records if r["event"] == "val")
    assert {"loss", "auroc", "auprc", "accuracy", "f1", "mcc", "brier", "recall"} <= set(val)


def test_run_fills_every_tensorboard_tab(cfg):
    from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer

    cfg["logging"]["tensorboard"] = True
    run_dir = run_experiment(cfg)
    multiplexer = EventMultiplexer().AddRunsFromDirectory(str(run_dir / "tensorboard"))
    multiplexer.Reload()
    tags = {kind: set() for kind in ("scalars", "histograms", "images", "tensors")}
    for run_tags in multiplexer.Runs().values():
        for kind in tags:
            tags[kind] |= set(run_tags.get(kind, []))

    expected_scalars = {
        "train/loss", "train/grad_norm", "train/lr_encoder", "train/lr_head",
        "system/pairs_per_s", "system/tokens_per_s", "system/process_rss_gb",
        "train_epoch/loss", "train_epoch/auroc",
        "val/loss", "val/auroc", "val/auprc", "val/f1", "val/mcc", "val/brier",
        "val_best/auroc", "test/auroc", "test/accuracy",
    }  # fmt: skip
    assert expected_scalars <= tags["scalars"]
    assert {"val/prob_positives", "val/prob_negatives", "test/prob_positives"} <= tags["histograms"]
    assert "weights/encoder.token_embedding.weight" in tags["histograms"]
    assert {"val/roc_curve", "test/roc_curve"} <= tags["images"]
    assert {"val/pr_curve", "test/pr_curve"} <= tags["tensors"]
    assert any(t.startswith("config") for t in tags["tensors"])
    hparams = multiplexer.PluginRunToTagToContent("hparams")
    assert any("_hparams_/experiment" in run_tags for run_tags in hparams.values())
    assert {"hparam/test_auroc", "hparam/best_val_auroc", "hparam/train_time_min"} <= tags[
        "scalars"
    ]


def test_run_launches_tensorboard_when_configured(cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(run_module, "launch_tensorboard", lambda *a, **k: calls.append((a, k)))
    cfg["logging"].update(tensorboard=True, launch_tensorboard=True, tensorboard_port=6123)
    run_experiment(cfg)
    assert calls == [((cfg["experiment"]["output_dir"],), {"port": 6123})]


def test_run_does_not_launch_tensorboard_when_disabled(cfg, monkeypatch):
    monkeypatch.setattr(run_module, "launch_tensorboard", lambda *a, **k: pytest.fail("launched"))
    run_experiment(cfg)


def test_early_stopping(cfg):
    """With lr = 0 the validation AUROC never improves: stop after `patience` epochs."""
    cfg["training"]["classification"].update(lr=0.0, epochs=10, early_stopping_patience=2)
    results = json.loads((run_experiment(cfg) / "results.json").read_text())
    assert results["best_epoch"] == 1 and results["epochs_run"] == 3


def test_grad_accumulation_reduces_optimizer_steps(cfg):
    cfg["training"]["classification"]["grad_accumulation"] = 2
    records = read_metrics(run_experiment(cfg) / "metrics.jsonl")
    assert sum(r["event"] == "train" for r in records) == 4  # 2 steps per epoch, 2 epochs


def test_limit_batches(cfg):
    run_dir = run_experiment(cfg, limit_train_batches=1, limit_eval_batches=1)
    records = read_metrics(run_dir / "metrics.jsonl")
    assert sum(r["event"] == "train" for r in records) == 2  # 1 step per epoch
    assert json.loads((run_dir / "results.json").read_text())["test"]["n_pairs"] == 4


def test_limited_evaluation_still_sees_both_classes(cfg):
    """Regression: evaluation loaders list positives first, so the first batches alone
    held one class and the AUROC of a --limit-eval-batches run was NaN."""
    run_dir = run_experiment(cfg, limit_eval_batches=1)
    for record in read_metrics(run_dir / "metrics.jsonl"):
        if record["event"] in ("val", "test"):
            assert record["tp"] + record["fn"] > 0 and record["tn"] + record["fp"] > 0
            assert not math.isnan(record["auroc"])


def test_run_is_reproducible(cfg):
    first = json.loads((run_experiment(cfg) / "results.json").read_text())
    second = json.loads((run_experiment(cfg) / "results.json").read_text())
    assert first["test"] == second["test"] and first["best_val"] == second["best_val"]


def test_bf16_run(cfg):
    cfg["training"]["precision"] = "bf16"
    results = json.loads((run_experiment(cfg) / "results.json").read_text())
    assert math.isfinite(results["test"]["loss"])


def test_strategy_b_is_not_implemented_yet(cfg):
    cfg["experiment"]["strategy"] = "B"
    with pytest.raises(NotImplementedError, match="only A"):
        run_experiment(cfg)


def test_cli(cfg, tmp_path, capsys):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main(["--config", str(path), "--name", "cli", "--device", "cpu"])
    assert "test AUROC" in capsys.readouterr().out
    assert any(p.name.startswith("cli_") for p in (tmp_path / "runs").iterdir())


@pytest.mark.parametrize("flag, expected", [([], False), (["--caffeinate"], True)])
def test_cli_caffeinate_flag(cfg, tmp_path, monkeypatch, flag, expected):
    seen = []

    @contextmanager
    def fake_keep_awake(enabled):
        seen.append(enabled)
        yield

    monkeypatch.setattr(run_module, "keep_awake", fake_keep_awake)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    main(["--config", str(path), *flag])
    assert seen == [expected]
