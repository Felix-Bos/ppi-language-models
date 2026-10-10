"""Post-training analyses on a tiny run trained on the synthetic mini-dataset (CPU)."""

import importlib.util
import os

import plotly.graph_objects as go
import pytest
import torch
from conftest import make_tiny_cfg, make_tiny_data

from ppi_lm.analysis import plots
from ppi_lm.analysis.attention import (
    a_to_b_block,
    attention_focus,
    cross_protein_mass,
    pair_attention,
    sink_mass,
)
from ppi_lm.analysis.importance import integrated_gradients
from ppi_lm.analysis.inference import (
    find_latest_run,
    load_run,
    make_loader,
    predict_pairs,
    protein_embeddings,
    sample_pairs,
    swapped,
)
from ppi_lm.analysis.latent import amino_acid_embeddings, project_2d
from ppi_lm.analysis.network import interaction_graph, outcome, rank_partners
from ppi_lm.training.run import run_experiment


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis")
    data_dir = make_tiny_data(root / "data")
    run_dir = run_experiment(make_tiny_cfg(root / "runs", data_dir))
    return load_run(run_dir, data_dir=data_dir)


@pytest.fixture(scope="module")
def test_pairs(run):
    return sample_pairs(run.data, "test")


@pytest.fixture(scope="module")
def preds(run, test_pairs):
    return predict_pairs(run.model, run.data, test_pairs, batch_size=3)


@pytest.fixture(scope="module")
def positive_pair(test_pairs):
    return next(p for p in test_pairs if p[2] == 1)


# --- loading and inference ------------------------------------------------------


def test_load_run_restores_the_best_weights(run):
    state = torch.load(run.run_dir / "best.pt", weights_only=True)["model"]
    for key, value in run.model.state_dict().items():
        assert torch.equal(value, state[key]), key
    assert not run.model.training
    assert run.results["best_epoch"] >= 1 and run.metrics[0]["event"] == "setup"


def test_load_run_knows_the_best_epoch_and_history(run):
    assert run.finished and run.best_epoch == run.results["best_epoch"]
    history = run.validation_history()
    assert [r["epoch"] for r in history] == list(range(1, len(history) + 1))


def test_an_interrupted_run_can_be_loaded(run, tmp_path):
    """Ctrl+C after an epoch: best.pt and metrics.jsonl exist, results.json does not."""
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    for name in ("config.yaml", "best.pt", "metrics.jsonl"):
        (interrupted / name).write_bytes((run.run_dir / name).read_bytes())
    loaded = load_run(interrupted, data_dir=run.data.data_dir)
    assert not loaded.finished and loaded.results == {}
    assert loaded.best_epoch == run.best_epoch and loaded.validation_history()


def test_find_latest_run(tmp_path):
    for name in ("smoke_1", "transformer_A_direct_1", "transformer_A_direct_2", "no_checkpoint"):
        (tmp_path / name).mkdir()
    for name in ("smoke_1", "transformer_A_direct_1", "transformer_A_direct_2"):
        (tmp_path / name / "best.pt").write_bytes(b"")
    os.utime(tmp_path / "transformer_A_direct_1" / "best.pt", (1e9, 1e9))  # older
    os.utime(tmp_path / "transformer_A_direct_2" / "best.pt", (2e9, 2e9))  # newest
    os.utime(tmp_path / "smoke_1" / "best.pt", (1.5e9, 1.5e9))
    assert find_latest_run(tmp_path).name == "transformer_A_direct_2"
    assert find_latest_run(tmp_path, "smoke_*").name == "smoke_1"


def test_find_latest_run_without_any_run_explains_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="ppi_lm.training.run"):
        find_latest_run(tmp_path / "runs")


def test_sample_pairs(run):
    assert len(sample_pairs(run.data, "test", n=3, seed=1)) == 3
    assert sample_pairs(run.data, "test", 3, seed=1) == sample_pairs(run.data, "test", 3, seed=1)
    assert sample_pairs(run.data, "test") == run.data.pairs["test"]


def test_predictions_are_aligned_with_pairs(run, preds, test_pairs):
    n, d = len(test_pairs), run.model.encoder.d_model
    assert len(preds) == n and preds.labels.tolist() == [p[2] for p in test_pairs]
    assert preds.pair_vectors.shape == (n, run.model.head.input_dim)
    assert preds.protein_a.shape == preds.protein_b.shape == (n, d)
    assert preds.length_a.tolist() == [len(run.data.seqs[a]) for a, _, _ in test_pairs]
    torch.testing.assert_close(preds.probs, torch.sigmoid(preds.logits))


def test_predictions_match_the_model(run, preds, test_pairs):
    batch = next(iter(make_loader(run.model, run.data, test_pairs, batch_size=len(test_pairs))))
    with torch.no_grad():
        logits = run.model(batch["input_ids"], batch["segment_ids"], batch["attention_mask"])
    torch.testing.assert_close(preds.logits, logits)


def test_predictions_do_not_depend_on_batch_size(run, preds, test_pairs):
    other = predict_pairs(run.model, run.data, test_pairs, batch_size=1)
    torch.testing.assert_close(other.logits, preds.logits, atol=1e-5, rtol=1e-5)


def test_swapped():
    assert swapped([("A", "B", 1)]) == [("B", "A", 1)]


def test_protein_embeddings_average_over_pairs(preds):
    ids, vectors, counts = protein_embeddings(preds)
    assert vectors.shape == (len(ids), preds.protein_a.shape[1])
    assert counts.sum() == 2 * len(preds)
    pid = ids[0]
    own = [preds.protein_a[i] for i, p in enumerate(preds.pairs) if p[0] == pid] + [
        preds.protein_b[i] for i, p in enumerate(preds.pairs) if p[1] == pid
    ]
    torch.testing.assert_close(vectors[0], torch.stack(own).mean(0))


# --- latent space ---------------------------------------------------------------


@pytest.mark.parametrize("method", ["pca", "tsne"])
def test_project_2d(method):
    x = torch.randn(30, 8)
    assert project_2d(x, method).shape == (30, 2)


def test_project_2d_unknown_method_fails():
    with pytest.raises(ValueError, match="Unknown method"):
        project_2d(torch.randn(5, 3), "lda")


@pytest.mark.skipif(importlib.util.find_spec("umap") is not None, reason="umap is installed")
def test_project_2d_umap_without_package_explains_how_to_install():
    with pytest.raises(ImportError, match="umap-learn"):
        project_2d(torch.randn(5, 3), "umap")


def test_amino_acid_embeddings(run):
    letters, vectors = amino_acid_embeddings(run.model, run.tokenizer)
    assert len(letters) == 20 and "X" not in letters
    row = run.model.encoder.token_embedding.weight[run.tokenizer.token2id[letters[0]]]
    torch.testing.assert_close(vectors[0], row.detach())


# --- attention ------------------------------------------------------------------


@pytest.fixture(scope="module")
def att(run, positive_pair):
    return pair_attention(run.model, run.data, positive_pair)


def test_attention_shapes_and_rows(run, att):
    cfg = run.model.encoder.config
    L = len(att.tokens)
    assert len(att.weights) == cfg.n_layers
    assert all(w.shape == (cfg.n_heads, L, L) for w in att.weights)
    torch.testing.assert_close(att.stacked().sum(-1), torch.ones(cfg.n_layers, cfg.n_heads, L))
    assert att.tokens[0] == "<CLS>" and att.tokens[-1] == "<EOS>"


def test_attention_logit_matches_predictions(run, att, positive_pair):
    expected = predict_pairs(run.model, run.data, [positive_pair]).logits[0].item()
    assert att.logit == pytest.approx(expected, abs=1e-5)


def test_residue_positions_exclude_special_tokens(run, att, positive_pair):
    a, b, _ = positive_pair
    n_a = min(len(run.data.seqs[a]), run.data.max_protein_length)
    n_b = min(len(run.data.seqs[b]), run.data.max_protein_length)
    assert len(att.residues_a) == n_a and len(att.residues_b) == n_b


def test_cross_protein_mass(att):
    mass = cross_protein_mass(att)
    assert mass.shape == (len(att.weights), att.weights[0].shape[0], 2)
    assert ((mass >= 0) & (mass <= 1)).all()
    expected = att.weights[0][0][att.residues_a][:, att.residues_b].sum(-1).mean()
    torch.testing.assert_close(mass[0, 0, 0], expected)


def test_sink_mass_and_focus_are_shares(att):
    for values in (sink_mass(att), attention_focus(att)):
        assert values.shape == (len(att.weights), att.weights[0].shape[0])
        assert ((values >= -1e-6) & (values <= 1 + 1e-6)).all()


def test_a_to_b_block(att):
    assert a_to_b_block(att).shape == (len(att.residues_a), len(att.residues_b))
    assert a_to_b_block(att, layer=0).shape == (len(att.residues_a), len(att.residues_b))


# --- integrated gradients -------------------------------------------------------


@pytest.fixture(scope="module")
def imp(run, positive_pair):
    return integrated_gradients(run.model, run.data, positive_pair, steps=64)


def test_integrated_gradients_completeness(imp):
    """The scores add up to logit(input) - logit(baseline)."""
    expected = imp.logit - imp.baseline_logit
    assert imp.scores.sum().item() == pytest.approx(expected, rel=0.05, abs=1e-3)


def test_integrated_gradients_logit_is_the_model_logit(run, imp, positive_pair):
    expected = predict_pairs(run.model, run.data, [positive_pair]).logits[0].item()
    assert imp.logit == pytest.approx(expected, abs=1e-5)


def test_integrated_gradients_per_residue(run, imp, positive_pair):
    letters, scores = imp.residue_scores(0)
    a = positive_pair[0]
    assert "".join(letters) == run.data.seqs[a][: run.data.max_protein_length]
    assert len(scores) == len(letters)


def test_integrated_gradients_leave_the_model_clean(run, positive_pair):
    integrated_gradients(run.model, run.data, positive_pair, steps=4)
    assert all(p.grad is None for p in run.model.parameters())
    assert not run.model.encoder.token_embedding._forward_hooks


def test_integrated_gradients_with_a_frozen_encoder(run, positive_pair):
    run.model.encoder.requires_grad_(False)
    try:
        imp = integrated_gradients(run.model, run.data, positive_pair, steps=8)
        assert torch.isfinite(imp.scores).all()
    finally:
        run.model.encoder.requires_grad_(True)


# --- interaction network --------------------------------------------------------


@pytest.mark.parametrize(
    "label, prob, expected", [(1, 0.9, "TP"), (1, 0.1, "FN"), (0, 0.9, "FP"), (0, 0.1, "TN")]
)
def test_outcome(label, prob, expected):
    assert outcome(label, prob) == expected


def test_interaction_graph(preds):
    graph = interaction_graph(preds)
    assert graph.number_of_edges() <= len(preds)
    for _, _, data in graph.edges(data=True):
        assert data["outcome"] == outcome(data["label"], data["prob"])


def test_rank_partners(run, test_pairs):
    protein = test_pairs[0][0]
    candidates = sorted({p for a, b, _ in test_pairs for p in (a, b)})
    ranking = rank_partners(run.model, run.data, protein, candidates)
    assert protein not in [c for c, _ in ranking]
    assert len(ranking) == len(candidates) - 1
    probs = [p for _, p in ranking]
    assert probs == sorted(probs, reverse=True)


# --- figures -------------------------------------------------------------------


@pytest.fixture(scope="module")
def figures(run, preds, att, imp, test_pairs):
    other = predict_pairs(run.model, run.data, swapped(test_pairs))
    ids, vectors, counts = protein_embeddings(preds)
    letters, aa = amino_acid_embeddings(run.model, run.tokenizer)
    negative = next(p for p in test_pairs if p[2] == 0)
    att_neg = pair_attention(run.model, run.data, negative)
    graph = interaction_graph(preds)
    protein = test_pairs[0][0]
    candidates = sorted({p for a, b, _ in test_pairs for p in (a, b)})
    ranking = rank_partners(run.model, run.data, protein, candidates)
    coords = project_2d(vectors, "pca")
    return {
        "training": plots.plot_training_curves(run.metrics),
        "system": plots.plot_system(run.metrics),
        "roc_pr": plots.plot_roc_pr(preds, "test"),
        "calibration": plots.plot_calibration(preds, n_bins=3),
        "confusion": plots.plot_confusion(preds),
        "probabilities": plots.plot_probability_distributions(preds),
        "length": plots.plot_metric_by_length(preds, run.data.max_protein_length, n_bins=2),
        "symmetry": plots.plot_order_symmetry(preds, other),
        "proteins": plots.plot_latent(
            coords, counts.numpy(), "proteins", hover={"protein": ids}, color_title="paires"
        ),
        "proteins_categorical": plots.plot_latent(
            coords, [f"c{i % 5}" for i in range(len(ids))], "proteins", categorical=True
        ),
        "pairs": plots.plot_latent_pairs(project_2d(preds.pair_vectors), preds, "pairs"),
        "amino_acids": plots.plot_amino_acids(letters, project_2d(aa), "amino acids"),
        "attention": plots.plot_attention_map(att),
        "attention_layer": plots.plot_attention_map(att, layer=0),
        "attention_stats": plots.plot_attention_statistics([att], [att_neg]),
        "importance": plots.plot_residue_importance(imp),
        "graph": plots.plot_interaction_graph(graph, outcomes=("TP", "FP", "FN", "TN")),
        "ranking": plots.plot_partner_ranking(ranking, {candidates[1]}, protein),
    }


def test_every_figure_is_plotly_with_a_title(figures):
    for name, fig in figures.items():
        assert isinstance(fig, go.Figure), name
        assert fig.layout.title.text, name
        assert len(fig.data) > 0, name
        fig.to_json()  # serializable: renders in Jupyter / VS Code


def test_every_figure_uses_the_theme_surface(figures):
    surface = plots.theme()["surface"]
    for name, fig in figures.items():
        assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == surface, name


def test_every_trace_has_a_hover(figures):
    for name, fig in figures.items():
        for trace in fig.data:
            has_hover = trace.hovertemplate or trace.hoverinfo == "skip" or trace.x == (None,)
            assert has_hover, f"{name}: {trace.name}"


def test_interacting_pairs_are_always_slot_1(figures):
    blue, orange = plots.theme()["categorical"][:2]
    for name in ("probabilities", "symmetry"):
        colors = {
            t.name: t.marker.color
            for t in figures[name].data
            if t.name in plots.LABEL_NAMES.values()
        }
        assert colors == {"interaction": blue, "pas d'interaction": orange}, name


def test_scatter_uses_at_most_three_hues(figures):
    """Beyond 3 categories, the extra ones fold into the muted gray."""
    colors = {t.marker.color for t in figures["proteins_categorical"].data}
    assert colors <= set(plots.theme()["categorical"][:3]) | {plots.theme()["muted"]}


def test_errors_are_marked_by_shape(figures):
    symbols = {t.name.split(" (")[0]: t.marker.symbol for t in figures["pairs"].data}
    assert symbols["interaction · correct"] == "circle"
    assert symbols["interaction · FN, ratée"] == "x"


def test_dark_theme(preds):
    plots.set_theme("dark")
    try:
        fig = plots.plot_probability_distributions(preds)
        assert fig.layout.paper_bgcolor == plots.THEMES["dark"]["surface"]
        assert fig.data[0].marker.color == plots.THEMES["dark"]["categorical"][0]
    finally:
        plots.set_theme("light")


def test_unknown_theme_fails():
    with pytest.raises(ValueError, match="theme"):
        plots.set_theme("sepia")


def test_calibration_bins_cover_every_pair(preds):
    bins = plots.calibration_bins(preds, n_bins=3)
    assert bins["count"].sum() == len(preds)
    assert ((bins["positive_rate"] >= 0) & (bins["positive_rate"] <= 1)).all()


def test_text_inside_fills_is_readable():
    assert plots._ink_on("#0d366b") == "#ffffff"  # dark blue -> white text
    assert plots._ink_on("#cde2fb") == "#0b0b0b"  # light blue -> ink
