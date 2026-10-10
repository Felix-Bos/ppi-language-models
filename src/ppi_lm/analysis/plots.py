"""Interactive Plotly figures for the analysis notebook, in one consistent theme.

Colors come from the reference data-viz palette (checked for color-vision deficiency):
- identity: categorical slots in a fixed order; a scatter uses at most 3 hues. In every
  figure, interacting pairs are slot 1 (blue) and non-interacting pairs slot 2 (orange);
  prediction errors are shown by the marker shape (x), never by an extra color;
- magnitude (attention, length, rates): one blue ramp, light -> dark;
- sign (residue importance): blue <-> red poles;
- grid and axes: solid hairlines one step off the surface; text never wears a data color.

set_theme("dark") switches every figure built afterwards to the dark steps.
"""

import numpy as np
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.metrics import precision_recall_curve, roc_curve

from ppi_lm.analysis.attention import (
    PairAttention,
    a_to_b_block,
    attention_focus,
    cross_protein_mass,
    sink_mass,
)
from ppi_lm.analysis.importance import ResidueImportance
from ppi_lm.analysis.inference import PairPredictions
from ppi_lm.analysis.latent import AA_CLASS, AA_GROUP, AA_GROUPS, AA_NAMES
from ppi_lm.training.metrics import binary_metrics

# --- theme ----------------------------------------------------------------------

_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]  # fmt: skip

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "ink_secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "categorical": [
            "#2a78d6",
            "#eb6834",
            "#1baf7a",
            "#eda100",
            "#e87ba4",
            "#008300",
            "#6250d6",
            "#e34948",
        ],  # fmt: skip
        "ramp": _BLUE_RAMP,  # near zero = light, recedes into the light surface
        "positive": "#e34948",
        "negative": "#2a78d6",
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#f0efec",
        "ink_secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "categorical": [
            "#3987e5",
            "#d95926",
            "#199e70",
            "#c98500",
            "#d55181",
            "#008300",
            "#9085e9",
            "#e66767",
        ],  # fmt: skip
        "ramp": _BLUE_RAMP[::-1],  # anchor flipped: near zero = dark, recedes into the surface
        "positive": "#e66767",
        "negative": "#3987e5",
    },
}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
LABEL_NAMES = {1: "interaction", 0: "pas d'interaction"}
_current = {"name": "light"}


def set_theme(name: str) -> None:
    """light | dark, for every figure built afterwards."""
    if name not in THEMES:
        raise ValueError(f"Unknown theme '{name}', expected one of {list(THEMES)}")
    _current["name"] = name


def theme() -> dict:
    return THEMES[_current["name"]]


def _colorscale(start: int = 0) -> list:
    """The blue ramp as a Plotly colorscale; start > 0 skips the lightest steps (for marks)."""
    ramp = theme()["ramp"]
    ramp = ramp[start:] if _current["name"] == "light" else ramp[: len(ramp) - start]
    return [[i / (len(ramp) - 1), color] for i, color in enumerate(ramp)]


def label_color(label: int) -> str:
    return theme()["categorical"][0 if label == 1 else 1]


def _ink_on(fill_hex: str) -> str:
    """White or ink text inside a colored fill, by the fill's luminance."""
    r, g, b = (int(fill_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else "#ffffff"


def _subplots(**kwargs) -> go.Figure:
    """make_subplots with the subplot titles in secondary ink (styled before any other
    annotation exists, so cell labels and direct labels keep their own colors)."""
    fig = make_subplots(**kwargs)
    fig.update_annotations(font={"color": theme()["ink_secondary"], "size": 13})
    fig.layout.meta = {"subplot_titles": bool(kwargs.get("subplot_titles"))}
    return fig


def _layout(
    fig: go.Figure,
    title: str,
    subtitle: str = "",
    height: int = 450,
    axes: bool = True,
    legend: bool = True,
) -> go.Figure:
    """Theme + a top block in fixed rows, so nothing overlaps:
    title, subtitle, legend, subplot titles, then the plot area."""
    t = theme()
    title_block = 62 if subtitle else 44  # px from the top to the end of the title block
    legend_block = 28 if legend else 0
    has_subplot_titles = bool((fig.layout.meta or {}).get("subplot_titles"))
    top = title_block + legend_block + (30 if has_subplot_titles else 14)
    fig.update_layout(
        template="none",
        title={
            "text": f"<b>{title}</b>",
            "subtitle": {"text": subtitle, "font": {"size": 12, "color": t["ink_secondary"]}},
            "x": 0.01,
            "xanchor": "left",
            "yref": "container",
            "y": 1 - 16 / height,
            "yanchor": "top",
            "font": {"size": 16, "color": t["ink"]},
        },
        font={"family": FONT, "size": 12, "color": t["ink_secondary"]},
        paper_bgcolor=t["surface"],
        plot_bgcolor=t["surface"],
        height=height,
        margin={"l": 70, "r": 30, "t": top + 8, "b": 60},
        showlegend=legend,
        legend={
            "orientation": "h",
            "x": 0.01,
            "xanchor": "left",
            "yref": "container",
            "y": 1 - (title_block + 2) / height,
            "yanchor": "top",
            "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": t["ink_secondary"]},
        },
        hoverlabel={
            "bgcolor": t["surface"],
            "bordercolor": t["border"],
            "font": {"family": FONT, "color": t["ink"]},
        },
    )
    axis = {
        "showgrid": axes,
        "gridcolor": t["grid"],
        "gridwidth": 1,
        "zeroline": False,
        "showline": axes,
        "linecolor": t["axis"],
        "ticks": "",
        "showticklabels": axes,
        "tickfont": {"color": t["muted"]},
        "title_font": {"color": t["ink_secondary"]},
        "automargin": True,
    }
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _scatter(n: int):
    """SVG scatter up to 5,000 points (renders everywhere, exports cleanly), WebGL beyond."""
    return go.Scatter if n <= 5000 else go.Scattergl


def _marker(color, size: int = 8, symbol="circle") -> dict:
    """Filled dot with a thin surface-colored ring, legible where dots overlap."""
    return {
        "color": color,
        "size": size,
        "symbol": symbol,
        "line": {"color": theme()["surface"], "width": 1},
    }


def _reference_line(fig, x, y, text: str = "", row=None, col=None) -> None:
    """Muted solid hairline (chance, diagonal, threshold), with an optional direct label."""
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line={"color": theme()["muted"], "width": 1},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )
    if text:
        fig.add_annotation(
            x=x[-1],
            y=y[-1],
            text=text,
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font={"color": theme()["muted"], "size": 11},
            row=row,
            col=col,
        )


def _series(metrics: list[dict], event: str, key: str) -> tuple[list, list]:
    rows = [r for r in metrics if r["event"] == event and key in r]
    return [r["step"] for r in rows], [r[key] for r in rows]


# --- training -------------------------------------------------------------------


def plot_training_curves(metrics: list[dict]) -> go.Figure:
    """Loss and AUROC against the optimizer step: train (blue) and validation (orange)."""
    c = theme()["categorical"]
    fig = _subplots(rows=1, cols=2, subplot_titles=("Loss (BCE)", "AUROC"))
    lines = [
        ("train", "loss", "train · par step", c[0], 1, 0.35, "lines", True),
        ("train_epoch", "loss", "train", c[0], 1, 1, "lines+markers", True),
        ("val", "loss", "validation", c[1], 1, 1, "lines+markers", True),
        ("train_epoch", "auroc", "train", c[0], 2, 1, "lines+markers", False),
        ("val", "auroc", "validation", c[1], 2, 1, "lines+markers", False),
    ]
    for event, key, name, color, col, opacity, mode, legend in lines:
        x, y = _series(metrics, event, key)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name=name,
                mode=mode,
                opacity=opacity,
                legendgroup=name,
                showlegend=legend,
                line={"color": color, "width": 1 if opacity < 1 else 2},
                marker=_marker(color),
                hovertemplate=f"{name} : %{{y:.4f}}<extra></extra>",
            ),
            row=1,
            col=col,
        )

    x, y = _series(metrics, "val", "auroc")
    if x:
        _reference_line(fig, [min(x), max(x)], [0.5, 0.5], "hasard", row=1, col=2)
        valid = [(xi, yi) for xi, yi in zip(x, y, strict=True) if yi == yi]  # drop NaN
        if valid:
            bx, by = max(valid, key=lambda p: p[1])
            fig.add_annotation(
                x=bx,
                y=by,
                text=f"meilleure : {by:.3f}",
                showarrow=True,
                arrowcolor=theme()["muted"],
                ax=0,
                ay=-30,
                font={"color": theme()["ink"]},
                row=1,
                col=2,
            )
    fig.update_xaxes(title="step d'optimisation")
    fig.update_layout(hovermode="x unified")
    return _layout(fig, "Courbes d'apprentissage", "train : bleu · validation : orange")


def plot_system(metrics: list[dict]) -> go.Figure:
    """Learning rates, speed and memory during training (one legend per panel)."""
    c = theme()["categorical"]
    fig = _subplots(
        rows=1, cols=3, subplot_titles=("Learning rate", "Vitesse (paires / s)", "Mémoire (Go)")
    )
    memory_keys = [
        k for k in ("process_rss_gb", "mps_driver_gb", "cuda_allocated_gb")
        if any(k in r for r in metrics)
    ][:2]  # fmt: skip
    panels = [
        (1, "legend", [("train", "lr_encoder", "encodeur"), ("train", "lr_head", "head")]),
        (2, None, [("system", "pairs_per_s", "paires / s")]),
        (3, "legend2", [("system", k, k.replace("_gb", "")) for k in memory_keys]),
    ]
    for col, legend, series in panels:
        for i, (event, key, name) in enumerate(series):
            x, y = _series(metrics, event, key)
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=name,
                    mode="lines",
                    line={"color": c[i], "width": 2},
                    legend=legend or "legend",
                    showlegend=legend is not None,
                    hovertemplate=f"{name} : %{{y:.4g}}<extra></extra>",
                ),
                row=1,
                col=col,
            )
    fig.update_xaxes(title="step d'optimisation")
    fig = _layout(fig, "Optimisation et ressources", height=400, legend=False)
    # one legend under each panel that has several series
    below = {"yref": "paper", "y": -0.3, "yanchor": "top", "orientation": "h"}
    fig.update_layout(
        hovermode="x unified",
        showlegend=True,
        legend={**below, "x": fig.layout.xaxis.domain[0], "xanchor": "left"},
        legend2={**below, "x": fig.layout.xaxis3.domain[0], "xanchor": "left"},
        margin={"b": 110},
    )
    return fig


# --- performance ----------------------------------------------------------------


def plot_roc_pr(preds: PairPredictions, title: str = "Performance") -> go.Figure:
    """ROC and precision-recall curves, with the threshold of every point on hover."""
    c, labels, probs = theme()["categorical"], preds.labels.numpy(), preds.probs.numpy()
    m = binary_metrics(preds.labels, preds.logits)
    fig = _subplots(rows=1, cols=2, subplot_titles=("ROC", "Précision-rappel"))
    fpr, tpr, thr = roc_curve(labels, probs)
    precision, recall, thr_pr = precision_recall_curve(labels, probs)
    curves = [
        (fpr, tpr, thr, "FPR", "TPR", 1, f"AUROC {m['auroc']:.3f}"),
        (recall[:-1], precision[:-1], thr_pr, "rappel", "précision", 2, f"AUPRC {m['auprc']:.3f}"),
    ]
    for x, y, t, xname, yname, col, label in curves:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line={"color": c[0], "width": 2},
                name=label,
                customdata=np.clip(t, 0, 1),
                showlegend=False,
                hovertemplate=(
                    f"{xname} %{{x:.3f}} · {yname} %{{y:.3f}}"
                    "<br>seuil %{customdata:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=col,
        )
        fig.add_annotation(
            x=0.97,
            y=0.05 if col == 1 else 0.95,
            xref=f"x{col if col > 1 else ''} domain",
            yref=f"y{col if col > 1 else ''} domain",
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="right",
            font={"color": theme()["ink"]},
        )
    _reference_line(fig, [0, 1], [0, 1], "hasard", row=1, col=1)
    _reference_line(fig, [0, 1], [labels.mean()] * 2, "hasard", row=1, col=2)
    fig.update_xaxes(title="taux de faux positifs", range=[0, 1], row=1, col=1)
    fig.update_yaxes(title="taux de vrais positifs", range=[0, 1.02], row=1, col=1)
    fig.update_xaxes(title="rappel", range=[0, 1], row=1, col=2)
    fig.update_yaxes(title="précision", range=[0, 1.02], row=1, col=2)
    return _layout(fig, title, f"{len(preds)} paires", height=460, legend=False)


def calibration_bins(preds: PairPredictions, n_bins: int = 10) -> dict[str, np.ndarray]:
    """Quantile bins of the predicted probability: mean prediction, positive rate, count."""
    probs, labels = preds.probs.numpy(), preds.labels.numpy()
    edges = np.unique(np.quantile(probs, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.digitize(probs, edges[1:-1]), 0, len(edges) - 2)
    used = [b for b in range(len(edges) - 1) if (which == b).any()]
    return {
        "mean_prob": np.array([probs[which == b].mean() for b in used]),
        "positive_rate": np.array([labels[which == b].mean() for b in used]),
        "count": np.array([(which == b).sum() for b in used]),
    }


def plot_calibration(preds: PairPredictions, n_bins: int = 10) -> go.Figure:
    """Reliability diagram: when the model says 0.8, do 80 % of those pairs interact?"""
    bins = calibration_bins(preds, n_bins)
    fig = go.Figure()
    _reference_line(fig, [0, 1], [0, 1], "parfaitement calibré")
    fig.add_trace(
        go.Scatter(
            x=bins["mean_prob"],
            y=bins["positive_rate"],
            mode="lines+markers",
            name="modèle",
            line={"color": theme()["categorical"][0], "width": 2},
            marker=_marker(theme()["categorical"][0], 9),
            customdata=bins["count"],
            hovertemplate=(
                "prédit %{x:.3f} · observé %{y:.3f}<br>%{customdata} paires<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title="probabilité prédite (moyenne du bin)", range=[0, 1])
    fig.update_yaxes(title="part de vraies interactions", range=[0, 1])
    subtitle = f"{len(bins['count'])} bins de même effectif · ligne grise : parfaitement calibré"
    return _layout(fig, "Calibration", subtitle, 460, legend=False)


def plot_confusion(preds: PairPredictions, threshold: float = 0.5) -> go.Figure:
    m = binary_metrics(preds.labels, preds.logits, threshold)
    counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    rates = counts / counts.sum(1, keepdims=True).clip(min=1)
    names = ["pas d'interaction", "interaction"]
    fig = go.Figure(
        go.Heatmap(
            z=rates,
            x=names,
            y=names,
            colorscale=_colorscale(),
            zmin=0,
            zmax=1,
            showscale=False,
            xgap=2,
            ygap=2,
            customdata=counts,
            hovertemplate=(
                "vrai : %{y}<br>prédit : %{x}<br>%{customdata} paires (%{z:.1%})<extra></extra>"
            ),
        )
    )
    ramp = theme()["ramp"]
    for i in range(2):
        for j in range(2):
            fill = ramp[min(int(rates[i, j] * (len(ramp) - 1)), len(ramp) - 1)]
            fig.add_annotation(
                x=names[j],
                y=names[i],
                text=f"<b>{counts[i, j]:,}</b><br>{rates[i, j]:.0%}",
                showarrow=False,
                font={"color": _ink_on(fill), "size": 14},
            )
    fig.update_xaxes(title="prédit", showgrid=False, showline=False)
    fig.update_yaxes(title="vrai", showgrid=False, showline=False, autorange="reversed")
    subtitle = f"seuil {threshold} · MCC {m['mcc']:.3f} · F1 {m['f1']:.3f}"
    return _layout(fig, "Matrice de confusion", subtitle, height=420, legend=False)


def plot_probability_distributions(preds: PairPredictions) -> go.Figure:
    """Predicted probabilities of interacting and non-interacting pairs: separated = good."""
    probs, labels = preds.probs.numpy(), preds.labels.numpy()
    fig = go.Figure()
    for label in (1, 0):
        fig.add_trace(
            go.Histogram(
                x=probs[labels == label],
                name=LABEL_NAMES[label],
                xbins={"start": 0, "end": 1, "size": 0.05},
                marker={
                    "color": label_color(label),
                    "line": {"color": theme()["surface"], "width": 1},
                },
                hovertemplate=f"{LABEL_NAMES[label]}<br>p ∈ %{{x}} : %{{y}} paires<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.15, bargroupgap=0)
    fig.update_xaxes(title="probabilité prédite", range=[0, 1])
    fig.update_yaxes(title="paires")
    return _layout(fig, "Probabilités prédites", "bleu : interaction · orange : pas d'interaction")


def plot_metric_by_length(
    preds: PairPredictions, max_protein_length: int, n_bins: int = 5
) -> go.Figure:
    """AUROC and accuracy per length bin of the longer protein (proteins beyond the crop
    length were cut)."""
    longer = torch.maximum(preds.length_a, preds.length_b).numpy()
    edges = np.unique(np.quantile(longer, np.linspace(0, 1, n_bins + 1)))
    which = np.clip(np.digitize(longer, edges[1:-1]), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        sel = which == b
        if not sel.any():
            continue
        sub = binary_metrics(preds.labels[sel], preds.logits[sel])
        rows.append(
            (
                np.median(longer[sel]),
                sub["auroc"],
                sub["accuracy"],
                sel.sum(),
                edges[b],
                edges[b + 1],
            )
        )
    rows = np.array(rows, dtype=float)
    c = theme()["categorical"]
    fig = go.Figure()
    for col, name, color in ((1, "AUROC", c[0]), (2, "accuracy", c[1])):
        fig.add_trace(
            go.Scatter(
                x=rows[:, 0],
                y=rows[:, col],
                name=name,
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker=_marker(color, 9),
                customdata=rows[:, 3:6],
                hovertemplate=(
                    f"{name} %{{y:.3f}}<br>longueur %{{customdata[1]:.0f}}–%{{customdata[2]:.0f}}"
                    "<br>%{customdata[0]:.0f} paires<extra></extra>"
                ),
            )
        )
    # on a log axis, shapes take data values but annotations take log10 values
    fig.add_shape(
        type="line",
        x0=max_protein_length,
        x1=max_protein_length,
        y0=0,
        y1=1,
        yref="paper",
        line={"color": theme()["muted"], "width": 1},
    )
    fig.add_annotation(
        x=np.log10(max_protein_length),
        y=1,
        yref="paper",
        text=f"coupe à {max_protein_length}",
        showarrow=False,
        xanchor="left",
        font={"color": theme()["muted"], "size": 11},
    )
    fig.update_xaxes(
        title="longueur de la plus longue protéine (résidus, médiane du bin)", type="log"
    )
    fig.update_yaxes(title="score")
    return _layout(
        fig, "Performance selon la longueur", "au-delà de la ligne, les protéines sont coupées"
    )


def plot_order_symmetry(preds: PairPredictions, preds_swapped: PairPredictions) -> go.Figure:
    """p(A, B) against p(B, A): the input is not symmetric, so the model may disagree."""
    p, q, labels = preds.probs.numpy(), preds_swapped.probs.numpy(), preds.labels.numpy()
    ids = np.array([f"{a} + {b}" for a, b, _ in preds.pairs])
    fig = go.Figure()
    _reference_line(fig, [0, 1], [0, 1], "symétrique")
    for label in (1, 0):
        sel = labels == label
        fig.add_trace(
            _scatter(len(p))(
                x=p[sel],
                y=q[sel],
                mode="markers",
                name=LABEL_NAMES[label],
                marker=_marker(label_color(label), 7),
                opacity=0.75,
                customdata=ids[sel],
                hovertemplate="%{customdata}<br>p(A,B) %{x:.3f} · p(B,A) %{y:.3f}<extra></extra>",
            )
        )
    mean = binary_metrics(
        preds.labels, torch.logit(torch.tensor((p + q) / 2).clamp(1e-6, 1 - 1e-6))
    )
    subtitle = (
        f"|Δp| moyen {np.abs(p - q).mean():.3f} · AUROC (A,B) "
        f"{binary_metrics(preds.labels, preds.logits)['auroc']:.3f} · (B,A) "
        f"{binary_metrics(preds_swapped.labels, preds_swapped.logits)['auroc']:.3f} · "
        f"moyenne des deux ordres {mean['auroc']:.3f}"
    )
    fig.update_xaxes(title="p(A, B)", range=[0, 1], constrain="domain")
    fig.update_yaxes(title="p(B, A)", range=[0, 1], scaleanchor="x", constrain="domain")
    return _layout(fig, "Symétrie de l'ordre A / B", subtitle, height=560)


# --- latent space ---------------------------------------------------------------


def _hover(hover: dict[str, list] | None, n: int) -> tuple[np.ndarray | None, str]:
    if not hover:
        return None, ""
    data = np.array([list(map(str, values)) for values in hover.values()]).T.reshape(n, -1)
    template = "<br>".join(f"{k} : %{{customdata[{i}]}}" for i, k in enumerate(hover))
    return data, template


def plot_latent(
    coords: np.ndarray,
    values,
    title: str,
    hover: dict[str, list] | None = None,
    categorical: bool = False,
    color_title: str = "",
) -> go.Figure:
    """2D projection colored by a category (3 hues max, the rest in gray) or a number."""
    values = np.asarray(values)
    customdata, template = _hover(hover, len(coords))
    fig = go.Figure()
    if categorical:
        for i, value in enumerate(dict.fromkeys(values.tolist())):
            sel = values == value
            color = theme()["categorical"][i] if i < 3 else theme()["muted"]
            fig.add_trace(
                _scatter(len(coords))(
                    x=coords[sel, 0],
                    y=coords[sel, 1],
                    mode="markers",
                    name=str(value),
                    marker=_marker(color),
                    opacity=0.8,
                    customdata=None if customdata is None else customdata[sel],
                    hovertemplate=f"<b>{value}</b><br>{template}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            _scatter(len(coords))(
                x=coords[:, 0],
                y=coords[:, 1],
                mode="markers",
                showlegend=False,
                marker={
                    **_marker(values),
                    "colorscale": _colorscale(start=3),
                    "showscale": True,
                    "colorbar": {"title": {"text": color_title}, "outlinewidth": 0},
                },
                customdata=customdata,
                hovertemplate=f"{color_title} %{{marker.color:.3g}}<br>{template}<extra></extra>",
            )
        )
    return _layout(fig, title, height=560, axes=False, legend=categorical)


def plot_latent_pairs(coords: np.ndarray, preds: PairPredictions, title: str) -> go.Figure:
    """Pair vectors: color = true label, shape = prediction (dot correct, x wrong)."""
    probs, labels = preds.probs.numpy(), preds.labels.numpy()
    correct = (probs >= 0.5) == labels.astype(bool)
    ids = np.array([f"{a} + {b}" for a, b, _ in preds.pairs])
    fig = go.Figure()
    for label in (1, 0):
        for ok in (True, False):
            sel = (labels == label) & (correct == ok)
            kind = "correct" if ok else ("FN, ratée" if label == 1 else "FP, fausse alerte")
            fig.add_trace(
                _scatter(len(coords))(
                    x=coords[sel, 0],
                    y=coords[sel, 1],
                    mode="markers",
                    name=f"{LABEL_NAMES[label]} · {kind} ({sel.sum()})",
                    marker=_marker(label_color(label), 8 if ok else 9, "circle" if ok else "x"),
                    opacity=0.8 if ok else 1,
                    customdata=np.stack([ids[sel], probs[sel]], -1),
                    hovertemplate="%{customdata[0]}<br>p = %{customdata[1]}<extra></extra>",
                )
            )
    subtitle = "couleur : vrai label · forme : ● correct, ✕ erreur"
    return _layout(fig, title, subtitle, 560, axes=False)


def plot_amino_acids(letters: list[str], coords: np.ndarray, title: str) -> go.Figure:
    """Learned amino-acid embeddings in 2D, colored by physico-chemical group."""
    fig = go.Figure()
    for i, group in enumerate(AA_GROUPS):
        members = [k for k, aa in enumerate(letters) if AA_GROUP[aa] == group]
        color = theme()["categorical"][i]
        fig.add_trace(
            go.Scatter(
                x=coords[members, 0],
                y=coords[members, 1],
                mode="markers+text",
                name=group,
                text=[letters[k] for k in members],
                textfont={"color": _ink_on(color), "size": 13},
                marker=_marker(color, 30),
                customdata=[[AA_NAMES[letters[k]], AA_CLASS[letters[k]]] for k in members],
                hovertemplate=(
                    "<b>%{text}</b> %{customdata[0]}<br>classe : %{customdata[1]}<extra></extra>"
                ),
            )
        )
    subtitle = "acides aminés proches = le modèle les traite de façon similaire"
    return _layout(fig, title, subtitle, 520, axes=False)


# --- attention ------------------------------------------------------------------


def _token_labels(att: PairAttention) -> list[str]:
    labels, counter = [], {0: 0, 1: 0}
    for token, segment in zip(att.tokens, att.segment_ids.tolist(), strict=True):
        if token.startswith("<"):
            labels.append(token)
        else:
            counter[segment] += 1
            labels.append(f"{'AB'[segment]}·{token}{counter[segment]}")
    return labels


def plot_attention_map(att: PairAttention, layer: int | None = None) -> go.Figure:
    """Whole pair attention (mean over heads) with the A | B boundary, and the A -> B block."""
    w = att.stacked().mean(1)
    w = (w.mean(0) if layer is None else w[layer]).numpy()
    labels = _token_labels(att)
    a_labels = [labels[i] for i in att.residues_a.tolist()]
    b_labels = [labels[i] for i in att.residues_b.tolist()]
    fig = _subplots(
        rows=1,
        cols=2,
        column_widths=[0.55, 0.45],
        horizontal_spacing=0.12,
        subplot_titles=("Toute la paire (requête → clé)", "Bloc A → B (résidus)"),
    )
    hover = "requête %{y}<br>clé %{x}<br>poids %{z:.4f}<extra></extra>"
    fig.add_trace(
        go.Heatmap(z=w, x=labels, y=labels, coloraxis="coloraxis", hovertemplate=hover), 1, 1
    )
    fig.add_trace(
        go.Heatmap(
            z=a_to_b_block(att, layer).numpy(),
            x=b_labels,
            y=a_labels,
            coloraxis="coloraxis2",
            hovertemplate=hover,
        ),
        1,
        2,
    )
    boundary = int((att.segment_ids == 0).sum()) - 0.5
    for kind in ("x", "y"):
        line = {
            "type": "line",
            "line": {"color": theme()["surface"], "width": 2},
            "xref": "x",
            "yref": "y",
        }
        coords = {"x0": boundary, "x1": boundary, "y0": -0.5, "y1": len(labels) - 0.5}
        if kind == "y":
            coords = {"y0": boundary, "y1": boundary, "x0": -0.5, "x1": len(labels) - 0.5}
        fig.add_shape(**line, **coords)
    a, b, label = att.pair
    p = torch.sigmoid(torch.tensor(att.logit)).item()
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False)
    fig.update_xaxes(showgrid=False, showline=False, tickangle=-90)
    if len(labels) > 60:
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
    which = "moyenne des couches" if layer is None else f"couche {layer}"
    fig = _layout(
        fig,
        f"Attention · {a} + {b}",
        f"{LABEL_NAMES[label]} · p = {p:.2f} · {which}, moyenne des têtes",
        560,
        legend=False,
    )
    fig.update_layout(
        coloraxis={
            "colorscale": _colorscale(),
            "colorbar": {"x": 0.50, "len": 0.8, "outlinewidth": 0},
        },
        coloraxis2={
            "colorscale": _colorscale(),
            "colorbar": {"x": 1.0, "len": 0.8, "outlinewidth": 0},
        },
    )
    return fig


def plot_attention_statistics(
    positives: list[PairAttention], negatives: list[PairAttention]
) -> go.Figure:
    """Per layer: share of attention to the partner (interacting vs not); per layer and
    head: attention sinks and focus."""

    def mean_of(fn, atts):
        return torch.stack([fn(a) for a in atts]).mean(0).numpy()

    fig = _subplots(
        rows=1,
        cols=3,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Attention vers le partenaire",
            "Sinks (vers <CLS>/<SEP>/<EOS>)",
            "Focus (1 − entropie normalisée)",
        ),
    )
    for atts, label in ((positives, 1), (negatives, 0)):
        cross = mean_of(cross_protein_mass, atts).mean(-1).mean(-1)  # (layers,)
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(cross)),
                y=cross,
                mode="lines+markers",
                name=LABEL_NAMES[label],
                line={"color": label_color(label), "width": 2},
                marker=_marker(label_color(label), 9),
                hovertemplate=f"{LABEL_NAMES[label]}<br>couche %{{x}} : %{{y:.3f}}<extra></extra>",
            ),
            1,
            1,
        )
    everything = positives + negatives
    for col, fn in ((2, sink_mass), (3, attention_focus)):
        fig.add_trace(
            go.Heatmap(
                z=mean_of(fn, everything),
                coloraxis="coloraxis",
                xgap=2,
                ygap=2,
                hovertemplate="couche %{y} · tête %{x}<br>%{z:.3f}<extra></extra>",
            ),
            1,
            col,
        )
        fig.update_xaxes(title="tête", showgrid=False, showline=False, dtick=1, row=1, col=col)
        fig.update_yaxes(
            title="couche",
            showgrid=False,
            showline=False,
            dtick=1,
            autorange="reversed",
            row=1,
            col=col,
        )
    fig.update_xaxes(title="couche", dtick=1, row=1, col=1)
    fig.update_yaxes(title="part de l'attention", rangemode="tozero", row=1, col=1)
    subtitle = f"{len(positives)} paires en interaction, {len(negatives)} sans"
    fig = _layout(fig, "Statistiques d'attention", subtitle, height=430)
    bar = {"len": 0.8, "outlinewidth": 0}
    fig.update_layout(
        coloraxis={"colorscale": _colorscale(), "cmin": 0, "cmax": 1, "colorbar": bar}
    )
    return fig


# --- residue importance ---------------------------------------------------------


def plot_residue_importance(imp: ResidueImportance, top_k: int = 3) -> go.Figure:
    """Integrated gradients per residue of A and B (red: towards interaction, blue: against).
    The top_k residues pushing towards interaction are labeled."""
    t = theme()
    names = imp.pair[:2]
    fig = _subplots(
        rows=2,
        cols=1,
        vertical_spacing=0.2,
        subplot_titles=(f"Protéine A · {names[0]}", f"Protéine B · {names[1]}"),
    )
    for row, segment in ((1, 0), (2, 1)):
        letters, scores = imp.residue_scores(segment)
        s = scores.numpy()
        positions = np.arange(1, len(s) + 1)
        fig.add_trace(
            go.Bar(
                x=positions,
                y=s,
                showlegend=False,
                marker={
                    "color": [t["positive"] if v > 0 else t["negative"] for v in s],
                    "cornerradius": 2,
                },
                customdata=letters,
                hovertemplate="%{customdata}%{x} : %{y:+.4f}<extra></extra>",
            ),
            row,
            1,
        )
        for k in np.argsort(-s)[:top_k]:
            if s[k] > 0:
                fig.add_annotation(
                    x=positions[k],
                    y=s[k],
                    text=f"{letters[k]}{positions[k]}",
                    showarrow=False,
                    yanchor="bottom",
                    font={"color": t["ink"], "size": 11},
                    row=row,
                    col=1,
                )
        fig.update_xaxes(title="position (résidu)" if row == 2 else None, row=row, col=1)
        fig.update_yaxes(title="IG", row=row, col=1)
    for name, color in (("vers interaction", t["positive"]), ("contre", t["negative"])):
        fig.add_trace(go.Bar(x=[None], y=[None], name=name, marker={"color": color}))
    fig.update_layout(bargap=0.15)
    p = torch.sigmoid(torch.tensor(imp.logit)).item()
    subtitle = (
        f"{LABEL_NAMES[imp.pair[2]]} · p = {p:.2f} · contrôle : somme IG {imp.scores.sum():.3f} "
        f"≈ logit − référence {imp.logit - imp.baseline_logit:.3f}"
    )
    fig.update_yaxes(exponentformat="e")  # IG values are small: 5e-5, not "50µ"
    return _layout(fig, "Importance des résidus (integrated gradients)", subtitle, height=580)


# --- interaction network --------------------------------------------------------

EDGE_STYLES = {
    "TP": ("interaction trouvée (TP)", 0, "solid"),
    "FN": ("interaction ratée (FN)", 0, "dot"),
    "FP": ("fausse alerte (FP)", 1, "solid"),
    "TN": ("absence correcte (TN)", None, "solid"),
}


def plot_interaction_graph(
    graph, outcomes: tuple[str, ...] = ("TP", "FP", "FN"), title: str = "", seed: int = 0
) -> go.Figure:
    """Proteins as nodes, evaluated pairs as edges: blue = true interaction (solid found,
    dotted missed), orange = false alarm. Hover an edge midpoint for its probability."""
    import networkx as nx

    t = theme()
    edges = [(a, b, d) for a, b, d in graph.edges(data=True) if d["outcome"] in outcomes]
    sub = nx.Graph()
    sub.add_edges_from(edges)
    fig = go.Figure()
    if sub.number_of_nodes() == 0:
        return _layout(fig, title or "Graphe d'interactions", "aucune arête à afficher", axes=False)
    pos = nx.spring_layout(sub, seed=seed, k=1.5 / np.sqrt(sub.number_of_nodes()))
    for outcome_name in outcomes:
        name, slot, dash = EDGE_STYLES[outcome_name]
        chosen = [(a, b, d) for a, b, d in sub.edges(data=True) if d["outcome"] == outcome_name]
        color = t["muted"] if slot is None else t["categorical"][slot]
        xs, ys, mx, my, text = [], [], [], [], []
        for a, b, d in chosen:
            xs += [pos[a][0], pos[b][0], None]
            ys += [pos[a][1], pos[b][1], None]
            mx.append((pos[a][0] + pos[b][0]) / 2)
            my.append((pos[a][1] + pos[b][1]) / 2)
            text.append(f"{a} — {b}<br>p = {d['prob']:.3f} · {outcome_name}")
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines",
                name=f"{name} · {len(chosen)}",
                legendgroup=outcome_name,
                line={"color": color, "width": 2, "dash": dash},
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mx,
                y=my,
                mode="markers",
                legendgroup=outcome_name,
                showlegend=False,
                marker={"size": 14, "color": "rgba(0,0,0,0)"},
                text=text,
                hovertemplate="%{text}<extra></extra>",
            )
        )
    nodes = list(sub.nodes)
    degrees = [sub.degree[n] for n in nodes]
    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text" if len(nodes) <= 60 else "markers",
            text=nodes,
            textposition="top center",
            textfont={"color": t["ink_secondary"], "size": 10},
            marker={
                "size": [8 + 2 * d for d in degrees],
                "color": t["ink_secondary"],
                "line": {"color": t["surface"], "width": 2},
            },
            customdata=degrees,
            showlegend=False,
            hovertemplate="<b>%{text}</b><br>%{customdata} paire(s) affichée(s)<extra></extra>",
        )
    )
    subtitle = f"{len(nodes)} protéines · survole le milieu d'une arête pour sa probabilité"
    return _layout(fig, title or "Graphe d'interactions", subtitle, height=640, axes=False)


def plot_partner_ranking(
    ranking: list[tuple[str, float]], known: set[str], protein: str, top: int = 20
) -> go.Figure:
    """Top predicted partners of one protein; known interactions highlighted in blue."""
    t = theme()
    rows = ranking[:top][::-1]  # highest at the top
    fig = go.Figure()
    for is_known, name, color in (
        (True, "interaction connue", t["categorical"][0]),
        (False, "inconnue du dataset", t["muted"]),
    ):
        sel = [(p, prob) for p, prob in rows if (p in known) == is_known]
        fig.add_trace(
            go.Bar(
                x=[prob for _, prob in sel],
                y=[p for p, _ in sel],
                orientation="h",
                name=name,
                marker={"color": color, "cornerradius": 4},
                text=[f"{prob:.2f}" for _, prob in sel],
                textposition="outside",
                textfont={"color": t["ink_secondary"]},
                hovertemplate="%{y} : p = %{x:.3f}<extra></extra>",
            )
        )
    fig.update_yaxes(categoryorder="array", categoryarray=[p for p, _ in rows], showgrid=False)
    fig.update_xaxes(title="probabilité d'interaction prédite", range=[0, 1.08])
    fig.update_layout(bargap=0.3, barmode="overlay")
    height = 120 + 28 * len(rows)
    return _layout(
        fig,
        f"Partenaires prédits pour {protein}",
        f"top {len(rows)} parmi les protéines testées",
        height,
    )
