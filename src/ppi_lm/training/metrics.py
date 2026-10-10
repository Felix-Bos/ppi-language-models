"""Binary classification metrics, computed from raw logits, and the ROC curve figure."""

import math

import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _ratio(num: float, den: float) -> float:
    """num / den, 0 when den is 0 (same convention as sklearn's zero_division=0)."""
    return num / den if den else 0.0


def confusion_counts(labels: torch.Tensor, preds: torch.Tensor) -> dict[str, int]:
    """labels, preds: (N,) bool. True / false positives and negatives."""
    return {
        "tp": int((preds & labels).sum()),
        "fp": int((preds & ~labels).sum()),
        "tn": int((~preds & ~labels).sum()),
        "fn": int((~preds & labels).sum()),
    }


def binary_metrics(
    labels: torch.Tensor, logits: torch.Tensor, threshold: float = 0.5
) -> dict[str, float]:
    """Every metric of a binary classifier.

    labels: (N,) 0 / 1
    logits: (N,) raw scores (before the sigmoid)
    threshold: on the probability, for the metrics that need hard predictions

    Ranking (threshold-free): auroc, auprc (NaN if only one class is present)
    Thresholded: accuracy, balanced_accuracy, precision, recall (= sensitivity),
                 specificity, f1, mcc, tp / fp / tn / fn, predicted_positive_rate
    Calibration: brier, mean_prob_positives, mean_prob_negatives
    """
    labels = labels.detach().cpu().bool()
    probs = torch.sigmoid(logits.detach().float().cpu())
    c = confusion_counts(labels, probs >= threshold)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]

    precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    metrics = {
        "auroc": math.nan,
        "auprc": math.nan,
        "accuracy": _ratio(tp + tn, len(labels)),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "mcc": _ratio(tp * tn - fp * fn, mcc_den),
        "brier": ((probs - labels.float()) ** 2).mean().item(),
        "mean_prob_positives": probs[labels].mean().item() if labels.any() else math.nan,
        "mean_prob_negatives": probs[~labels].mean().item() if (~labels).any() else math.nan,
        "predicted_positive_rate": _ratio(tp + fp, len(labels)),
        **c,
    }
    if labels.unique().numel() == 2:
        metrics["auroc"] = float(roc_auc_score(labels.numpy(), probs.numpy()))
        metrics["auprc"] = float(average_precision_score(labels.numpy(), probs.numpy()))
    return metrics


def roc_figure(labels: torch.Tensor, logits: torch.Tensor, title: str = "ROC"):
    """ROC curve as a matplotlib Figure (for TensorBoard), or None if one class only."""
    from matplotlib.figure import Figure  # no pyplot: works without a display

    labels = labels.detach().cpu().bool().numpy()
    if labels.min() == labels.max():
        return None
    probs = torch.sigmoid(logits.detach().float().cpu()).numpy()
    fpr, tpr, _ = roc_curve(labels, probs)
    auroc = roc_auc_score(labels, probs)

    fig = Figure(figsize=(4, 4), dpi=100)
    ax = fig.add_subplot()
    ax.plot(fpr, tpr, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="chance")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title=title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
