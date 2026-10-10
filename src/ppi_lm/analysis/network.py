"""Who interacts with whom: interaction graph of the predictions, and partner ranking."""

import networkx as nx
import torch

from ppi_lm.analysis.inference import PairPredictions, predict_pairs

OUTCOMES = ("TP", "FP", "FN", "TN")


def outcome(label: int, prob: float, threshold: float = 0.5) -> str:
    """TP: true interaction found, FP: predicted but false, FN: missed, TN: correctly absent."""
    predicted = prob >= threshold
    if label == 1:
        return "TP" if predicted else "FN"
    return "FP" if predicted else "TN"


def interaction_graph(preds: PairPredictions, threshold: float = 0.5) -> nx.Graph:
    """Nodes = proteins, one edge per evaluated pair, with its label, probability, outcome.

    The same pair may appear twice in a split (A, B) and (B, A): the last one is kept.
    """
    graph = nx.Graph()
    for (a, b, label), prob in zip(preds.pairs, preds.probs.tolist(), strict=True):
        graph.add_edge(a, b, label=label, prob=prob, outcome=outcome(label, prob, threshold))
    return graph


def subgraph_around(graph: nx.Graph, protein: str, radius: int = 1) -> nx.Graph:
    """The protein, its neighbours up to `radius` steps, and the edges between them."""
    return nx.ego_graph(graph, protein, radius=radius)


def rank_partners(
    model, data, protein: str, candidates: list[str], batch_size: int = 32
) -> list[tuple[str, float]]:
    """Interaction probability of `protein` with every candidate, highest first.

    Useful for "which proteins would this one interact with?". Candidates should come
    from the same split (no protein is shared between splits).
    """
    pairs = [(protein, c, 0) for c in candidates if c != protein]  # labels unused here
    probs = predict_pairs(model, data, pairs, batch_size).probs
    order = torch.argsort(probs, descending=True).tolist()
    return [(pairs[i][1], probs[i].item()) for i in order]
