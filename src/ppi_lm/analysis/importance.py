"""Residue importance by integrated gradients, for any encoder with a token embedding.

Integrated gradients (Sundararajan et al., 2017) attribute the interaction logit to each
input token: the token embeddings are scaled from 0 (baseline) to their real value, and
the gradient of the logit is integrated along the way:

    IG_i = sum_d  e_id * mean_over_alpha( d logit / d e_id  at  alpha * e )

Completeness: sum_i IG_i ~= logit(x) - logit(baseline). Positive scores push towards
"interaction", negative ones against. Unlike attention, this works for all four
architectures (they all start with a token embedding).
"""

from dataclasses import dataclass

import torch

from ppi_lm.analysis.inference import make_loader


@dataclass
class ResidueImportance:
    pair: tuple[str, str, int]
    tokens: list[str]  # (L,)
    segment_ids: torch.Tensor  # (L,)
    scores: torch.Tensor  # (L,) integrated gradients per token
    logit: float  # logit of the real input
    baseline_logit: float  # logit with all token embeddings at 0

    def residue_scores(self, segment: int) -> tuple[list[str], torch.Tensor]:
        """Residues of one protein (special tokens excluded) and their scores."""
        keep = [
            i
            for i, t in enumerate(self.tokens)
            if self.segment_ids[i] == segment and not t.startswith("<")
        ]
        return [self.tokens[i] for i in keep], self.scores[keep]


def integrated_gradients(
    model, data, pair: tuple[str, str, int], steps: int = 32
) -> ResidueImportance:
    """IG of the interaction logit with respect to every token of one pair.

    steps: points on the path 0 -> embedding (midpoint rule), computed in one batch
    """
    model.eval()
    device = next(model.parameters()).device
    batch = next(iter(make_loader(model, data, [pair], batch_size=1)))
    n_rows = steps + 2
    inputs = [batch[k].to(device).expand(n_rows, -1) for k in ("input_ids", "segment_ids")]
    mask = batch["attention_mask"].to(device).expand(n_rows, -1)

    # row 0: real input (alpha = 1), row 1: baseline (alpha = 0), then the path midpoints
    midpoints = (torch.arange(steps) + 0.5) / steps
    alphas = torch.cat([torch.ones(1), torch.zeros(1), midpoints]).to(device)
    captured = {}

    def scale(module, args, output):
        captured["embedding"] = output[0].detach()
        scaled = output * alphas[:, None, None]
        if scaled.requires_grad:
            scaled.retain_grad()
        else:  # frozen encoder: the embedding itself has no gradient
            scaled.requires_grad_(True)
        captured["scaled"] = scaled
        return scaled

    handle = model.encoder.token_embedding.register_forward_hook(scale)
    try:
        logits = model(inputs[0], inputs[1], mask)
        logits[2:].sum().backward()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    mean_grad = captured["scaled"].grad[2:].mean(0)  # (L, d)
    scores = (captured["embedding"] * mean_grad).sum(-1)  # (L,)
    return ResidueImportance(
        pair=pair,
        tokens=[data.tokenizer.id2token[i] for i in batch["input_ids"][0].tolist()],
        segment_ids=batch["segment_ids"][0],
        scores=scores.detach().cpu(),
        logit=logits[0].item(),
        baseline_logit=logits[1].item(),
    )
