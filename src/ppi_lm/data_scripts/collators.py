import torch
from torch.nn.utils.rnn import pad_sequence

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer


def pad_batch(seqs: list[torch.Tensor], pad_value: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pads sequences to the longest one.

    Returns the padded tensor (B, L) and an attention mask (B, L), True on real tokens.
    """
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    lengths = torch.tensor([len(s) for s in seqs])
    attention_mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]
    return padded, attention_mask


class ClassificationCollator:
    """Collator for classification tasks (e.g., PPI prediction)."""

    def __init__(self, tokenizer: ProteinTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = pad_batch(
            [item["input_ids"] for item in batch], self.tokenizer.pad_id
        )
        segment_ids, _ = pad_batch([item["segment_ids"] for item in batch], 0)
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.float)
        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class MLMCollator:
    """Collator for masked language modeling: dynamic span masking + BERT 80/10/10 rule.
    labels = original token on selected positions, -100 elsewhere (ignored by the loss).
    """

    def __init__(self, tokenizer: ProteinTokenizer, mask_prob: float = 0.15, max_span: int = 3):
        self.tokenizer = tokenizer
        self.mask_prob = mask_prob
        self.max_span = max_span
        self.special_ids = torch.tensor(tokenizer.special_ids)
        self.aa_ids = torch.tensor(tokenizer.aa_ids)

    def _select_positions(self, maskable: torch.Tensor) -> torch.Tensor:
        """Picks mask_prob of the maskable positions of ONE sequence, as short spans."""
        selected = torch.zeros_like(maskable)
        candidates = maskable.nonzero().squeeze(1)
        if len(candidates) == 0:
            return selected
        n_target = max(1, round(self.mask_prob * len(candidates)))
        while (selected & maskable).sum() < n_target:
            start = candidates[torch.randint(len(candidates), (1,))].item()
            span = torch.randint(1, self.max_span + 1, (1,)).item()
            selected[start : start + span] = True
        return selected & maskable  # a span never covers a special token

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        input_ids, attention_mask = pad_batch(
            [item["input_ids"] for item in batch], self.tokenizer.pad_id
        )
        segment_ids, _ = pad_batch([item["segment_ids"] for item in batch], 0)

        # 1. which positions to predict: never special tokens nor padding
        maskable = ~torch.isin(input_ids, self.special_ids)
        selected = torch.stack([self._select_positions(row) for row in maskable])

        # 2. labels: the true token where selected, -100 elsewhere
        labels = input_ids.clone()
        labels[~selected] = -100

        # 3. 80 % -> <MASK>, 10 % -> random amino acid, 10 % -> unchanged
        r = torch.rand(input_ids.shape)
        to_mask = selected & (r < 0.8)
        to_random = selected & (r >= 0.8) & (r < 0.9)

        input_ids = input_ids.clone()
        input_ids[to_mask] = self.tokenizer.mask_id
        random_aa = self.aa_ids[torch.randint(len(self.aa_ids), input_ids.shape)]
        input_ids[to_random] = random_aa[to_random]

        return {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
