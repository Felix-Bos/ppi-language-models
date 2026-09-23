from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from ppi_lm.data_scripts.collators import ClassificationCollator, MLMCollator
from ppi_lm.data_scripts.tokenizer import ProteinTokenizer


class PairDataset(Dataset):
    """One protein pair per item: <CLS> A <SEP> B <EOS>, with segment ids and label."""

    def __init__(
        self,
        pairs: list[tuple[str, str, int]],
        seqs: dict[str, str],
        tokenizer: ProteinTokenizer,
        max_protein_length: int = 1000,
        train: bool = True,
        augmentation_percent: float = 0.0,
        seed: int = 0,
    ):
        self.pairs = self._augment(pairs, augmentation_percent, seed) if train else pairs
        self.seqs = seqs
        self.tokenizer = tokenizer
        self.max_protein_length = max_protein_length
        self.train = train
        self.augmentation_percent = augmentation_percent

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _augment(
        pairs: list[tuple[str, str, int]], percent: float, seed: int
    ) -> list[tuple[str, str, int]]:
        """Adds `percent` of the pairs again, in reversed order (B, A)."""
        if percent <= 0:
            return pairs
        g = torch.Generator().manual_seed(seed)
        n_extra = round(percent * len(pairs))
        idx = torch.randperm(len(pairs), generator=g)[:n_extra].tolist()
        reversed_pairs = [(pairs[i][1], pairs[i][0], pairs[i][2]) for i in idx]
        return pairs + reversed_pairs

    def _crop(self, ids: list[int]) -> list[int]:
        """Random window in training, fixed start in evaluation."""
        if len(ids) <= self.max_protein_length:
            return ids
        if self.train:
            start = torch.randint(0, len(ids) - self.max_protein_length + 1, (1,)).item()
        else:
            start = 0
        return ids[start : start + self.max_protein_length]

    def __getitem__(self, idx: int) -> dict:
        id_a, id_b, label = self.pairs[idx]
        a = self._crop(self.tokenizer.tokenize(self.seqs[id_a]))
        b = self._crop(self.tokenizer.tokenize(self.seqs[id_b]))

        tok = self.tokenizer
        input_ids = [tok.cls_id] + a + [tok.sep_id] + b + [tok.eos_id]
        segment_ids = [0] * (len(a) + 2) + [1] * (len(b) + 1)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "label": label,
        }


class GoldStandardDataset:
    # naming convention from the original dataset
    SPLITS = {"train": "Intra1", "val": "Intra0", "test": "Intra2"}

    def __init__(
        self,
        data_dir: str = "./data/gold_standard_data",
        tokenizer: ProteinTokenizer | None = None,
        batch_size: int = 32,
        max_protein_length: int = 512,
        mask_prob: float = 0.15,
        augmentation_percent: float = 0.0,
        num_workers: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer or ProteinTokenizer()
        self.max_protein_length = max_protein_length
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.seqs = self._read_fasta(self.data_dir / "human_swissprot_oneliner.fasta")
        self.pairs = {split: self._load_split(prefix) for split, prefix in self.SPLITS.items()}

        self.augmentation_percent = augmentation_percent
        self.mlm_collator = MLMCollator(self.tokenizer, mask_prob)
        self.classif_collator = ClassificationCollator(self.tokenizer)

    def _read_fasta(self, file_path: Path) -> dict[str, str]:
        """Reads a FASTA file and returns a dictionary mapping sequence IDs to sequences."""
        seqs: dict[str, str] = {}
        with open(file_path) as f:
            lines = [line.strip() for line in f if line.strip()]
        for header, seq in zip(lines[0::2], lines[1::2], strict=True):
            assert header.startswith(">")
            seqs[header[1:].split()[0]] = seq
        return seqs

    def _read_pairs(self, file_path: Path, label: int) -> list[tuple[str, str, int]]:
        """Reads a pair file (one pair of UniProt IDs per line)
        and attaches the given label."""

        pairs: list[tuple[str, str, int]] = []

        with open(file_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    pairs.append((parts[0], parts[1], label))
        return pairs

    def _load_split(self, prefix: str) -> list[tuple[str, str, int]]:
        """Loads positive and negative pairs of one split."""
        pos = self._read_pairs(self.data_dir / f"{prefix}_pos_rr.txt", label=1)
        neg = self._read_pairs(self.data_dir / f"{prefix}_neg_rr.txt", label=0)
        return pos + neg  # combine positive and negative pairs

    def _dataset(self, split: str, only_positives: bool) -> PairDataset:
        pairs = self.pairs[split]
        if only_positives:
            pairs = [p for p in pairs if p[2] == 1]
        return PairDataset(
            pairs,
            self.seqs,
            self.tokenizer,
            max_protein_length=self.max_protein_length,
            train=(split == "train"),
            augmentation_percent=self.augmentation_percent,
        )

    def _make_loader(self, dataset: Dataset, collator, split: str) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=self.num_workers,
            collate_fn=collator,
        )

    def mlm_loader(self, split: str) -> DataLoader:
        """Positive pairs only, with masking (pre-training on pairs)."""
        return self._make_loader(
            self._dataset(split, only_positives=True), self.mlm_collator, split
        )

    def classif_loader(self, split: str) -> DataLoader:
        """All pairs, with labels (PPI classification)."""
        return self._make_loader(
            self._dataset(split, only_positives=False), self.classif_collator, split
        )
