import os
from pathlib import Path

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer


class GoldStandardDataset:
    def __init__(
        self,
        data_dir: str = "./data/gold_standard_data",
        tokenizer: ProteinTokenizer = None,
        batch_size: int = 32,
        max_protein_length: int = 512,
        mask_prob: float = 0.15,
        num_workers: int = 2,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer or ProteinTokenizer()
        self.max_protein_length = max_protein_length
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _read_fasta(self, file_path: Path) -> dict[str, str]:
        """ Reads a FASTA file and returns a dictionary mapping sequence IDs to sequences."""
        seqs: dict[str, str] = {}
        with open(file_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        for header, seq in zip(lines[0::2], lines[1::2]):
            assert header.startswith(">") 
            seqs[header[1:].split()[0]] = seq
        return seqs
    
    def _read_pairs(self):
        pass
    
    
    
    
                