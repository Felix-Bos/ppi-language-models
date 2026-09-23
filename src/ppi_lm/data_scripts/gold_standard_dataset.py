import os

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer


class GoldStandardDataset:
    def __init__(
        self,
        data_dir: str = "./data/gold_standard_data",
        tokenizer: ProteinTokenizer = ProteinTokenizer,
        batch_size: int = 32,
        max_protein_length: int = 512,
        mask_prob: float = 0.15,
        num_workers: int = 2,
    ):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_protein_length = max_protein_length
        self.mask_prob = mask_prob
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _read_fasta(self, file_path: str):
        with open(file_path) as f:
            for line in f:
                print(line.strip())


if __name__ == "__main__":
    dataset = GoldStandardDataset()
    dataset._read_fasta(os.path.join(dataset.data_dir, "human_swissprot_oneliner.fasta"))
