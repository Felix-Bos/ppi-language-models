class ProteinTokenizer:
    """Basic tokenizer for protein sequences.
    Maps amino acids to unique integer IDs and vice versa."""

    SPECIAL_TOKENS = ["<PAD>", "<MASK>", "<CLS>", "<SEP>", "<EOS>"]
    AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY") + ["X"]
    RARE_AMINO_ACIDS = "UOBZ"  # sélénocystéine, pyrrolysine -> X

    def __init__(self):
        self.vocab = self.SPECIAL_TOKENS + self.AMINO_ACIDS
        self.token2id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id2token = {idx: token for idx, token in enumerate(self.vocab)}

        self.pad_id = self.token2id["<PAD>"]
        self.cls_id = self.token2id["<CLS>"]
        self.eos_id = self.token2id["<EOS>"]
        self.sep_id = self.token2id["<SEP>"]
        self.mask_id = self.token2id["<MASK>"]
        self.x_id = self.token2id["X"]

        # les rares pointent vers l'id de X (après id2token : le décodage donne "X")
        for aa in self.RARE_AMINO_ACIDS:
            self.token2id[aa] = self.x_id

        self.special_ids = [self.token2id[t] for t in self.SPECIAL_TOKENS]
        self.aa_ids = [self.token2id[a] for a in self.AMINO_ACIDS if a != "X"]

    @property
    def vocab_size(self):
        return len(self.vocab)

    def tokenize(self, sequence: str) -> list[int]:
        # tout caractère inconnu devient X : on ne perd jamais un résidu
        return [self.token2id.get(aa, self.x_id) for aa in sequence.upper()]

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self.id2token[token_id] for token_id in token_ids)
