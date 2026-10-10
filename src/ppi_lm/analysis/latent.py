"""Latent space: 2D projections of protein / pair vectors and of the amino-acid embeddings."""

import importlib.util

import numpy as np
import torch

from ppi_lm.data_scripts.tokenizer import ProteinTokenizer

# physico-chemical classes of the 20 standard amino acids
AA_CLASSES = {
    "hydrophobic": "AVILM",
    "aromatic": "FWY",
    "polar": "STNQ",
    "positive": "KRH",
    "negative": "DE",
    "special": "CGP",
}
AA_CLASS = {aa: name for name, letters in AA_CLASSES.items() for aa in letters}

# three coarse groups, for scatter plots (at most 3 colors stay distinguishable)
AA_GROUPS = {
    "hydrophobe": "AVILMFWC",
    "polaire": "STNQYGP",
    "chargé": "DEKRH",
}
AA_GROUP = {aa: name for name, letters in AA_GROUPS.items() for aa in letters}

AA_NAMES = {
    "A": "alanine", "C": "cystéine", "D": "aspartate", "E": "glutamate",
    "F": "phénylalanine", "G": "glycine", "H": "histidine", "I": "isoleucine",
    "K": "lysine", "L": "leucine", "M": "méthionine", "N": "asparagine",
    "P": "proline", "Q": "glutamine", "R": "arginine", "S": "sérine",
    "T": "thréonine", "V": "valine", "W": "tryptophane", "Y": "tyrosine",
}  # fmt: skip


def project_2d(x: torch.Tensor | np.ndarray, method: str = "pca", seed: int = 0) -> np.ndarray:
    """(N, d) -> (N, 2) with PCA, t-SNE or UMAP (UMAP needs `pip install umap-learn`).

    PCA keeps global distances (linear); t-SNE and UMAP keep neighbourhoods
    (clusters are meaningful, distances between clusters much less).
    """
    x = np.asarray(x, dtype=np.float32)
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30.0, max(2.0, (len(x) - 1) / 3))
        return TSNE(n_components=2, perplexity=perplexity, random_state=seed).fit_transform(x)
    if method == "umap":
        if importlib.util.find_spec("umap") is None:
            raise ImportError("UMAP needs `pip install umap-learn` (or use method='tsne')")
        import umap

        return umap.UMAP(n_components=2, random_state=seed).fit_transform(x)
    raise ValueError(f"Unknown method '{method}', expected pca, tsne or umap")


def amino_acid_embeddings(
    model: torch.nn.Module, tokenizer: ProteinTokenizer
) -> tuple[list[str], torch.Tensor]:
    """Rows of the learned token embedding for the 20 standard amino acids.

    returns: letters (20,), vectors (20, d_model)
    """
    letters = [aa for aa in tokenizer.AMINO_ACIDS if aa != "X"]
    ids = torch.tensor([tokenizer.token2id[aa] for aa in letters])
    weight = model.encoder.token_embedding.weight.detach().cpu()
    return letters, weight[ids]
