# ppi-language-models

![CI](https://github.com/Felix-Bos/ppi-language-models/actions/workflows/ci.yml/badge.svg)

From-scratch PyTorch implementations of four sequence architectures — a **bidirectional Mamba (SSM)**, a **bidirectional (BERT-style) Transformer encoder**, a **bidirectional LSTM** and a **dilated CNN** — used as protein language models for **protein–protein interaction (PPI) prediction**, on a leakage-free benchmark.

> **Status: work in progress.** Results will be added below as they come in.
>
> - [x] Data pipeline: tokenizer, pair datasets, cropping, augmentation, MLM masking, loaders + tests
> - [ ] Models: Transformer → dilated CNN → BiLSTM → BiMamba
> - [ ] Training loops (MLM, classification) and evaluation
> - [ ] Experiments A and B
> - [ ] Analyses and visualizations

---

## Research question

**Does masked language modeling (MLM) pre-training help a small protein language model predict protein–protein interactions, compared with training directly on the interaction labels — and does the answer depend on the architecture?**

Everything is kept at "toy" scale on purpose: small models (a few million parameters), a single benchmark, and every architecture written by hand, so that the comparison isolates the effect of the architecture and of the training strategy rather than of scale.

---

## Data

### Gold-standard human PPI dataset (Bernett et al.)

Sequence-based PPI prediction is known to suffer from data leakage: many published models reach 95–99 % accuracy only because test proteins (or close homologs) also appear in training. On a leakage-free split, their performance drops close to random.

This project uses the **gold-standard dataset** of Bernett, Blumenthal & List (2024), built to avoid this:

- the human proteome is split so that **no protein is shared** between train, validation and test;
- sequence similarity between splits is minimized (KaHIP partitioning on length-normalized bitscores);
- redundancy is reduced inside each split (CD-HIT, < 40 % pairwise identity);
- each split is balanced, 50 % positive / 50 % negative pairs.

| Split | Files | Pairs | Positives | Negatives |
|---|---|---|---|---|
| Train | `Intra1_pos_rr.txt`, `Intra1_neg_rr.txt` | 163,192 | 81,596 | 81,596 |
| Validation | `Intra0_pos_rr.txt`, `Intra0_neg_rr.txt` | 59,260 | 29,630 | 29,630 |
| Test | `Intra2_pos_rr.txt`, `Intra2_neg_rr.txt` | 52,048 | 26,024 | 26,024 |

Sequences come from `human_swissprot_oneliner.fasta` (20,386 human Swiss-Prot proteins).

**Protein lengths** (whole file): median 415 residues, mean 558, 90th percentile 1,076, maximum 34,350. Most human proteins are longer than a toy model's context, which drives the cropping strategy described below.

For reference, even large pre-trained embeddings (ESM-2) plateau at an accuracy of about **0.65** on this benchmark. Small models trained from scratch are expected to land between chance (0.5) and that ceiling.

### Download

1. Download the dataset from figshare: <https://doi.org/10.6084/m9.figshare.21591618> ("Download all", ~15 MB).
2. Extract it into `data/gold_standard_data/`:

```
data/gold_standard_data/
├── human_swissprot_oneliner.fasta
├── Intra0_pos_rr.txt   Intra0_neg_rr.txt
├── Intra1_pos_rr.txt   Intra1_neg_rr.txt
└── Intra2_pos_rr.txt   Intra2_neg_rr.txt
```

The `data/` folder is git-ignored: the data is not redistributed in this repository.

---

## Installation

```bash
git clone https://github.com/Felix-Bos/ppi-language-models.git
cd ppi-language-models
pip install -e ".[dev]"
```

The project is installed as an editable package (`ppi_lm`), so modules can be imported from anywhere (scripts, tests, notebooks):

```python
from ppi_lm.data_scripts.gold_standard_dataset import GoldStandardDataset
```

---

## Repository structure

```
ppi-language-models/
├── .github/workflows/ci.yml          # lint (ruff) + tests (pytest) on every push
├── data/                             # datasets (not versioned)
├── notebooks/                        # exploration and visualizations
├── src/ppi_lm/
│   ├── data_scripts/
│   │   ├── tokenizer.py              # character-level protein tokenizer
│   │   ├── gold_standard_dataset.py  # file loading, PairDataset, DataLoaders
│   │   └── collators.py              # padding, MLM masking, classification batches
│   └── models/                       # Transformer, CNN, BiLSTM, BiMamba (in progress)
├── tests/                            # unit tests, run on a synthetic mini-dataset
├── pyproject.toml
└── README.md
```

---

## Data pipeline

### 1. Tokenizer

Character-level: one amino acid = one token. The vocabulary is fixed (no training involved) and shared by every stage, so pre-trained weights can be reused without changing the embedding size.

| Tokens | IDs |
|---|---|
| `<PAD>` `<MASK>` `<CLS>` `<SEP>` `<EOS>` | 0–4 |
| 20 standard amino acids | 5–24 |
| `X` (unknown / ambiguous) | 25 |

Rare residues (`U`, `O`, `B`, `Z`) and any unexpected character are mapped to `X`, so that no residue is ever silently dropped. `<PAD>` is 0 so that `nn.Embedding(padding_idx=0)` can be used.

### 2. Pair input format

Both proteins are concatenated into a single sequence, with segment ids telling the model which protein each token belongs to:

```
tokens      <CLS>  A₁ A₂ … Aₙ  <SEP>  B₁ B₂ … Bₘ  <EOS>
segments      0    0  0 …  0     0     1  1 …  1     1
```

**Cropping.** Each protein is limited to `max_protein_length` residues. During training, a random contiguous window is taken (a form of data augmentation, like a random crop on images); during evaluation, the window always starts at the beginning, so that metrics are deterministic.

**Pair-order augmentation.** The input is not symmetric in A and B (positions and segment ids differ, and reading `A,B` backwards is not `B,A`), so a bidirectional model is not automatically order-invariant. The option `augmentation_percent` adds a fraction of the training pairs again in the reversed order `(B, A)`, with a fixed seed for reproducibility. Validation and test are never augmented.

### 3. Collators (from a list of examples to a batch)

- **Padding**: sequences are right-padded to the longest one in the batch, and an `attention_mask` (True on real tokens) is returned so that every architecture can ignore padding — in attention, in mean pooling, and when reversing sequences for the backward pass of bidirectional recurrent models.
- **Classification collator**: returns padded tokens, segments, attention mask and one float label per pair.
- **MLM collator**: masking is recomputed for every batch (dynamic masking):
  - about 15 % of residues are selected, as short **spans** (1–3 residues), which forces the model to use longer-range context — including the partner protein — rather than immediate neighbors;
  - special tokens and padding are never selected;
  - BERT's **80/10/10** rule: 80 % → `<MASK>`, 10 % → random standard amino acid, 10 % → unchanged;
  - labels hold the original token on selected positions and `-100` elsewhere (ignored by `CrossEntropyLoss`).

### 4. Loaders

```python
data = GoldStandardDataset("data/gold_standard_data", max_protein_length=512, batch_size=32)

data.mlm_loader("train")  # positive pairs only, with masking   → pair pre-training
data.classif_loader("train")  # all pairs, with 0/1 labels          → classification
data.classif_loader("val")  # model selection, early stopping
data.classif_loader("test")  # final evaluation only
```

---

## Models

All four encoders are **bidirectional**: each residue is represented using context on both sides, as required by masked language modeling (a causal, left-to-right model could not use the residues after a masked position). They share the same interface — `(input_ids, segment_ids, attention_mask) → one vector per token` — and are sized to a comparable number of parameters.

| Model | Mixing mechanism | Cost in sequence length | Notes |
|---|---|---|---|
| Transformer (BERT-style encoder) | bidirectional self-attention, no causal mask | O(L²) | every token attends to the whole pair, left and right |
| Dilated CNN | stacked dilated convolutions | O(L) | receptive field must cover both proteins |
| BiLSTM | recurrence, both directions | O(L), sequential | backward pass must skip padding |
| BiMamba | selective state-space model, both directions | O(L) | selective scan implemented by hand |

**Heads**
- **MLM head**: linear layer from token vectors to the 26-token vocabulary.
- **Classification head**: mean pooling over real tokens (masked), then a small MLP → one logit. Mean pooling is used rather than the `<CLS>` vector because a `<CLS>` token in position 0 does not see the whole pair equally well in every architecture (e.g. a CNN only sees its receptive field), which would bias the comparison towards the Transformer.

The Transformer is implemented first: it is the best documented of the four and validates the whole pipeline before the others are added.

---

## Experiments

Two training strategies are compared, for each of the four architectures. Everything except the pre-training is identical between them: input format, encoder, classification head, training / validation / test data and metric.

### Experiment A — direct training

1. **Classification**: encoder + MLP head trained together from random initialization, with binary cross-entropy only.
   - Train: `Intra1`, positives + negatives
   - Validation: `Intra0` (hyper-parameters, early stopping)
2. **Test** on `Intra2`.

### Experiment B — masked pre-training, then classification

1. **Single-protein MLM** — input `<CLS> sequence <EOS>`, on the human proteins of the dataset (loader not implemented yet).
2. **Pair MLM** — input `<CLS> A <SEP> B <EOS>`, masking inside both proteins.
   - Train: **positive** pairs of `Intra1` only (81,596) — in a negative pair the partner carries no information about the masked residues, which would dilute the signal.
   - Validation: positive pairs of `Intra0` (perplexity).
3. **Classification** — the pre-trained encoder gets the same MLP head as in A, trained on `Intra1` (positives + negatives), in two variants:
   - **frozen encoder**: only the head is trained → measures the quality of the learned representations;
   - **fine-tuned encoder**: encoder and head trained together, with a lower learning rate for the encoder.
4. **Test** on `Intra2`.

### Leakage safeguards

- Pair MLM and classification only ever use **training** pairs. Validation is used for model selection only; the test set is evaluated once, at the very end.
- Step B.1 sees single sequences only (no interaction information), which is standard practice for protein language models (ESM does the same). To be fully conservative, it can be restricted to training proteins; both variants can be compared.
- The absence of protein overlap between splits is checked by a unit test (on the synthetic mini-dataset in CI; it can be run on the real data locally).

### Fairness of the comparison

B receives more total compute than A (two extra pre-training stages). To make sure a gain comes from what B learns rather than from longer training, A is given at least as many classification epochs as B, and the total training time of each run is reported.

### Results (to be filled)

**Test AUROC**

| Architecture | A (direct) | B, frozen encoder | B, fine-tuned |
|---|---|---|---|
| Transformer | | | |
| Dilated CNN | | | |
| BiLSTM | | | |
| BiMamba | | | |

**Pair MLM (validation)**

| Architecture | Perplexity with true partner | Perplexity with random partner |
|---|---|---|
| Transformer | | |
| Dilated CNN | | |
| BiLSTM | | |
| BiMamba | | |

Reference points for the MLM loss: guessing uniformly among 20 amino acids gives ln(20) ≈ 3.0; predicting amino-acid frequencies alone gives about 2.9. A model has to go clearly below that to be using context.

---

## Planned analyses and visualizations

- **Does the model use the partner?** MLM perplexity on protein A with its true partner vs. with a random protein as B. A lower perplexity with the true partner means the model exploits interaction information.
- **Amino-acid embeddings.** PCA of the learned embedding matrix, colored by physico-chemical class (hydrophobic, charged, polar, aromatic).
- **MLM confusion matrix vs. BLOSUM62.** Which amino acid is predicted instead of the true one, compared with evolutionary substitution rates.
- **Latent space.** UMAP of mean-pooled protein embeddings (colored by length, family, subcellular location) and of pair embeddings (colored by label), for A vs. B.
- **Attention maps** (Transformer): which regions of B each residue of A attends to.
- **Known shortcuts.** Performance as a function of protein length and of node degree (number of partners in training), since previous work showed that PPI models often exploit degree rather than sequence.
- **Order symmetry.** Gap between the predictions for `(A, B)` and `(B, A)`, with and without pair-order augmentation, and test-time averaging of both orders.
- **Effect of cropping.** Test AUROC on the full test set vs. on pairs that fit entirely in the context window.
- **Scaling with length.** Training time and memory per architecture as `max_protein_length` grows (e.g. 256 → 512 → 1000), where linear-time models (CNN, LSTM, Mamba) are expected to scale better than attention.
- Standard curves: training / validation losses, ROC curves.

---

## Development

```bash
pytest -v          # unit tests (synthetic mini-dataset, no real data needed)
ruff check .       # lint
ruff format .      # format
```

Tests cover the tokenizer, file loading, pair construction, cropping, augmentation, padding, MLM masking (never on special tokens or padding, ~15 % selection, spans, 80/10/10 rule, dynamic masking) and the DataLoaders. The GitHub Actions workflow runs ruff and pytest (CPU PyTorch) on every push to `main` and on pull requests.

---

## References

- Bernett, J., Blumenthal, D. B., & List, M. (2024). *Cracking the black box of deep sequence-based protein–protein interaction prediction.* Briefings in Bioinformatics, 25(2), bbae076.
- Bernett, J. (2022). *PPI prediction from sequence, gold standard dataset.* figshare. <https://doi.org/10.6084/m9.figshare.21591618>
- Reim, T., Hartebrodt, A., Blumenthal, D. B., Bernett, J., & List, M. (2025). *Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65.* Bioinformatics.
- Liu, D., et al. (2025). *PLM-interact: extending protein language models to predict protein–protein interactions.* Nature Communications.
- Devlin, J., et al. (2019). *BERT: Pre-training of deep bidirectional transformers for language understanding.* NAACL.
- Gu, A., & Dao, T. (2023). *Mamba: Linear-time sequence modeling with selective state spaces.*
- Lin, Z., et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model (ESM-2).* Science.