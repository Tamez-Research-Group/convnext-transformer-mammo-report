# ConvNeXt-Transformer conclusion generation from mammography

Code for *"A ConvNeXt-Transformer Approach for Automated Conclusion Generation from
Mammography"* (Bosques-Palomo, de Avila-Armenta, *et al.*, IEEE JBHI, under revision,
JBHI-02890-2025).

The model takes the four standard screening views of a study (LCC, LMLO, RCC, RMLO),
encodes each with a ConvNeXt-Small network, and writes the report conclusion with a
Transformer encoder-decoder. The conclusion includes the BI-RADS category and the
follow-up or biopsy recommendation.

```
4 DICOM views ──► preprocessing ──► ConvNeXt-S (shared) ──► 4 × 768 view embeddings
                  (flip, normalise,      1280×1280 input         │  + learned view embedding
                   0.1 mm, histogram                             ▼
                   matching to RSNA CDF)               Transformer encoder (2 layers, self-attention across views)
                                                                 │  keys / values
                                                                 ▼
                                    [CLS] ─► Transformer decoder (2 layers, causal self-attn + cross-attn) ─► "Conclusion: BI-RADS category 4. ... Biopsy ... is recommended."
```

## Repository layout

| Path | Content |
|---|---|
| `mammo_report/preprocessing.py` | Image preprocessing (Sec. II-B) |
| `mammo_report/convnext.py` | Plain-PyTorch ConvNeXt-S that loads the mmcls checkpoint, so mmcv is not needed for inference |
| `mammo_report/data.py` | Cohort, BI-RADS balancing, train / validation / hold-out split (Sec. II-E) |
| `mammo_report/model.py` | Transformer encoder-decoder (TensorFlow/Keras) |
| `mammo_report/train.py` | Training CLI (teacher forcing, masked cross-entropy, warm-up, early stopping) |
| `mammo_report/generate.py` | Tokenisation (bert-base-uncased) and greedy generation, plus the BI-RADS digit probabilities |
| `mammo_report/evaluate.py` | BLEU, ROUGE-L, METEOR, BERTScore (Bio-ClinicalBERT), Cohen's κ with bootstrap CI, confusion matrices, PR curves / AUPRC |
| `scripts/extract_features.py` | DICOM/TIFF → 768-d ConvNeXt embeddings |
| `scripts/ablations.py` | Classifier and single-view baselines on the same embeddings and split |
| `scripts/benchmark.py` | Parameters, model size, GPU memory, end-to-end latency per study |
| `convnext_training/` | ConvNeXt-S training (mmcls 1.0.0rc5), adapted from the [RSNA 2nd-place solution](https://github.com/ShuzhiLiu/RSNABreast2ndPlace) (MIT) |
| `assets/average_cdf.npy` | Reference CDF (100 high-contrast RSNA images) used for histogram matching |

## Installation

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -r requirements.txt
```
ConvNeXt *training* needs the separate, pinned mmcls stack in `convnext_training/requirements-mmcls.txt`.
Feature extraction and everything downstream do not.

## Data and weights

The TecSalud mammograms and reports are private patient data. They were collected under
ethics protocol P000542-MIRAI-MODIFICADO-CEIC-CR002 and are not distributed. The RSNA
data is available from the
[RSNA Screening Mammography Breast Cancer Detection challenge](https://www.kaggle.com/competitions/rsna-breast-cancer-detection).

| File | Description | Availability |
|---|---|---|
| `trained_model.pth` | ConvNeXt-S checkpoint (RSNA + TecSalud 2014–2018), 198 MB, SHA-256 `96dccd16…6ec0f` | *TODO: public link* |
| `features.csv` | One row per image: `year, patient_id, image_id, laterality, view, path, Column1..Column768` | on request (de-identified) |
| `reports.xlsx` | One row per study: `year_ID, BIRADS, GeneratedConclusion` | on request (de-identified) |

`GeneratedConclusion` is the radiologist's conclusion standardised to the BI-RADS
category and its recommendation (7 distinct sentences, see the paper). That makes the
task close to a 5-class BI-RADS problem. The classifier baselines in `scripts/ablations.py`
are reported for that reason.

## Usage

```bash
# 1) image embeddings (DICOMs are preprocessed on the fly)
python scripts/extract_features.py --images images.csv --checkpoint trained_model.pth --out features.csv

# 2) train + evaluate the three configurations (1vs5, 1245, all) under each split design;
#    finished runs are skipped, so the command can be re-run to resume
python scripts/run_experiments.py --features features.csv --reports reports.xlsx \
    --modes shared per-config --split-seeds 42 --model-seeds 42 1 2 --bertscore
python scripts/run_experiments.py --features features.csv --reports reports.xlsx \
    --modes per-config --split-seeds 1 2 3 4 --model-seeds 42 --bertscore   # repeated splits

# tables (Markdown / LaTeX / CSV) and figures, incl. the full-cohort evaluation
python scripts/summarize.py --runs runs --out results --ablations runs/ablations_all

# a single run by hand
python -m mammo_report.train --features features.csv --reports reports.xlsx \
    --config all --split-mode per-config --split-seed 42 --seed 42 --predict-excluded --out runs/example
python -m mammo_report.evaluate runs/example/predictions_test.csv --out runs/example/eval_test

# 3) baselines / ablations and computational cost
python scripts/ablations.py --features features.csv --reports reports.xlsx --config all --out runs/ablations_all
python scripts/benchmark.py --checkpoint trained_model.pth --transformer runs/all_seed42 --out benchmark.json
python scripts/benchmark.py --checkpoint trained_model.pth --transformer runs/all_seed42 --device cpu --out benchmark_cpu.json
```

### Protocol
* Balanced cohort: 356 studies per BI-RADS category (the size of the smallest class), from 2014–2019 TecSalud studies with all four views.
* Split designs (`--split-mode`):
  * `per-config` (original experiments): each configuration is balanced and split 85/15 on its own. With split seed 42 this reproduces the original training sets exactly. Test sets have 107 / 214 / 267 studies.
  * `shared`: one 15 % hold-out of the 5-class balanced cohort (267 studies, identical to the original BI-RADS 1–5 test set). The 1 & 5 and 1, 2, 4, 5 configurations use its subsets (107 and 214 studies).
  * repeated: `per-config` with split seeds 42, 1, 2, 3, 4. This measures how much the metrics depend on which studies end up in the test set.
* Validation: a stratified 15 % of the training data is used for early stopping (patience 3). The test set is never used for model selection.
* Transformer: d = 768, FFN 1024, 1 encoder head, 2 decoder heads, 2 + 2 layers, dropout 0.2, learned positional embeddings, BERT WordPiece vocabulary (30,522 tokens), max 40 tokens, Adam with linear warm-up to 1e-4, batch 64, ≤ 80 epochs.
* Biopsy recommendation = BI-RADS 4 or 5. κ 95 % CIs: 1,000 bootstrap resamples.

## Results (hold-out test set, mean ± SD over 3 seeds)

| Configuration | n | BLEU | ROUGE-L | METEOR | BI-RADS κ | Biopsy κ | High-risk AUPRC |
|---|---|---|---|---|---|---|---|
| BI-RADS 1 & 5 | 107 | 0.951 ± 0.004 | 0.960 ± 0.003 | 0.973 ± 0.002 | 0.84 ± 0.01 | 0.84 ± 0.01 | 0.97 ± 0.01 |
| BI-RADS 1, 2, 4, 5 | 214 | 0.838 ± 0.020 | 0.876 ± 0.014 | 0.911 ± 0.010 | 0.51 ± 0.02 | 0.64 ± 0.07 | 0.91 ± 0.01 |
| BI-RADS 1–5 | 267 | 0.772 ± 0.004 | 0.802 ± 0.003 | 0.853 ± 0.002 | 0.44 ± 0.02 | 0.59 ± 0.01 | 0.85 ± 0.01 |

High-risk = BI-RADS 4–5 vs. rest, scored by P(4) + P(5) at the step where the category is generated.

**Ablation (BI-RADS 1–5, same embeddings and split):**

| Model | BI-RADS κ | Biopsy κ | High-risk AUPRC |
|---|---|---|---|
| Transformer, 4 views | 0.44 ± 0.02 | 0.59 ± 0.01 | 0.85 |
| Logistic regression, mean of 4 views | 0.39 | 0.58 | 0.87 |
| Logistic regression, 4 views concatenated | 0.38 | 0.53 | 0.85 |
| Logistic regression, single view | 0.28–0.32 | 0.40–0.48 | 0.74–0.77 |

**Computational cost per study (4 views), RTX 4090 / i9 (24 cores):**

| Stage | Parameters | Size | Peak GPU memory | GPU time | CPU-only time |
|---|---|---|---|---|---|
| Preprocessing (4 views, sequential) | – | – | – | 1.44 s (CPU) | 1.35 s |
| ConvNeXt-S, 4 × 1280² | 49.5 M | 189 MB | 1.9 GB (fp32) / 1.3 GB (fp16) | 109 ms (fp32) / 55 ms (fp16) | 4.0 s |
| Transformer, greedy generation | 76.9 M | 293 MB | 0.35 GB | 330 ms | 1.16 s |
| **Total** | **126.3 M** | **≈ 480 MB** | **< 2 GB** | **≈ 1.9 s** | **≈ 6.5 s** |

## Differences from the submitted manuscript

The original experiments were run from notebooks. This repository re-implements them and
was checked against the original outputs. The transformer gives identical outputs given
the original weights, the hold-out set has the same 267 studies in the same order, and
the BLEU, METEOR and κ of the original BI-RADS 1–5 predictions are reproduced. The
following differences matter for the revision:

1. **Model selection.** The original runs used the hold-out set for early stopping. Here a separate validation split is used. `--val-frac 0` restores the original behaviour.
2. **ROUGE-L.** The manuscript reported the lower bootstrap bound (0.765) rather than the point estimate (0.79 for the original predictions).
3. **Positional encoding** is learned (`Embedding`), not sinusoidal. The **vocabulary** has 30,522 tokens (bert-base-uncased), not 3,052. Classes have **356** studies, not 365.
4. **Pixel spacing.** The published features were computed with a fixed 0.065 mm source spacing because the DICOM tag was never parsed. This is kept as the default for consistency, and `--use-dicom-spacing` uses the real value.
5. **Inference time.** The manuscript's 85 ms per study does not cover the full pipeline. The end-to-end figure, including preprocessing and generation, is ≈ 1.9 s on the same GPU.

## Citation
*To be added on acceptance.*

## License
*TODO: choose a license.* `convnext_training/` includes code under the MIT license from
ShuzhiLiu/RSNABreast2ndPlace (see `convnext_training/LICENSE-RSNABreast2ndPlace`).

## Acknowledgements
Secihti and the Microsoft AI for Good Lab (cloud computing). ConvNeXt training code by
M. A. Monsivais-Molina, based on the RSNA 2nd-place solution. Preprocessing by
B. Bosques-Palomo and J. G. Tamez-Peña.
