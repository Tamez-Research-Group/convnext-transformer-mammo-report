# ConvNeXt-Transformer conclusion generation from mammography

Official code for *"A ConvNeXt-Transformer Approach for Automated Conclusion Generation from
Mammography"* (Bosques-Palomo, de Avila-Armenta, *et al.*, IEEE JBHI).

Given the four screening views of a study (LCC, LMLO, RCC, RMLO), the model writes the
conclusion of the radiology report: the BI-RADS category and its follow-up or biopsy
recommendation. A ConvNeXt-Small network encodes each view, and a Transformer
encoder-decoder generates the text.

```
4 DICOM views ──► preprocessing ──► ConvNeXt-S (shared) ──► 4 × 768 view embeddings
                  (flip, normalise,      1280×1280 input         │  + learned view embedding
                   0.1 mm, histogram                             ▼
                   matching to RSNA CDF)               Transformer encoder (2 layers, self-attention across views)
                                                                 │  keys / values
                                                                 ▼
                                    [CLS] ─► Transformer decoder (2 layers, causal self-attn + cross-attn) ─► "Conclusion: BI-RADS category 4. ... Biopsy ... is recommended."
```

## Repository

| Path | Content |
|---|---|
| `mammo_report/` | Preprocessing, ConvNeXt encoder, data split, Transformer, training, generation, evaluation |
| `scripts/` | Feature extraction, experiments, summary tables/figures, baselines, compute benchmark |
| `convnext_training/` | ConvNeXt-S training (mmcls), adapted from the [RSNA 2nd-place solution](https://github.com/ShuzhiLiu/RSNABreast2ndPlace) |
| `assets/average_cdf.npy` | Reference CDF for histogram matching |

## Installation

```bash
pip install -r requirements.txt
```
Training the ConvNeXt encoder needs the separate stack in `convnext_training/requirements-mmcls.txt`.

## Data and weights

The TecSalud mammograms and reports are private patient data and are not distributed.

| File | Description | Availability |
|---|---|---|
| `trained_model.pth` | ConvNeXt-S checkpoint | *TODO: link* |
| `features.csv` | One row per image: `year, patient_id, image_id, laterality, view, path, Column1..Column768` | on request |
| `reports.xlsx` | One row per study: `year_ID, BIRADS, GeneratedConclusion` | on request |

## Usage

```bash
# 1. Image embeddings (DICOM or preprocessed TIFF)
python scripts/extract_features.py --images images.csv --checkpoint trained_model.pth --out features.csv

# 2. Train and evaluate one model
python -m mammo_report.train --features features.csv --reports reports.xlsx --config all --out runs/all
python -m mammo_report.evaluate runs/all/predictions_test.csv --out runs/all/eval

# 3. All experiments, tables and figures
python scripts/run_experiments.py --features features.csv --reports reports.xlsx --bertscore
python scripts/summarize.py --runs runs --out results

# 4. Baselines and computational cost
python scripts/ablations.py --features features.csv --reports reports.xlsx --config all --out runs/ablations
python scripts/benchmark.py --checkpoint trained_model.pth --transformer runs/all --out benchmark.json
```

`--config` selects the BI-RADS categories: `1vs5`, `1245` or `all` (1–5).
Run any script with `--help` for all options.

## Citation

*To be added on publication.*

## License

*TODO.* `convnext_training/` includes MIT-licensed code from ShuzhiLiu/RSNABreast2ndPlace
(see `convnext_training/LICENSE-RSNABreast2ndPlace`).

## Acknowledgements

Secihti and the Microsoft AI for Good Lab (cloud computing).
