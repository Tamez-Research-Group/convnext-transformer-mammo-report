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
| `scripts/` | Feature extraction, prediction, experiments, summary tables/figures, baselines, compute benchmark |
| `convnext_training/` | ConvNeXt-S training (mmcls), adapted from the [RSNA 2nd-place solution](https://github.com/ShuzhiLiu/RSNABreast2ndPlace) |
| `assets/average_cdf.npy` | Reference CDF for histogram matching |

## Installation

```bash
pip install -r requirements.txt
```
Training the ConvNeXt encoder needs the separate stack in `convnext_training/requirements-mmcls.txt`.

## Pretrained weights

The trained models are on Hugging Face:
[lalodeavilaarm/convnext-transformer-mammo-report](https://huggingface.co/lalodeavilaarm/convnext-transformer-mammo-report).

```bash
huggingface-cli download lalodeavilaarm/convnext-transformer-mammo-report --local-dir weights
```

| Path | Content |
|---|---|
| `convnext/trained_model.pth` | ConvNeXt-S image encoder |
| `transformer/all/` | Conclusion generator, BI-RADS 1–5 (general use) |
| `transformer/1245/`, `transformer/1vs5/` | Generators for the other two configurations of the paper |

## Generate conclusions for new studies

```bash
# images.csv: year, patient_id, image_id, laterality (L/R), view (CC/MLO), path (DICOM)
python scripts/extract_features.py --images images.csv --checkpoint weights/convnext/trained_model.pth --out features.csv
python scripts/predict.py --features features.csv --transformer weights/transformer/all --out conclusions.csv
```

Each study needs the four views (LCC, LMLO, RCC, RMLO). The output has the generated
conclusion, its BI-RADS category and the model's probabilities for BI-RADS 1–5.

## Reproduce the paper

The TecSalud mammograms and reports are private patient data and are not distributed.
Training needs `features.csv` (one row per image: `year, patient_id, image_id, laterality, view, path, Column1..Column768`)
and `reports.xlsx` (one row per study: `year_ID, BIRADS, GeneratedConclusion`).

```bash
# 1. Train and evaluate one model
python -m mammo_report.train --features features.csv --reports reports.xlsx --config all --out runs/all
python -m mammo_report.evaluate runs/all/predictions_test.csv --out runs/all/eval

# 2. All experiments (3 configurations x 3 seeds), tables and figures
python scripts/run_experiments.py --features features.csv --reports reports.xlsx --bertscore
python scripts/summarize.py --runs runs --out results
python scripts/paper_figures.py --runs runs --out paper_figures

# 3. Baselines and computational cost
python scripts/ablations.py --features features.csv --reports reports.xlsx --config all --out runs/ablations
python scripts/benchmark.py --checkpoint weights/convnext/trained_model.pth --transformer weights/transformer/all --out benchmark.json
```

`--config` selects the BI-RADS categories: `1vs5`, `1245` or `all` (1–5).
Run any script with `--help` for all options.

## Citation

*To be added on publication.*

## License

[MIT](LICENSE).

## Acknowledgements

Secihti and the Microsoft AI for Good Lab (cloud computing).
