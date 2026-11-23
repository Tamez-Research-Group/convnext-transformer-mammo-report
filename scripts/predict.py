"""Generate report conclusions for new studies with a trained (or downloaded) model.

Input: the features CSV written by ``scripts/extract_features.py`` (4 views per study:
LCC, LMLO, RCC, RMLO; studies with missing views are skipped). Output: one row per
study with the generated conclusion, the BI-RADS category written in it, and the
model's probabilities for BI-RADS 1-5.

Example (weights from https://huggingface.co/lalodeavilaarm/convnext-transformer-mammo-report):
    python scripts/extract_features.py --images images.csv --checkpoint convnext/trained_model.pth --out features.csv
    python scripts/predict.py --features features.csv --transformer transformer/all --out conclusions.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mammo_report.generate import get_tokenizer  # noqa: E402

get_tokenizer()  # load transformers/torch before TensorFlow initialises the GPU (Windows DLL order)
import tensorflow as tf  # noqa: E402

from mammo_report.data import load_features  # noqa: E402
from mammo_report.evaluate import extract_birads  # noqa: E402
from mammo_report.generate import decode, greedy_generate  # noqa: E402
from mammo_report.model import TransformerConfig, build_transformer  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="features CSV from scripts/extract_features.py")
    p.add_argument("--transformer", required=True, help="directory with model_config.json and weights/")
    p.add_argument("--out", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    a = p.parse_args(argv)

    for g in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(g, True)
    run = Path(a.transformer)
    cfg = TransformerConfig.load(run / "model_config.json")
    model = build_transformer(cfg)
    model.load_weights(str(run / "weights" / "transformer")).expect_partial()

    feats = load_features(a.features)
    if not feats:
        sys.exit("No study has all four views (LCC, LMLO, RCC, RMLO).")
    ids = list(feats)
    ids_tok, probs = greedy_generate(model, np.stack([feats[i] for i in ids]), cfg.max_len, a.batch_size)
    text = [t.replace("[CLS]", "").replace("[SEP]", "").strip() for t in decode(ids_tok)]
    out = pd.DataFrame({"study_id": ids, "conclusion": text, "birads": [extract_birads(t) for t in text]})
    for i in range(5):
        out[f"p_birads{i + 1}"] = probs[:, i]
    out.to_csv(a.out, index=False)
    print(f"Wrote {len(out)} conclusions to {a.out}")


if __name__ == "__main__":
    main()
