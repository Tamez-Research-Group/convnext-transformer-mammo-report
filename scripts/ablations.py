"""Component ablations on the same ConvNeXt embeddings and the same split as the transformer.

Because every target conclusion is one of a few BI-RADS-standardised templates,
a direct classifier on the image embeddings + template lookup is the natural
baseline for the Transformer decoder. This script trains logistic-regression
classifiers on:
  * all four views concatenated (4 x 768)
  * the mean of the four views (order-free fusion)
  * each single view
and writes predictions in the same format as ``mammo_report.train`` so they are
scored by exactly the same code (``mammo_report.evaluate``).

Example:
    python scripts/ablations.py --features weights2.csv --reports generated_conclusions.xlsx --config all --out runs/ablations_all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mammo_report.data import CONFIGS, SPLIT_MODES, VIEW_NAMES, load_features, load_reports, make_split, stack  # noqa: E402
from mammo_report.evaluate import evaluate, load_predictions  # noqa: E402
from mammo_report.generate import decode, tokenize  # noqa: E402


def fusion(x: np.ndarray, how: str) -> np.ndarray:
    if how == "concat":
        return x.reshape(len(x), -1)
    if how == "mean":
        return x.mean(1)
    return x[:, VIEW_NAMES.index(how)]


def main(argv=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--config", choices=list(CONFIGS), default="all")
    p.add_argument("--out", required=True)
    p.add_argument("--split-mode", choices=SPLIT_MODES, default="per-config")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--C-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                   help="inverse L2 strengths tried; the best on the validation set is used")
    a = p.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feats = load_features(a.features)
    reports = load_reports(a.reports)
    reports = reports[reports["year_ID"].isin(feats)]
    split = make_split(reports, CONFIGS[a.config], mode=a.split_mode, seed=a.split_seed)
    train = pd.concat([split.train, split.val])  # no early stopping needed
    classes = CONFIGS[a.config]
    templates = (reports.groupby("BIRADS")["GeneratedConclusion"].agg(lambda s: s.value_counts().index[0]))
    tmpl_text = dict(zip(templates.index, decode(tokenize(templates.tolist()))))
    refs = decode(tokenize(split.test["GeneratedConclusion"].tolist()))

    from sklearn.metrics import cohen_kappa_score

    summary = {}
    for how in ["concat", "mean", *VIEW_NAMES]:
        # choose the L2 strength on the validation set, then refit on train + val
        x_tr, x_va = fusion(stack(split.train, feats), how), fusion(stack(split.val, feats), how)
        val_kappa = {}
        for c_reg in a.C_grid:
            m = make_pipeline(StandardScaler(), LogisticRegression(C=c_reg, max_iter=5000)).fit(x_tr, split.train["BIRADS"])
            val_kappa[c_reg] = cohen_kappa_score(split.val["BIRADS"], m.predict(x_va))
        best_c = max(val_kappa, key=val_kappa.get)
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=best_c, max_iter=5000))
        clf.fit(fusion(stack(train, feats), how), train["BIRADS"])
        proba = clf.predict_proba(fusion(stack(split.test, feats), how))
        pred = clf.classes_[proba.argmax(1)]
        df = pd.DataFrame({"year_ID": split.test["year_ID"], "BIRADS": split.test["BIRADS"],
                           "reference": refs, "prediction": [tmpl_text[c] for c in pred]})
        for c in range(1, 6):
            df[f"p_birads{c}"] = proba[:, list(clf.classes_).index(c)] if c in clf.classes_ else 0.0
        path = out / f"predictions_test_logreg_{how}.csv"
        df.to_csv(path, index=False)
        res = evaluate(load_predictions(path), out / f"eval_logreg_{how}", bertscore=False)
        summary[how] = {"C": best_c, "birads_kappa": res["birads_kappa"]["kappa"], "biopsy_kappa": res["biopsy_kappa"]["kappa"],
                        "BLEU": res["text"]["BLEU"], "high_risk_AUPRC": res.get("high_risk", {}).get("AUPRC")}
        print(how, json.dumps(summary[how]))
    (out / "summary.json").write_text(json.dumps({"config": a.config, "classes": classes, **summary}, indent=2))


if __name__ == "__main__":
    main()
