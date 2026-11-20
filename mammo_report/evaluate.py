"""Evaluation of generated conclusions (paper Sec. III-IV).

Input: a predictions CSV with columns ``reference``, ``prediction`` and,
optionally, ``p_birads1..5`` (written by ``mammo_report.train``). Also accepts the
original ``results*.xlsx`` files (columns 'True Captions' / 'Predicted Caption').

Example:
    python -m mammo_report.evaluate runs/all_seed42/predictions_test.csv --out runs/all_seed42/eval
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:  # On Windows, torch's OpenMP runtime must load before numpy/sklearn's or BERTScore crashes silently.
    import torch  # noqa: F401
except ImportError:
    pass
import numpy as np
import pandas as pd

BIRADS_RE = re.compile(r"bi\s*-\s*rads category (\d)")
HIGH_RISK = (4, 5)


# --------------------------------------------------------------------- parsing
def extract_birads(text: str) -> int:
    """BI-RADS category written in a conclusion (0 if none could be parsed)."""
    m = BIRADS_RE.search(str(text).lower())
    return int(m.group(1)) if m else 0


def biopsy_from_birads(cat: np.ndarray) -> np.ndarray:
    """Biopsy recommendation as defined in the paper: BI-RADS 4 or 5."""
    return np.isin(cat, HIGH_RISK).astype(int)


def load_predictions(path: str) -> pd.DataFrame:
    if str(path).endswith(".xlsx"):
        df = pd.read_excel(path).rename(columns={"Patient ID": "year_ID", "True Captions": "reference",
                                                 "Predicted Caption": "prediction"})
    else:
        df = pd.read_csv(path)
    df["birads_true"] = df["reference"].map(extract_birads)
    df["birads_pred"] = df["prediction"].map(extract_birads)
    return df


# ---------------------------------------------------------------- text metrics
def text_metrics(refs: list[str], preds: list[str], bertscore: bool = True) -> dict:
    """Corpus BLEU (sacreBLEU), ROUGE-L F (bootstrap mid), mean METEOR, mean BERTScore F1.

    Strings are scored as written by the original evaluator, i.e. including the
    '[CLS]'/'[SEP]' markers; use ``--strip-special`` to remove them.
    """
    import sacrebleu
    from nltk import word_tokenize
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer, scoring

    out = {"BLEU": sacrebleu.corpus_bleu(preds, [refs]).score / 100}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    agg = scoring.BootstrapAggregator()
    np.random.seed(0)
    for r, p in zip(refs, preds):
        agg.add_scores(scorer.score(r, p))
    rl = agg.aggregate()["rougeL"]
    # NB: the submitted manuscript reported the *low* bootstrap bound (0.765) as ROUGE-L.
    out["ROUGE-L"] = rl.mid.fmeasure
    out["ROUGE-L_ci95"] = [rl.low.fmeasure, rl.high.fmeasure]
    out["METEOR"] = float(np.mean([meteor_score([word_tokenize(r)], word_tokenize(p)) for r, p in zip(refs, preds)]))
    if bertscore:
        from bert_score import score

        _, _, f1 = score(preds, refs, model_type="emilyalsentzer/Bio_ClinicalBERT", num_layers=9, verbose=False)
        out["BERTScore_F1"] = float(f1.mean())
    return out


# ------------------------------------------------------------- agreement stats
def kappa_ci(y_true, y_pred, n_boot: int = 1000, seed: int = 0) -> dict:
    """Cohen's kappa with a percentile bootstrap 95 % CI."""
    from sklearn.metrics import cohen_kappa_score

    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(np.unique(y_true)) < 2:  # undefined when the reference has a single class
        return {"kappa": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        k = cohen_kappa_score(y_true[i], y_pred[i])
        if not np.isnan(k):
            boots.append(k)
    return {"kappa": float(cohen_kappa_score(y_true, y_pred)),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


def high_risk_operating_points(y: np.ndarray, score: np.ndarray) -> list[dict]:
    """Sensitivity / specificity / PPV / NPV of 'BI-RADS 4-5' at several thresholds."""
    rows = []
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5]:
        pred = score >= thr
        tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
        tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
        rows.append({"threshold": thr, "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
                     "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "missed_high_risk": fn})
    return rows


def risk_metrics(df: pd.DataFrame, out: Path | None) -> dict:
    """AUPRC / AUROC for high-risk (BI-RADS 4-5) and per-class one-vs-rest."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

    pcols = [f"p_birads{i}" for i in range(1, 6)]
    if not set(pcols) <= set(df.columns) or df[pcols].isna().all().all():
        return {}
    d = df.dropna(subset=pcols)
    present = sorted(d["birads_true"].unique())
    res = {"n_scored": int(len(d))}
    y_hr = np.isin(d["birads_true"], HIGH_RISK).astype(int)
    s_hr = d["p_birads4"].values + d["p_birads5"].values
    if 0 < y_hr.sum() < len(y_hr):
        res["high_risk"] = {"prevalence": float(y_hr.mean()), "AUPRC": float(average_precision_score(y_hr, s_hr)),
                            "AUROC": float(roc_auc_score(y_hr, s_hr)),
                            "operating_points": high_risk_operating_points(y_hr, s_hr)}
    res["per_class"] = {}
    curves = {"BI-RADS 4-5": (y_hr, s_hr)} if "high_risk" in res else {}
    for c in present:
        y = (d["birads_true"] == c).astype(int).values
        s = d[f"p_birads{c}"].values
        if 0 < y.sum() < len(y):
            res["per_class"][str(c)] = {"prevalence": float(y.mean()), "AUPRC": float(average_precision_score(y, s)),
                                        "AUROC": float(roc_auc_score(y, s))}
            if c in HIGH_RISK:
                curves[f"BI-RADS {c}"] = (y, s)
    if out is not None and curves:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4.5, 4))
        for name, (y, s) in curves.items():
            p, r, _ = precision_recall_curve(y, s)
            ax.plot(r, p, label=f"{name} (AUPRC={average_precision_score(y, s):.2f}, prev={y.mean():.2f})")
            ax.axhline(y.mean(), ls=":", lw=0.8, color=ax.lines[-1].get_color())
        ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title="High-risk BI-RADS")
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        fig.savefig(out / "pr_curves_high_risk.png", dpi=200)
        plt.close(fig)
    return res


def confusion_plot(y_true, y_pred, labels, title, path: Path) -> list[list[int]]:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Model", ylabel="Radiologist", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return cm.tolist()


# ------------------------------------------------------------------------ main
def evaluate(df: pd.DataFrame, out: Path | None = None, bertscore: bool = True, strip_special: bool = False) -> dict:
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    refs, preds = df["reference"].astype(str).tolist(), df["prediction"].astype(str).tolist()
    if strip_special:
        clean = lambda s: s.replace("[CLS]", "").replace("[SEP]", "").strip()  # noqa: E731
        refs, preds = [clean(s) for s in refs], [clean(s) for s in preds]
    labels = sorted(set(df["birads_true"]) | set(df["birads_pred"]))
    res = {"n": int(len(df)),
           "class_counts": {str(k): int(v) for k, v in df["birads_true"].value_counts().sort_index().items()},
           "unparseable_predictions": int((df["birads_pred"] == 0).sum()),
           "text": text_metrics(refs, preds, bertscore),
           "birads_accuracy": float((df["birads_true"] == df["birads_pred"]).mean()),
           "birads_kappa": kappa_ci(df["birads_true"], df["birads_pred"]),
           "biopsy_kappa": kappa_ci(biopsy_from_birads(df["birads_true"]), biopsy_from_birads(df["birads_pred"]))}
    res.update(risk_metrics(df, out))
    if out is not None:
        res["confusion_birads"] = confusion_plot(df["birads_true"], df["birads_pred"], labels, "BI-RADS", out / "cm_birads.png")
        res["confusion_biopsy"] = confusion_plot(biopsy_from_birads(df["birads_true"]),
                                                 biopsy_from_birads(df["birads_pred"]), [0, 1], "Biopsy", out / "cm_biopsy.png")
        (out / "metrics.json").write_text(json.dumps(res, indent=2))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("predictions", help="predictions_*.csv or original results*.xlsx")
    p.add_argument("--out", help="directory for metrics.json and figures")
    p.add_argument("--classes", help="restrict to these BI-RADS classes, e.g. 1,5")
    p.add_argument("--no-bertscore", action="store_true")
    p.add_argument("--strip-special", action="store_true", help="remove [CLS]/[SEP] before text metrics")
    a = p.parse_args(argv)
    df = load_predictions(a.predictions)
    if a.classes:
        df = df[df["birads_true"].isin([int(c) for c in a.classes.split(",")])].reset_index(drop=True)
    res = evaluate(df, Path(a.out) if a.out else None, bertscore=not a.no_bertscore, strip_special=a.strip_special)
    summary = {k: res[k] for k in ["n", "text", "birads_kappa", "biopsy_kappa"]}
    if "high_risk" in res:
        summary["high_risk_AUPRC"] = res["high_risk"]["AUPRC"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
