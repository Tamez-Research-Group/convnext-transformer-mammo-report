"""Confusion-matrix figures of the paper (Figs. 7-8) and of the Supplementary Material.

Uses the representative run of each configuration (per-config split, split seed 42,
model seed 42): ``eval_test`` for the main figures and ``eval_excluded`` for the
full-cohort figures.

Example:
    python scripts/paper_figures.py --runs runs --out paper_figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

CONFIGS = {  # run name -> (classes, title label, main-figure suffix, supplementary suffix)
    "1vs5": ([1, 5], "1 & 5", "1and5", None),
    "1245": ([1, 2, 4, 5], "1,2,4,5", "1245", "w3"),
    "all": ([1, 2, 3, 4, 5], "1,2,3,4,5", "All", "all"),
}


def heatmap(cm, labels, title, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues", annot_kws={"size": 20},
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(xlabel="Predicted labels", ylabel="True labels", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--run", default="split42_seed42")
    p.add_argument("--out", default="paper_figures")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for cfg, (classes, label, main_sfx, supp_sfx) in CONFIGS.items():
        for which, sfx, prefix in [("test", main_sfx, "{kind}{sfx}.png"), ("excluded", supp_sfx, "supp_{kind}_{sfx}.png")]:
            f = Path(a.runs) / "per-config" / cfg / a.run / f"eval_{which}" / "metrics.json"
            if sfx is None or not f.exists():
                continue
            m = json.loads(f.read_text())
            # evaluate.py labels the matrix with sorted(true | predicted), i.e. the configuration's classes
            cm = m["confusion_birads"]
            assert len(cm) == len(classes), f"{f}: unexpected label set"
            name = "Full cohort, " if which == "excluded" else ""
            heatmap(cm, classes, f"{name}BI-RADS {label} Deep Learning Confusion Matrix",
                    out / prefix.format(kind="BIRADS", sfx=sfx))
            heatmap(m["confusion_biopsy"], ["No biopsy", "Biopsy"], f"{name}Biopsy ({label}) Deep Learning Confusion Matrix",
                    out / prefix.format(kind="Biopsy", sfx=sfx))
    print("\n".join(sorted(str(x) for x in out.glob("*.png"))))


if __name__ == "__main__":
    main()
