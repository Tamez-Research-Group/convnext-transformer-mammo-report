"""Collect finished runs into the paper's tables and figures.

Designs (all read from ``runs/``):
  shared      one hold-out shared by all configurations, split seed 42, model seeds 42/1/2
  per-config  each configuration split on its own,       split seed 42, model seeds 42/1/2
  repeated    per-config splits with split seeds 42/1/2/3/4, model seed 42

Outputs (``--out``): results_<design>.md/.tex/.csv, full_cohort.md/.csv, and
figures pr_curves_<design>.png, kappa_by_design.png, ablation_<design>.png.

Example:
    python scripts/summarize.py --runs runs --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CONFIG_LABELS = {"1vs5": "BI-RADS 1 & 5", "1245": "BI-RADS 1, 2, 4, 5", "all": "BI-RADS 1–5"}
# Categorical slots 1-3 of the reference palette (fixed order, one per configuration)
CONFIG_COLORS = {"1vs5": "#2a78d6", "1245": "#eb6834", "all": "#1baf7a"}
CONFIG_STYLES = {"1vs5": "-", "1245": "--", "all": "-."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DESIGNS = {
    "shared": ("shared", [(42, 42), (42, 1), (42, 2)]),
    "per-config": ("per-config", [(42, 42), (42, 1), (42, 2)]),
    "repeated": ("per-config", [(42, 42), (1, 42), (2, 42), (3, 42), (4, 42)]),
}
METRICS = [("BLEU", "text.BLEU"), ("ROUGE-L", "text.ROUGE-L"), ("METEOR", "text.METEOR"),
           ("BERTScore", "text.BERTScore_F1"), ("BI-RADS κ", "birads_kappa.kappa"),
           ("Biopsy κ", "biopsy_kappa.kappa"), ("High-risk AUPRC", "high_risk.AUPRC"),
           ("High-risk AUROC", "high_risk.AUROC"), ("High-risk prevalence", "high_risk.prevalence")]


def get(d: dict, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return np.nan
        d = d[k]
    return d


def run_dirs(runs: Path, design: str, cfg: str) -> list[Path]:
    mode, pairs = DESIGNS[design]
    return [runs / mode / cfg / f"split{s}_seed{m}" for s, m in pairs]


def collect(runs: Path, design: str, which: str = "test") -> pd.DataFrame:
    rows = []
    for cfg in CONFIG_LABELS:
        for d in run_dirs(runs, design, cfg):
            f = d / f"eval_{which}" / "metrics.json"
            if not f.exists():
                continue
            m = json.loads(f.read_text())
            rows.append({"config": cfg, "run": d.name, "n": m["n"], **{name: get(m, p) for name, p in METRICS}})
    return pd.DataFrame(rows)


def table(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cfg, g in df.groupby("config", sort=False):
        row = {"Configuration": CONFIG_LABELS[cfg], "n (test)": int(g["n"].iloc[0]), "runs": len(g)}
        for name, _ in METRICS:
            v = g[name].dropna()
            row[name] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else (f"{v.mean():.3f}" if len(v) else "–")
        out.append(row)
    return pd.DataFrame(out)


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def pr_figure(runs: Path, design: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    for cfg in CONFIG_LABELS:
        d = run_dirs(runs, design, cfg)[0]
        df = pd.read_csv(d / "predictions_test.csv")
        y = df["BIRADS"].isin([4, 5]).astype(int).values
        s = (df["p_birads4"] + df["p_birads5"]).values
        p, r, _ = precision_recall_curve(y, s)
        c = CONFIG_COLORS[cfg]
        ax.plot(r, p, color=c, ls=CONFIG_STYLES[cfg], lw=2,
                label=f"{CONFIG_LABELS[cfg]}  AUPRC {average_precision_score(y, s):.2f}")
        ax.axhline(y.mean(), color=c, ls=":", lw=1)  # chance level = prevalence
    ax.set(xlim=(0, 1), ylim=(0, 1.02))
    ax.set_xlabel("Recall (sensitivity) for BI-RADS 4–5", fontsize=8, color=INK)
    ax.set_ylabel("Precision (PPV)", fontsize=8, color=INK)
    style_axes(ax)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False, labelcolor=INK)
    ax.text(0.99, 0.02, "dotted: prevalence (chance)", transform=ax.transAxes, ha="right", fontsize=6, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def kappa_by_design_figure(summaries: dict[str, pd.DataFrame], path: Path) -> None:
    """Dot + SD whisker of BI-RADS κ for each configuration under each split design."""
    import matplotlib.pyplot as plt

    designs = list(summaries)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    offsets = np.linspace(-0.22, 0.22, len(designs))
    markers = {"shared": "o", "per-config": "s", "repeated": "D"}
    for j, cfg in enumerate(CONFIG_LABELS):
        for off, design in zip(offsets, designs):
            v = summaries[design].query("config == @cfg")["BI-RADS κ"].dropna()
            if v.empty:
                continue
            y = j + off
            sd = v.std(ddof=1) if len(v) > 1 else 0.0
            ax.hlines(y, v.mean() - sd, v.mean() + sd, color=CONFIG_COLORS[cfg], lw=1.5, zorder=2)
            ax.plot(v.mean(), y, markers[design], color=CONFIG_COLORS[cfg], ms=6, mec="white", mew=1, zorder=3)
            ax.text(v.mean() + sd + 0.012, y, f"{design}  {v.mean():.2f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(CONFIG_LABELS)), list(CONFIG_LABELS.values()), fontsize=8, color=INK)
    ax.set_xlim(0.3, 1.08)
    ax.invert_yaxis()
    ax.set_xlabel("BI-RADS Cohen's κ (mean ± SD)", fontsize=8, color=INK)
    style_axes(ax)
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def ablation_figure(ablation_dir: Path, transformer: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    s = json.loads((ablation_dir / "summary.json").read_text())
    t = transformer.query("config == 'all'")["BI-RADS κ"]
    names = ["Trans-\nformer", "4 views\nmean", "4 views\nconcat", "LCC\nonly", "LMLO\nonly", "RCC\nonly", "RMLO\nonly"]
    vals = [t.mean(), *(s[k]["birads_kappa"] for k in ["mean", "concat", "LCC", "LMLO", "RCC", "RMLO"])]
    errs = [t.std(ddof=1)] + [0] * 6
    colors = [CONFIG_COLORS["1vs5"]] + ["#b8b7b2"] * 6  # emphasis on the proposed model, neutral baselines
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    x = np.arange(len(vals))
    ax.bar(x, vals, width=0.6, color=colors, edgecolor="white", linewidth=2)
    ax.errorbar(x[0], vals[0], yerr=errs[0], color=INK, elinewidth=1, capsize=0)
    for xi, v, e in zip(x, vals, errs):
        ax.text(xi, v + e + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.set_xticks(x, names, fontsize=6.5, color=INK)
    ax.set_xlabel("grey: logistic regression on the same ConvNeXt embeddings", fontsize=6.5, color=INK2)
    ax.set_ylabel("BI-RADS κ (BI-RADS 1–5)", fontsize=8, color=INK)
    ax.set_ylim(0, max(vals) * 1.18)
    style_axes(ax)
    ax.grid(False, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results")
    p.add_argument("--ablations", help="output dir of scripts/ablations.py (BI-RADS 1-5)")
    a = p.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = {}
    for design in DESIGNS:
        df = collect(runs, design)
        if df.empty:
            continue
        raw[design] = df
        df.to_csv(out / f"runs_{design}.csv", index=False)
        t = table(df)
        t.to_csv(out / f"results_{design}.csv", index=False)
        (out / f"results_{design}.md").write_text(t.to_markdown(index=False), encoding="utf-8")
        (out / f"results_{design}.tex").write_text(t.to_latex(index=False).replace("±", r"$\pm$").replace("κ", r"$\kappa$"),
                                                   encoding="utf-8")
        pr_figure(runs, design, out / f"pr_curves_{design}.png")
        print(f"\n## {design}\n{t.to_markdown(index=False)}")

    kappa_by_design_figure(raw, out / "kappa_by_design.png")

    fc = []
    for mode in ["shared", "per-config"]:
        for cfg in CONFIG_LABELS:
            f = runs / mode / cfg / "split42_seed42" / "eval_excluded" / "metrics.json"
            if f.exists():
                m = json.loads(f.read_text())
                fc.append({"design": mode, "Configuration": CONFIG_LABELS[cfg], "n": m["n"],
                           **{name: round(get(m, pth), 3) for name, pth in METRICS if name != "BERTScore"},
                           "birads_accuracy": round(m["birads_accuracy"], 3),
                           "class counts": m["class_counts"]})
    if fc:
        fc = pd.DataFrame(fc)
        fc.to_csv(out / "full_cohort.csv", index=False)
        (out / "full_cohort.md").write_text(fc.to_markdown(index=False), encoding="utf-8")
        print(f"\n## full cohort (studies excluded by balancing)\n{fc.to_markdown(index=False)}")

    if a.ablations and "shared" in raw:
        ablation_figure(Path(a.ablations), raw["shared"], out / "ablation_shared.png")


if __name__ == "__main__":
    main()
