"""Run the training / evaluation grid used in the paper.

Each run is trained and evaluated in its own process and written to
``runs/<mode>/<config>/split<S>_seed<M>``. Finished runs are skipped, so the
grid can be resumed.

Example (everything reported in the paper):
    python scripts/run_experiments.py --features weights2.csv --reports reports.xlsx \
        --modes per-config --split-seeds 42 --model-seeds 42 1 2   # paper (add "shared" for the shared hold-out)
    python scripts/run_experiments.py ... --modes per-config --split-seeds 1 2 3 4 --model-seeds 42
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--reports", required=True)
    p.add_argument("--runs", default=str(ROOT / "runs"))
    p.add_argument("--modes", nargs="+", default=["per-config"])
    p.add_argument("--configs", nargs="+", default=["1vs5", "1245", "all"])
    p.add_argument("--split-seeds", nargs="+", type=int, default=[42])
    p.add_argument("--model-seeds", nargs="+", type=int, default=[42, 1, 2])
    p.add_argument("--excluded-for", default="split42_seed42",
                   help="run name for which the full cohort (excluded studies) is also predicted")
    p.add_argument("--bertscore", action="store_true", help="also compute BERTScore (slower)")
    a = p.parse_args(argv)

    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "2"}
    for mode in a.modes:
        for cfg in a.configs:
            for ss in a.split_seeds:
                for ms in a.model_seeds:
                    name = f"split{ss}_seed{ms}"
                    out = Path(a.runs) / mode / cfg / name
                    if (out / "eval_test" / "metrics.json").exists():
                        print(f"skip {mode}/{cfg}/{name}")
                        continue
                    out.mkdir(parents=True, exist_ok=True)
                    cmd = [sys.executable, "-m", "mammo_report.train", "--features", a.features, "--reports", a.reports,
                           "--config", cfg, "--split-mode", mode, "--split-seed", str(ss), "--seed", str(ms),
                           "--out", str(out)]
                    if name == a.excluded_for:
                        cmd.append("--predict-excluded")
                    with open(out / "train.log", "w") as log:
                        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
                    if rc:
                        print(f"FAILED {mode}/{cfg}/{name} (exit {rc}), see {out / 'train.log'}")
                        continue
                    for which in ["test", "excluded"]:
                        pred = out / f"predictions_{which}.csv"
                        if pred.exists():
                            ev = [sys.executable, "-m", "mammo_report.evaluate", str(pred), "--out", str(out / f"eval_{which}")]
                            if not a.bertscore or which == "excluded":
                                ev.append("--no-bertscore")
                            subprocess.run(ev, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"done {mode}/{cfg}/{name}", flush=True)


if __name__ == "__main__":
    main()
