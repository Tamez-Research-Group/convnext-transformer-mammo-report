"""Cohort assembly, BI-RADS balancing and train/val/test splits (paper Sec. II-E).

Inputs (not distributed; see README):
  * ``features_csv``: one row per image with columns year, patient_id, image_id,
    laterality, view, path, Column1..Column768 (ConvNeXt embeddings,
    produced by ``scripts/extract_features.py``).
  * ``reports_xlsx``: one row per study with year_ID, BIRADS and
    GeneratedConclusion (the BI-RADS-standardised conclusion used as target).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

VIEW_ORDER = [("L", "CC"), ("L", "MLO"), ("R", "CC"), ("R", "MLO")]
VIEW_NAMES = ["LCC", "LMLO", "RCC", "RMLO"]
CONFIGS = {"1vs5": [1, 5], "1245": [1, 2, 4, 5], "all": [1, 2, 3, 4, 5]}
EMBED_COLS = [f"Column{i}" for i in range(1, 769)]


def load_features(features_csv: str) -> dict[str, np.ndarray]:
    """Return {year_ID: array (4, 768)} with views ordered LCC, LMLO, RCC, RMLO."""
    df = pd.read_csv(features_csv)
    df["year_ID"] = df["year"].astype(str) + "_" + df["patient_id"].astype(str)
    order = {v: i for i, v in enumerate(VIEW_ORDER)}
    df["view_idx"] = [order[(lat, view)] for lat, view in zip(df["laterality"], df["view"])]
    df = df.sort_values(["year_ID", "view_idx"], kind="stable")
    counts = df.groupby("year_ID").size()
    complete = counts[counts == 4].index
    df = df[df["year_ID"].isin(complete)]
    feats = df[EMBED_COLS].to_numpy(np.float32).reshape(-1, 4, 768)
    ids = df["year_ID"].to_numpy()[::4]
    return dict(zip(ids, feats))


def load_reports(reports_xlsx: str) -> pd.DataFrame:
    """Studies with a usable BI-RADS (0 = incomplete study is excluded; 4a/4b/4c -> 4)."""
    df = pd.read_excel(reports_xlsx)
    df["BIRADS"] = df["BIRADS"].astype(str).replace({"4a": "4", "4b": "4", "4c": "4"}).astype(int)
    df = df[df["BIRADS"] != 0]
    return df[["year_ID", "BIRADS", "GeneratedConclusion"]].reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    excluded: pd.DataFrame  # studies left out by balancing (full-cohort evaluation)

    def restrict(self, classes: list[int]) -> "Split":
        keep = lambda d: d[d["BIRADS"].isin(classes)].reset_index(drop=True)  # noqa: E731
        return Split(keep(self.train), keep(self.val), keep(self.test), keep(self.excluded))

    def to_frame(self) -> pd.DataFrame:
        parts = [d.assign(split=n) for n, d in
                 [("train", self.train), ("val", self.val), ("test", self.test), ("excluded", self.excluded)]]
        return pd.concat(parts, ignore_index=True)[["year_ID", "BIRADS", "split"]]


def make_split(
    reports: pd.DataFrame,
    n_per_class: int = 356,
    test_frac: float = 0.15,
    val_frac: float = 0.15,
    seed: int = 42,
) -> Split:
    """Balanced 5-class cohort -> stratified hold-out test set -> validation set.

    The balanced set and the test set reproduce the original experiments exactly
    (``prepFeatVecs.py``): the smallest class (BI-RADS 1, n=356) is used whole and
    in file order, the others are subsampled with ``random_state=seed``, and the
    test set is a stratified 15 % split with the same seed.

    Unlike the original runs, early stopping uses a validation set carved out of
    the training data (``val_frac``), so the test set is never seen during
    training. ``val_frac=0`` restores the original behaviour of selecting on the
    test set (only for reproducing the submitted numbers).
    """
    from sklearn.model_selection import train_test_split

    parts = []
    for c in sorted(reports["BIRADS"].unique()):
        d = reports[reports["BIRADS"] == c]
        parts.append(d if len(d) == n_per_class else d.sample(n=n_per_class, random_state=seed))
    balanced = pd.concat(parts, ignore_index=True)
    train, test = train_test_split(balanced, test_size=test_frac, random_state=seed,
                                   stratify=balanced["BIRADS"])
    if val_frac > 0:
        train, val = train_test_split(train, test_size=val_frac, random_state=seed,
                                      stratify=train["BIRADS"])
    else:
        val = test.copy()
    excluded = reports[~reports["year_ID"].isin(balanced["year_ID"])]
    reset = lambda d: d.reset_index(drop=True)  # noqa: E731
    return Split(reset(train), reset(val), reset(test), reset(excluded))


def stack(df: pd.DataFrame, features: dict[str, np.ndarray], views: list[str] | None = None) -> np.ndarray:
    """(N, n_views, 768) array for the studies in ``df`` (optionally a subset of views)."""
    x = np.stack([features[i] for i in df["year_ID"]])
    if views:
        x = x[:, [VIEW_NAMES.index(v) for v in views]]
    return x
