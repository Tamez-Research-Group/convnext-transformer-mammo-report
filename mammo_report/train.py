"""Train the conclusion-generation transformer on precomputed ConvNeXt embeddings.

Example:
    python -m mammo_report.train --features weights2.csv --reports generated_conclusions.xlsx \
        --config all --out runs/all_seed42
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Load transformers (and torch's CUDA DLLs) before TensorFlow: the reverse order
# intermittently crashes at start-up on Windows (conflicting cuBLAS builds).
import transformers  # noqa: F401,E402
import tensorflow as tf
from tensorflow import keras

from .data import CONFIGS, SPLIT_MODES, VIEW_NAMES, load_features, load_reports, make_split, stack
from .generate import decode, greedy_generate, tokenize
from .model import TransformerConfig, build_transformer


class CaptioningModel(keras.Model):
    """Teacher-forced training wrapper (masked cross-entropy over non-pad tokens)."""

    def __init__(self, transformer: keras.Model):
        super().__init__()
        self.transformer = transformer
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.acc_tracker = keras.metrics.Mean(name="acc")
        self.xent = keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction="none")

    def _step(self, x, seq, training):
        inp, target = seq[:, :-1], seq[:, 1:]
        mask = tf.cast(tf.not_equal(target, 0), tf.float32)
        probs = self.transformer({"visual_context": x, "dec_input_ids": inp}, training=training)
        loss = tf.reduce_sum(self.xent(target, probs) * mask) / tf.reduce_sum(mask)
        correct = tf.cast(tf.equal(target, tf.argmax(probs, -1, output_type=tf.int32)), tf.float32)
        acc = tf.reduce_sum(correct * mask) / tf.reduce_sum(mask)
        return loss, acc

    def train_step(self, data):
        x, seq = data
        with tf.GradientTape() as tape:
            loss, acc = self._step(x, seq, True)
        grads = tape.gradient(loss, self.transformer.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.transformer.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, seq = data
        loss, acc = self._step(x, seq, False)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    """Linear warm-up to ``peak`` over ``warmup_steps`` then constant."""

    def __init__(self, peak: float, warmup_steps: int):
        super().__init__()
        self.peak, self.warmup_steps = peak, warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warm = tf.cast(self.warmup_steps, tf.float32)
        return tf.where(step < warm, self.peak * step / warm, self.peak)

    def get_config(self):
        return {"peak": self.peak, "warmup_steps": self.warmup_steps}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def predictions_frame(model, df: pd.DataFrame, x: np.ndarray, cfg: TransformerConfig) -> pd.DataFrame:
    ids, cat = greedy_generate(model, x, max_len=cfg.max_len)
    out = pd.DataFrame({
        "year_ID": df["year_ID"].values,
        "BIRADS": df["BIRADS"].values,
        "reference": decode(tokenize(df["GeneratedConclusion"].tolist(), cfg.max_len)),
        "prediction": decode(ids),
    })
    for i in range(5):
        out[f"p_birads{i + 1}"] = cat[:, i]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True, help="ConvNeXt embeddings CSV (one row per image)")
    p.add_argument("--reports", required=True, help="Reports xlsx with year_ID, BIRADS, GeneratedConclusion")
    p.add_argument("--config", choices=list(CONFIGS), default="all", help="BI-RADS classes used")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42, help="seed for weight init / batching")
    p.add_argument("--split-mode", choices=SPLIT_MODES, default="shared",
                   help="shared = one hold-out for all configurations; per-config = each configuration split on its own")
    p.add_argument("--split-seed", type=int, default=42, help="seed for the cohort split (42 = paper)")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="validation fraction for early stopping; 0 = select on the test set (original runs)")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--views", default=",".join(VIEW_NAMES), help="comma-separated subset of LCC,LMLO,RCC,RMLO")
    p.add_argument("--num-encoders", type=int, default=2, help="0 = ablate the view-encoder")
    p.add_argument("--predict-excluded", action="store_true",
                   help="also generate for all studies excluded by balancing (full-cohort evaluation)")
    a = p.parse_args(argv)

    set_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    views = a.views.split(",")
    cfg = TransformerConfig(num_encoders=a.num_encoders, n_views=len(views))

    feats = load_features(a.features)
    reports = load_reports(a.reports)
    reports = reports[reports["year_ID"].isin(feats)]
    split = make_split(reports, CONFIGS[a.config], mode=a.split_mode, val_frac=a.val_frac, seed=a.split_seed)
    split.to_frame().to_csv(out / "split.csv", index=False)

    def ds(df, shuffle):
        d = tf.data.Dataset.from_tensor_slices((stack(df, feats, views), tokenize(df["GeneratedConclusion"].tolist(), cfg.max_len)))
        if shuffle:
            d = d.shuffle(len(df), seed=a.seed, reshuffle_each_iteration=True)
        return d.batch(a.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds, val_ds = ds(split.train, True), ds(split.val, False)
    steps = len(train_ds) * a.epochs
    transformer = build_transformer(cfg)
    model = CaptioningModel(transformer)
    model.compile(optimizer=keras.optimizers.Adam(WarmupSchedule(a.lr, steps // 15)))
    hist = model.fit(train_ds, validation_data=val_ds, epochs=a.epochs, verbose=2,
                     callbacks=[keras.callbacks.EarlyStopping(patience=a.patience, restore_best_weights=True)])

    transformer.save_weights(str(out / "weights" / "transformer"))  # TF checkpoint format
    cfg.save(out / "model_config.json")
    (out / "train_args.json").write_text(json.dumps(vars(a), indent=2))
    (out / "history.json").write_text(json.dumps({k: [float(v) for v in vs] for k, vs in hist.history.items()}, indent=2))

    predictions_frame(transformer, split.test, stack(split.test, feats, views), cfg).to_csv(out / "predictions_test.csv", index=False)
    if a.predict_excluded and len(split.excluded):
        predictions_frame(transformer, split.excluded, stack(split.excluded, feats, views), cfg).to_csv(
            out / "predictions_excluded.csv", index=False)
    print(f"Saved model and predictions to {out}")


if __name__ == "__main__":
    main()
