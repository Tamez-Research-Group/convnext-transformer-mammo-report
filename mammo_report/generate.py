"""Tokenisation and autoregressive (greedy) conclusion generation."""
from __future__ import annotations

import re

import numpy as np
import tensorflow as tf

_TOKENIZER = None
CLS_ID, SEP_ID, PAD_ID = 101, 102, 0


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import BertTokenizer

        _TOKENIZER = BertTokenizer.from_pretrained("bert-base-uncased")
    return _TOKENIZER


def tokenize(captions: list[str], max_len: int = 40) -> np.ndarray:
    tok = get_tokenizer()
    enc = tok(list(captions), max_length=max_len, padding="max_length", truncation=True)
    return np.asarray(enc["input_ids"], dtype=np.int32)


def decode(ids: np.ndarray) -> list[str]:
    """Token ids -> '[CLS] ... [SEP]' strings, as scored in the paper."""
    tok = get_tokenizer()
    out = []
    for row in ids:
        text = tok.decode(row)
        m = re.search(r"\[CLS\](.*?)\[SEP\]", text)
        out.append(m.group(0) if m else text.replace("[PAD]", "").strip())
    return out


def category_token_ids() -> dict[int, int]:
    tok = get_tokenizer()
    return {c: tok.convert_tokens_to_ids(str(c)) for c in range(1, 6)}


def greedy_generate(model: tf.keras.Model, features: np.ndarray, max_len: int = 40, batch_size: int = 64):
    """Generate conclusions from view embeddings.

    Returns ``(token_ids (N, max_len), category_probs (N, 5))``. ``category_probs``
    are the model's probabilities for the BI-RADS digits 1-5 at the step where the
    category is written (right after the token "category"), renormalised over the
    five digits. They give a continuous score for PR curves without changing the
    generated text.
    """
    tok = get_tokenizer()
    category_id = tok.convert_tokens_to_ids("category")
    digit_ids = np.array(list(category_token_ids().values()))

    @tf.function(reduce_retracing=True)
    def step(x, ids):
        return model({"visual_context": x, "dec_input_ids": ids}, training=False)

    all_ids, all_cat = [], []
    for start in range(0, len(features), batch_size):
        x = tf.convert_to_tensor(features[start:start + batch_size], tf.float32)
        b = x.shape[0]
        ids = np.zeros((b, max_len), np.int32)
        ids[:, 0] = CLS_ID
        cat = np.full((b, 5), np.nan, np.float32)
        done = np.zeros(b, bool)
        for t in range(max_len - 1):
            probs = step(x, tf.constant(ids))[:, t, :].numpy()
            nxt = probs.argmax(-1)
            at_cat = (ids[:, t] == category_id) & np.isnan(cat[:, 0])
            if at_cat.any():
                p = probs[at_cat][:, digit_ids]
                cat[at_cat] = p / p.sum(1, keepdims=True)
            nxt[done] = PAD_ID
            ids[:, t + 1] = nxt
            done |= nxt == SEP_ID
            if done.all():
                break
        all_ids.append(ids)
        all_cat.append(cat)
    return np.concatenate(all_ids), np.concatenate(all_cat)
