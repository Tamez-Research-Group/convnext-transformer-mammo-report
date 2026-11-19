"""Transformer encoder-decoder that writes the conclusion from ConvNeXt view embeddings.

Faithful re-implementation of the TensorFlow/Keras model used in the paper
(layer names are kept so weights of the original runs can be transferred).

Input: ``visual_context`` (batch, n_views=4, 768), one ConvNeXt embedding per
standard view (LCC, LMLO, RCC, RMLO). The four embeddings are treated as a
4-token sequence: a learned positional (view) embedding is added, two
self-attention encoder layers mix information across views, and every decoder
layer cross-attends to the four encoded view tokens.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, LayerNormalization, MultiHeadAttention


@dataclass
class TransformerConfig:
    vocab_size: int = 30522        # bert-base-uncased WordPiece vocabulary
    embed_dim: int = 768
    ff_dim: int = 1024
    encoder_heads: int = 1
    decoder_heads: int = 2
    num_encoders: int = 2
    num_decoders: int = 2
    dropout: float = 0.2
    n_views: int = 4
    max_len: int = 40              # tokenised caption length incl. [CLS]/[SEP]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TransformerConfig":
        return cls(**json.loads(Path(path).read_text()))


def positional_encoder(seq_len: int, cfg: TransformerConfig, name: str = "pos_enc") -> tf.keras.Model:
    """Adds a *learned* positional embedding (one vector per position)."""
    inputs = tf.keras.Input(shape=(None, cfg.embed_dim))
    table = Embedding(seq_len, cfg.embed_dim, name="positional_matrix")
    positions = tf.range(tf.shape(inputs)[1])
    return tf.keras.Model(inputs, inputs + table(positions), name=name)


def _ffn(cfg: TransformerConfig) -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [Dense(cfg.ff_dim, activation="gelu", name="ffnn_dense1"), Dense(cfg.embed_dim, name="ffnn_dense2")],
        name="feed_forward_network",
    )


def encoder_layer(cfg: TransformerConfig, name: str) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, cfg.embed_dim))
    attn = MultiHeadAttention(cfg.encoder_heads, key_dim=cfg.embed_dim, name="multihead_attention")
    x = attn(query=inputs, key=inputs, value=inputs)
    x = LayerNormalization(epsilon=1e-6, name="attention_layer_norm")(
        inputs + Dropout(cfg.dropout, name="attention_dropout")(x))
    y = Dropout(cfg.dropout, name="ffnn_dropout")(_ffn(cfg)(x))
    out = LayerNormalization(epsilon=1e-6, name="ffnn_layer_norm")(x + y)
    return tf.keras.Model(inputs, out, name=name)


def encoder_block(cfg: TransformerConfig) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(None, cfg.embed_dim), name="cnn_features")
    x = positional_encoder(cfg.n_views, cfg)(inputs)
    for i in range(cfg.num_encoders):
        x = encoder_layer(cfg, name=f"encoder_stack_layer_{i}")(x)
    return tf.keras.Model(inputs, x, name="transformer_encoder")


def decoder_layer(cfg: TransformerConfig, name: str) -> tf.keras.Model:
    dec_in = tf.keras.Input(shape=(None, cfg.embed_dim), name="dec_inputs")
    enc_out = tf.keras.Input(shape=(None, cfg.embed_dim), name="enc_outputs")
    self_attn = MultiHeadAttention(cfg.decoder_heads, key_dim=cfg.embed_dim, name="self_attention")
    cross_attn = MultiHeadAttention(cfg.decoder_heads, key_dim=cfg.embed_dim, name="encoder_decoder_attention")

    a1 = self_attn(query=dec_in, key=dec_in, value=dec_in, use_causal_mask=True)
    x1 = LayerNormalization(epsilon=1e-6, name="attention_layer_norm1")(
        dec_in + Dropout(cfg.dropout, name="attention_dropout1")(a1))
    a2 = cross_attn(query=x1, key=enc_out, value=enc_out)  # text queries, view tokens as keys/values
    x2 = LayerNormalization(epsilon=1e-6, name="attention_layer_norm2")(
        x1 + Dropout(cfg.dropout, name="attention_dropout2")(a2))
    y = Dropout(cfg.dropout, name="ffnn_dropout")(_ffn(cfg)(x2))
    out = LayerNormalization(epsilon=1e-6, name="ffnn_layer_norm")(x2 + y)
    return tf.keras.Model({"dec_inputs": dec_in, "enc_outputs": enc_out}, out, name=name)


def decoder_block(cfg: TransformerConfig) -> tf.keras.Model:
    ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="dec_input_ids")
    enc_out = tf.keras.Input(shape=(None, cfg.embed_dim), name="enc_outputs")
    emb_in = tf.keras.Input(shape=(None,), dtype=tf.int32, name="dec_input_ids")
    emb = Embedding(cfg.vocab_size, cfg.embed_dim, name="embedding_layer")(emb_in)
    initial = tf.keras.Model(emb_in, positional_encoder(cfg.max_len, cfg)(emb), name="initial_decoder_layer")

    x = initial(ids)
    for i in range(cfg.num_decoders):
        x = decoder_layer(cfg, name=f"decoder_layer_{i}")({"dec_inputs": x, "enc_outputs": enc_out})
    probs = Dense(cfg.vocab_size, activation="softmax", name="decoder_output")(x)
    return tf.keras.Model({"dec_input_ids": ids, "enc_outputs": enc_out}, probs, name="transformer_decoder")


def build_transformer(cfg: TransformerConfig | None = None) -> tf.keras.Model:
    """Full model: {'visual_context': (B, V, 768), 'dec_input_ids': (B, T)} -> (B, T, vocab) probabilities."""
    cfg = cfg or TransformerConfig()
    visual = tf.keras.Input(shape=(None, cfg.embed_dim), name="visual_context")
    ids = tf.keras.Input(shape=(None,), dtype=tf.int32, name="dec_input_ids")
    enc = encoder_block(cfg)(visual)
    probs = decoder_block(cfg)({"dec_input_ids": ids, "enc_outputs": enc})
    return tf.keras.Model({"visual_context": visual, "dec_input_ids": ids}, probs, name="transformer")
