from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


class InputLevelMask(layers.Layer):

    def call(self, inputs):
        embeddings, mask = inputs
        mask = tf.expand_dims(mask, axis=-1)
        return embeddings * mask


def _tcn_residual_block(x, filters, kernel_size, dilation):
    prev = x
    conv = layers.Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu"
    )(x)
    conv = layers.Conv1D(
        filters, kernel_size, padding="causal", dilation_rate=dilation, activation="relu"
    )(conv)
    if prev.shape[-1] != filters:
        prev = layers.Conv1D(filters, 1, padding="same")(prev)
    return layers.add([prev, conv])


def _build_tcn(x, hidden_units, n_blocks=3, kernel_size=3):
    for i in range(n_blocks):
        x = _tcn_residual_block(x, hidden_units, kernel_size, dilation=2 ** i)
    return layers.GlobalAveragePooling1D()(x)


def _transformer_encoder_block(x, num_heads, ff_dim):
    d_model = x.shape[-1]
    attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.add([x, attn]))
    ff = layers.Dense(ff_dim, activation="relu")(x)
    ff = layers.Dense(d_model)(ff)
    x = layers.LayerNormalization(epsilon=1e-6)(layers.add([x, ff]))
    return x


def _build_transformer(x, num_heads=4, ff_dim=128, n_blocks=2):
    for _ in range(n_blocks):
        x = _transformer_encoder_block(x, num_heads, ff_dim)
    return layers.GlobalAveragePooling1D()(x)


def build_masked_detector(
    vocab_size: int,
    embed_dim: int = 64,
    hidden_units: int = 128,
    backbone: str = "lstm",
    num_classes: int = 2,
    name: str = "masked_detector",
) -> tf.keras.Model:
    event_ids = layers.Input(shape=(None,), dtype="int32", name="event_ids")
    mask_in = layers.Input(shape=(None,), dtype="float32", name="mask")

    emb = layers.Embedding(
        input_dim=vocab_size, output_dim=embed_dim, name="embedding"
    )(event_ids)
    masked = InputLevelMask(name="input_level_mask")([emb, mask_in])

    backbone = backbone.lower()
    if backbone == "lstm":
        feats = layers.LSTM(hidden_units, name="backbone_lstm")(masked)
    elif backbone == "tcn":
        feats = _build_tcn(masked, hidden_units)
    elif backbone == "transformer":
        feats = _build_transformer(masked)
    else:
        raise ValueError(f"未知 backbone: {backbone}")

    feats = layers.Dropout(0.2)(feats)
    logits = layers.Dense(num_classes, name="logits")(feats)
    return models.Model([event_ids, mask_in], logits, name=f"{name}_{backbone}")
