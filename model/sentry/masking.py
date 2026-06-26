from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


class NormalPatternReferenceModel:

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_units: int = 128,
        max_len: int = 200,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_units = hidden_units
        self.max_len = max_len
        self.model = self._build()

    def _build(self) -> tf.keras.Model:
        inp = layers.Input(shape=(None,), dtype="int32", name="event_ids")
        x = layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_dim,
            mask_zero=True,
            name="ref_embedding",
        )(inp)
        x = layers.LSTM(self.hidden_units, return_sequences=True, name="ref_lstm")(x)
        out = layers.Dense(self.vocab_size, activation="softmax", name="next_event")(x)
        return models.Model(inp, out, name="normal_pattern_reference_model")

    @staticmethod
    def _make_next_step_targets(
        sequences: Sequence[Sequence[int]], max_len: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        X, Y = [], []
        for s in sequences:
            s = list(s)[:max_len]
            if len(s) < 2:
                continue
            inp = s[:-1]
            tgt = s[1:]
            X.append(inp)
            Y.append(tgt)
        X = tf.keras.preprocessing.sequence.pad_sequences(
            X, maxlen=max_len, padding="post", value=0
        )
        Y = tf.keras.preprocessing.sequence.pad_sequences(
            Y, maxlen=max_len, padding="post", value=0
        )
        return X, Y

    def train(
        self,
        normal_sequences: Sequence[Sequence[int]],
        epochs: int = 10,
        batch_size: int = 64,
        verbose: int = 1,
    ) -> None:
        X, Y = self._make_next_step_targets(normal_sequences, self.max_len)
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
        )
        sample_weight = (Y != 0).astype(np.float32)
        self.model.fit(
            X,
            Y,
            sample_weight=sample_weight,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

    def normal_confidence(self, sequence: Sequence[int]) -> np.ndarray:
        seq = list(sequence)[: self.max_len]
        T = len(seq)
        conf = np.zeros(T, dtype=np.float32)
        if T == 0:
            return conf

        inp = np.array(seq[:-1], dtype=np.int32)[None, :]
        if inp.shape[1] == 0:
            return conf
        probs = self.model.predict(inp, verbose=0)[0]

        for p in range(probs.shape[0]):
            next_token = seq[p + 1]
            conf[p + 1] = float(probs[p, next_token])
        return conf


def build_confidence_mask(normal_confidence: np.ndarray) -> np.ndarray:
    m = 1.0 - np.asarray(normal_confidence, dtype=np.float32)
    return np.clip(m, 0.0, 1.0)


def generate_masks(
    ref_model: NormalPatternReferenceModel,
    normal_sequences: Sequence[Sequence[int]],
    abnormal_sequences: Sequence[Sequence[int]],
    max_len: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    seqs: List[List[int]] = []
    masks: List[np.ndarray] = []
    labels: List[int] = []

    for s in normal_sequences:
        s = list(s)[:max_len]
        if len(s) == 0:
            continue
        seqs.append(s)
        masks.append(np.ones(len(s), dtype=np.float32))
        labels.append(0)

    for s in abnormal_sequences:
        s = list(s)[:max_len]
        if len(s) == 0:
            continue
        cn = ref_model.normal_confidence(s)
        m = build_confidence_mask(cn)
        seqs.append(s)
        masks.append(m)
        labels.append(1)

    seqs_padded = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=max_len, padding="post", value=0
    ).astype(np.int32)
    masks_padded = tf.keras.preprocessing.sequence.pad_sequences(
        masks, maxlen=max_len, padding="post", value=0.0, dtype="float32"
    )
    labels_arr = np.asarray(labels, dtype=np.int32)
    return seqs_padded, masks_padded, labels_arr
