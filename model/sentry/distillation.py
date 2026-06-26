from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def compute_teacher_probs(teacher: tf.keras.Model, inputs) -> np.ndarray:
    logits = teacher.predict(inputs, verbose=0)
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return probs


def identify_difficult_samples(
    teacher: tf.keras.Model,
    inputs,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels).astype(np.int32)
    probs = compute_teacher_probs(teacher, inputs)
    pred = np.argmax(probs, axis=-1)
    p_correct = probs[np.arange(len(labels)), labels]

    difficulty = 1.0 - p_correct
    weights = 1.0 - p_correct
    is_difficult = (pred != labels)
    return is_difficult, difficulty.astype(np.float32), weights.astype(np.float32)


def hard_label_loss(y_true, student_logits) -> tf.Tensor:
    return tf.keras.losses.sparse_categorical_crossentropy(
        y_true, student_logits, from_logits=True
    )


def soft_label_loss(teacher_logits, student_logits, temperature: float) -> tf.Tensor:
    T = float(temperature)
    teacher_soft = tf.nn.softmax(teacher_logits / T, axis=-1)
    student_log_soft = tf.nn.log_softmax(student_logits / T, axis=-1)
    kl = tf.reduce_sum(
        teacher_soft * (tf.math.log(teacher_soft + 1e-12) - student_log_soft),
        axis=-1,
    )
    return kl * (T * T)


def difficult_sample_loss(
    y_true,
    student_logits,
    weights: tf.Tensor,
    is_difficult: tf.Tensor,
) -> tf.Tensor:
    ce = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, student_logits, from_logits=True
    )
    mask = tf.cast(is_difficult, tf.float32)
    weighted = ce * weights * mask
    denom = tf.reduce_sum(mask) + 1e-8
    return tf.reduce_sum(weighted) / denom


class DifficultSampleDistiller:

    def __init__(
        self,
        teacher: tf.keras.Model,
        student: tf.keras.Model,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        temperature: float = 4.0,
        learning_rate: float = 1e-3,
    ):
        self.teacher = teacher
        self.student = student
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    @tf.function
    def _train_step(self, x_ids, x_mask, y, teacher_logits, weights, is_difficult):
        with tf.GradientTape() as tape:
            student_logits = self.student([x_ids, x_mask], training=True)

            l_hard = tf.reduce_mean(hard_label_loss(y, student_logits))
            l_soft = tf.reduce_mean(
                soft_label_loss(teacher_logits, student_logits, self.temperature)
            )
            l_diff = difficult_sample_loss(y, student_logits, weights, is_difficult)

            loss = self.alpha * l_hard + self.beta * l_soft + self.gamma * l_diff

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))
        return loss, l_hard, l_soft, l_diff

    def train(
        self,
        x_ids: np.ndarray,
        x_mask: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        verbose: int = 1,
    ) -> None:
        y = np.asarray(y).astype(np.int32)

        teacher_logits_all = self.teacher.predict([x_ids, x_mask], verbose=0)
        is_difficult, _difficulty, weights = identify_difficult_samples(
            self.teacher, [x_ids, x_mask], y
        )

        N = len(y)
        ds = tf.data.Dataset.from_tensor_slices(
            (
                x_ids,
                x_mask,
                y,
                teacher_logits_all.astype(np.float32),
                weights.astype(np.float32),
                is_difficult.astype(np.float32),
            )
        ).shuffle(buffer_size=min(N, 10000)).batch(batch_size)

        for epoch in range(epochs):
            agg = np.zeros(4, dtype=np.float64)
            nb = 0
            for bx, bm, by, btl, bw, bd in ds:
                loss, lh, ls, ld = self._train_step(bx, bm, by, btl, bw, bd)
                agg += [float(loss), float(lh), float(ls), float(ld)]
                nb += 1
            if verbose:
                agg /= max(nb, 1)
                print(
                    f"[Distill] epoch {epoch + 1}/{epochs} "
                    f"loss={agg[0]:.4f} | L_hard={agg[1]:.4f} "
                    f"L_soft={agg[2]:.4f} L_diff={agg[3]:.4f} "
                    f"| #difficult={int(is_difficult.sum())}"
                )
