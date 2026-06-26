from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import tensorflow as tf

from .backbone import build_masked_detector
from .distillation import DifficultSampleDistiller
from .masking import NormalPatternReferenceModel, generate_masks
from .pattern_unit import PatternUnitExtractor


@dataclass
class SentryConfig:
    vocab_size: int
    max_len: int = 200
    embed_dim: int = 64
    hidden_units: int = 128
    backbone: str = "lstm"

    l_min: int = 2
    l_max: int = 5
    tau: float = 1.0
    gamma_pu: float = 2.0
    distance_metric: str = "matching"
    max_pattern_units: Optional[int] = 200

    ref_epochs: int = 10

    alpha: float = 1.0
    beta: float = 1.0
    gamma_distill: float = 1.0
    temperature: float = 4.0
    teacher_epochs: int = 10
    student_epochs: int = 10
    batch_size: int = 64


class SENTRY:

    def __init__(self, config: SentryConfig):
        self.cfg = config
        self.extractor: Optional[PatternUnitExtractor] = None
        self.ref_model: Optional[NormalPatternReferenceModel] = None
        self.teacher: Optional[tf.keras.Model] = None
        self.student: Optional[tf.keras.Model] = None

    def _reconstruct(
        self,
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int],
        fit: bool,
    ) -> List[List[int]]:
        if fit:
            self.extractor = PatternUnitExtractor(
                l_min=self.cfg.l_min,
                l_max=self.cfg.l_max,
                tau=self.cfg.tau,
                gamma=self.cfg.gamma_pu,
                metric=self.cfg.distance_metric,
                max_candidates_per_class=self.cfg.max_pattern_units,
            )
            return self.extractor.fit_transform(sequences, labels)
        assert self.extractor is not None
        return self.extractor.transform(sequences)

    def fit(
        self,
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int],
        verbose: int = 1,
    ) -> "SENTRY":
        labels = np.asarray(labels).astype(np.int32)

        if verbose:
            print(">>> Stage 1: pattern unit 提取与序列重构")
        cleaned = self._reconstruct(sequences, labels, fit=True)

        normal_seqs = [s for s, y in zip(cleaned, labels) if y == 0]
        abnormal_seqs = [s for s, y in zip(cleaned, labels) if y == 1]

        if verbose:
            print(">>> Stage 2: 训练正常模式参考模型并生成置信度掩码")
        self.ref_model = NormalPatternReferenceModel(
            vocab_size=self.cfg.vocab_size,
            embed_dim=self.cfg.embed_dim,
            hidden_units=self.cfg.hidden_units,
            max_len=self.cfg.max_len,
        )
        self.ref_model.train(normal_seqs, epochs=self.cfg.ref_epochs, verbose=verbose)

        x_ids, x_mask, y = generate_masks(
            self.ref_model, normal_seqs, abnormal_seqs, self.cfg.max_len
        )

        if verbose:
            print(">>> Stage 3: 训练教师模型")
        self.teacher = build_masked_detector(
            vocab_size=self.cfg.vocab_size,
            embed_dim=self.cfg.embed_dim,
            hidden_units=self.cfg.hidden_units,
            backbone=self.cfg.backbone,
            name="teacher",
        )
        self.teacher.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        self.teacher.fit(
            [x_ids, x_mask], y,
            epochs=self.cfg.teacher_epochs,
            batch_size=self.cfg.batch_size,
            verbose=verbose,
        )

        if verbose:
            print(">>> Stage 3: 蒸馏训练学生模型 (L = α·L_hard + β·L_soft + γ·L_difficult)")
        self.student = build_masked_detector(
            vocab_size=self.cfg.vocab_size,
            embed_dim=self.cfg.embed_dim,
            hidden_units=self.cfg.hidden_units,
            backbone=self.cfg.backbone,
            name="student",
        )
        distiller = DifficultSampleDistiller(
            teacher=self.teacher,
            student=self.student,
            alpha=self.cfg.alpha,
            beta=self.cfg.beta,
            gamma=self.cfg.gamma_distill,
            temperature=self.cfg.temperature,
        )
        distiller.train(
            x_ids, x_mask, y,
            epochs=self.cfg.student_epochs,
            batch_size=self.cfg.batch_size,
            verbose=verbose,
        )
        return self

    def predict(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> np.ndarray:
        assert self.student is not None and self.ref_model is not None
        cleaned = self._reconstruct(sequences, labels=[0] * len(sequences), fit=False)

        x_ids, x_mask, _ = generate_masks(
            self.ref_model,
            normal_sequences=[],
            abnormal_sequences=cleaned,
            max_len=self.cfg.max_len,
        )
        logits = self.student.predict([x_ids, x_mask], verbose=0)
        return np.argmax(logits, axis=-1)
