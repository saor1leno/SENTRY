from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

from .distance import candidate_to_sequence_distance


def lcs_matched_indices(seq: Sequence[int], pattern: Sequence[int]) -> List[int]:
    seq = list(seq)
    pattern = list(pattern)
    n, m = len(seq), len(pattern)
    if n == 0 or m == 0:
        return []

    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if seq[i - 1] == pattern[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    i, j = n, m
    matched: List[int] = []
    while i > 0 and j > 0:
        if seq[i - 1] == pattern[j - 1]:
            matched.append(i - 1)
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    matched.reverse()
    return matched


@dataclass
class PatternUnitExtractor:

    l_min: int = 2
    l_max: int = 5
    tau: float = 1.0
    gamma: float = 2.0
    metric: str = "matching"
    max_candidates_per_class: int | None = None

    pattern_units_: List[np.ndarray] = field(default_factory=list, init=False)

    def _generate_candidates(self, sequences: Sequence[Sequence[int]]) -> List[Tuple[int, ...]]:
        candidates = set()
        for Sk in sequences:
            Sk = np.asarray(Sk)
            n = len(Sk)
            for l in range(self.l_min, self.l_max + 1):
                if n < l:
                    continue
                for i in range(n - l + 1):
                    candidates.add(tuple(int(v) for v in Sk[i : i + l]))
        return list(candidates)

    def fit(
        self,
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int],
    ) -> "PatternUnitExtractor":
        labels = np.asarray(labels)
        candidates = self._generate_candidates(sequences)

        scored: List[Tuple[float, float, np.ndarray]] = []  # (delta, min_mu, Cs)
        for Cs in candidates:
            d_pos, d_neg = [], []
            for Sk, y in zip(sequences, labels):
                d = candidate_to_sequence_distance(Cs, Sk, metric=self.metric)
                if np.isinf(d):
                    continue
                if y == 1:
                    d_pos.append(d)
                else:
                    d_neg.append(d)

            if len(d_pos) == 0 or len(d_neg) == 0:
                continue

            mu_pos = float(np.mean(d_pos))
            mu_neg = float(np.mean(d_neg))
            delta = abs(mu_pos - mu_neg)
            min_mu = min(mu_pos, mu_neg)

            if delta > self.tau and min_mu < self.gamma:
                scored.append((delta, min_mu, np.asarray(Cs)))

        scored.sort(key=lambda x: (-x[0], x[1]))
        if self.max_candidates_per_class is not None:
            scored = scored[: self.max_candidates_per_class]

        self.pattern_units_ = [cs for _, _, cs in scored]
        return self

    # --------------------------- 序列重构 ---------------------------
    def reconstruct_sequence(self, Sk: Sequence[int]) -> List[int]:
        Sk = list(Sk)
        keep = set()
        for Cp in self.pattern_units_:
            for idx in lcs_matched_indices(Sk, Cp):
                keep.add(idx)
        if not keep:
            return Sk
        return [Sk[i] for i in sorted(keep)]

    def transform(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> List[List[int]]:
        if not self.pattern_units_:
            raise RuntimeError("请先调用 fit() 提取 pattern units。")
        return [self.reconstruct_sequence(Sk) for Sk in sequences]

    def fit_transform(
        self,
        sequences: Sequence[Sequence[int]],
        labels: Sequence[int],
    ) -> List[List[int]]:
        self.fit(sequences, labels)
        return self.transform(sequences)


def extract_benign_behavior_units(
    normal_sequences: Sequence[Sequence[int]],
    l_min: int = 2,
    l_max: int = 5,
    metric: str = "matching",
    support_threshold: float = 1.0,
    top_k: int | None = None,
) -> List[np.ndarray]:
    candidates = set()
    for Sk in normal_sequences:
        Sk = np.asarray(Sk)
        n = len(Sk)
        for l in range(l_min, l_max + 1):
            if n < l:
                continue
            for i in range(n - l + 1):
                candidates.add(tuple(int(v) for v in Sk[i : i + l]))

    scored = []
    for Cs in candidates:
        support = 0
        for Sk in normal_sequences:
            d = candidate_to_sequence_distance(Cs, Sk, metric=metric)
            if d <= support_threshold:
                support += 1
        scored.append((support, np.asarray(Cs)))

    scored.sort(key=lambda x: -x[0])
    if top_k is not None:
        scored = scored[:top_k]
    return [cs for _, cs in scored]
