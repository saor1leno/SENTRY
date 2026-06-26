from __future__ import annotations

from typing import Sequence

import numpy as np


def matching_distance_window(Cs: Sequence[int], CR: Sequence[int]) -> float:
    Cs = np.asarray(Cs)
    CR = np.asarray(CR)
    l = len(Cs)
    if len(CR) != l:
        raise ValueError("matching_distance_window 要求 Cs 与 CR 等长")

    total = 0.0
    for i in range(l):
        matches = np.where(CR == Cs[i])[0]
        if matches.size > 0:
            total += float(np.min(np.abs(matches - i)))
        else:
            total += float(l)
    return total


def hamming_distance_window(Cs: Sequence[int], CR: Sequence[int]) -> float:
    Cs = np.asarray(Cs)
    CR = np.asarray(CR)
    if len(Cs) != len(CR):
        raise ValueError("hamming_distance_window 要求 Cs 与 CR 等长")
    return float(np.sum(Cs != CR))


def euclidean_distance_window(Cs: Sequence[int], CR: Sequence[int]) -> float:
    Cs = np.asarray(Cs, dtype=np.float64)
    CR = np.asarray(CR, dtype=np.float64)
    if len(Cs) != len(CR):
        raise ValueError("euclidean_distance_window 要求 Cs 与 CR 等长")
    return float(np.sqrt(np.sum((Cs - CR) ** 2)))


WINDOW_DISTANCE_FUNCS = {
    "matching": matching_distance_window,
    "hamming": hamming_distance_window,
    "euclidean": euclidean_distance_window,
}


def candidate_to_sequence_distance(
    Cs: Sequence[int],
    Sk: Sequence[int],
    metric: str = "matching",
) -> float:
    if metric not in WINDOW_DISTANCE_FUNCS:
        raise ValueError(f"未知的距离度量: {metric}")
    dist_fn = WINDOW_DISTANCE_FUNCS[metric]

    Cs = np.asarray(Cs)
    Sk = np.asarray(Sk)
    l = len(Cs)
    if len(Sk) < l:
        return float("inf")

    best = float("inf")
    for start in range(len(Sk) - l + 1):
        CR = Sk[start : start + l]
        d = dist_fn(Cs, CR)
        if d < best:
            best = d
            if best == 0.0:
                break
    return best
