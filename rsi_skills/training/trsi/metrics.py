"""Validation metrics used to compare training schemes. Pure Python (no numpy, no sklearn).

- auprc: average precision, computed as sklearn's average_precision_score (sum over distinct thresholds,
  highest first, of (R_n - R_{n-1}) * P_n).
- auc: ROC AUC from average ranks (ties count half).
- ece: 15 equal-width bins on p, as the Elliptic team's eval_decisions.py.
- brier: mean squared error of p.
Thresholds, calibration and test-set evaluation belong to the evaluation side, not to this skill.
"""
from __future__ import annotations

from typing import Sequence

HIGHER_IS_BETTER = {"auprc": True, "auc": True, "ece": False, "brier": False}
ECE_BINS = 15


def _check(y: Sequence[int], p: Sequence[float]) -> None:
    if len(y) != len(p) or not y:
        raise ValueError(f"y and p must be non-empty and of equal length (got {len(y)} and {len(p)})")
    if any(v not in (0, 1) for v in y):
        raise ValueError("y must be 0/1")
    if any(not 0.0 <= v <= 1.0 for v in p):
        raise ValueError("p must be probabilities in [0, 1]")


def auprc(y: Sequence[int], p: Sequence[float]) -> float:
    pos = sum(y)
    if pos == 0:
        raise ValueError("AUPRC needs at least one positive row")
    pairs = sorted(zip(p, y), key=lambda t: -t[0])
    tp = fp = 0
    ap = prev_recall = 0.0
    i, n = 0, len(pairs)
    while i < n:
        thr = pairs[i][0]
        while i < n and pairs[i][0] == thr:  # one step per distinct threshold
            tp += pairs[i][1]
            fp += 1 - pairs[i][1]
            i += 1
        recall = tp / pos
        ap += (recall - prev_recall) * tp / (tp + fp)
        prev_recall = recall
    return ap


def auc(y: Sequence[int], p: Sequence[float]) -> float:
    pos = sum(y)
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        raise ValueError("AUC needs both classes")
    order = sorted(range(len(p)), key=lambda k: p[k])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and p[order[j + 1]] == p[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average rank of a tie block
        rank_sum += avg_rank * sum(y[order[k]] for k in range(i, j + 1))
        i = j + 1
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def ece(y: Sequence[int], p: Sequence[float], bins: int = ECE_BINS) -> float:
    count = [0] * bins
    sum_y = [0.0] * bins
    sum_p = [0.0] * bins
    for yi, pi in zip(y, p):
        b = min(int(pi * bins), bins - 1)
        count[b] += 1
        sum_y[b] += yi
        sum_p[b] += pi
    n = len(y)
    return sum(abs(sum_y[b] - sum_p[b]) / n for b in range(bins) if count[b])


def brier(y: Sequence[int], p: Sequence[float]) -> float:
    return sum((pi - yi) ** 2 for yi, pi in zip(y, p)) / len(y)


def evaluate(y: Sequence[int], p: Sequence[float]) -> dict:
    """All comparison metrics on one set of validation predictions."""
    y = [int(v) for v in y]
    p = [float(v) for v in p]
    _check(y, p)
    return {"auprc": auprc(y, p), "auc": auc(y, p), "ece": ece(y, p), "brier": brier(y, p),
            "n": len(y), "positives": sum(y)}
