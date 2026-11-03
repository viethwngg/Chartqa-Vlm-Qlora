"""Metrics for ChartQA-style VQA: exact match, relaxed accuracy, bootstrap CI."""
import random
import re
from typing import List, Optional, Tuple


def normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip(".")


def to_float(text: str) -> Optional[float]:
    t = normalize(text).replace(",", "")
    is_pct = t.endswith("%")
    t = t.rstrip("%").strip()
    try:
        return float(t)
    except ValueError:
        return None


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    """Numeric answers pass within a relative tolerance; others need exact match."""
    p, g = to_float(pred), to_float(gold)
    if p is not None and g is not None:
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= tol
    return exact_match(pred, gold)


def accuracy(flags: List[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0


def bootstrap_ci(flags: List[bool], n_samples: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> Tuple[float, float]:
    rng = random.Random(seed)
    n = len(flags)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_samples):
        sample = [flags[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_samples)]
    hi = means[min(int((1 - alpha / 2) * n_samples), n_samples - 1)]
    return lo, hi


def evaluate(preds: List[str], golds: List[str], tol: float = 0.05,
             n_boot: int = 1000) -> dict:
    em = [exact_match(p, g) for p, g in zip(preds, golds)]
    rx = [relaxed_match(p, g, tol) for p, g in zip(preds, golds)]
    return {
        "n": len(golds),
        "exact_match": accuracy(em),
        "relaxed_accuracy": accuracy(rx),
        "relaxed_ci95": bootstrap_ci(rx, n_boot),
    }
