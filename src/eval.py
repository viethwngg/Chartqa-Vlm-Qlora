"""ChartQA-style local metrics; see docs/EXPERIMENT_PLAN.md for protocol details."""
import math
import random
import re
from collections import defaultdict
from decimal import Decimal, DecimalException
from typing import List, Optional, Sequence, Tuple

PROTOCOL = "chartqa_local_v1"
_NUMBER = re.compile(r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?%?")


def normalize(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Answers must be strings.")
    text = re.sub(r"\s+", " ", text.strip().lower())
    if text.endswith(".") and not text.endswith(".."):
        text = text[:-1].rstrip()
    return text


def _number(text: str) -> Optional[Decimal]:
    text = normalize(text)
    if not _NUMBER.fullmatch(text):
        return None
    try:
        value = Decimal(text.rstrip("%").replace(",", ""))
        if text.endswith("%"):
            value /= 100
        # Keep numeric eligibility bounded to finite float range for interoperability.
        if not math.isfinite(float(value)) or (value != 0 and float(value) == 0):
            return None
        return value
    except (DecimalException, OverflowError):
        return None


def to_float(text: str) -> Optional[float]:
    value = _number(text)
    return float(value) if value is not None else None


def exact_match(pred: str, gold: str) -> bool:
    return normalize(pred) == normalize(gold)


def _check_tolerance(tol: float) -> None:
    if not math.isfinite(tol) or not 0 <= tol <= 1:
        raise ValueError("Tolerance must be finite and between 0 and 1.")


def relaxed_match(pred: str, gold: str, tol: float = 0.05) -> bool:
    """5% relative error; percent strings are fractions; numeric zero needs zero."""
    _check_tolerance(tol)
    p, g = _number(pred), _number(gold)
    if p is not None and g is not None:
        if g == 0:
            return p == 0
        return abs(p - g) <= Decimal(str(tol)) * abs(g)
    return exact_match(pred, gold)


def accuracy(flags: Sequence[float]) -> float:
    if not flags:
        raise ValueError("Cannot score an empty sample.")
    return sum(flags) / len(flags)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def bootstrap_ci(flags: Sequence[float], n_samples: int = 1000,
                 alpha: float = 0.05, seed: int = 42,
                 image_ids: Optional[Sequence[str]] = None) -> Tuple[float, float]:
    """Question-weighted percentile CI, optionally resampling entire charts.

    Values may be paired score differences (-1, 0, 1). Duplicate chart draws
    retain all questions each time. No IDs means individual-question bootstrap.
    """
    if not flags or any(not math.isfinite(v) for v in flags):
        raise ValueError("Bootstrap requires nonempty finite values.")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if image_ids is not None:
        if len(image_ids) != len(flags):
            raise ValueError("image_ids and scores must have equal length.")
        if any(not isinstance(i, str) or not i.strip() for i in image_ids):
            raise ValueError("Chart IDs must be nonempty strings.")
    ids = image_ids if image_ids is not None else list(range(len(flags)))
    groups = defaultdict(list)
    for chart, value in zip(ids, flags):
        groups[chart].append(value)
    # Aggregate clusters once instead of repeatedly expanding all their questions.
    clusters = [(sum(values), len(values)) for values in groups.values()]
    rng = random.Random(seed)
    means = []
    for _ in range(n_samples):
        total, count = 0.0, 0
        for _ in range(len(clusters)):
            subtotal, size = rng.choice(clusters)
            total += subtotal
            count += size
        means.append(total / count)
    means.sort()
    return _percentile(means, alpha / 2), _percentile(means, 1 - alpha / 2)


def _validate_answers(preds: Sequence[str], golds: Sequence[str]) -> None:
    if not golds or len(preds) != len(golds):
        raise ValueError("Predictions and golds must be nonempty and equal length.")
    for pred, gold in zip(preds, golds):
        normalize(pred)
        if not normalize(gold):
            raise ValueError("Gold answers must not be empty.")


def evaluate(preds: List[str], golds: List[str], tol: float = 0.05,
             n_boot: int = 1000, image_ids: Optional[Sequence[str]] = None,
             seed: int = 42) -> dict:
    _validate_answers(preds, golds)
    _check_tolerance(tol)
    em = [exact_match(p, g) for p, g in zip(preds, golds)]
    rx = [relaxed_match(p, g, tol) for p, g in zip(preds, golds)]
    return {
        "n": len(golds),
        "n_charts": len(set(image_ids)) if image_ids is not None else None,
        "exact_match": accuracy(em),
        "relaxed_accuracy": accuracy(rx),
        "exact_ci95": bootstrap_ci(em, n_boot, seed=seed, image_ids=image_ids),
        "relaxed_ci95": bootstrap_ci(rx, n_boot, seed=seed, image_ids=image_ids),
    }


def paired_compare(baseline: List[str], predictions: List[str], golds: List[str],
                   image_ids: Sequence[str], tol: float = 0.05,
                   n_boot: int = 1000, seed: int = 42) -> dict:
    """Compare already aligned predictions; the CLI validates sample identities."""
    _validate_answers(baseline, golds)
    _validate_answers(predictions, golds)
    _check_tolerance(tol)
    result = {}
    for name, scorer in (("exact_match", exact_match),
                         ("relaxed_accuracy", lambda p, g: relaxed_match(p, g, tol))):
        differences = [int(scorer(p, g)) - int(scorer(b, g))
                       for b, p, g in zip(baseline, predictions, golds)]
        result[name] = {
            "delta": accuracy(differences),
            "ci95": bootstrap_ci(differences, n_boot, seed=seed, image_ids=image_ids),
        }
    return result
