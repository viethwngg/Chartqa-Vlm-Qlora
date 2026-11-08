"""Artificial metric fixtures only; these are not ChartQA model results."""
import pytest

from src.eval import bootstrap_ci, evaluate, exact_match, normalize, paired_compare, relaxed_match, to_float


@pytest.mark.parametrize("pred,gold,expected", [
    ("10.5", "10", True), ("10.5001", "10", False),
    ("0.105", "0.1", True), ("-105", "-100", True),
    ("0.0", "0", True), ("0.00001", "0", False),
    ("50%", "0.5", True), ("50%", "50", False),
    ("1,000", "1000", True), ("1,2", "12", False),
    ("1e3", "1000", True), ("10 kg", "10", False),
    ("The answer is 10", "10", False), ("", "10", False),
    ("[1, 2]", "12", False), ("Yes", "yes.", True),
])
def test_relaxed_policy(pred, gold, expected):
    assert relaxed_match(pred, gold) is expected


def test_normalization_preserves_semantics():
    assert normalize("  YES. \n") == "yes"
    assert normalize("a   b") == "a b"
    assert normalize("wait...") == "wait..."
    assert not exact_match("50%", "0.5")
    assert not exact_match("1,000", "1000")
    assert not exact_match("-10", "10")
    assert not exact_match("a-b", "ab")


@pytest.mark.parametrize("text", ["nan", "inf", "-inf", "1e999", "1e-999", "1e999999999%", "-1e999999999%", "1_000", "1,00", "50%%", "$10"])
def test_numeric_parser_rejects_invalid_values(text):
    assert to_float(text) is None


@pytest.mark.parametrize("preds,golds", [([], []), (["a"], []), ([], ["a"]), (["a"], ["a", "b"]), (["a"], [""]), ([1], ["1"])])
def test_evaluate_rejects_invalid_pairs(preds, golds):
    with pytest.raises(ValueError):
        evaluate(preds, golds)


@pytest.mark.parametrize("kwargs", [{"n_samples": 0}, {"n_samples": -1}, {"n_samples": True},
                                       {"alpha": 0}, {"alpha": 1}, {"image_ids": []}, {"image_ids": [""]}])
def test_bootstrap_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci([True], **kwargs)


def test_cluster_resampling_preserves_whole_charts_and_weighting():
    # One chart is inseparable; question bootstrap could invent mixed proportions.
    assert bootstrap_ci([1, 0, 0], image_ids=["a"] * 3) == (1 / 3, 1 / 3)
    # Unequal group sizes: the observed point estimate stays question-weighted.
    result = evaluate(["yes", "no", "no", "no"], ["yes"] * 4,
                      image_ids=["a", "b", "b", "b"], n_boot=200)
    assert result["relaxed_accuracy"] == 0.25
    assert result["n_charts"] == 2
    assert result["relaxed_ci95"] == (0, 1)
    assert bootstrap_ci([1, 0, 0], seed=9) == bootstrap_ci([1, 0, 0], seed=9)


def test_paired_ci_uses_within_chart_differences():
    # Each chart has one fix and one regression: paired delta is always zero.
    result = paired_compare(["yes", "no", "yes", "no"], ["no", "yes", "no", "yes"],
                            ["yes"] * 4, ["a", "a", "b", "b"])
    assert result["relaxed_accuracy"] == {"delta": 0, "ci95": (0, 0)}
    perfect = paired_compare(["no"] * 2, ["yes"] * 2, ["yes"] * 2, ["a", "b"])
    assert perfect["exact_match"] == {"delta": 1, "ci95": (1, 1)}


@pytest.mark.parametrize("tol", [-0.1, 2, float("nan"), float("inf")])
def test_invalid_tolerance(tol):
    with pytest.raises(ValueError):
        relaxed_match("a", "a", tol)
