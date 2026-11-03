from src.data import IGNORE_INDEX, mask_prompt_labels, split_by_image
from src.eval import evaluate, relaxed_match


def test_relaxed_numeric():
    assert relaxed_match("10.2", "10")
    assert relaxed_match("1,000", "1000")
    assert not relaxed_match("12", "10")
    assert relaxed_match("Yes", "yes.")


def test_split_no_image_leakage():
    recs = [{"image_id": f"img{i // 3}", "q": i} for i in range(60)]
    tr, va, te = split_by_image(recs, 30, 9, 12, seed=1)
    ids = [set(r["image_id"] for r in s) for s in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_mask_labels():
    labels = mask_prompt_labels([1, 2, 3, 4, 5], 3)
    assert labels == [IGNORE_INDEX] * 3 + [4, 5]


def test_evaluate_runs():
    r = evaluate(["10", "a"], ["10", "b"], n_boot=50)
    assert r["n"] == 2 and r["exact_match"] == 0.5
