import pytest

from src.data import IGNORE_INDEX, mask_prompt_labels, split_by_image


def test_split_is_deterministic_disjoint_and_keeps_whole_groups():
    records = [{"image_id": f"chart{i // 2}", "q": i} for i in range(20)]
    splits = split_by_image(records, 6, 4, 4)
    assert splits == split_by_image(records, 6, 4, 4)
    assert [len(s) for s in splits] == [6, 4, 4]
    ids = [set(r["image_id"] for r in s) for s in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for split in splits:
        for chart in set(r["image_id"] for r in split):
            assert sum(r["image_id"] == chart for r in split) == 2


def test_oversized_groups_and_zero_caps_remain_unassigned():
    records = [{"image_id": "a"}] * 4
    assert split_by_image(records, 3, 2, 1) == ([], [], [])
    assert split_by_image(records, 0, 0, 0) == ([], [], [])


@pytest.mark.parametrize("cap", [-1, 1.5, True])
def test_invalid_caps(cap):
    with pytest.raises(ValueError):
        split_by_image([], cap, 0, 0)


@pytest.mark.parametrize("record", [{}, {"image_id": None}, {"image_id": " "}, {"image_id": 1}])
def test_invalid_image_ids(record):
    with pytest.raises(ValueError):
        split_by_image([record], 1, 1, 1)


def test_mask_boundaries_do_not_mutate_input():
    tokens = [1, 2, 3]
    assert mask_prompt_labels(tokens, 0) == tokens
    assert mask_prompt_labels(tokens, 2) == [IGNORE_INDEX, IGNORE_INDEX, 3]
    assert mask_prompt_labels(tokens, 10) == [IGNORE_INDEX] * 3
    assert tokens == [1, 2, 3]
    with pytest.raises(ValueError):
        mask_prompt_labels(tokens, -1)
