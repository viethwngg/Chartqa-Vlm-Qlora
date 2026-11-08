"""Data utilities: split by image (no leakage) and prompt-masked labels."""
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

IGNORE_INDEX = -100


def split_by_image(records: List[dict], n_train: int, n_val: int, n_test: int,
                   seed: int = 42, image_key: str = "image_id"
                   ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Group questions by image, shuffle images, then fill train/val/test.

    The same supplied image ID never appears in two splits. Fill test, val,
    then train, skipping any split that cannot fit a whole group. Sizes are
    approximate question caps; groups that fit nowhere remain unassigned.
    Content deduplication and preservation of official splits are caller duties.
    """
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 0
           for n in (n_train, n_val, n_test)):
        raise ValueError("Split caps must be nonnegative integers.")
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        if not isinstance(r.get(image_key), str) or not r[image_key].strip():
            raise ValueError(f"Each record needs a nonempty string {image_key}.")
        groups[r[image_key]].append(r)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)

    splits = {"test": [], "val": [], "train": []}
    caps = {"test": n_test, "val": n_val, "train": n_train}
    for k in keys:
        for name in ("test", "val", "train"):
            if len(splits[name]) + len(groups[k]) <= caps[name]:
                splits[name].extend(groups[k])
                break
    return splits["train"], splits["val"], splits["test"]


def mask_prompt_labels(input_ids: Sequence[int], prompt_len: int) -> List[int]:
    """Loss only on the answer: set labels of the prompt part to -100."""
    if isinstance(prompt_len, bool) or not isinstance(prompt_len, int) or prompt_len < 0:
        raise ValueError("prompt_len must be a nonnegative integer.")
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX
    return labels


def load_records(dataset_name: str, split: str):
    """TODO: load ChartQA from Hugging Face and return a list of dicts with
    keys: image_id, image, question, answer. Verify the dataset id and schema."""
    raise NotImplementedError("Implement dataset loading for the chosen dataset.")
