"""Data utilities: split by image (no leakage) and prompt-masked labels."""
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

IGNORE_INDEX = -100


def split_by_image(records: List[dict], n_train: int, n_val: int, n_test: int,
                   seed: int = 42, image_key: str = "image_id"
                   ) -> Tuple[List[dict], List[dict], List[dict]]:
    """Group questions by image, shuffle images, then fill train/val/test.

    The same image never appears in two splits. Sizes are approximate caps
    (a split stops once adding another image would exceed its size).
    """
    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
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
    labels = list(input_ids)
    for i in range(min(prompt_len, len(labels))):
        labels[i] = IGNORE_INDEX
    return labels


def load_records(dataset_name: str, split: str):
    """TODO: load ChartQA from Hugging Face and return a list of dicts with
    keys: image_id, image, question, answer. Verify the dataset id and schema."""
    raise NotImplementedError("Implement dataset loading for the chosen dataset.")
