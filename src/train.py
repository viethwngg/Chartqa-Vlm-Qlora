"""QLoRA fine-tuning entry point (skeleton).

Model-specific parts (processor, chat template, collator) are TODO because
they differ between SmolVLM and Qwen2-VL. Implement and test them first with
the 'overfit 20-50 samples' check before running a full training.
"""
import argparse

from src.utils import load_config, set_seed


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    # TODO 1: load processor + model (4-bit if cfg["model"]["load_in_4bit"]).
    # TODO 2: load records, split with src.data.split_by_image, LOCK the test set.
    # TODO 3: build Dataset + collator: apply chat template, then
    #         src.data.mask_prompt_labels so loss is only on the answer.
    # TODO 4: wrap with peft LoraConfig using cfg["lora"], train with Trainer.
    # TODO 5: save adapter to cfg["train"]["output_dir"].
    raise NotImplementedError("Fill in the TODOs for the chosen model.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
