# chartqa-vlm-qlora

Fine-tuning a small open-source vision-language model (VLM) for chart question answering with QLoRA, with a zero-shot baseline, ablations and error analysis.

> Status: work in progress. Results below are placeholders until experiments are run.

## Goal
Measure how much QLoRA fine-tuning improves a small VLM on ChartQA compared with zero-shot, and which design choices (LoRA rank, tuning the vision encoder) matter.

## Setup
- Model: `TODO`
- Dataset: ChartQA subset, split **by image** (train `N`, val `N`, test `N`), fixed seed
- Method: QLoRA (4-bit), loss on answer tokens only
- Metrics: exact match, relaxed accuracy (5% numeric tolerance), bootstrap 95% CI

## Results (fill with real numbers)
| Setting | Exact match | Relaxed acc. | 95% CI |
|---|---|---|---|
| Zero-shot | - | - | - |
| QLoRA (r=16) | - | - | - |
| QLoRA (r=8 / r=32) | - | - | - |
| QLoRA + vision tuning | - | - | - |

## Error analysis
TODO: error categories with examples (wrong value read, arithmetic, small text, answer format).

## Limitations
TODO: small training subset, one dataset, small model, limited seeds.

## Run
```bash
pip install -r requirements.txt
pytest -q
python -m src.train --config configs/base.yaml
```

## Structure
```
configs/   experiment configs
src/       data.py (split, label masking), eval.py (metrics), train.py, utils.py
tests/     unit tests
results/   tables, plots, failure examples
notebooks/ error analysis
```
