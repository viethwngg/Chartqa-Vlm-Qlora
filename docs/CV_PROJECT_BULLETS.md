# CV project wording

No CV/resume file was found in the repository. The text below concerns this project only. Do not imply completed training, production deployment, TAO integration, or measured accuracy/memory/speed gains.

## Implemented achievements — CV-ready

**ChartQA VLM QLoRA — evaluation and reproducibility toolkit (ongoing)**  
Built and tested the data utilities and offline evaluation foundation for a controlled study of parameter-efficient chart question answering.

- Implemented deterministic chart-group splitting and prompt-prefix label masking, with tests for image-ID isolation, size limits, and invalid inputs.
- Built exact-match and 5% numeric-tolerance scoring with explicit percentage/zero handling and chart-clustered paired bootstrap confidence intervals.
- Developed a validated prediction-reporting CLI that aligns samples, preserves input provenance, and exports paired error transitions for manual review.
- Designed a controlled zero-shot/QLoRA experiment matrix and documented reproducibility, checkpoint-recovery, GPU measurement, and result-reporting acceptance criteria.

Evidence: `src/data.py`, `src/eval.py`, `src/report.py`, CPU tests, and `docs/EXPERIMENT_PLAN.md`. These bullets describe implemented utilities and completed design work; they do not claim that model experiments ran.

## Work-in-progress contributions — use with an explicit ongoing label

- Developing a PyTorch/Transformers/PEFT QLoRA workflow for small VLMs, with model-specific multimodal collation, frozen-base adapter training, and checkpoint recovery still to be integrated.
- Preparing matched baseline and rank/vision ablations, including validation-only selection and synchronized GPU timing/memory measurement.
- Planning image-grounded error review to distinguish value-reading, arithmetic, visual-text, legend/axis, reasoning, formatting, and unsupported-answer failures.

Installed packages and a tiny NF4 backward check demonstrate environment feasibility only. Do not write “fine-tuned a VLM,” “optimized GPU training,” or “completed ablation studies” until model runs provide evidence.

## Potential future achievements — examples, not CV-ready claims

Only replace bracketed fields with measured values linked to retained run artifacts:

- Fine-tuned [pinned model] using PyTorch, Transformers, PEFT and NF4 QLoRA on [N questions / G charts]; changed held-out relaxed accuracy from [A]% to [B]%, paired 95% CI for the difference [L, U] percentage points.
- Compared ranks 8/16/32 at fixed data, effective batch and update budget, measuring [quality difference], [training GiB], and [wall seconds] on [actual GPU].
- Verified resumed training and adapter reload, recording [tested steps/tolerance], and measured inference P50/P95 [values] ms under [declared boundaries].

A negative or inconclusive result is valid. Never turn pending fields, upstream model-card scores, or synthetic unit-test outputs into achievements.

## Interview explanation

“The project investigates whether small VLMs can answer chart questions better with parameter-efficient adaptation. I have implemented the data utilities and offline evaluation/reporting foundation, including shared-chart uncertainty and paired failure exports. The model training path is still a skeleton. I designed the next gates to establish a clean baseline, correct answer-only supervision, reproducible checkpoint recovery, and comparable GPU measurements before making improvement claims.”

Explain why question-level random splits can leak charts, why an adapter export is not a resumable checkpoint, why a quantized baseline controls precision confounds, and why paired clustered intervals are more appropriate than independent question resampling. When experiments exist, replace the pending narrative with evidence rather than changing the standard of proof.
