# Experiment and measurement plan

**Experiment status: all E0–E4 Not run.** The offline evaluator/report path is implemented. This document specifies future model work; it does not certify a training pipeline. MVP is E0 plus E2 on one selected base, one locked split, seed 42.

## Selection and data gates

Evaluate the three candidates in the [README](../README.md) for license, processor/chat support, visual detail, exact PEFT targets, NF4 backward support, and measured memory fit. Start feasibility with SmolVLM-500M; this is a recommendation, not a selection. Save the immutable base/processor revisions and complete module map before choosing. Do not copy upstream accuracy into project results. A second base model is a future independent comparison, not an extra point in the rank ablation.

Use [ChartQA](https://github.com/vis-nlp/ChartQA), with the [HuggingFaceM4 distribution](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) as the candidate loader source. Pin the data revision; validate schema and original membership. Store `sample_id`, content-derived `image_id`, relative image path, question, raw gold, source split, question source, dimensions and hash. The current report format supports one string gold; reject unsupported multi-label records instead of silently selecting one.

Hash RGB pixels plus dimensions to group identical charts, audit cross-split duplicates, and quarantine overlapping groups with an exclusion record. Exact hashes do not detect resized/cropped near-duplicates; review suspicious near-duplicates separately. Preserve official test charts; sample within official splits deterministically at chart level. If using a custom development pool, label its scores as a custom subset, never official full-test performance. Freeze sorted manifests and SHA-256 hashes. Report approximate caps, actual question/chart counts, unused groups, answer types, human/generated counts, image sizes, and any corrupted images.

Validation is for prompt/hyperparameter/checkpoint selection. Lock configurations and checkpoint selection rules before final held-out testing. Do not tune on failed test examples. If test leakage is discovered, invalidate and regenerate affected reports with new manifest versions; never silently rewrite prior results.

## Matrix and controlled variables

| ID | Variant | Rank | Alpha | Dropout | Trainable scope | Primary comparison | Status |
|---|---|---|---|---|---|---|---|
| E0 | Zero-shot matched NF4 base | N/A | N/A | N/A | None | Reference | Not run |
| E1 | QLoRA small | 8 | 16 | 0.05 | Language attention adapters | E1 vs E2 | Not run |
| E2 | QLoRA main | 16 | 32 | 0.05 | Same language attention adapters | E2 vs E0 | Not run |
| E3 | QLoRA large | 32 | 64 | 0.05 | Same language attention adapters | E3 vs E2 | Not run |
| E4 | QLoRA plus vision | 16 | 32 | 0.05 | E2 plus enumerated vision adapters | E4 vs E2 | Not run |

All trainable updates are LoRA; base vision weights and projector remain frozen even for E4. If the architecture lacks suitable vision linear targets or memory does not fit, record E4 as blocked, with evidence. Full vision unfreezing would be a different experiment, not an undocumented implementation of E4. Generic suffix targeting can accidentally adapt both language and vision: inspect exact names.

Hold base/revision, quantized inference precision, processor/image-token budget, prompt, manifests/order, seed, optimizer, scheduler, number of updates, effective batch, checkpoint selection, and decoding constant. Keep `alpha/r = 2` so scaling is not an unintended rank confound. Log every resolved config diff. Rank still changes trainable capacity and optimizer memory by design.

Planned training defaults: one epoch, batch 1, accumulation 16, one GPU, LR 2e-4, FP16, gradient checkpointing, NF4 plus double quantization. AdamW is proposed; betas/decay/scheduler/warmup must be explicitly resolved before runs. The original YAML does not encode all these settings and remains a sketch. Effective batch is nominally 16; record treatment of the final partial accumulation window and actual consumed samples.

Use the same concise answer prompt, greedy decoding, one beam and 32-token cap. Decode generated continuation only, preserve raw output, and pass the entire continuation to the evaluator. Do not extract whichever number happens to match gold. Record cap-hit/truncated outputs and effective processor settings, not merely a nominal pixel cap. Model eval mode disables dropout for all inference.

## Training acceptance gates

1. A real selected-model example must pass processor, generation, NF4 forward/backward and LoRA gradient checks with finite loss; measure memory.
2. Inspect a mixed-length batch's labels. Prompt, padding and image tokens must be ignored; answer and intended EOS must remain. Fail on zero supervised tokens or answer truncation. Assert base weights remain fixed after an optimizer step.
3. Overfit 20–50 training samples before a full run. Save loss, actual optimizer steps, supervised token counts, trainable parameter count, precision and memory logs. This is a debugging check, not a test benchmark.
4. Compare an uninterrupted short run with a deliberately interrupted/resumed run. Restore optimizer, scheduler, scaler, step, RNG and data position; compare subsequent loss/weights within documented tolerance. Export adapters separately and compare fresh-process reloaded inference.
5. Complete E0/E2 validation and inspect throughput/memory before budgeting rank or vision variants. Select checkpoints by validation relaxed accuracy, with earliest checkpoint breaking ties. Freeze this rule before running.

## Quality metric specification

The implemented local policy is `chartqa_local_v1`, following the numeric tolerance in the [ChartQA paper](https://aclanthology.org/2022.findings-acl.177/) and the percent-to-fraction convention in [Pix2Struct](https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py). Differences from upstream scoring are disclosed below; this is not a certified official leaderboard implementation.

Let `N(s)` strip/collapse whitespace, lowercase, and remove one terminal period unless part of an ellipsis. Do not remove all punctuation, minus signs, percent signs, units, or articles. `EM = sum(1[N(pred_i) = N(gold_i)]) / n`. Numeric formatting is deliberately not canonicalized for EM.

For relaxed scoring, accept a full finite decimal/scientific string or valid comma-grouped thousands, optionally followed by one `%` (divide that value by 100). `50%` and `0.5` are relaxed-equivalent; `50%` and `50` are not. For nonzero numeric gold `g`, correctness is `abs(p-g) / abs(g) <= 0.05`; for zero, `p == 0`. Negative gold uses `abs(g)`. Otherwise fall back to EM. No ANLS, list reordering, unit conversion, or chain-of-thought answer extraction. NaN/infinity and malformed thousands are not numeric. Empty prediction is an incorrect answer; empty gold is invalid.

Local deviations: normalized text case/whitespace/sentence period; strict finite numeric grammar; comma support; numeric equality for zero (`0.0` vs `0` passes). Upstream zero fallback and text normalization may differ. Percent semantics in source annotations can be ambiguous; inspect these in error analysis without changing scoring after test results are known. Years and IDs remain numeric under this generic tolerance; report that limitation and EM alongside relaxed accuracy.

For every prediction file, require nonempty unique sample IDs and chart IDs, nonempty questions/golds, and string predictions. Paired files must have exactly the same IDs, chart IDs, questions, and raw golds. Sort by sample ID before scoring. No missing sample may be silently dropped. Separately compare IDs to the locked manifest, because matching two incomplete files is not proof of complete evaluation.

## Confidence intervals and comparisons

Point estimates are question-weighted. For `G` distinct charts, draw `G` chart IDs uniformly with replacement and include every question of each drawn chart, retaining repeated draws. Divide summed correct answers by sampled question count. Repeat 1,000 times with seed 42; take linearly interpolated 2.5th and 97.5th percentiles of the sorted replicates. This respects within-chart dependence without changing the point estimate to a macro chart average.

For paired comparisons, compute per-question `adapted_correct - baseline_correct` and use the same chart draws for the difference. Report delta and interval for EM and relaxed accuracy. CLI JSON uses fractions in [0,1], deltas in [-1,1]; display quality as percent and differences as percentage points. The Python evaluator can use question bootstrap if no chart IDs are supplied, but experiment reports must use chart IDs. Report n questions and n charts. One-chart intervals collapse; few charts are unreliable. Do not infer significance from overlapping or non-overlapping marginal CIs. Intervals do not include training-seed uncertainty, model selection bias, or multiplicity across ablations. Treat E2–E0 as the predeclared primary comparison; other comparisons are exploratory.

## Systems measurement specification (instrumentation pending)

All timing uses a monotonic wall clock and CUDA synchronization at interval boundaries. Store exact GPU name/count/VRAM/driver, CPU/RAM, OS, CUDA/runtime, package versions, attention implementation, precision, image/token limits, input length distribution, batch size, decoding, warmup count, and software commit. Never assume a T4 from the original YAML comment.

| Measurement | Boundary, unit, and interpretation |
|---|---|
| Model load time | Seconds from local cached load start through processor/model/optional adapter setup, device placement and synchronization; downloads excluded and separately recorded |
| Training wall time | Seconds from first train iteration to final checkpoint completion; include validation/checkpoint I/O, exclude data download and initial load; record exclusions and interruption time |
| Training throughput | Actual consumed training examples / training wall seconds; includes defined validation/checkpoint overhead; tokens/s optional |
| Effective batch | Per-device microbatch × accumulation × GPU count; record tail handling |
| Trainable parameters | `requires_grad` count plus logical total parameter count and exact names; beware packed quantized storage when computing denominator |
| Adapter size | Sum of exported adapter weight/config bytes; exclude base model, optimizer and checkpoints; list included files |
| Peak train GPU allocation | Reset PyTorch peak stats after model setup, measure training interval; peak includes resident model; bytes and GiB; record reserved peak separately |
| Peak inference allocation | Separate fresh measurement after load/warmup, reset peak stats, same timed cohort; includes resident model; do not reuse training peak |
| Device VRAM context | GPU capacity and optional process/device readings separately; PyTorch allocation is not whole-device utilization |
| GPU-hours | Defined phase wall seconds × actually used GPU count / 3600; report training and inference separately, with load hours separately if tracked |
| Estimated cost | Only with actual hourly rate, currency, source/date and billable boundary; otherwise null/Not measured; do not invent local GPU prices |
| Inference latency | Batch=1, 10 fixed warmup examples excluded; at least 100 timed examples if available, otherwise report n and limitation; synchronized per-request milliseconds |
| Inference scope | Preloaded decoded RGB image + question through preprocessing, transfer, generation and decode; exclude disk access/model load/report scoring; record raw per-request times |
| Inference P50/P95 | Linear percentiles of per-request milliseconds; match sample order, hardware, precision, image settings, output cap and warmup |
| Inference throughput | Timed examples / total timed request seconds; store generated token counts because variable answer lengths affect comparison |

Keep warmup/evaluation subsets fixed across E0–E4. Run one benchmark workload at a time, record other GPU activity, and retain failed/OOM runs with status rather than imputing metrics. Repeat short timing runs if variance matters; do not spend GPU time during this documentation upgrade. NF4 support on a tiny layer is not a memory-fit or speed benchmark.

## Error collection and annotation

The report CLI exports all paired rows and relaxed-score transitions. `fixed`: E0 wrong, adapted correct; `remaining`: both wrong; `new`: E0 correct, adapted wrong; `both_correct`: both correct. Summary counts cover the full supplied cohort. Exporting all rows allows consistent denominators and later inspection of false-positive relaxed matches.

For a manageable first review, sort IDs and use a fixed seed to sample up to 20 cases from each of fixed/remaining/new; record population and reviewed counts, then inspect at least a few both-correct cases. This sampling/annotation is manual work, not implemented automatic classification. Open chart images using the locked manifest; retain sample/image ID, question, raw gold, zero-shot prediction, adapted prediction, transition, primary category, and notes. Add image-relative path in notes or a companion manifest. Store sanitized excerpts, not copied third-party chart collections, in Git as permitted by the dataset terms.

| Category code | Required evidence / collection rule |
|---|---|
| value_extraction | A single requested mark has an identifiable value and the prediction reads a different value; inspect original chart |
| arithmetic | Required operands can be identified but the operation/aggregation result is wrong; record operands and calculation; do not assume hidden reasoning |
| small_text | Visible label/tick ambiguity plausibly explains the answer; inspect original-resolution crop; mark uncertain if attribution is unsupported |
| legend_axis | Wrong series, unit, scale, or axis association; record the correct mapping |
| multi_point_reasoning | Incorrect comparison/order/logical relation across multiple points beyond a simple arithmetic slip; state the required relation |
| output_format | Correct content is present but units, prose, multiple answers, or formatting violate the answer policy; do not change scores retrospectively |
| unsupported_answer | Prediction cannot be grounded in the visible chart; avoid using this as a catch-all for every numeric mistake |

Leave category blank and explain uncertainty if evidence is insufficient. Use a primary category and note overlaps. For remaining errors, notes must explain whether the failure changed between models. Review a small second pass for consistency; do not claim inter-rater reliability without a second annotator. Category comparisons must report reviewed denominators and selection procedure; a fixed case does not prove that a category is solved. Representative real examples, category counts, and error-resolution claims are Pending.

## Artifacts and completion criteria

Use [results contracts](../results/README.md). Each run needs raw predictions, resolved config/revisions, dataset manifest/hash, code/environment/hardware metadata, checkpoint/adapter locations and metrics. Reports contain input SHA-256 hashes and metric protocol. Test fixtures belong only in temporary test directories. Do not overwrite prior reports; use new run/report IDs. Update measured tables only from traceable outputs, keep missing fields null, and retain experiment status.

MVP complete means E0/E2 have passed all data/model/recovery gates, test predictions cover the frozen manifest, paired uncertainty and systems measurements are present, and manually reviewed examples explain limitations. It does not require a positive accuracy improvement. E1/E3/E4, extra seeds, plots, and additional base models extend the project after this gate.
