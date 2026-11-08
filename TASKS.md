# Engineering checklist

Status reflects verification, not file existence. **DONE** means the stated artifact/acceptance gate was checked; model experiments remain unrun. Priorities: P0 = MVP blocker, P1 = next value, P2 = extension. Dependencies refer to task IDs. Commands are run from the root with the activated environment. “Pending implementation” is deliberately not a runnable command.

**MVP:** validated immutable data manifests → one pinned base model → E0 → answer-only E2 with overfit/recovery proof → paired report. Rank/vision experiments are subsequent work. No cloud services or distributed training required.

## Phase 0 — Repository Audit

- [x] **A0 | P0 | DONE — Inspect repository and existing behavior.** Dependencies: none. Output: `docs/REPOSITORY_AUDIT.md`. Acceptance: all tracked source/config/test files and empty artifact folders inspected; stubs identified. Verification: `git ls-files`; original `python -m pytest -q` passed four tests.
- [x] **A1 | P0 | DONE — Validate environment and risks.** Dependencies: A0. Output: audit environment snapshot. Acceptance: package versions, CUDA availability, real device, native BF16 capability, and tiny NF4 forward/backward recorded separately from model feasibility. Verification: audit probe in `docs/REPOSITORY_AUDIT.md`.

## Phase 1 — Dataset Pipeline

- [ ] **D0 | P0 | TODO — Obtain and validate ChartQA.** Dependencies: A0. Output: pinned source revision and validated records. Acceptance: decode images; map question/answer schema; preserve raw labels; reject missing/empty/multiple unsupported labels; retain human/generated/source split fields. Verification: pending loader integration test; `load_records` still raises.
- [x] **D1 | P0 | DONE — Verify supplied-image grouping utility.** Dependencies: A0. Output: `src/data.py`, `tests/test_data.py`. Acceptance: deterministic, disjoint supplied IDs; whole groups; caps and invalid arguments tested. Does not certify real-data deduplication. Verification: `python -m pytest tests/test_data.py tests/test_basic.py -q`.
- [ ] **D2 | P0 | TODO — Lock train/validation/test manifests and statistics.** Dependencies: D0, D1. Output: manifests, image hashes, exclusions, statistics JSON. Acceptance: official test retained; no pixel-hash overlap; caps and unassigned groups reported; question/chart and subgroup counts saved. Verification: pending manifest validator; manually audit hash intersections before E0.
- [x] **D3 | P0 | DONE — Define and test answer normalization.** Dependencies: A0. Output: `src/eval.py`, metric policy in README. Acceptance: whitespace/case, punctuation, thousands, zero, percentage scale, invalid numeric strings tested; raw answers retained by reporting. Verification: `python -m pytest tests/test_eval.py -q`.

## Phase 2 — Zero-Shot Baseline

- [ ] **B0 | P0 | TODO — Select and load one pretrained VLM.** Dependencies: A1. Output: pinned model/processor IDs and revisions, module map, resolved environment. Acceptance: selected candidate loads on target hardware, processes an image, generates an answer; quantized dtype and target paths inspected. Verification: pending model smoke script; no weights downloaded in this upgrade.
- [ ] **B1 | P0 | TODO — Implement deterministic inference and saved predictions.** Dependencies: B0, D2. Output: inference entry point, validation predictions, metadata. Acceptance: greedy settings fixed; generated continuation only; stable IDs; complete manifest coverage; identical repeat under documented environment; load time recorded. Verification: pending inference integration test.
- [ ] **B2 | P0 | TODO — Execute and report E0.** Dependencies: B1, M0, M1. Output: validation E0 report, later locked-test report, latency/memory metadata. Acceptance: full declared manifest; raw outputs; warmup/timing protocol; no test-driven tuning. Verification: `python -m src.report --predictions outputs/E0/predictions.jsonl --output-dir outputs/E0/report` after artifacts exist.

## Phase 3 — QLoRA Training

- [ ] **T0 | P0 | TODO — Integrate NF4 and supported LoRA modules.** Dependencies: B0. Output: model loader, resolved adapter config, trainable parameter listing. Acceptance: NF4/double quantization confirmed; only intended adapters train; no accidental vision matches for E1–E3; frozen base parameters unchanged after a step. Verification: pending model backward/parameter test.
- [ ] **T1 | P0 | TODO — Build answer-only multimodal collator.** Dependencies: B0, D0. Output: collator and token-level tests. Acceptance: mask prompt, image, padding, non-answer specials; preserve answer/EOS; fail if truncation leaves zero targets; inspect variable-length batch. Existing prefix helper alone is insufficient. Verification: pending processor-backed collator test.
- [ ] **T2 | P0 | TODO — Implement training and validation logging.** Dependencies: T0, T1, D2. Output: training loop, resolved config, loss/validation history. Acceptance: overfit 20–50 training samples with finite loss; evaluate generation on validation; identical optimizer/step schedule across ablations. Verification: pending short GPU integration run.
- [ ] **T3 | P0 | TODO — Save, resume, and reload adapters.** Dependencies: T2. Output: full checkpoint and separate adapter export. Acceptance: interrupted run resumes step/optimizer/scheduler/scaler/RNG/sampler; compare next-step loss/weights to uninterrupted run within stated tolerance; fresh process adapter inference agrees with saved model. Verification: pending checkpoint recovery and reload tests.

## Phase 4 — Ablation Experiments

- [ ] **X0 | P0 | TODO — Freeze experiment controls.** Dependencies: D2, B1, T3. Output: resolved E0–E4 configs and hashes. Acceptance: same base/revision, data, image/token limits, prompt, decoding, steps, optimizer, seed; alpha/rank=2; only intended changes; validation-only selection. Verification: pending resolved-config diff check.
- [ ] **X2 | P0 | TODO — Run E2 rank 16.** Dependencies: X0, B2. Output: adapter, validation/test predictions, training and systems records. Acceptance: pass smoke/recovery gate before full run; paired report on locked test after selection. Verification: report CLI using E2 and E0 inputs.
- [ ] **X1 | P1 | TODO — Run E1 rank 8 and E3 rank 32.** Dependencies: X2. Output: two run directories and rank comparison. Acceptance: identical budget/data/order/controls; report quality deltas, memory, time; failed/OOM runs retained as such. Verification: offline CLI per experiment; manual run metadata comparison.
- [ ] **X4 | P2 | TODO — Run E4 vision adaptation.** Dependencies: X2. Output: explicit vision target map and E4 artifacts. Acceptance: only added vision adapters differ from E2; base vision weights remain frozen; feasibility measured; scope changes documented. Verification: pending vision backward check then paired report against E2 and E0.
- [ ] **X5 | P2 | TODO — Repeat important comparisons across seeds.** Dependencies: X2; compute budget. Output: E0/E2 seed-specific reports. Acceptance: predeclare seeds and distinguish between-seed variation from within-test bootstrap. Verification: inspect run manifests; no pooled pseudo-replicated significance claim.

## Phase 5 — Benchmark & Performance

- [x] **M0 | P0 | DONE — Implement offline quality and paired uncertainty.** Dependencies: D3. Output: evaluator/report CLI. Acceptance: EM, 5% relaxed accuracy, chart bootstrap, paired differences; reject empty/mismatched/duplicate samples; deterministic tests. Verification: `python -m pytest tests/test_eval.py tests/test_report.py -q`.
- [ ] **M1 | P0 | TODO — Instrument training and inference separately.** Dependencies: B0, T2 for training portion. Output: systems records matching `results/run_metadata.template.json`. Acceptance: synchronized wall time, warmup, P50/P95, throughput, load time, train/inference peaks, effective batch, trainable count, adapter bytes, actual hardware, GPU-hours and optional cost/rate source. Verification: pending benchmark smoke run; measurement definitions in experiment plan.
- [x] **M2 | P1 | DONE — Define benchmark contracts and pending tables.** Dependencies: A0. Output: `results/README.md`, CSV/JSON templates, README comparison tables. Acceptance: units/boundaries explicit; unmeasured values null or Pending; no fabricated model numbers. Verification: parse JSON/CSV and inspect README tables.
- [ ] **M3 | P1 | TODO — Populate measured tables and plots.** Dependencies: X2, M1. Output: comparison tables plus quality/memory/latency plots. Acceptance: every point traceable to a run; uncertainty labeled; missing runs stay Pending. Verification: report regeneration and manual provenance audit; plotting script pending.

## Phase 6 — Error Analysis

- [x] **F0 | P0 | DONE — Export paired correctness transitions.** Dependencies: M0. Output: `errors.csv` and summary counts. Acceptance: stable IDs, question/gold/both predictions, fixed/remaining/new/both_correct, blank manual category/notes; metadata mismatch rejected. Verification: `python -m pytest tests/test_report.py -q`.
- [ ] **F1 | P1 | TODO — Inspect and categorize actual failures.** Dependencies: X2, F0. Output: annotated error CSV and representative image-linked cases. Acceptance: review value extraction, arithmetic, small text, legend/axis, multi-point reasoning, formatting, hallucination; evidence notes; leave uncertain cases unclassified. Verification: manual review using `docs/EXPERIMENT_PLAN.md` rubric.
- [ ] **F2 | P1 | TODO — Compare category outcomes.** Dependencies: F1. Output: fixed/remaining/new category counts with denominators and examples. Acceptance: same review cohort; sampled versus complete counts labeled; no unsupported category-level improvement claim. Verification: recompute counts from annotations; report selection bias.

## Phase 7 — Engineering Quality

- [x] **Q0 | P0 | DONE — Run meaningful CPU tests and offline integration.** Dependencies: D1, D3, M0, F0. Output: tests and audit validation record. Acceptance: metric edge cases, cluster pairing, CLI generation, non-overwrite and invalid input behavior pass. Verification: `python -m pytest -q`; `python -m compileall -q src tests`.
- [ ] **Q1 | P0 | TODO — Validate model/training configuration.** Dependencies: B0, X0. Output: schema and startup preflight. Acceptance: reject TODO IDs, unknown keys, unsupported dtypes/modules, invalid caps/LR/batch, incompatible model/processor before loading weights. Verification: pending config tests; current `load_config` is parsing only.
- [ ] **Q2 | P0 | TODO — Capture reproducible runs and logs.** Dependencies: B0, T2. Output: resolved lock, run metadata, seed/kernel settings, logs. Acceptance: revisions/commit/diff, dataset manifest hash, environment/hardware, finite-loss/OOM failures, validation selection and checkpoint locations recorded. Verification: fresh environment smoke and recovery test, pending. Existing `set_seed` alone is not a determinism guarantee.
- [x] **Q3 | P1 | DONE — Automate offline report regeneration.** Dependencies: M0. Output: JSON metrics, Markdown summary, paired review CSV. Acceptance: input hashes and protocol/seed saved; new destination required; artificial integration fixtures stay outside results. Verification: `python -m pytest tests/test_report.py -q`.

## Phase 8 — Documentation & Interview Preparation

- [x] **I0 | P0 | DONE — Document architecture, methodology, and limits.** Dependencies: A0, M2. Output: README and experiment plan. Acceptance: two Mermaid diagrams; commands map to implemented entry points; future integration labeled; no measured claims for unrun experiments. Verification: command help, Markdown link checks, Mermaid parsing/rendering recorded in audit.
- [x] **I1 | P0 | DONE — Write defensible CV and interview narrative.** Dependencies: A0, Q0. Output: `docs/CV_PROJECT_BULLETS.md`. Acceptance: concise description and 3–4 verified bullets separate from WIP/future outcomes; no unrelated CV edits. Verification: cross-check every claim against source/tests and run status.
- [ ] **I2 | P1 | TODO — Add results-backed interview examples.** Dependencies: M3, F2. Output: measured figures and an actual failure walkthrough. Acceptance: accurately explain what ablations demonstrate, including negative/null results; CV metrics trace to artifacts. Verification: manual evidence review.
