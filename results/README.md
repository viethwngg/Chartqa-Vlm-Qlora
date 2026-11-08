# Experiment artifact contracts

**No model experiment has been run.** This folder contains templates only. Empty CSV headers and JSON `null` values mean unmeasured, not zero. No synthetic benchmark rows belong here. Unit tests create artificial fixtures under temporary directories only.

## Prediction input, version 1

Use UTF-8 JSONL, one object per question, with required string fields:

| Field | Contract |
|---|---|
| `sample_id` | Nonempty unique stable ID from the locked manifest |
| `image_id` | Nonempty canonical chart identity shared by all questions on that chart |
| `question` | Original nonempty question |
| `gold` | Original nonempty single answer string; retain numeric formatting |
| `prediction` | Raw decoded generated continuation; empty string permitted as a wrong answer |

The report CLI rejects duplicates, missing fields, non-string answers, and empty files. Extra fields are allowed but not propagated into reports. Paired files require identical sample sets, image IDs, questions, and raw golds; ordering may differ. It cannot verify that supplied predictions came from a model or that two files both cover the full test manifest. Maintain run metadata and independently audit manifest coverage/comparability.

## Offline report outputs

`python -m src.report --help` documents the implemented CLI. Run it on existing predictions, with an optional `--baseline` file. A **new** output directory is required; existing destinations are rejected. `--bootstrap-samples` defaults to 1000, `--seed` to 42, and `--tolerance` to 0.05. Altered tolerances are recorded but are not the primary E0–E4 protocol.

| File | Contents |
|---|---|
| `metrics.json` | `schema_version`, metric protocol, input paths/SHA-256, tolerance, chart-bootstrap settings, question/chart counts, EM and relaxed accuracy and marginal CIs; optionally baseline quality, paired deltas/CIs, transition counts |
| `summary.md` | Quality percentages, paired percentage-point deltas, and limits; systems values remain unmeasured |
| `errors.csv` | Paired rows only: sample/image ID, question, gold, baseline and fine-tuned predictions, transition, blank category and notes |

JSON quality/CIs are fractions in [0,1]; paired deltas/CIs are in [-1,1]. All CIs are percentile 95% intervals. The baseline is normally E0 (zero-shot); for an ablation comparison it may be E2, so preserve the baseline input identity rather than assuming every baseline is zero-shot. `errors.csv` contains all rows, including both-correct, to retain denominators. Categories are manual; the CLI only computes correctness transitions.

## Run layout and metadata

Suggested future run layout under Git-ignored `outputs/<run-id>/`: resolved config, code commit/dirty diff, environment lock, dataset manifests/hash, `run_metadata.json`, raw `predictions.jsonl`, latency samples, training logs, checkpoints, final adapter, and uniquely named report directories. Archive immutable artifacts before publishing selected summaries in `results/`.

- `run_metadata.template.json` defines unmeasured provenance/systems fields; fill from instrumentation, not estimates presented as measurements. It is a contract, not a validated run or enforced JSON Schema.
- `model_quality.template.csv` defines display-ready quality and paired-delta columns; CI columns distinguish metric and marginal versus paired uncertainty.
- `systems_performance.template.csv` separates training/inference peaks, times, throughput, GPU-hours, load time, effective batch, parameter counts and adapter size.
- `ablation_analysis.template.csv` compares an explicit experiment/reference pair and requires an evidence-based interpretation.
- `error_analysis.template.csv` matches the report review columns. Add manual annotations to a copy, leaving raw machine outputs immutable.

CSV templates contain headers only. Human-facing unmeasured tables in README use Pending/Not measured. Numeric machine fields use null (JSON) or empty cells (CSV), never fabricated zeros. E0 training fields are not applicable. Use the [experiment plan](../docs/EXPERIMENT_PLAN.md) for all timing boundaries and annotation rules.

Full experiment collection, run-schema enforcement, plot generation, category aggregation and systems instrumentation remain TODO. The CLI automatically produces quality reports only; it does not populate these systems templates or certify experimental comparability.
