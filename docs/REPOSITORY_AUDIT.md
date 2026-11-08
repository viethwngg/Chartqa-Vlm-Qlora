# Repository audit and verification record

Audit date: **2026-11-08** (user timezone: Asia/Saigon). Starting commit: `fb592e0` (`Initial project skeleton`). Working tree was clean before this upgrade. No `AGENTS.md` was found in the repository or checked parent directories. No CV/resume file was present.

## Inspection before edits

Read the complete README, every `src/` file, `configs/base.yaml`, `tests/test_basic.py`, `requirements.txt`, `.gitignore`, Git file list/history, and `results/` and `notebooks/` including hidden contents. Both artifact directories contained only empty `.gitkeep` files. There were no script directories, real results, saved models, dataset files, or notebooks in the tracked tree.

| Component at audit start | Verified behavior / gap |
|---|---|
| README | Work-in-progress sketch; listed training command could not train; model/data IDs were TODO |
| `src/data.py` | Supplied-ID grouping/splitting and prefix masking worked; no data ingestion; loader raises `NotImplementedError` |
| Split behavior | Deterministic group shuffling; test-first/val/train cap allocation; whole groups may be unused; docstring did not fully describe behavior |
| Label masking | Prefix-only helper; not a processor-aware multimodal answer-only collator |
| `src/eval.py` | Basic metrics and question bootstrap; `zip` silently ignored unequal prediction lengths; empty evaluation looked like 0 accuracy; invalid bootstrap sizes unguarded |
| Numeric parsing | Removed arbitrary commas/percent signs, so malformed numeric strings and percent scale could be mis-scored; unused percent flag |
| `src/utils.py` | YAML parsed into a dictionary; seed 42 and learning rate float verified; no configuration schema or placeholder rejection |
| `src/train.py` | `--help` functional; calling main parsed config and seeded, then raised `NotImplementedError`; no model, loop, checkpoints, resume or export |
| Dependencies | ML stack present locally; `requirements.txt` entirely unpinned; PATH `pytest` referred to a different Python install, so used `.venv` Python explicitly |
| Tests | Four CPU utility tests passed; no model, reporting or checkpoint integration tests |
| Git hygiene | Existing ignore rules covered data, virtualenv, checkpoints, outputs and model binaries; no hardcoded credentials observed in tracked files |

## Actual environment observed

These are environment observations, **not training benchmark results** and not a dependency lock verified for a chosen model.

| Item | Observed value |
|---|---|
| OS / interpreter | Windows; existing `.venv`, Python 3.11.9 |
| GPU | NVIDIA GeForce GTX 1660 Ti |
| Device memory reported by `nvidia-smi` | 6,144 MiB |
| NVIDIA driver | 576.88 |
| Compute capability | 7.5 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime / availability | 12.8 / True |
| Native BF16 support reported by PyTorch | False |
| Transformers / PEFT | 5.22.1 / 0.21.2 |
| bitsandbytes / Accelerate | 0.50.2 / 1.15.0 |
| datasets / Pillow | 5.1.0 / 13.0.0 |
| PyYAML / NumPy / pytest | 6.0.3 / 2.4.6 / 9.2.0 |

All six ML imports (`torch`, `transformers`, `peft`, `bitsandbytes`, `accelerate`, `datasets`) succeeded. A small `Linear4bit(64, 32)` NF4 layer with double quantization and FP16 compute completed forward propagation and input-gradient backward propagation on CUDA with finite values. This tests a kernel path, not LoRA integration, optimizer updates, selected-model compatibility, training speed, or VLM memory fit. The original YAML's T4 comment is not a description of the detected machine.

Observed warnings concerned Windows distributed output redirection and missing Triton flop counting. Neither prevented the tiny test; neither demonstrates readiness for all training kernels. No model weights, dataset, or expensive training job was downloaded/launched.

Reproduce the hardware inventory with `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader`. For package versions use `python -m pip list` inside the activated environment. The lightweight GPU probe used this Python code (requires the existing ML stack; this is not part of CPU tests):

```python
import torch
import bitsandbytes as bnb

print(torch.__version__, torch.version.cuda, torch.cuda.is_available())
print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print(torch.cuda.is_bf16_supported(including_emulation=False))
x = torch.randn(2, 64, device="cuda", dtype=torch.float16, requires_grad=True)
layer = bnb.nn.Linear4bit(
    64, 32, bias=False, compute_dtype=torch.float16,
    quant_type="nf4", compress_statistics=True,
).cuda()
y = layer(x)
y.float().square().mean().backward()
torch.cuda.synchronize()
assert torch.isfinite(y).all() and torch.isfinite(x.grad).all()
```

## Changes justified by findings

- Rewrote README with research questions, actual status, architecture/inference diagrams, candidate review, evaluation rules, honest tables and reproducibility commands.
- Added actionable `TASKS.md`, experiment/measurement plan, defensible CV bullets, and artifact contracts/templates. The checklist distinguishes completed utility/design work from unrun experiments.
- Fixed metric length/empty/argument handling and numeric grammar; made percent scale and zero behavior explicit under versioned `chartqa_local_v1`. Added seeded chart-clustered and paired bootstrap. This intentionally changes the flawed legacy percent handling; no existing experimental data required migration.
- Added an offline JSONL report CLI with strict sample alignment, input hashes, quality/paired intervals, correctness transitions, manual-review CSV, and non-overwrite behavior. It does not run inference or certify experiment provenance.
- Added data argument validation and corrected splitting documentation without changing the valid-input allocation algorithm. Prefix masking remains a helper, not a finished collator.
- Added lightweight development requirements and meaningful metric/data/report tests. Original training code, YAML and ML requirements remain intact because selecting/integrating a model would exceed the documentation upgrade's safe verification scope.

## Validation after upgrade

| Check | Observed outcome |
|---|---|
| `.venv` Python `-m pytest -q` | 73 passed; artificial fixtures only |
| `python -m compileall -q src tests` | Passed |
| `python -m src.report --help` | Passed; implemented arguments match README |
| `python -m src.train --help` | Passed; actual training still raises as documented |
| Offline CLI integration | Paired report/CSV/JSON generation, reordering, ID/metadata mismatch rejection, duplicate rejection, deterministic settings and overwrite protection passed |
| JSON/CSV templates | Valid JSON; four CSVs with unique headers and no data rows |
| Local documentation links | Six Markdown files, 12 local links, and balanced code fences checked after all deliverables were created |
| Mermaid diagrams | Both README diagrams parsed and rendered to SVG with Mermaid CLI 11.12.0 using installed Chrome; generated artifacts under ignored `outputs/document-validation/` |
| `git diff --check` | Passed; only Git's platform line-ending notices |

No GPU model benchmarks, fine-tuning, dataset downloads, or cloud experiments were executed. Generated test metrics stayed in temporary test folders, not `results/`. The tiny CUDA probe has no reported latency, accuracy or memory-performance claim.

Mermaid syntax/render validation succeeded for both SVGs. An optional subsequent PNG preview stalled and its dedicated renderer was stopped; no pixel-level visual-review claim is made. SVG artifacts remain under the ignored validation directory.

## Remaining risks and next three priorities

1. **Pin and validate dataset/model interfaces:** implement source schema mapping, content identities, official-split overlap audit, immutable manifests, selected model/processor revisions, and config preflight. A local 6 GiB GPU may require a smaller model or image budget; measure before choosing.
2. **Implement and measure E0:** deterministic continuation-only inference with complete manifest coverage, prediction provenance, synchronized latency/memory and a validation report. Then freeze final test evaluation policy.
3. **Prove E2 training correctness before full runs:** model-specific answer masks, verified trainable targets, finite-loss overfit check, full checkpoint recovery and fresh-process adapter reload. Only then run controlled comparisons and rank/vision extensions.

Other limits: unpinned dependencies; no clean-environment model test; no full run metadata/schema validation; no automatic systems instrumentation, category annotation, plotting or benchmark aggregation. Image hashing and manifest verification are designed but unimplemented. The local scorer is explicitly distinct from upstream leaderboard implementations. CI outputs describe resampling uncertainty, not guaranteed experimental improvement.
