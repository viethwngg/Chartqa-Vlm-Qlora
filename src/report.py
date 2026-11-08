"""Offline report generation from saved JSONL predictions; no model loading."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.eval import PROTOCOL, evaluate, normalize, paired_compare, relaxed_match

FIELDS = ("sample_id", "image_id", "question", "gold", "prediction")


def load_predictions(path):
    raw = Path(path).read_bytes()
    rows = {}
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) for k in FIELDS):
            raise ValueError(f"{path}:{line_number}: required fields must be strings: {FIELDS}")
        if any(not row[k].strip() for k in FIELDS if k != "prediction") or not normalize(row["gold"]):
            raise ValueError(f"{path}:{line_number}: empty identity, question, or gold")
        if row["sample_id"] in rows:
            raise ValueError(f"{path}:{line_number}: duplicate sample_id {row['sample_id']}")
        rows[row["sample_id"]] = row
    if not rows:
        raise ValueError(f"{path}: no predictions")
    return rows, hashlib.sha256(raw).hexdigest()


def build_report(predictions_path, baseline_path=None, n_boot=1000, seed=42, tolerance=0.05):
    rows, digest = load_predictions(predictions_path)
    ids = sorted(rows)
    ordered = [rows[i] for i in ids]
    predictions = [r["prediction"] for r in ordered]
    golds = [r["gold"] for r in ordered]
    images = [r["image_id"] for r in ordered]
    settings = dict(tol=tolerance, n_boot=n_boot, image_ids=images, seed=seed)
    report = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "bootstrap": {"unit": "chart", "replicates": n_boot, "seed": seed, "confidence": 0.95},
        "relaxed_tolerance": tolerance,
        "inputs": {"predictions": {"path": str(predictions_path), "sha256": digest}},
        "quality": evaluate(predictions, golds, **settings),
        "limitations": ["Manifest coverage and run comparability require external validation.",
                        "Categories require manual image-grounded review; no systems metrics measured."],
    }
    errors = []
    if baseline_path is not None:
        baseline_rows, baseline_digest = load_predictions(baseline_path)
        if set(baseline_rows) != set(rows):
            raise ValueError("Paired files must contain exactly the same sample IDs.")
        for sample_id in ids:
            for field in ("image_id", "question", "gold"):
                if rows[sample_id][field] != baseline_rows[sample_id][field]:
                    raise ValueError(f"Paired {field} mismatch for sample {sample_id}.")
        baseline = [baseline_rows[i]["prediction"] for i in ids]
        report["inputs"]["baseline"] = {"path": str(baseline_path), "sha256": baseline_digest}
        report["baseline_quality"] = evaluate(baseline, golds, **settings)
        report["paired"] = paired_compare(baseline, predictions, golds, **settings)
        counts = dict(fixed=0, remaining=0, new=0, both_correct=0)
        for row, base in zip(ordered, baseline):
            before = relaxed_match(base, row["gold"], tolerance)
            after = relaxed_match(row["prediction"], row["gold"], tolerance)
            transition = {(False, True): "fixed", (False, False): "remaining",
                          (True, False): "new", (True, True): "both_correct"}[before, after]
            counts[transition] += 1
            errors.append({
                "sample_id": row["sample_id"], "image_id": row["image_id"],
                "question": row["question"], "gold": row["gold"],
                "baseline_prediction": base, "fine_tuned_prediction": row["prediction"],
                "transition": transition, "error_category": "", "notes": "",
            })
        report["transitions"] = counts
    return report, errors


def render_summary(report):
    quality = report["quality"]
    lines = ["# Offline prediction report", "", f"Protocol: `{report['protocol']}`.", "",
             f"Questions: {quality['n']}; charts: {quality['n_charts']}.", "",
             "| Metric | Score | Chart-bootstrap 95% CI |", "|---|---|---|"]
    for key, ci_key in (("exact_match", "exact_ci95"), ("relaxed_accuracy", "relaxed_ci95")):
        lo, hi = quality[ci_key]
        lines.append(f"| {key} | {100 * quality[key]:.2f}% | [{100 * lo:.2f}, {100 * hi:.2f}]% |")
    if "paired" in report:
        lines += ["", "Deltas are prediction minus supplied baseline, in percentage points.", "",
                  "| Metric | Delta | Paired 95% CI |", "|---|---|---|"]
        for key, values in report["paired"].items():
            lo, hi = values["ci95"]
            lines.append(f"| {key} | {100 * values['delta']:.2f} | [{100 * lo:.2f}, {100 * hi:.2f}] |")
        lines += ["", "Relaxed-score transitions: " + json.dumps(report["transitions"], sort_keys=True),
                  "", "Review errors.csv against the chart images before assigning categories."]
    lines += ["", "Input hashes and bootstrap settings are stored in metrics.json.", "",
              "Manifest coverage and run comparability were not validated here. Systems performance: Not measured.",
              "Model origin and experimental authenticity must be established by the run metadata."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; never overwrites an existing report")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.05)
    args = parser.parse_args(argv)
    try:
        if args.output_dir.exists():
            raise ValueError("Output directory already exists; choose a new destination.")
        report, errors = build_report(args.predictions, args.baseline,
                                      args.bootstrap_samples, args.seed, args.tolerance)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output_dir / "summary.md").write_text(render_summary(report), encoding="utf-8")
        if errors:
            with (args.output_dir / "errors.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(errors[0]))
                writer.writeheader()
                writer.writerows(errors)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Report saved to {args.output_dir}")


if __name__ == "__main__":
    main()
