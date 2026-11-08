"""Offline CLI integration using artificial data in temporary directories only."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.report import build_report, load_predictions

ROOT = Path(__file__).resolve().parents[1]


def save_rows(path, predictions):
    rows = [{"sample_id": str(i), "image_id": f"chart{i // 2}", "question": "Synthetic test?",
             "gold": "yes", "prediction": p} for i, p in enumerate(predictions)]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return rows


def test_paired_cli_writes_artifacts_and_refuses_overwrite(tmp_path):
    baseline, tuned = tmp_path / "base.jsonl", tmp_path / "tuned.jsonl"
    save_rows(baseline, ["no", "no", "yes", "yes"])
    rows = save_rows(tuned, ["yes", "no", "no", "yes"])
    tuned.write_text("\n".join(json.dumps(r) for r in reversed(rows)), encoding="utf-8")
    out = tmp_path / "report"
    args = [sys.executable, "-m", "src.report", "--predictions", str(tuned), "--baseline", str(baseline),
            "--output-dir", str(out), "--bootstrap-samples", "100"]
    completed = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    report = json.loads((out / "metrics.json").read_text())
    assert report["quality"]["n"] == 4
    assert report["transitions"] == dict(fixed=1, remaining=1, new=1, both_correct=1)
    assert report["paired"]["relaxed_accuracy"]["delta"] == 0
    assert len(report["inputs"]["baseline"]["sha256"]) == 64
    assert report["bootstrap"]["unit"] == "chart"
    with (out / "errors.csv").open(newline="", encoding="utf-8") as stream:
        errors = list(csv.DictReader(stream))
    assert [r["sample_id"] for r in errors] == ["0", "1", "2", "3"]
    assert all(r["error_category"] == r["notes"] == "" for r in errors)
    original = (out / "metrics.json").read_bytes()
    repeated = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    assert repeated.returncode != 0 and "already exists" in repeated.stderr
    assert (out / "metrics.json").read_bytes() == original
    assert "percentage points" in (out / "summary.md").read_text()


def test_single_report_and_reproducibility(tmp_path):
    path = tmp_path / "pred.jsonl"
    save_rows(path, ["yes", "no"])
    report, errors = build_report(path, n_boot=30)
    assert report == build_report(path, n_boot=30)[0]
    assert errors == [] and "paired" not in report
    assert report["quality"]["relaxed_accuracy"] == 0.5


@pytest.mark.parametrize("field,value", [("sample_id", "other"), ("image_id", "wrong"),
                                         ("question", "changed"), ("gold", "no")])
def test_paired_alignment_validation(tmp_path, field, value):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    save_rows(a, ["yes"])
    rows = save_rows(b, ["yes"])
    rows[0][field] = value
    b.write_text(json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError):
        build_report(a, b)


@pytest.mark.parametrize("content", ["", "{", "[]", '{}',
    '{"sample_id":"a","image_id":"b","question":"q","gold":"g","prediction":null}'])
def test_bad_prediction_files(tmp_path, content):
    path = tmp_path / "bad.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_predictions(path)


def test_duplicate_ids(tmp_path):
    path = tmp_path / "duplicates.jsonl"
    rows = save_rows(path, ["yes"])
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[0]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_predictions(path)
