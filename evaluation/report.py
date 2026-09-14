"""
evaluation/report.py

FILE I/O for the evaluation infrastructure.

- MeasurementRecord streams are stored as JSONL (one JSON object per
  line): append-friendly for live capture, deterministic, easy to
  inspect and diff, trivially re-readable.
- EvaluationSummary is stored as a single JSON document.
- No database, no external formats, no pandas.
"""

import json
from pathlib import Path
from typing import List, Union

from evaluation.record import MeasurementRecord
from evaluation.metrics import EvaluationSummary


def write_jsonl(
    path: Union[str, Path],
    records: List[MeasurementRecord],
    append: bool = False,
) -> int:
    """
    Write records as JSONL. Returns the number of lines written.
    append=True keeps an existing capture file and adds to it (the
    live-capture pattern); append=False replaces the file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    count = 0
    with p.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(rec.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Union[str, Path]) -> List[MeasurementRecord]:
    """Read a JSONL record file. Blank lines are skipped."""
    records: List[MeasurementRecord] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(MeasurementRecord.from_dict(json.loads(line)))
    return records


def write_summary_json(
    path: Union[str, Path],
    summary: EvaluationSummary,
) -> None:
    """Write one evaluation summary as a JSON document."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")
