"""
evaluation/__main__.py

CLI for the offline evaluation analysis layer (no new dependencies):

    python -m evaluation analyze RECORDS.jsonl
        [--events EVENTS.jsonl]
        [--role CALIBRATION|VALIDATION|UNSPECIFIED]
        [--name report_name]
        [--json OUT.json] [--markdown OUT.md] [--csv OUT.csv]

Behavior:
- validates input; malformed files produce a one-line actionable error
  and exit code 2 (never a traceback dump)
- empty datasets are reported clearly (exit 2 from the CLI; the library
  API handles empty datasets gracefully)
- prints a concise deterministic summary
- optionally writes the JSON report, Markdown rendering and CSV summary

Exit codes: 0 success, 2 input error.
"""

import argparse
import sys
from typing import List, Optional

from evaluation.analysis import DatasetError, load_jsonl_records
from evaluation.reporting import (
    ROLES,
    analyze_dataset,
    build_report,
    load_jsonl_events,
    write_csv_summary,
    write_json_report,
    write_markdown_report,
)


def _print_summary(report: dict) -> None:
    """Concise deterministic stdout summary."""
    ds = report["dataset_summary"]
    provs = report["provenance_summary"]
    om = report["overall_metrics"]
    print(f"Dataset: {ds['total_records']} records | "
          f"scenes: {ds['scene_count']} | tracks: {ds['track_count']} | "
          f"frames: {ds['frame_range']}")
    counts = " ".join(f"{prov}={row['count']}"
                      for prov, row in sorted(provs.items()))
    print(f"Provenance: {counts}")
    print(f"Overall: valid={om['n_valid']} invalid={om['n_invalid']} "
          f"(invalid_rate={om['invalid_rate']:.3f}) | "
          f"MAE={om['mae_m']} | {om['validation_class']}")
    nav = report.get("navigation_summary")
    if nav and nav.get("frames"):
        counts = " ".join(f"{a}={n}"
                          for a, n in nav["action_counts"].items()
                          if n)
        print(f"Navigation (descriptive): {nav['frames']} frames | {counts} | "
              f"changes={nav['action_change_count']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation",
        description="Offline evaluation analysis for SINA measurement "
                    "records (deterministic, hardware-free).",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_analyze = sub.add_parser(
        "analyze", help="analyze a MeasurementRecord JSONL file")
    p_analyze.add_argument("records", help="records JSONL file")
    p_analyze.add_argument(
        "--events", default=None,
        help="optional EventLogger JSONL trace for navigation/risk/"
             "temporal analysis")
    p_analyze.add_argument(
        "--role", default="UNSPECIFIED", choices=list(ROLES),
        help="dataset role (never assumed to be validation data)")
    p_analyze.add_argument(
        "--name", default="sina_evaluation_report", help="report name")
    p_analyze.add_argument("--json", default=None,
                           help="write JSON report to this path")
    p_analyze.add_argument("--markdown", default=None,
                           help="write Markdown report to this path")
    p_analyze.add_argument("--csv", default=None,
                           help="write CSV summary to this path")

    args = parser.parse_args(argv)

    try:
        records = load_jsonl_records(args.records)
    except DatasetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if not records:
        print(
            f"ERROR: {args.records} contains no records (empty dataset)",
            file=sys.stderr)
        return 2

    events = None
    if args.events:
        try:
            events = load_jsonl_events(args.events)
        except DatasetError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    report = build_report(
        analyze_dataset(records, events),
        dataset_role=args.role,
        name=args.name,
        source=args.records,
    )

    _print_summary(report)

    if args.json:
        write_json_report(args.json, report)
        print(f"JSON report written: {args.json}")
    if args.markdown:
        write_markdown_report(args.markdown, report)
        print(f"Markdown report written: {args.markdown}")
    if args.csv:
        write_csv_summary(args.csv, report)
        print(f"CSV summary written: {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
