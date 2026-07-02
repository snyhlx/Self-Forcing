#!/usr/bin/env python3
"""Summarize V-Bench results by video source suffix.

The V-Bench aggregate JSON mixes all videos in a directory. For eval directories
that contain paired files such as prompt_0000_draft_head.mp4 and
prompt_0000_target_only.mp4, this script writes source-level summaries back into
each V-Bench run directory.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_METRICS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]
RAW_SCALE_METRICS = {"imaging_quality"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_source(video_path: str) -> str:
    name = Path(video_path).stem
    for suffix in ("draft_head", "target_only", "teacher", "wan21_1p3b_teacher"):
        if name.endswith(f"_{suffix}"):
            return suffix
    parts = name.split("_")
    if len(parts) >= 3 and parts[0] == "prompt" and parts[1].isdigit():
        return "_".join(parts[2:])
    return "unknown"


def to_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


def latest_eval_result(run_dir: Path) -> Path | None:
    files = sorted((run_dir / "vbench_outputs").glob("*_eval_results.json"))
    return files[-1] if files else None


def summarize_result(result_path: Path, metrics: list[str]) -> dict[str, Any]:
    payload = load_json(result_path)
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    video_paths_by_source: dict[str, set[str]] = defaultdict(set)

    for metric in metrics:
        if metric not in payload:
            continue
        entries = payload[metric][1]
        for entry in entries:
            source = infer_source(str(entry["video_path"]))
            values[source][metric].append(to_float(entry["video_results"]))
            video_paths_by_source[source].add(str(entry["video_path"]))

    by_source = {}
    for source in sorted(values):
        metric_means = {
            metric: sum(metric_values) / len(metric_values)
            for metric, metric_values in sorted(values[source].items())
            if metric_values
        }
        normalized_metrics = [
            value
            for metric, value in metric_means.items()
            if metric not in RAW_SCALE_METRICS
        ]
        by_source[source] = {
            "video_count": len(video_paths_by_source[source]),
            "mean_excluding_raw_scale_metrics": (
                sum(normalized_metrics) / len(normalized_metrics)
                if normalized_metrics
                else None
            ),
            "metrics": metric_means,
            "raw_scale_metrics": sorted(RAW_SCALE_METRICS & metric_means.keys()),
        }

    return {
        "result_path": str(result_path),
        "metrics": metrics,
        "raw_scale_metrics": sorted(RAW_SCALE_METRICS),
        "by_source": by_source,
    }


def write_tsv(path: Path, run_name: str, summary: dict[str, Any]) -> None:
    metrics = [metric for metric in summary["metrics"] if metric not in RAW_SCALE_METRICS]
    raw_metrics = [metric for metric in summary["metrics"] if metric in RAW_SCALE_METRICS]
    headers = [
        "run",
        "source",
        "video_count",
        "mean_excluding_raw_scale_metrics",
        *metrics,
        *[f"{metric}_raw" for metric in raw_metrics],
    ]
    lines = ["\t".join(headers)]
    for source, data in sorted(summary["by_source"].items()):
        row = [
            run_name,
            source,
            str(data["video_count"]),
            f"{data['mean_excluding_raw_scale_metrics']:.6f}",
        ]
        for metric in metrics:
            row.append(f"{data['metrics'].get(metric, float('nan')):.6f}")
        for metric in raw_metrics:
            row.append(f"{data['metrics'].get(metric, float('nan')):.6f}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize V-Bench results by draft/target video source.")
    parser.add_argument("--vbench_runs_root", default="/root/Self-Forcing/eval/vbench_runs")
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    args = parser.parse_args()

    runs_root = Path(args.vbench_runs_root).resolve()
    aggregate_lines: list[str] = []
    aggregate_header: str | None = None
    processed = 0

    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        result_path = latest_eval_result(run_dir)
        if result_path is None:
            continue
        summary = summarize_result(result_path, args.metrics)
        json_path = run_dir / "vbench_source_summary.json"
        tsv_path = run_dir / "vbench_source_summary.tsv"
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        write_tsv(tsv_path, run_dir.name, summary)

        lines = tsv_path.read_text(encoding="utf-8").splitlines()
        if lines:
            if aggregate_header is None:
                aggregate_header = lines[0]
                aggregate_lines.append(aggregate_header)
            aggregate_lines.extend(lines[1:])
        processed += 1
        print(f"Wrote {json_path} and {tsv_path}")

    if aggregate_lines:
        aggregate_path = runs_root / "vbench_source_summary.tsv"
        aggregate_path.write_text("\n".join(aggregate_lines) + "\n", encoding="utf-8")
        print(f"Wrote {aggregate_path}")
    print(f"Processed {processed} V-Bench run directories")


if __name__ == "__main__":
    main()
