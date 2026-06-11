#!/usr/bin/env python3
"""Run VBench over every SDVG output directory containing videos.

The script evaluates each directory that directly contains .mp4 files under
outputs/sdvg. For each directory, it writes a VBench prompt map, runs VBench in
custom_input mode, and appends one JSON record to vbench_results.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


DEFAULT_DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def direct_video_dirs(root: Path) -> list[Path]:
    video_dirs = set()
    skip_names = {"vbench_eval", "vbench_results"}
    for mp4 in root.rglob("*.mp4"):
        if any(part in skip_names for part in mp4.parts):
            continue
        video_dirs.add(mp4.parent)
    return sorted(video_dirs)


def find_nearest_profile(video_dir: Path, root: Path) -> Path | None:
    current = video_dir
    while root in (current, *current.parents):
        candidate = current / "profile.json"
        if candidate.is_file():
            return candidate
        if current == root:
            break
        current = current.parent
    return None


def prompt_map_from_profile(profile_path: Path | None) -> dict[str, str]:
    if profile_path is None:
        return {}

    profile = load_json(profile_path)
    prompt_map = {}
    for run in profile.get("runs", []):
        video_path = run.get("video_path")
        prompt = run.get("prompt")
        if video_path and prompt:
            prompt_map[str(Path(video_path).resolve())] = prompt
            prompt_map[Path(video_path).name] = prompt
    return prompt_map


def build_prompt_file(video_dir: Path, eval_dir: Path, sdvg_root: Path) -> tuple[Path, int, Path | None]:
    videos = sorted(video_dir.glob("*.mp4"))
    profile_path = find_nearest_profile(video_dir, sdvg_root)
    profile_prompts = prompt_map_from_profile(profile_path)

    prompt_map = {}
    for video in videos:
        resolved = str(video.resolve())
        prompt = profile_prompts.get(resolved) or profile_prompts.get(video.name) or video.stem
        prompt_map[resolved] = prompt

    rel_dir = video_dir.relative_to(sdvg_root)
    prompt_file = eval_dir / "prompt_maps" / rel_dir / "prompts.json"
    prompt_file.parent.mkdir(parents=True, exist_ok=True)
    prompt_file.write_text(json.dumps(prompt_map, ensure_ascii=False, indent=2), encoding="utf-8")
    return prompt_file, len(videos), profile_path


def collect_result_jsons(output_dir: Path, started_at: float) -> list[dict[str, Any]]:
    results = []
    for path in sorted(output_dir.rglob("*.json")):
        if path.stat().st_mtime + 1e-6 < started_at:
            continue
        try:
            payload = load_json(path)
        except Exception as exc:  # noqa: BLE001 - keep eval sweep moving.
            payload = {"parse_error": str(exc)}
        results.append(
            {
                "path": str(path),
                "payload": payload,
            }
        )
    return results


def has_eval_results(output_dir: Path) -> bool:
    return output_dir.is_dir() and any(output_dir.glob("*_eval_results.json"))


def already_done(jsonl_path: Path, sdvg_root: Path, vbench_output_root: Path) -> set[str]:
    done = set()
    if not jsonl_path.is_file():
        jsonl_done = set()
    else:
        jsonl_done = set()
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("returncode") == 0 and record.get("video_dir"):
                    output_dir = Path(record.get("vbench_output_dir", ""))
                    if has_eval_results(output_dir):
                        jsonl_done.add(record["video_dir"])
        done.update(jsonl_done)

    # Also trust completed VBench outputs on disk. This keeps --resume useful
    # even after vbench_results.jsonl was deleted or an earlier run was stopped.
    for eval_result in vbench_output_root.rglob("*_eval_results.json"):
        rel_dir = eval_result.parent.relative_to(vbench_output_root)
        video_dir = sdvg_root / rel_dir
        if video_dir.is_dir():
            done.add(str(video_dir))
    return done


def run_vbench_for_dir(
    *,
    vbench_python: str,
    vbench_launch_py: Path,
    ngpus: int,
    video_dir: Path,
    output_dir: Path,
    prompt_file: Path,
    dimensions: list[str],
    extra_args: list[str],
    dry_run: bool,
) -> tuple[list[str], int | None, float, list[dict[str, Any]]]:
    command = [
        vbench_python,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(ngpus),
        str(vbench_launch_py),
        "--videos_path",
        str(video_dir),
        "--prompt_file",
        str(prompt_file),
        "--mode",
        "custom_input",
        "--output_path",
        str(output_dir),
        "--dimension",
        *dimensions,
        *extra_args,
    ]
    if dry_run:
        return command, None, 0.0, []

    started_at = time.time()
    env = os.environ.copy()
    vbench_bin_dir = str(Path(vbench_python).resolve().parent)
    env["PATH"] = f"{vbench_bin_dir}:{env.get('PATH', '')}"
    process = subprocess.run(command, check=False, env=env)  # noqa: S603 - command is user-configurable eval CLI.
    elapsed_s = time.time() - started_at
    result_jsons = collect_result_jsons(output_dir, started_at)
    return command, process.returncode, elapsed_s, result_jsons


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SDVG output videos with VBench.")
    parser.add_argument("--sdvg_root", default="/root/Self-Forcing/outputs/sdvg")
    parser.add_argument("--eval_root", default="/root/Self-Forcing/eval")
    parser.add_argument("--vbench_bin", default="/root/vbench_venv/bin/vbench")
    parser.add_argument("--vbench_python", default="/root/vbench_venv/bin/python")
    parser.add_argument("--vbench_launch_py", default=None)
    parser.add_argument("--ngpus", type=int, default=1)
    parser.add_argument("--results_jsonl", default=None)
    parser.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMENSIONS)
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N video directories. 0 means all.")
    parser.add_argument("--resume", action="store_true", help="Skip directories already successful in results JSONL.")
    parser.add_argument("--dry_run", action="store_true", help="Print planned commands without running VBench.")
    parser.add_argument(
        "--vbench_extra_arg",
        action="append",
        default=[],
        help="Additional argument passed to VBench. Repeat for multiple args.",
    )
    return parser.parse_args()


def resolve_vbench_launch_py(vbench_python: str, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        return Path(explicit_path).resolve()

    venv_root = Path(vbench_python).expanduser().parent.parent
    matches = sorted(venv_root.glob("lib/python*/site-packages/vbench/launch/evaluate.py"))
    if not matches:
        raise FileNotFoundError(
            "Could not locate vbench/launch/evaluate.py. "
            "Pass --vbench_launch_py explicitly."
        )
    return matches[0]


def main() -> None:
    args = parse_args()
    sdvg_root = Path(args.sdvg_root).resolve()
    eval_root = Path(args.eval_root).resolve()
    results_jsonl = Path(args.results_jsonl).resolve() if args.results_jsonl else eval_root / "vbench_results.jsonl"
    vbench_output_root = eval_root / "vbench_outputs"
    vbench_launch_py = resolve_vbench_launch_py(args.vbench_python, args.vbench_launch_py)

    if not sdvg_root.is_dir():
        raise FileNotFoundError(f"Missing SDVG root: {sdvg_root}")
    if not args.dry_run:
        if not Path(args.vbench_python).is_file():
            raise FileNotFoundError(f"Missing VBench Python: {args.vbench_python}")
        if not vbench_launch_py.is_file():
            raise FileNotFoundError(f"Missing VBench launch script: {vbench_launch_py}")

    completed = already_done(results_jsonl, sdvg_root, vbench_output_root) if args.resume else set()
    video_dirs = [path for path in direct_video_dirs(sdvg_root) if str(path) not in completed]
    if args.limit > 0:
        video_dirs = video_dirs[: args.limit]

    print(f"Found {len(video_dirs)} video directories to evaluate")
    print(f"Results JSONL: {results_jsonl}")

    for index, video_dir in enumerate(video_dirs, start=1):
        rel_dir = video_dir.relative_to(sdvg_root)
        prompt_file, video_count, profile_path = build_prompt_file(video_dir, eval_root, sdvg_root)
        output_dir = vbench_output_root / rel_dir

        print(f"[{index}/{len(video_dirs)}] {video_dir} ({video_count} videos)")
        command, returncode, elapsed_s, result_jsons = run_vbench_for_dir(
            vbench_python=args.vbench_python,
            vbench_launch_py=vbench_launch_py,
            ngpus=args.ngpus,
            video_dir=video_dir,
            output_dir=output_dir,
            prompt_file=prompt_file,
            dimensions=args.dimensions,
            extra_args=args.vbench_extra_arg,
            dry_run=args.dry_run,
        )
        print(" ".join(command))

        record = {
            "video_dir": str(video_dir),
            "relative_dir": str(rel_dir),
            "video_count": video_count,
            "profile_json": str(profile_path) if profile_path else None,
            "prompt_file": str(prompt_file),
            "vbench_output_dir": str(output_dir),
            "dimensions": args.dimensions,
            "command": command,
            "returncode": returncode,
            "elapsed_s": elapsed_s,
            "result_jsons": result_jsons,
            "dry_run": args.dry_run,
        }
        if not args.dry_run:
            append_jsonl(results_jsonl, record)
        if returncode not in (0, None):
            print(f"WARNING: VBench failed for {video_dir} with return code {returncode}")


if __name__ == "__main__":
    main()
