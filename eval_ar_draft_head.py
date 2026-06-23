#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm


def prompt_args(args: argparse.Namespace, *, prompt_override: str | None = None) -> list[str]:
    if prompt_override is not None:
        return ["--prompt", prompt_override]
    if args.prompt_file:
        return [
            "--prompt_file",
            args.prompt_file,
            "--start_index",
            str(args.start_index),
            "--max_prompts",
            str(args.max_prompts),
        ]
    return ["--prompt", args.prompt]


def split_indices(num_records: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    if val_fraction <= 0:
        return sorted(indices), []
    val_count = max(1, int(round(num_records * val_fraction)))
    val_count = min(val_count, num_records - 1)
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def select_manifest_prompt(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.video_manifest_path:
        return None
    if args.prompt_file:
        raise ValueError("--prompt_file and --video_manifest_path are mutually exclusive")
    from sdvg_draft_head import DraftHeadRecordDataset

    dataset = DraftHeadRecordDataset(args.video_manifest_path)
    dataset_index = args.video_dataset_index
    if args.video_prompt_index is not None:
        matched_index = None
        for index in range(len(dataset)):
            if int(dataset[index]["prompt_index"]) == int(args.video_prompt_index):
                matched_index = index
                break
        if matched_index is None:
            raise ValueError(f"video_prompt_index={args.video_prompt_index} not found in {args.video_manifest_path}")
        dataset_index = matched_index
    if dataset_index is None:
        if args.video_split == "all":
            dataset_index = args.video_split_index
        else:
            train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
            indices = train_indices if args.video_split == "train" else val_indices
            if not indices:
                raise ValueError(f"Requested split {args.video_split!r} is empty")
            if not 0 <= args.video_split_index < len(indices):
                raise ValueError(
                    f"video_split_index={args.video_split_index} out of range for "
                    f"{args.video_split} split length {len(indices)}"
                )
            dataset_index = indices[args.video_split_index]
    if not 0 <= dataset_index < len(dataset):
        raise ValueError(f"video_dataset_index={dataset_index} out of range for dataset length {len(dataset)}")
    record = dataset[dataset_index]
    return {
        "manifest_path": str(Path(args.video_manifest_path).resolve()),
        "dataset_index": int(dataset_index),
        "prompt_index": int(record["prompt_index"]),
        "block_index": int(record.get("block_index", -1)),
        "prompt": str(record["prompt"]),
        "split": args.video_split,
        "split_index": int(args.video_split_index),
    }


def run_sdvg(
    args: argparse.Namespace,
    *,
    mode: str,
    output_dir: Path,
    seed: int,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "sdvg_inference.py",
        "--mode",
        mode,
        "--config_path",
        args.config_path,
        "--model_root",
        args.model_root,
        "--target_model_name",
        args.target_model_name,
        "--target_checkpoint_path",
        args.target_checkpoint_path,
        "--draft_checkpoint_path",
        args.draft_checkpoint_path,
        "--draft_head_checkpoint_path",
        args.draft_head_checkpoint_path,
        "--output_dir",
        str(output_dir),
        "--num_blocks",
        str(args.num_blocks),
        "--denoising_step_list",
        args.denoising_step_list,
        "--seed",
        str(seed),
        "--fps",
        str(args.fps),
        *prompt_args(args, prompt_override=prompt_override),
    ]
    if args.no_force_target_first_block:
        command.append("--no_force_target_first_block")
    if args.draft_head_log_target_delta and mode == "draft_head":
        command.append("--draft_head_log_target_delta")
    if args.draft_head_oracle_context and mode == "draft_head":
        command.append("--draft_head_oracle_context")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            subprocess.run(command, cwd=Path(__file__).parent, check=True, stdout=log_file, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{mode} generation failed; see log: {log_path}") from exc
    profile_path = output_dir / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    return {
        "mode": mode,
        "seed": seed,
        "output_dir": str(output_dir),
        "profile_path": str(profile_path),
        "log_path": str(log_path),
        "runs": profile.get("runs", []),
        "aggregate": profile.get("aggregate", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an AR draft head with optional target-only references.")
    parser.add_argument("--draft_head_checkpoint_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--target_checkpoint_path", default="/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors")
    parser.add_argument("--draft_checkpoint_path", default="/mnt/lanxiangh/models/Self-Forcing/checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--prompt", default="A hyperrealistic close-up of ocean waves shimmering at sunset.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=1)
    parser.add_argument("--video_manifest_path", default=None)
    parser.add_argument("--video_dataset_index", type=int, default=None)
    parser.add_argument("--video_prompt_index", type=int, default=None)
    parser.add_argument("--video_split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--video_split_index", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--dataset_cache_dir", default="/mnt/lanxiangh/data/cache/specgen")
    parser.add_argument("--dataset_index_workers", type=int, default=8)
    parser.add_argument("--dataset_cache_wait_seconds", type=int, default=3600)
    parser.add_argument("--num_blocks", type=int, default=7)
    parser.add_argument("--denoising_step_list", default="999 969 922 841 666")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_reference_count", type=int, default=1)
    parser.add_argument("--target_reference_seed_stride", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--no_force_target_first_block", action="store_true")
    parser.add_argument("--draft_head_log_target_delta", action="store_true")
    parser.add_argument("--draft_head_oracle_context", action="store_true")
    args = parser.parse_args()

    if args.target_reference_count < 0:
        raise ValueError("--target_reference_count must be >= 0")
    if args.video_dataset_index is not None and args.video_dataset_index < 0:
        raise ValueError("--video_dataset_index must be >= 0")
    if args.video_prompt_index is not None and args.video_prompt_index < 0:
        raise ValueError("--video_prompt_index must be >= 0")
    if args.video_split_index < 0:
        raise ValueError("--video_split_index must be >= 0")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    manifest_selection = select_manifest_prompt(args)
    prompt_override = manifest_selection["prompt"] if manifest_selection is not None else None

    jobs = [("draft_head", output_dir / "draft_head", args.seed)]
    for ref_index in range(args.target_reference_count):
        ref_seed = args.seed + ref_index * args.target_reference_seed_stride
        jobs.append(("target_only", output_dir / f"target_ref_{ref_index:02d}", ref_seed))
    for mode, run_output_dir, seed in tqdm(jobs, desc="AR eval runs", unit="run"):
        summaries.append(run_sdvg(args, mode=mode, output_dir=run_output_dir, seed=seed, prompt_override=prompt_override))

    summary = {
        "args": vars(args),
        "manifest_selection": manifest_selection,
        "runs": summaries,
    }
    summary_path = output_dir / "profile.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote AR draft-head eval profile: {summary_path}")
    for item in summaries:
        for run in item["runs"]:
            print(f"{item['mode']} seed={item['seed']}: {run.get('video_path')}")


if __name__ == "__main__":
    main()
