#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def prompt_args(args: argparse.Namespace) -> list[str]:
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


def run_sdvg(args: argparse.Namespace, *, mode: str, output_dir: Path, seed: int) -> dict[str, Any]:
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
        *prompt_args(args),
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
        subprocess.run(command, cwd=Path(__file__).parent, check=True, stdout=log_file, stderr=subprocess.STDOUT)
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    summaries.append(run_sdvg(args, mode="draft_head", output_dir=output_dir / "draft_head", seed=args.seed))
    for ref_index in range(args.target_reference_count):
        ref_seed = args.seed + ref_index * args.target_reference_seed_stride
        summaries.append(
            run_sdvg(
                args,
                mode="target_only",
                output_dir=output_dir / f"target_ref_{ref_index:02d}",
                seed=ref_seed,
            )
        )

    summary = {
        "args": vars(args),
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
