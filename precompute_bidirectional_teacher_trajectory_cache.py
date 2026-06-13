#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

from train_bidirectional_draft_head import (
    BidirectionalPromptAnchorDataset,
    OnlineTeacherTrajectoryCache,
    collate_bidirectional_examples,
    split_indices,
)
from sdvg_inference import ensure_wan_symlinks
from utils.misc import set_seed
from utils.wan_wrapper import WanTextEncoder


def select_indices(
    *,
    dataset_len: int,
    split: str,
    split_index_start: int,
    max_examples: int,
    val_fraction: float,
    seed: int,
    num_shards: int,
    shard_index: int,
) -> list[int]:
    if split == "all":
        indices = list(range(dataset_len))
    else:
        train_indices, val_indices = split_indices(dataset_len, val_fraction, seed)
        indices = train_indices if split == "train" else val_indices
    if split_index_start:
        indices = indices[split_index_start:]
    if max_examples > 0:
        indices = indices[:max_examples]
    if num_shards > 1:
        indices = [index for ordinal, index in enumerate(indices) if ordinal % num_shards == shard_index]
    return indices


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline precompute compact bidirectional Wan teacher trajectory cache.")
    parser.add_argument("--manifest_path", default="/mnt/lanxiangh/data/ff_exec/bidirectional_wan_draft_head_dataset/tau_delta_0p0/manifest.json")
    parser.add_argument("--cache_dir", default="/mnt/lanxiangh/data/ff_exec/teacher_trajectory_cache")
    parser.add_argument("--dataset_cache_dir", default="/mnt/lanxiangh/data/cache/specgen")
    parser.add_argument("--dataset_index_workers", type=int, default=8)
    parser.add_argument("--dataset_cache_wait_seconds", type=int, default=3600)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--num_blocks", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trajectory_steps", type=int, default=5)
    parser.add_argument("--trajectory_solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--trajectory_shift", type=float, default=None)
    parser.add_argument("--trajectory_scope", choices=["future", "full"], default="future")
    parser.add_argument("--trajectory_dataset_key", default=None)
    parser.add_argument("--split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--split_index_start", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--progress_path", default=None)
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard_index must satisfy 0 <= shard_index < num_shards")

    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(args.model_root)
    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    dataset = BidirectionalPromptAnchorDataset(
        args.manifest_path,
        num_blocks=args.num_blocks,
        cache_dir=args.dataset_cache_dir,
        index_workers=args.dataset_index_workers,
        cache_wait_seconds=args.dataset_cache_wait_seconds,
    )
    if dataset.format != "bidirectional_wan_full_video_v1":
        raise ValueError("Teacher trajectory cache precompute currently requires Option-B full-video manifest")

    indices = select_indices(
        dataset_len=len(dataset),
        split=args.split,
        split_index_start=args.split_index_start,
        max_examples=args.max_examples,
        val_fraction=args.val_fraction,
        seed=args.seed,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )
    progress_path = Path(args.progress_path) if args.progress_path else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    cache = OnlineTeacherTrajectoryCache(
        cache_dir=args.cache_dir,
        manifest_path=args.manifest_path,
        model_name=args.target_model_name,
        model_root=args.model_root,
        config_path=args.config_path,
        num_blocks=args.num_blocks,
        sampling_steps=args.trajectory_steps,
        sample_solver=args.trajectory_solver,
        seed=args.seed,
        device=device,
        dtype=dtype,
        text_encoder=text_encoder,
        trajectory_scope=args.trajectory_scope,
        trajectory_shift=args.trajectory_shift,
        trajectory_dataset_key=args.trajectory_dataset_key,
    )

    pbar = tqdm(indices, desc=f"teacher trajectory shard {args.shard_index}/{args.num_shards}", unit="sample")
    done = 0
    for dataset_index in pbar:
        sample = dataset[dataset_index]
        batch = collate_bidirectional_examples([sample])
        payload = cache(batch)
        done += 1
        pbar.set_postfix(dataset_index=dataset_index, steps=int(payload["timesteps"].numel()))
        if progress_path is not None:
            progress_path.write_text(
                json.dumps(
                    {
                        "done": done,
                        "total": len(indices),
                        "dataset_index": int(dataset_index),
                        "cache_root": str(cache.cache_root),
                        "trajectory_steps": args.trajectory_steps,
                        "trajectory_solver": args.trajectory_solver,
                        "trajectory_shift": args.trajectory_shift,
                        "trajectory_scope": args.trajectory_scope,
                        "trajectory_dataset_key": args.trajectory_dataset_key,
                        "shard_index": args.shard_index,
                        "num_shards": args.num_shards,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    pbar.close()
    print(f"Teacher trajectory cache root: {cache.cache_root}")
    print(f"Cached samples: {done}/{len(indices)}")


if __name__ == "__main__":
    main()
