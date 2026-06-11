#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline
from sdvg_inference import ensure_wan_symlinks, load_config, load_prompts
from utils.misc import set_seed
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


class FullVideoDatasetWriter:
    def __init__(self, output_dir: str | Path, *, shard_size: int = 32):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.buffer: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.total_records = 0

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        self.total_records += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        shard_index = len(self.shards)
        shard_name = f"shard_{shard_index:05d}.pt"
        torch.save({"records": self.buffer}, self.output_dir / shard_name)
        self.shards.append({"path": shard_name, "num_records": len(self.buffer)})
        self.buffer = []

    def close(self, metadata: dict[str, Any]) -> Path:
        self.flush()
        manifest = {
            "format": "bidirectional_wan_full_video_v1",
            "num_records": self.total_records,
            "shard_size": self.shard_size,
            "shards": self.shards,
            "metadata": metadata,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Option-B full-video Wan teacher latents.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--prompt", default="A hyperrealistic close-up of ocean waves shimmering at sunset.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=1)
    parser.add_argument("--num_prompt_shards", type=int, default=1)
    parser.add_argument("--prompt_shard_index", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--progress_path", default=None)
    parser.add_argument("--num_blocks", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=32)
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--sample_solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument(
        "--negative_prompt",
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形，毁容",
    )
    args = parser.parse_args()

    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(args.model_root)
    set_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    config = load_config(args.config_path)
    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    model_kwargs["model_name"] = args.target_model_name
    config.model_kwargs = model_kwargs
    config.num_train_timestep = getattr(config, "num_train_timestep", 1000)
    config.negative_prompt = args.negative_prompt
    config.guidance_scale = args.guidance_scale

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    pipeline = BidirectionalDiffusionInferencePipeline(
        config,
        device=device,
        text_encoder=text_encoder,
        vae=vae,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    pipeline.sampling_steps = int(args.sampling_steps)
    pipeline.sample_solver = args.sample_solver

    if args.num_prompt_shards < 1:
        raise ValueError("--num_prompt_shards must be >= 1")
    if not 0 <= args.prompt_shard_index < args.num_prompt_shards:
        raise ValueError("--prompt_shard_index must satisfy 0 <= index < num_prompt_shards")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    prompts = load_prompts(args.prompt, args.prompt_file, args.start_index, args.max_prompts)
    prompts = [
        item
        for ordinal, item in enumerate(prompts)
        if ordinal % args.num_prompt_shards == args.prompt_shard_index
    ]
    writer = FullVideoDatasetWriter(args.output_dir, shard_size=args.shard_size)
    total_frames = int(args.num_blocks) * 3
    progress_path = Path(args.progress_path) if args.progress_path else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "done": 0,
                    "total": len(prompts),
                    "batch_size": args.batch_size,
                    "prompt_shard_index": args.prompt_shard_index,
                }
            ),
            encoding="utf-8",
        )

    pbar = tqdm(total=len(prompts), desc=f"collect bidirectional wan shard {args.prompt_shard_index}")
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start : start + args.batch_size]
        if progress_path is not None:
            progress_path.write_text(
                json.dumps(
                    {
                        "done": start,
                        "total": len(prompts),
                        "batch_size": args.batch_size,
                        "prompt_shard_index": args.prompt_shard_index,
                    }
                ),
                encoding="utf-8",
            )
        noise_items = []
        for prompt_index, _prompt in batch_prompts:
            seed_index = prompt_index or 0
            generator = torch.Generator(device=device).manual_seed(args.seed + seed_index)
            noise_items.append(
                torch.randn(
                    [1, total_frames, 16, 60, 104],
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
            )
        noise = torch.cat(noise_items, dim=0)
        _video, latents = pipeline.inference(
            noise=noise,
            text_prompts=[prompt for _prompt_index, prompt in batch_prompts],
            return_latents=True,
            decode_video=False,
        )
        noise_cpu = noise.detach().cpu()
        latents_cpu = latents.detach().cpu()
        for sample_index, (prompt_index, prompt) in enumerate(batch_prompts):
            writer.add(
                {
                    "prompt_index": prompt_index if prompt_index is not None else -1,
                    "prompt": prompt,
                    "noise": noise_cpu[sample_index : sample_index + 1].contiguous(),
                    "target_latents": latents_cpu[sample_index : sample_index + 1].contiguous(),
                }
            )
        pbar.update(len(batch_prompts))
        if progress_path is not None:
            progress_path.write_text(
                json.dumps(
                    {
                        "done": min(start + len(batch_prompts), len(prompts)),
                        "total": len(prompts),
                        "batch_size": args.batch_size,
                        "prompt_shard_index": args.prompt_shard_index,
                    }
                ),
                encoding="utf-8",
            )
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
    pbar.close()

    manifest_path = writer.close(
        {
            "teacher_setup": "bidirectional_wan",
            "target_model_name": args.target_model_name,
            "config_path": str(Path(args.config_path).resolve()),
            "num_blocks": args.num_blocks,
            "seed": args.seed,
            "sampling_steps": args.sampling_steps,
            "sample_solver": args.sample_solver,
            "guidance_scale": args.guidance_scale,
            "num_prompt_shards": args.num_prompt_shards,
            "prompt_shard_index": args.prompt_shard_index,
        }
    )
    print(f"Wrote Option-B bidirectional Wan manifest: {manifest_path}")


if __name__ == "__main__":
    main()
