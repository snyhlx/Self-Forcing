#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

from sdvg_draft_head import DraftHeadDatasetWriter, make_draft_head_record
from sdvg_inference import (
    build_pipeline,
    commit_clean_block,
    ensure_wan_symlinks,
    load_config,
    load_prompts,
    reset_pipeline_cache,
    restore_current_block_cache,
    snapshot_current_block_cache,
)
from utils.misc import set_seed
from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper


def parse_timestep_list(value: str) -> list[int]:
    timesteps = [int(item) for item in value.replace(",", " ").split() if item.strip()]
    if not timesteps:
        raise ValueError("--denoising_step_list must contain at least one timestep")
    return timesteps


def load_prompts_from_manifest(manifest_path: str | Path, start_index: int, max_prompts: int) -> list[tuple[int | None, str]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: list[tuple[int | None, str]] = []
    for shard in manifest.get("shards", []):
        payload = torch.load(manifest_path.parent / shard["path"], map_location="cpu", weights_only=False)
        for record in payload["records"]:
            prompt = str(record.get("prompt", "")).strip()
            if not prompt:
                continue
            prompt_index = record.get("prompt_index")
            if prompt_index == -1:
                prompt_index = None
            records.append((prompt_index, prompt))
    records = records[start_index:]
    if max_prompts > 0:
        records = records[:max_prompts]
    if not records:
        raise ValueError(f"No prompts selected from manifest {manifest_path}")
    return records


@torch.no_grad()
def denoise_block_with_teacher_trajectory(
    pipeline,
    noisy_input: torch.Tensor,
    conditional_dict: dict[str, Any],
    current_start_frame: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run causal teacher denoising and keep per-step x_t plus x0 predictions."""
    batch_size, current_num_frames = noisy_input.shape[:2]
    current = noisy_input
    noisy_states: list[torch.Tensor] = []
    clean_predictions: list[torch.Tensor] = []
    for index, current_timestep in enumerate(pipeline.denoising_step_list.tolist()):
        timestep = torch.full(
            (batch_size, current_num_frames),
            int(current_timestep),
            device=noisy_input.device,
            dtype=torch.int64,
        )
        noisy_states.append(current.detach().clone())
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=current,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        clean_predictions.append(denoised_pred.detach().clone())
        if index < len(pipeline.denoising_step_list) - 1:
            next_timestep = int(pipeline.denoising_step_list[index + 1].item())
            current = pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                torch.full(
                    (batch_size * current_num_frames,),
                    next_timestep,
                    device=noisy_input.device,
                    dtype=torch.long,
                ),
            ).unflatten(0, denoised_pred.shape[:2])
    return (
        denoised_pred,
        torch.stack(noisy_states, dim=0),
        torch.stack(clean_predictions, dim=0),
    )


def write_metadata(output_dir: Path, metadata: dict[str, Any]) -> None:
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect minimal AR teacher trajectories for draft-head flow training.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--target_checkpoint_path", default="/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors")
    parser.add_argument("--prompt", default="A hyperrealistic close-up of ocean waves shimmering at sunset.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--prompt_manifest_path", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=1)
    parser.add_argument("--num_prompt_shards", type=int, default=1)
    parser.add_argument("--prompt_shard_index", type=int, default=0)
    parser.add_argument("--num_blocks", type=int, default=7)
    parser.add_argument("--collect_start_block", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard_size", type=int, default=64)
    parser.add_argument("--denoising_step_list", default="999 969 922 841 666")
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    parser.add_argument("--progress_path", default=None)
    parser.add_argument("--decode_video", action="store_true")
    parser.add_argument(
        "--negative_prompt",
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形，毁容",
    )
    args = parser.parse_args()

    if args.num_prompt_shards < 1:
        raise ValueError("--num_prompt_shards must be >= 1")
    if not 0 <= args.prompt_shard_index < args.num_prompt_shards:
        raise ValueError("--prompt_shard_index must satisfy 0 <= index < num_prompt_shards")
    if args.num_blocks <= 0:
        raise ValueError("--num_blocks must be positive")
    if not 0 <= args.collect_start_block < args.num_blocks:
        raise ValueError("--collect_start_block must satisfy 0 <= collect_start_block < num_blocks")

    project_root = Path(__file__).parent
    os.chdir(project_root)
    ensure_wan_symlinks(args.model_root)
    set_seed(args.seed)

    device = torch.device("cuda")
    dtype = torch.bfloat16
    denoising_steps = parse_timestep_list(args.denoising_step_list)

    config = load_config(args.config_path)
    config.denoising_step_list = denoising_steps
    config.warp_denoising_step = False
    config.guidance_scale = args.guidance_scale
    config.negative_prompt = args.negative_prompt
    config.num_frame_per_block = int(getattr(config, "num_frame_per_block", 3))

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    pipeline = build_pipeline(
        config,
        args.target_model_name,
        args.target_checkpoint_path,
        device,
        dtype,
        text_encoder=text_encoder,
        vae=vae,
        use_ema=False,
    ).eval().requires_grad_(False)
    pipeline.denoising_step_list = torch.tensor(denoising_steps, device=device, dtype=torch.long)

    if args.prompt_manifest_path:
        prompts = load_prompts_from_manifest(args.prompt_manifest_path, args.start_index, args.max_prompts)
    else:
        prompts = load_prompts(args.prompt, args.prompt_file, args.start_index, args.max_prompts)
    prompts = [
        item
        for ordinal, item in enumerate(prompts)
        if ordinal % args.num_prompt_shards == args.prompt_shard_index
    ]
    output_dir = Path(args.output_dir)
    writer = DraftHeadDatasetWriter(output_dir, shard_size=args.shard_size)
    progress_path = Path(args.progress_path) if args.progress_path else None
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "format": "ar_teacher_trajectory_v1",
        "target_model_name": args.target_model_name,
        "target_checkpoint_path": args.target_checkpoint_path,
        "config_path": str(Path(args.config_path).resolve()),
        "num_blocks": args.num_blocks,
        "collect_start_block": args.collect_start_block,
        "num_frame_per_block": int(pipeline.num_frame_per_block),
        "denoising_step_list": denoising_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "num_prompt_shards": args.num_prompt_shards,
        "prompt_shard_index": args.prompt_shard_index,
    }
    write_metadata(output_dir, metadata)

    latent_shape = (1, args.num_blocks * int(pipeline.num_frame_per_block), 16, 60, 104)
    pbar = tqdm(prompts, desc=f"collect AR teacher traj shard {args.prompt_shard_index}")
    records_written = 0
    for prompt_ordinal, (prompt_index, prompt) in enumerate(pbar, start=1):
        seed_index = int(prompt_index) if prompt_index is not None else 0
        generator = torch.Generator(device=device).manual_seed(args.seed + seed_index)
        noise = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
        reset_pipeline_cache(pipeline, 1, dtype, device, total_frames=noise.shape[1])
        conditional_dict = pipeline.text_encoder([prompt])
        prompt_embeds = conditional_dict["prompt_embeds"].detach().cpu()
        output = torch.zeros_like(noise)
        current_num_frames = int(pipeline.num_frame_per_block)

        for block_index in range(args.num_blocks):
            start = block_index * current_num_frames
            end = start + current_num_frames
            block_noise = noise[:, start:end]
            context_latents = output[:, :start].detach().cpu() if start > 0 else None

            cache_before = snapshot_current_block_cache(pipeline, current_num_frames)
            target_latents, noisy_traj, clean_traj = denoise_block_with_teacher_trajectory(
                pipeline,
                block_noise,
                conditional_dict,
                start,
            )
            restore_current_block_cache(pipeline, cache_before)

            output[:, start:end] = target_latents
            commit_clean_block(pipeline, target_latents, conditional_dict, start)

            if block_index >= args.collect_start_block:
                writer.add(
                    make_draft_head_record(
                        prompt=prompt,
                        prompt_index=prompt_index,
                        block_index=block_index,
                        block_noise=block_noise.detach().cpu(),
                        target_latents=target_latents.detach().cpu(),
                        target_features={},
                        context_latents=context_latents,
                        prompt_embeds=prompt_embeds,
                        teacher_trajectory_latents=clean_traj[:, 0].detach().cpu(),
                        teacher_trajectory_noisy_latents=noisy_traj[:, 0].detach().cpu(),
                        teacher_trajectory_timesteps=torch.tensor(denoising_steps, dtype=torch.long),
                    )
                )
                records_written += 1

        if args.decode_video:
            # Optional smoke artifact; dataset collection itself only needs latents.
            video = pipeline.vae.decode_to_pixel(output, use_cache=False)
            torch.save(video.detach().cpu(), output_dir / f"prompt_{seed_index:05d}_video.pt")
        pipeline.vae.model.clear_cache()
        torch.cuda.empty_cache()
        pbar.set_postfix(prompt_index=seed_index, records=records_written)
        if progress_path is not None:
            progress_path.write_text(
                json.dumps(
                    {
                        "done_prompts": prompt_ordinal,
                        "total_prompts": len(prompts),
                        "records_written": records_written,
                        "output_dir": str(output_dir),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    manifest_path = writer.close()
    metadata["num_records"] = records_written
    metadata["manifest_path"] = str(manifest_path)
    write_metadata(output_dir, metadata)
    print(f"Wrote AR teacher trajectory manifest: {manifest_path}")
    print(f"Records: {records_written}")


if __name__ == "__main__":
    main()
