#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

from train_bidirectional_draft_head import (
    BidirectionalPromptAnchorDataset,
    BidirectionalPromptAnchorDraftHead,
    OnlineTeacherTrajectoryCache,
    OnlineTargetAnchorGenerator,
    WanFullVideoDraftHead,
    amp_context,
    collate_bidirectional_examples,
    compute_bidirectional_losses,
    compute_teacher_trajectory_losses,
    flow_prediction_to_clean_latent,
    flow_prediction_step,
    make_scheduler,
    parse_unroll_step_weights,
    split_indices,
)


def _arg_or_checkpoint(args: argparse.Namespace, train_args: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return train_args.get(name, default)


def load_bidirectional_checkpoint(path: str | Path, device: torch.device) -> tuple[BidirectionalPromptAnchorDraftHead, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "bidirectional_prompt_anchor_draft_head_v1":
        raise ValueError(f"Unsupported checkpoint format: {payload.get('format')}")
    model_config = dict(payload["model_config"])
    if "patch_size" in model_config:
        model_config["patch_size"] = tuple(model_config["patch_size"])
    model_config["gradient_checkpointing"] = False
    model_class = model_config.pop("model_class", "BidirectionalPromptAnchorDraftHead")
    if model_class == "WanFullVideoDraftHead":
        model_config.pop("temporal_mixer_layers", None)
        model_config.pop("temporal_mixer_ffn_dim", None)
        model = WanFullVideoDraftHead(**model_config)
    elif model_class == "BidirectionalPromptAnchorDraftHead":
        model = BidirectionalPromptAnchorDraftHead(**model_config)
    else:
        raise ValueError(f"Unsupported bidirectional draft-head model_class: {model_class}")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.to(device).eval(), payload


def make_descending_timestep_list(num_steps: int) -> list[int]:
    if num_steps < 2:
        raise ValueError("--head_sampling_steps must be >= 2")
    return torch.linspace(1000, 0, steps=num_steps).round().long().tolist()


def draft_output_to_clean_latent(
    *,
    model_output: torch.Tensor,
    prediction_type: str,
    scheduler,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "flow":
        return flow_prediction_to_clean_latent(scheduler, model_output, noisy_latents, timestep)
    if prediction_type == "clean_latent":
        return model_output
    raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")


@torch.no_grad()
def unipc_head_sample(
    *,
    model: torch.nn.Module,
    initial_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    anchor_latents: torch.Tensor | None,
    num_steps: int,
    shift: float,
) -> tuple[torch.Tensor, list[int]]:
    if num_steps < 2:
        raise ValueError("--head_sampling_steps must be >= 2 for --head_solver unipc")
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    sample_scheduler.set_timesteps(num_steps, device=initial_latents.device, shift=shift)
    current = initial_latents
    used_timesteps: list[int] = []
    for t in tqdm(sample_scheduler.timesteps, desc="draft head unipc", leave=True):
        timestep = t * torch.ones(current.shape[:2], device=current.device, dtype=torch.float32)
        flow_prediction = model(
            anchor_latents=anchor_latents,
            future_noise=current,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
        )
        current = sample_scheduler.step(
            flow_prediction.unsqueeze(0),
            t,
            current.unsqueeze(0),
            return_dict=False,
        )[0].squeeze(0)
        used_timesteps.append(int(round(float(t.detach().cpu().item()))))
    return current, used_timesteps


@torch.no_grad()
def generate_videos(
    *,
    model: BidirectionalPromptAnchorDraftHead,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    prompts: list[tuple[int | None, str]],
    model_root: str,
    config_path: str,
    target_model_name: str,
    target_checkpoint_path: str,
    num_blocks: int,
    seed: int,
    amp_dtype: str,
    fps: int,
    training_mode: str,
    denoising_step_list: list[int],
    prediction_type: str,
    anchor_conditioning: str,
    head_solver: str,
    head_solver_shift: float,
    teacher_sampling_steps: int | None,
    unroll_noise_mode: str,
    timestep_shift: float,
    target_refine_timestep: int,
    save_raw_video: bool,
    save_target_video: bool,
    teacher_setup: str,
    device: torch.device,
) -> dict[str, Any]:
    from sdvg_inference import ensure_wan_symlinks, load_config, write_video
    from utils.misc import set_seed
    from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

    if training_mode not in ("one_step", "unrolled"):
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    if anchor_conditioning not in ("clean", "none"):
        raise ValueError("--anchor_conditioning must be 'clean' or 'none'")
    if head_solver not in ("euler", "unipc"):
        raise ValueError("--head_solver must be 'euler' or 'unipc'")
    if head_solver == "unipc" and prediction_type != "flow":
        raise ValueError("--head_solver unipc requires --prediction_type flow")
    if num_blocks <= 1:
        raise ValueError("num_blocks must be greater than 1")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(model_root)
    set_seed(seed)
    dtype = torch.bfloat16 if amp_dtype == "bf16" and device.type == "cuda" else torch.float32
    config = load_config(config_path)
    config.denoising_step_list = list(denoising_step_list)
    config.warp_denoising_step = False
    config.num_frame_per_block = 3

    text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    if teacher_setup != "bidirectional_wan":
        raise ValueError("Only --teacher_setup bidirectional_wan is supported by this video path")
    from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline

    model_kwargs = dict(getattr(config, "model_kwargs", {}))
    model_kwargs["model_name"] = target_model_name
    config.model_kwargs = model_kwargs
    config.num_train_timestep = getattr(config, "num_train_timestep", 1000)
    config.negative_prompt = getattr(
        config,
        "negative_prompt",
        "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
    )
    config.guidance_scale = getattr(config, "guidance_scale", 5.0)
    target_pipeline = BidirectionalDiffusionInferencePipeline(
        config,
        device=device,
        text_encoder=text_encoder,
        vae=vae,
    ).to(device=device, dtype=dtype).eval().requires_grad_(False)
    if teacher_sampling_steps is not None:
        if teacher_sampling_steps < 2:
            raise ValueError("--teacher_sampling_steps must be >= 2")
        target_pipeline.sampling_steps = int(teacher_sampling_steps)
    model = model.to(device=device, dtype=dtype).eval()
    scheduler = make_scheduler(float(timestep_shift))
    summaries = []

    for prompt_index, prompt in tqdm(prompts, desc="video prompts"):
        prompt_seed_index = prompt_index or 0
        generator = torch.Generator(device=device).manual_seed(seed + prompt_seed_index)
        noise = torch.randn(
            [1, num_blocks * 3, model.latent_channels, 60, 104],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        teacher_video, teacher_latents = target_pipeline.inference(
            noise=noise.clone(),
            text_prompts=[prompt],
            return_latents=True,
            decode_video=save_target_video,
        )
        anchor = teacher_latents[:, :3].detach()
        prompt_embeds = target_pipeline.text_encoder([prompt])["prompt_embeds"].detach().to(device=device, dtype=dtype)
        future_noise = noise[:, 3:]
        model_input = noise if anchor_conditioning == "none" else future_noise
        model_anchor = None if anchor_conditioning == "none" else anchor

        actual_denoising_step_list = list(denoising_step_list)
        if head_solver == "unipc":
            output_prediction, actual_denoising_step_list = unipc_head_sample(
                model=model,
                initial_latents=model_input,
                prompt_embeds=prompt_embeds,
                anchor_latents=model_anchor,
                num_steps=len(denoising_step_list),
                shift=head_solver_shift,
            )
            prediction = output_prediction
        elif training_mode == "one_step":
            timestep = torch.full(model_input.shape[:2], int(denoising_step_list[0]), device=device, dtype=torch.long)
            model_output = model(anchor_latents=model_anchor, future_noise=model_input, prompt_embeds=prompt_embeds, timestep=timestep)
            prediction = draft_output_to_clean_latent(
                model_output=model_output,
                prediction_type=prediction_type,
                scheduler=scheduler,
                noisy_latents=model_input,
                timestep=timestep,
            )
        else:
            current = model_input
            current_noise = model_input
            prediction = current
            step_iter = tqdm(denoising_step_list, desc="draft head denoise", leave=True)
            for step_index, current_timestep in enumerate(step_iter):
                timestep = torch.full(current.shape[:2], int(current_timestep), device=device, dtype=torch.long)
                model_output = model(anchor_latents=model_anchor, future_noise=current, prompt_embeds=prompt_embeds, timestep=timestep)
                prediction = draft_output_to_clean_latent(
                    model_output=model_output,
                    prediction_type=prediction_type,
                    scheduler=scheduler,
                    noisy_latents=current,
                    timestep=timestep,
                )
                if step_index < len(denoising_step_list) - 1:
                    next_timestep = int(denoising_step_list[step_index + 1])
                    next_noise = torch.randn_like(prediction) if unroll_noise_mode == "fresh" else model_input
                    next_timestep_tensor = torch.full(current.shape[:2], next_timestep, device=device, dtype=torch.long)
                    if prediction_type == "flow":
                        current = flow_prediction_step(
                            scheduler,
                            model_output,
                            current,
                            timestep,
                            next_timestep_tensor,
                        )
                    else:
                        current = scheduler.add_noise(
                            prediction.flatten(0, 1),
                            next_noise.flatten(0, 1),
                            next_timestep_tensor.flatten(0, 1),
                        ).unflatten(0, prediction.shape[:2])

        output_prediction = prediction
        source = "bidirectional_draft_head"
        if target_refine_timestep > 0:
            print("WARNING: target_refine_timestep is ignored for bidirectional_wan video generation; refine path is causal-only.")

        output_stem = f"prompt_{prompt_index:04d}" if prompt_index is not None else "prompt_single"
        if save_target_video:
            target_video_path = output_dir / f"{output_stem}_bidirectional_target_teacher.mp4"
            write_video(target_video_path, 255.0 * rearrange(teacher_video, "b t c h w -> b t h w c")[0], fps=fps)
            summaries.append(
                {
                    "mode": "bidirectional_target_teacher",
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "video_path": str(target_video_path),
                    "num_blocks": num_blocks,
                    "target_refine_timestep": 0,
                }
            )
        if save_raw_video and target_refine_timestep > 0:
            raw_video = target_pipeline.vae.decode_to_pixel(torch.cat([anchor, prediction], dim=1), use_cache=False)
            raw_video = (raw_video * 0.5 + 0.5).clamp(0, 1)
            raw_path = output_dir / f"{output_stem}_bidirectional_draft_head_raw.mp4"
            write_video(raw_path, 255.0 * rearrange(raw_video, "b t c h w -> b t h w c")[0], fps=fps)

        latents = output_prediction if anchor_conditioning == "none" else torch.cat([anchor, output_prediction], dim=1)
        video = target_pipeline.vae.decode_to_pixel(latents, use_cache=False)
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video_path = output_dir / f"{output_stem}_{source}.mp4"
        write_video(video_path, 255.0 * rearrange(video, "b t c h w -> b t h w c")[0], fps=fps)
        summaries.append(
            {
                "mode": source,
                "prompt_index": prompt_index,
                "prompt": prompt,
                "video_path": str(video_path),
                "num_blocks": num_blocks,
                "target_refine_timestep": int(target_refine_timestep),
                    "head_solver": head_solver,
            }
        )

    profile = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "num_prompts": len(summaries),
        "training_mode": training_mode,
        "prediction_type": prediction_type,
        "anchor_conditioning": anchor_conditioning,
        "head_solver": head_solver,
        "head_solver_shift": float(head_solver_shift),
        "teacher_sampling_steps": int(target_pipeline.sampling_steps),
        "denoising_step_list": list(actual_denoising_step_list) if summaries else list(denoising_step_list),
        "unroll_noise_mode": unroll_noise_mode,
        "target_refine_timestep": int(target_refine_timestep),
        "runs": summaries,
    }
    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


@torch.no_grad()
def generate_manifest_videos(
    *,
    model: BidirectionalPromptAnchorDraftHead,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    manifest_path: str | Path,
    dataset_index: int | None,
    prompt_index: int | None,
    split: str,
    split_index: int,
    val_fraction: float,
    seed: int,
    model_root: str,
    config_path: str,
    num_blocks: int,
    amp_dtype: str,
    fps: int,
    training_mode: str,
    denoising_step_list: list[int],
    prediction_type: str,
    anchor_conditioning: str,
    head_solver: str,
    head_solver_shift: float,
    unroll_noise_mode: str,
    timestep_shift: float,
    dataset_cache_dir: str,
    dataset_index_workers: int,
    dataset_cache_wait_seconds: int,
    device: torch.device,
) -> dict[str, Any]:
    from sdvg_inference import ensure_wan_symlinks, write_video
    from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

    if training_mode not in ("one_step", "unrolled"):
        raise ValueError(f"Unsupported training_mode: {training_mode}")
    if anchor_conditioning not in ("clean", "none"):
        raise ValueError("--anchor_conditioning must be 'clean' or 'none'")
    if head_solver not in ("euler", "unipc"):
        raise ValueError("--head_solver must be 'euler' or 'unipc'")
    if head_solver == "unipc" and prediction_type != "flow":
        raise ValueError("--head_solver unipc requires --prediction_type flow")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(model_root)
    dtype = torch.bfloat16 if amp_dtype == "bf16" and device.type == "cuda" else torch.float32

    dataset = BidirectionalPromptAnchorDataset(
        manifest_path,
        num_blocks=num_blocks,
        cache_dir=dataset_cache_dir,
        index_workers=dataset_index_workers,
        cache_wait_seconds=dataset_cache_wait_seconds,
    )
    if prompt_index is not None:
        matched_index = None
        for index in range(len(dataset)):
            if int(dataset[index]["prompt_index"]) == int(prompt_index):
                matched_index = index
                break
        if matched_index is None:
            raise ValueError(f"prompt_index={prompt_index} not found in {manifest_path}")
        dataset_index = matched_index
    if dataset_index is None:
        if split == "all":
            dataset_index = split_index
        else:
            train_indices, val_indices = split_indices(len(dataset), val_fraction, seed)
            indices = train_indices if split == "train" else val_indices
            if not indices:
                raise ValueError(f"Requested split {split!r} is empty")
            if not 0 <= split_index < len(indices):
                raise ValueError(f"split_index={split_index} out of range for {split} split length {len(indices)}")
            dataset_index = indices[split_index]
    if not 0 <= dataset_index < len(dataset):
        raise ValueError(f"dataset_index={dataset_index} out of range for dataset length {len(dataset)}")
    record = dataset[dataset_index]

    text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)
    model = model.to(device=device, dtype=dtype).eval()
    scheduler = make_scheduler(float(timestep_shift))

    prompt = str(record["prompt"])
    record_prompt_index = int(record["prompt_index"])
    anchor = record["anchor_latents"].to(device=device, dtype=dtype)
    future_noise = record["future_noise"].to(device=device, dtype=dtype)
    target_future = record["future_target_latents"].to(device=device, dtype=dtype)
    prompt_embeds = text_encoder([prompt])["prompt_embeds"].detach().to(device=device, dtype=dtype)
    if anchor_conditioning == "none":
        if dataset.format != "bidirectional_wan_full_video_v1":
            raise ValueError("--anchor_conditioning none video generation requires the full-video Option-B manifest")
        raw_record = dataset._load_record(dataset.index[dataset_index])
        model_input = raw_record["noise"].to(device=device, dtype=dtype)
        model_anchor = None
        target_latents = torch.cat([anchor, target_future], dim=1)
    else:
        model_input = future_noise
        model_anchor = anchor

    actual_denoising_step_list = list(denoising_step_list)
    if head_solver == "unipc":
        prediction, actual_denoising_step_list = unipc_head_sample(
            model=model,
            initial_latents=model_input,
            prompt_embeds=prompt_embeds,
            anchor_latents=model_anchor,
            num_steps=len(denoising_step_list),
            shift=head_solver_shift,
        )
    elif training_mode == "one_step":
        timestep = torch.full(model_input.shape[:2], int(denoising_step_list[0]), device=device, dtype=torch.long)
        model_output = model(anchor_latents=model_anchor, future_noise=model_input, prompt_embeds=prompt_embeds, timestep=timestep)
        prediction = draft_output_to_clean_latent(
            model_output=model_output,
            prediction_type=prediction_type,
            scheduler=scheduler,
            noisy_latents=model_input,
            timestep=timestep,
        )
    else:
        current = model_input
        prediction = current
        step_iter = tqdm(denoising_step_list, desc="draft head denoise", leave=True)
        for step_index, current_timestep in enumerate(step_iter):
            timestep = torch.full(current.shape[:2], int(current_timestep), device=device, dtype=torch.long)
            model_output = model(anchor_latents=model_anchor, future_noise=current, prompt_embeds=prompt_embeds, timestep=timestep)
            prediction = draft_output_to_clean_latent(
                model_output=model_output,
                prediction_type=prediction_type,
                scheduler=scheduler,
                noisy_latents=current,
                timestep=timestep,
            )
            if step_index < len(denoising_step_list) - 1:
                next_timestep = int(denoising_step_list[step_index + 1])
                next_noise = torch.randn_like(prediction) if unroll_noise_mode == "fresh" else model_input
                next_timestep_tensor = torch.full(current.shape[:2], next_timestep, device=device, dtype=torch.long)
                if prediction_type == "flow":
                    current = flow_prediction_step(
                        scheduler,
                        model_output,
                        current,
                        timestep,
                        next_timestep_tensor,
                    )
                else:
                    current = scheduler.add_noise(
                        prediction.flatten(0, 1),
                        next_noise.flatten(0, 1),
                        next_timestep_tensor.flatten(0, 1),
                    ).unflatten(0, prediction.shape[:2])

    output_stem = f"train_prompt_{record_prompt_index:04d}_idx_{dataset_index:05d}"
    target_video = vae.decode_to_pixel(target_latents, use_cache=False)
    target_video = (target_video * 0.5 + 0.5).clamp(0, 1)
    target_path = output_dir / f"{output_stem}_stored_target_teacher.mp4"
    write_video(target_path, 255.0 * rearrange(target_video, "b t c h w -> b t h w c")[0], fps=fps)

    draft_latents = prediction if anchor_conditioning == "none" else torch.cat([anchor, prediction], dim=1)
    draft_video = vae.decode_to_pixel(draft_latents, use_cache=False)
    draft_video = (draft_video * 0.5 + 0.5).clamp(0, 1)
    draft_path = output_dir / f"{output_stem}_bidirectional_draft_head.mp4"
    write_video(draft_path, 255.0 * rearrange(draft_video, "b t c h w -> b t h w c")[0], fps=fps)

    profile = {
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "dataset_index": int(dataset_index),
        "prompt_index": record_prompt_index,
        "prompt": prompt,
        "training_mode": training_mode,
        "prediction_type": prediction_type,
        "anchor_conditioning": anchor_conditioning,
        "head_solver": head_solver,
        "head_solver_shift": float(head_solver_shift),
        "denoising_step_list": list(actual_denoising_step_list),
        "unroll_noise_mode": unroll_noise_mode,
        "runs": [
            {"mode": "stored_target_teacher", "video_path": str(target_path)},
            {"mode": "bidirectional_draft_head", "video_path": str(draft_path)},
        ],
    }
    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


@torch.no_grad()
def evaluate_split(
    *,
    model: BidirectionalPromptAnchorDraftHead,
    loader: DataLoader,
    anchor_generator: OnlineTargetAnchorGenerator,
    device: torch.device,
    amp_dtype: str,
    scheduler,
    training_mode: str,
    denoising_step_list: list[int],
    prediction_type: str,
    random_timestep_sampling: str,
    logit_normal_mean: float,
    logit_normal_std: float,
    unroll_step_weights: list[float],
    unroll_noise_mode: str,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    detail_loss_weight: float,
    temporal_delta_weight: float,
    boundary_weight: float,
    max_examples: int,
) -> dict[str, float]:
    total_loss = 0.0
    total_clean_mse = 0.0
    total_unrolled_clean_mse = 0.0
    total_flow_mse = 0.0
    flow_count = 0
    total_examples = 0
    progress = tqdm(loader, desc="eval", unit="batch")
    for batch in progress:
        anchor, prompt_embeds = anchor_generator(batch)
        dtype = torch.bfloat16 if amp_dtype == "bf16" and device.type == "cuda" else torch.float32
        anchor = anchor.to(device=device, dtype=dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        future_noise = batch["future_noise"].to(device=device, dtype=dtype)
        target = batch["future_target_latents"].to(device=device, dtype=dtype)
        with amp_context(device, amp_dtype):
            _loss, metrics = compute_bidirectional_losses(
                model,
                anchor=anchor,
                initial_noise=future_noise,
                target=target,
                prompt_embeds=prompt_embeds,
                scheduler=scheduler,
                training_mode=training_mode,
                denoising_step_list=denoising_step_list,
                prediction_type=prediction_type,
                random_timestep_sampling=random_timestep_sampling,
                logit_normal_mean=logit_normal_mean,
                logit_normal_std=logit_normal_std,
                unroll_step_weights=unroll_step_weights,
                unroll_noise_mode=unroll_noise_mode,
                clean_latent_loss_weight=clean_latent_loss_weight,
                flow_loss_weight=flow_loss_weight,
                detail_loss_weight=detail_loss_weight,
                temporal_delta_weight=temporal_delta_weight,
                boundary_weight=boundary_weight,
            )
        batch_size = int(target.shape[0])
        total_examples += batch_size
        total_loss += metrics["loss"] * batch_size
        total_clean_mse += metrics["clean_latent_mse"] * batch_size
        total_unrolled_clean_mse += metrics.get("unrolled_clean_latent_mse", metrics["clean_latent_mse"]) * batch_size
        if "unrolled_flow_mse" in metrics:
            total_flow_mse += metrics["unrolled_flow_mse"] * batch_size
            flow_count += batch_size
        progress.set_postfix(rmse=math.sqrt(total_clean_mse / max(1, total_examples)), loss=total_loss / max(1, total_examples))
        if max_examples > 0 and total_examples >= max_examples:
            break
    result = {
        "examples": float(total_examples),
        "loss": total_loss / max(1, total_examples),
        "clean_latent_mse": total_clean_mse / max(1, total_examples),
        "clean_latent_rmse": math.sqrt(total_clean_mse / max(1, total_examples)),
        "unrolled_clean_latent_mse": total_unrolled_clean_mse / max(1, total_examples),
        "unrolled_clean_latent_rmse": math.sqrt(total_unrolled_clean_mse / max(1, total_examples)),
    }
    if flow_count:
        result["unrolled_flow_mse"] = total_flow_mse / flow_count
    return result


@torch.no_grad()
def evaluate_teacher_trajectory_split(
    model: BidirectionalPromptAnchorDraftHead,
    loader: DataLoader,
    text_encoder: torch.nn.Module,
    teacher_trajectory_cache: OnlineTeacherTrajectoryCache,
    device: torch.device,
    amp_dtype: str,
    scheduler,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    detail_loss_weight: float,
    anchor_conditioning: str,
    max_examples: int,
) -> dict[str, float]:
    metric_sums: dict[str, float] = {}
    total_examples = 0
    progress = tqdm(loader, desc="teacher-trajectory eval", unit="batch")
    for batch in progress:
        dtype = torch.bfloat16 if amp_dtype == "bf16" and device.type == "cuda" else torch.float32
        anchor = batch["anchor_latents"].to(device=device, dtype=dtype) if anchor_conditioning == "clean" else None
        prompt_embeds = text_encoder(batch["prompts"])["prompt_embeds"].detach().to(device=device, dtype=dtype)
        trajectory = teacher_trajectory_cache(batch)
        with amp_context(device, amp_dtype):
            _loss, metrics = compute_teacher_trajectory_losses(
                model,
                anchor=anchor,
                prompt_embeds=prompt_embeds,
                trajectory=trajectory,
                scheduler=scheduler,
                clean_latent_loss_weight=clean_latent_loss_weight,
                flow_loss_weight=flow_loss_weight,
                detail_loss_weight=detail_loss_weight,
                anchor_conditioning=anchor_conditioning,
            )
        batch_size = int(prompt_embeds.shape[0])
        total_examples += batch_size
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value) * batch_size
        progress.set_postfix(
            loss=metric_sums.get("loss", 0.0) / max(1, total_examples),
            flow_mse=metric_sums.get("teacher_trajectory_flow_mse", 0.0) / max(1, total_examples),
        )
        if max_examples > 0 and total_examples >= max_examples:
            break
    result = {
        "examples": float(total_examples),
        **{key: value / max(1, total_examples) for key, value in sorted(metric_sums.items())},
    }
    if "clean_latent_mse" in result:
        result["clean_latent_rmse"] = math.sqrt(result["clean_latent_mse"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a bidirectional prompt-anchor draft head checkpoint.")
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--manifest_path", default=None)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--video_output_dir", default=None)
    parser.add_argument("--video_manifest_path", default=None)
    parser.add_argument("--video_dataset_index", type=int, default=None)
    parser.add_argument("--video_prompt_index", type=int, default=None)
    parser.add_argument("--video_split", choices=["all", "train", "val"], default="all")
    parser.add_argument("--video_split_index", type=int, default=0)
    parser.add_argument("--prompt", default="A hyperrealistic close-up of ocean waves shimmering at sunset.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=1)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--target_refine_timestep", type=int, default=0)
    parser.add_argument("--save_raw_video", action="store_true")
    parser.add_argument("--save_target_video", action="store_true")
    parser.add_argument("--model_root", default=None)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--target_model_name", default=None)
    parser.add_argument("--target_checkpoint_path", default=None)
    parser.add_argument("--teacher_setup", choices=["bidirectional_wan"], default="bidirectional_wan")
    parser.add_argument("--anchor_noise_seed", type=int, default=None)
    parser.add_argument("--num_blocks", type=int, default=None)
    parser.add_argument("--val_fraction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dataset_cache_dir", default="/mnt/lanxiangh/data/cache/specgen")
    parser.add_argument("--dataset_index_workers", type=int, default=8)
    parser.add_argument("--dataset_cache_wait_seconds", type=int, default=3600)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default=None)
    parser.add_argument("--training_mode", choices=["one_step", "unrolled", "random_timestep", "teacher_trajectory"], default=None)
    parser.add_argument("--prediction_type", choices=["flow", "clean_latent"], default=None)
    parser.add_argument("--anchor_conditioning", choices=["clean", "none"], default=None)
    parser.add_argument("--denoising_step_list", nargs="+", type=int, default=None)
    parser.add_argument("--head_sampling_steps", type=int, default=None)
    parser.add_argument("--head_solver", choices=["euler", "unipc"], default="euler")
    parser.add_argument("--head_solver_shift", type=float, default=8.0)
    parser.add_argument("--teacher_sampling_steps", type=int, default=None)
    parser.add_argument("--random_timestep_sampling", choices=["uniform_schedule", "logit_normal"], default=None)
    parser.add_argument("--logit_normal_mean", type=float, default=None)
    parser.add_argument("--logit_normal_std", type=float, default=None)
    parser.add_argument("--unroll_step_weights", nargs="+", type=float, default=None)
    parser.add_argument("--unroll_noise_mode", choices=["fixed", "fresh"], default=None)
    parser.add_argument("--clean_latent_loss_weight", type=float, default=None)
    parser.add_argument("--flow_loss_weight", type=float, default=None)
    parser.add_argument("--detail_loss_weight", type=float, default=None)
    parser.add_argument("--temporal_delta_weight", type=float, default=None)
    parser.add_argument("--boundary_weight", type=float, default=None)
    parser.add_argument("--timestep_shift", type=float, default=None)
    parser.add_argument("--teacher_trajectory_cache_dir", default=None)
    parser.add_argument("--teacher_trajectory_steps", type=int, default=None)
    parser.add_argument("--teacher_trajectory_solver", choices=["unipc", "dpm++"], default=None)
    parser.add_argument("--overfit_num_examples", type=int, default=None)
    parser.add_argument("--overfit_start_index", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_bidirectional_checkpoint(args.checkpoint_path, device)
    train_args = payload.get("train_args", {})
    amp_dtype = str(_arg_or_checkpoint(args, train_args, "amp_dtype", "bf16"))
    num_blocks = int(_arg_or_checkpoint(args, train_args, "num_blocks", 9))
    denoising_step_list = list(_arg_or_checkpoint(args, train_args, "denoising_step_list", [1000, 750, 500, 250, 0]))
    training_mode = str(_arg_or_checkpoint(args, train_args, "training_mode", "unrolled"))
    prediction_type = str(_arg_or_checkpoint(args, train_args, "prediction_type", "clean_latent"))
    anchor_conditioning = str(_arg_or_checkpoint(args, train_args, "anchor_conditioning", "clean"))
    if args.head_sampling_steps is not None:
        training_mode = "unrolled"
        denoising_step_list = make_descending_timestep_list(args.head_sampling_steps)
    elif args.video_output_dir and training_mode == "random_timestep":
        training_mode = "unrolled"
    random_timestep_sampling = str(_arg_or_checkpoint(args, train_args, "random_timestep_sampling", "uniform_schedule"))
    logit_normal_mean = float(_arg_or_checkpoint(args, train_args, "logit_normal_mean", 0.0))
    logit_normal_std = float(_arg_or_checkpoint(args, train_args, "logit_normal_std", 1.0))
    unroll_noise_mode = str(_arg_or_checkpoint(args, train_args, "unroll_noise_mode", "fixed"))
    timestep_shift = float(_arg_or_checkpoint(args, train_args, "timestep_shift", 5.0))
    model_root = str(_arg_or_checkpoint(args, train_args, "model_root", "/mnt/lanxiangh/models"))
    config_path = str(_arg_or_checkpoint(args, train_args, "config_path", "configs/self_forcing_dmd.yaml"))
    target_model_name = str(_arg_or_checkpoint(args, train_args, "target_model_name", "Wan2.1-T2V-14B"))
    target_checkpoint_path = str(_arg_or_checkpoint(args, train_args, "target_checkpoint_path"))
    anchor_noise_seed = int(_arg_or_checkpoint(args, train_args, "anchor_noise_seed", 42))

    if args.video_output_dir:
        from sdvg_inference import load_prompts

        if args.video_manifest_path:
            profile = generate_manifest_videos(
                model=model,
                checkpoint_path=args.checkpoint_path,
                output_dir=args.video_output_dir,
                manifest_path=args.video_manifest_path,
                dataset_index=args.video_dataset_index,
                prompt_index=args.video_prompt_index,
                split=args.video_split,
                split_index=args.video_split_index,
                val_fraction=float(_arg_or_checkpoint(args, train_args, "val_fraction", 0.05)),
                seed=int(_arg_or_checkpoint(args, train_args, "seed", 0)),
                model_root=model_root,
                config_path=config_path,
                num_blocks=num_blocks,
                amp_dtype=amp_dtype,
                fps=args.fps,
                training_mode=training_mode,
                denoising_step_list=denoising_step_list,
                prediction_type=prediction_type,
                anchor_conditioning=anchor_conditioning,
                head_solver=args.head_solver,
                head_solver_shift=args.head_solver_shift,
                unroll_noise_mode=unroll_noise_mode,
                timestep_shift=timestep_shift,
                dataset_cache_dir=args.dataset_cache_dir,
                dataset_index_workers=args.dataset_index_workers,
                dataset_cache_wait_seconds=args.dataset_cache_wait_seconds,
                device=device,
            )
            print(json.dumps(profile, indent=2), flush=True)
            return

        prompts = load_prompts(args.prompt, args.prompt_file, args.start_index, args.max_prompts)
        profile = generate_videos(
            model=model,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.video_output_dir,
            prompts=prompts,
            model_root=model_root,
            config_path=config_path,
            target_model_name=target_model_name,
            target_checkpoint_path=target_checkpoint_path,
            num_blocks=num_blocks,
            seed=anchor_noise_seed,
            amp_dtype=amp_dtype,
            fps=args.fps,
            training_mode=training_mode,
            denoising_step_list=denoising_step_list,
            prediction_type=prediction_type,
            anchor_conditioning=anchor_conditioning,
            head_solver=args.head_solver,
            head_solver_shift=args.head_solver_shift,
            teacher_sampling_steps=args.teacher_sampling_steps,
            unroll_noise_mode=unroll_noise_mode,
            timestep_shift=timestep_shift,
            target_refine_timestep=args.target_refine_timestep,
            save_raw_video=args.save_raw_video,
            save_target_video=args.save_target_video,
            teacher_setup=args.teacher_setup,
            device=device,
        )
        print(json.dumps(profile, indent=2), flush=True)
        return

    manifest_path = _arg_or_checkpoint(args, train_args, "manifest_path")
    if manifest_path is None:
        raise ValueError("--manifest_path is required when checkpoint train_args does not include it")
    val_fraction = float(_arg_or_checkpoint(args, train_args, "val_fraction", 0.05))
    seed = int(_arg_or_checkpoint(args, train_args, "seed", 0))
    dataset = BidirectionalPromptAnchorDataset(
        manifest_path,
        num_blocks=num_blocks,
        cache_dir=args.dataset_cache_dir,
        index_workers=args.dataset_index_workers,
        cache_wait_seconds=args.dataset_cache_wait_seconds,
    )
    overfit_num_examples = int(_arg_or_checkpoint(args, train_args, "overfit_num_examples", 0))
    overfit_start_index = int(_arg_or_checkpoint(args, train_args, "overfit_start_index", 0))
    if overfit_num_examples > 0:
        train_indices = list(range(overfit_start_index, min(len(dataset), overfit_start_index + overfit_num_examples)))
        val_indices = []
    else:
        train_indices, val_indices = split_indices(len(dataset), val_fraction, seed)
    indices = val_indices if args.split == "val" else train_indices
    eval_dataset = Subset(dataset, indices)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bidirectional_examples,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    if args.batch_size != 1:
        raise ValueError("Online target anchor generation currently requires --batch_size 1")

    from utils.wan_wrapper import WanTextEncoder

    text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    scheduler = make_scheduler(timestep_shift)
    if training_mode == "teacher_trajectory":
        teacher_trajectory_cache = OnlineTeacherTrajectoryCache(
            cache_dir=str(_arg_or_checkpoint(args, train_args, "teacher_trajectory_cache_dir", "/mnt/lanxiangh/data/ff_exec/teacher_trajectory_cache")),
            manifest_path=manifest_path,
            model_name=target_model_name,
            model_root=model_root,
            config_path=config_path,
            num_blocks=num_blocks,
            sampling_steps=int(_arg_or_checkpoint(args, train_args, "teacher_trajectory_steps", 5)),
            sample_solver=str(_arg_or_checkpoint(args, train_args, "teacher_trajectory_solver", "unipc")),
            seed=anchor_noise_seed,
            device=device,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            text_encoder=text_encoder,
            trajectory_scope="full" if anchor_conditioning == "none" else "future",
        )
        precompute_indices = indices[: args.max_examples] if args.max_examples > 0 else indices
        teacher_trajectory_cache.precompute(dataset, precompute_indices)
        teacher_trajectory_cache.unload_pipeline()
        result = evaluate_teacher_trajectory_split(
            model=model,
            loader=loader,
            text_encoder=text_encoder,
            teacher_trajectory_cache=teacher_trajectory_cache,
            device=device,
            amp_dtype=amp_dtype,
            scheduler=scheduler,
            clean_latent_loss_weight=float(_arg_or_checkpoint(args, train_args, "clean_latent_loss_weight", 1.0)),
            flow_loss_weight=float(_arg_or_checkpoint(args, train_args, "flow_loss_weight", 0.25)),
            detail_loss_weight=float(_arg_or_checkpoint(args, train_args, "detail_loss_weight", 0.0)),
            anchor_conditioning=anchor_conditioning,
            max_examples=args.max_examples,
        )
        result.update(
            {
                "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
                "split": args.split,
                "manifest_path": str(Path(manifest_path).resolve()),
                "num_dataset_examples": len(dataset),
                "num_split_examples": len(eval_dataset),
                "training_mode": training_mode,
                "anchor_conditioning": anchor_conditioning,
                "teacher_trajectory_steps": int(_arg_or_checkpoint(args, train_args, "teacher_trajectory_steps", 5)),
                "teacher_trajectory_solver": str(_arg_or_checkpoint(args, train_args, "teacher_trajectory_solver", "unipc")),
            }
        )
        print(json.dumps(result, indent=2), flush=True)
        if args.output_path:
            output_path = Path(args.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Wrote metrics: {output_path}", flush=True)
        return

    anchor_generator = OnlineTargetAnchorGenerator(
        model_name=target_model_name,
        checkpoint_path=target_checkpoint_path,
        model_root=model_root,
        config_path=config_path,
        num_blocks=num_blocks,
        seed=anchor_noise_seed,
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        text_encoder=text_encoder,
    )
    unroll_step_weights = parse_unroll_step_weights(
        _arg_or_checkpoint(args, train_args, "unroll_step_weights", None),
        len(denoising_step_list),
    )
    result = evaluate_split(
        model=model,
        loader=loader,
        anchor_generator=anchor_generator,
        device=device,
        amp_dtype=amp_dtype,
        scheduler=scheduler,
        training_mode=training_mode,
        denoising_step_list=denoising_step_list,
        prediction_type=prediction_type,
        random_timestep_sampling=random_timestep_sampling,
        logit_normal_mean=logit_normal_mean,
        logit_normal_std=logit_normal_std,
        unroll_step_weights=unroll_step_weights,
        unroll_noise_mode=unroll_noise_mode,
        clean_latent_loss_weight=float(_arg_or_checkpoint(args, train_args, "clean_latent_loss_weight", 1.0)),
        flow_loss_weight=float(_arg_or_checkpoint(args, train_args, "flow_loss_weight", 0.25)),
        detail_loss_weight=float(_arg_or_checkpoint(args, train_args, "detail_loss_weight", 0.0)),
        temporal_delta_weight=float(_arg_or_checkpoint(args, train_args, "temporal_delta_weight", 0.0)),
        boundary_weight=float(_arg_or_checkpoint(args, train_args, "boundary_weight", 0.0)),
        max_examples=args.max_examples,
    )
    result.update(
        {
            "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
            "split": args.split,
            "manifest_path": str(Path(manifest_path).resolve()),
            "num_dataset_examples": len(dataset),
            "num_split_examples": len(eval_dataset),
        }
    )
    print(json.dumps(result, indent=2), flush=True)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote metrics: {output_path}", flush=True)


if __name__ == "__main__":
    main()
