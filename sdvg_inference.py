import argparse
import gc
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import torch
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import load_file as load_safetensors
from torchvision import transforms
from tqdm import tqdm

from pipeline import CausalInferencePipeline
from sdvg_draft_head import (
    DraftHeadDatasetWriter,
    DraftHeadRecordDataset,
    CausalWanARDraftHead,
    FeatureCaptureConfig,
    KVCacheInjectedLatentDraftHead,
    KVInjectedLatentDraftHead,
    LatentDraftHead,
    TargetFeatureCapture,
    WanDFlashLatentDraftHead,
    load_draft_head_checkpoint,
    make_draft_head_record,
    pool_target_features,
)
from sdvg_latent_verifier import LatentBlockVerifier, verifier_features
from train_bidirectional_draft_head import BidirectionalPromptAnchorDraftHead
from utils.dataset import TextDataset
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


@dataclass
class BlockProfile:
    block_index: int
    source: str
    accepted: bool
    score: float | None
    agreement_delta: float | None = None
    old_draft_delta: float | None = None
    draft_ms: float = 0.0
    target_ms: float = 0.0
    route_ms: float = 0.0
    decode_ms: float = 0.0
    score_ms: float = 0.0
    commit_ms: float = 0.0
    output_decode_ms: float = 0.0
    overhead_profile: dict[str, Any] | None = None


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def record_profile_ms(profile: dict[str, Any] | None, key: str, start_time: float) -> None:
    if profile is None:
        return
    profile[key] = float(profile.get(key, 0.0)) + (sync_time() - start_time) * 1000.0


def write_video(output_path: str | Path, video: torch.Tensor, fps: int = 16):
    video = video.detach().cpu().clamp(0, 255).to(torch.uint8).numpy()
    iio.imwrite(output_path, video, fps=fps, codec="libx264", pixelformat="yuv420p")


def normalize_video(video: torch.Tensor) -> torch.Tensor:
    return (video * 0.5 + 0.5).clamp(0, 1)


def decode_latents_by_chunk(
    vae: WanVAEWrapper,
    latents: torch.Tensor,
    chunk_frames: int,
    *,
    use_cache: bool,
) -> torch.Tensor:
    if use_cache:
        vae.model.clear_cache()
    chunks = []
    for start in range(0, latents.shape[1], chunk_frames):
        chunk = latents[:, start:start + chunk_frames]
        chunks.append(normalize_video(vae.decode_to_pixel(chunk, use_cache=use_cache)).detach())
    if use_cache:
        vae.model.clear_cache()
    return torch.cat(chunks, dim=1)


def clone_vae_cache(vae: WanVAEWrapper):
    cache = []
    for item in vae.model._feat_map:
        if isinstance(item, torch.Tensor):
            cache.append(item.clone())
        else:
            cache.append(item)
    return cache


def restore_vae_cache(vae: WanVAEWrapper, cache):
    vae.model._feat_map = [
        item.clone() if isinstance(item, torch.Tensor) else item
        for item in cache
    ]


def cleanup_cuda_runtime_state(*pipelines: CausalInferencePipeline | None) -> None:
    for pipeline in pipelines:
        if pipeline is None:
            continue
        if hasattr(pipeline, "kv_cache1"):
            pipeline.kv_cache1 = []
        if hasattr(pipeline, "crossattn_cache"):
            pipeline.crossattn_cache = []
        vae_model = getattr(getattr(pipeline, "vae", None), "model", None)
        clear_cache = getattr(vae_model, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def snapshot_current_block_cache(
    pipeline: CausalInferencePipeline,
    current_num_frames: int,
):
    """Snapshot only the KV span that the current block probe will overwrite.

    Cloning the full cache for Wan 14B can consume tens of GB. For these
    non-rolling caches, each block probe writes into the current local tail.
    Restoring that tail plus the end indices is enough before committing the
    accepted/replaced clean block.
    """
    num_new_tokens = current_num_frames * pipeline.frame_seq_length
    snapshot = []
    for cache in pipeline.kv_cache1:
        local_end_index = int(cache["local_end_index"].item())
        global_end_index = int(cache["global_end_index"].item())
        slice_end = local_end_index + num_new_tokens
        snapshot.append(
            {
                "global_end_index": global_end_index,
                "local_end_index": local_end_index,
                "slice_start": local_end_index,
                "slice_end": slice_end,
                "k": cache["k"][:, local_end_index:slice_end].clone(),
                "v": cache["v"][:, local_end_index:slice_end].clone(),
            }
        )
    return snapshot


def restore_current_block_cache(pipeline: CausalInferencePipeline, snapshot):
    for cache, saved in zip(pipeline.kv_cache1, snapshot, strict=True):
        cache["k"][:, saved["slice_start"]:saved["slice_end"]] = saved["k"]
        cache["v"][:, saved["slice_start"]:saved["slice_end"]] = saved["v"]
        cache["global_end_index"].fill_(saved["global_end_index"])
        cache["local_end_index"].fill_(saved["local_end_index"])


def load_config(config_path: str) -> Any:
    config = OmegaConf.load(config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    return OmegaConf.merge(default_config, config)


def ensure_wan_symlinks(model_root: str):
    target_dir = Path("wan_models")
    target_dir.mkdir(exist_ok=True)
    for model_name in ("Wan2.1-T2V-1.3B", "Wan2.1-T2V-14B"):
        source = Path(model_root) / "wan_models" / model_name
        if not source.is_dir():
            raise FileNotFoundError(f"Missing local Wan model directory: {source}")
        target = target_dir / model_name
        if target.is_symlink():
            try:
                if target.resolve() == source.resolve():
                    continue
            except FileNotFoundError:
                pass
            target.unlink(missing_ok=True)
        if not target.exists():
            try:
                target.symlink_to(source, target_is_directory=True)
            except FileExistsError:
                # Another DDP rank may have created it between exists() and symlink_to().
                if not target.exists() and not target.is_symlink():
                    raise
        elif not target.is_dir():
            raise FileExistsError(f"Expected {target} to be a directory or symlink")


def load_checkpoint_into_generator(generator: WanDiffusionWrapper, checkpoint_path: str, use_ema: bool):
    if checkpoint_path.endswith((".safetensors", ".sft")):
        state_dict = load_safetensors(checkpoint_path, device="cpu")
        # Krea realtime checkpoint stores raw module keys, usually prefixed with "model.".
        generator.load_state_dict(state_dict, strict=True)
        return

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if use_ema and "generator_ema" in state_dict:
        key = "generator_ema"
    elif "generator" in state_dict:
        key = "generator"
    elif "generator_ema" in state_dict:
        key = "generator_ema"
    else:
        raise KeyError(f"Checkpoint {checkpoint_path} has no generator or generator_ema weights")
    generator.load_state_dict(state_dict[key], strict=True)


def build_pipeline(
    config: Any,
    model_name: str,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    text_encoder: WanTextEncoder | None = None,
    vae: WanVAEWrapper | None = None,
    use_ema: bool = True,
) -> CausalInferencePipeline:
    generator = WanDiffusionWrapper(
        model_name=model_name,
        **getattr(config, "model_kwargs", {}),
        is_causal=True,
    )
    load_checkpoint_into_generator(generator, checkpoint_path, use_ema=use_ema)
    pipeline = CausalInferencePipeline(
        config,
        device=device,
        generator=generator,
        text_encoder=text_encoder,
        vae=vae,
    )
    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device=device)
    return pipeline


def reset_pipeline_cache(
    pipeline: CausalInferencePipeline,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    total_frames: int,
):
    """Initialize caches large enough for the requested number of latent frames.

    The upstream Self-Forcing inference path defaults to 32760 tokens, which is
    21 latent frames at 1560 tokens/frame. SDVG paper-style runs use 9 blocks,
    i.e. 27 latent frames, so the cache must be sized from total_frames.
    """
    num_heads = pipeline.generator.model.num_heads
    dim = pipeline.generator.model.dim
    head_dim = dim // num_heads
    kv_cache_size = max(pipeline.generator.seq_len, total_frames * pipeline.frame_seq_length)

    pipeline.kv_cache1 = [
        {
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(pipeline.num_transformer_blocks)
    ]
    pipeline.crossattn_cache = [
        {
            "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(pipeline.num_transformer_blocks)
    ]


def initialize_causal_generator_caches(
    generator: WanDiffusionWrapper,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    total_frames: int,
    frame_seq_length: int,
) -> tuple[list[dict], list[dict]]:
    """Create standalone causal caches for a Wan generator without wrapping it in a pipeline."""
    num_heads = generator.model.num_heads
    dim = generator.model.dim
    head_dim = dim // num_heads
    kv_cache_size = max(generator.seq_len, total_frames * frame_seq_length)
    num_blocks = len(generator.model.blocks)
    kv_cache = [
        {
            "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(num_blocks)
    ]
    crossattn_cache = [
        {
            "k": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "v": torch.zeros([batch_size, 512, num_heads, head_dim], dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(num_blocks)
    ]
    return kv_cache, crossattn_cache


@torch.no_grad()
def commit_causal_wan_ar_draft_head_block(
    draft_head: CausalWanARDraftHead,
    block_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    *,
    current_start_frame: int,
    frame_seq_length: int,
    context_noise: int,
    kv_cache: list[dict],
    crossattn_cache: list[dict],
    overhead_profile: dict[str, Any] | None = None,
) -> None:
    t_profile = sync_time() if overhead_profile is not None else 0.0
    timestep = torch.ones(
        [block_latents.shape[0], block_latents.shape[1]],
        device=block_latents.device,
        dtype=torch.int64,
    ) * int(context_noise)
    record_profile_ms(overhead_profile, "timestep_alloc_ms", t_profile)
    t_profile = sync_time() if overhead_profile is not None else 0.0
    draft_head(
        noisy_latents=block_latents,
        prompt_embeds=prompt_embeds,
        timestep=timestep,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=current_start_frame * frame_seq_length,
    )
    record_profile_ms(overhead_profile, "forward_ms", t_profile)


def denoise_block(
    pipeline: CausalInferencePipeline,
    noisy_input: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    overhead_profile: dict[str, Any] | None = None,
    stochastic_generator: torch.Generator | None = None,
) -> torch.Tensor:
    batch_size, current_num_frames = noisy_input.shape[:2]
    current = noisy_input
    step_profiles = overhead_profile.setdefault("steps", []) if overhead_profile is not None else None
    for index, current_timestep in enumerate(pipeline.denoising_step_list):
        step_profile: dict[str, Any] | None = (
            {"step_index": int(index), "timestep": int(current_timestep.item() if torch.is_tensor(current_timestep) else current_timestep)}
            if step_profiles is not None
            else None
        )
        t_profile = sync_time() if overhead_profile is not None else 0.0
        timestep = torch.ones(
            [batch_size, current_num_frames],
            device=noisy_input.device,
            dtype=torch.int64,
        ) * current_timestep
        record_profile_ms(step_profile, "timestep_alloc_ms", t_profile)
        t_profile = sync_time() if overhead_profile is not None else 0.0
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=current,
            conditional_dict=conditional_dict,
            timestep=timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=current_start_frame * pipeline.frame_seq_length,
        )
        record_profile_ms(step_profile, "forward_ms", t_profile)
        if index < len(pipeline.denoising_step_list) - 1:
            next_timestep = pipeline.denoising_step_list[index + 1]
            t_profile = sync_time() if overhead_profile is not None else 0.0
            next_noise = torch.randn_like(denoised_pred.flatten(0, 1), generator=stochastic_generator)
            record_profile_ms(step_profile, "next_noise_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            next_timestep_tensor = next_timestep * torch.ones(
                [batch_size * current_num_frames],
                device=noisy_input.device,
                dtype=torch.long,
            )
            record_profile_ms(step_profile, "next_timestep_alloc_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            current = pipeline.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                next_noise,
                next_timestep_tensor,
            ).unflatten(0, denoised_pred.shape[:2])
            record_profile_ms(step_profile, "add_noise_ms", t_profile)
        if step_profiles is not None and step_profile is not None:
            step_profile["total_ms"] = sum(
                float(value) for key, value in step_profile.items() if key.endswith("_ms")
            )
            step_profiles.append(step_profile)
    return denoised_pred


def commit_clean_block(
    pipeline: CausalInferencePipeline,
    block_latents: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    overhead_profile: dict[str, Any] | None = None,
):
    t_profile = sync_time() if overhead_profile is not None else 0.0
    timestep = torch.ones(
        [block_latents.shape[0], block_latents.shape[1]],
        device=block_latents.device,
        dtype=torch.int64,
    ) * pipeline.args.context_noise
    record_profile_ms(overhead_profile, "timestep_alloc_ms", t_profile)
    t_profile = sync_time() if overhead_profile is not None else 0.0
    pipeline.generator(
        noisy_image_or_video=block_latents,
        conditional_dict=conditional_dict,
        timestep=timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=current_start_frame * pipeline.frame_seq_length,
    )
    record_profile_ms(overhead_profile, "forward_ms", t_profile)


def commit_clean_block_with_feature_capture(
    pipeline: CausalInferencePipeline,
    block_latents: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    layer_names: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    config = FeatureCaptureConfig(layer_names=layer_names, detach=True, clone=True)
    with TargetFeatureCapture(pipeline.generator.model, config) as capture:
        commit_clean_block(pipeline, block_latents, conditional_dict, current_start_frame)
        return capture.latest()


def target_kv_cache_from_pipeline(
    pipeline: CausalInferencePipeline,
    layer_names: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for layer_name in layer_names:
        parts = layer_name.split(".")
        if len(parts) < 2 or parts[0] != "blocks" or not parts[1].isdigit():
            raise ValueError(
                "KV-cache draft-head capture expects layer names like 'blocks.8', "
                f"got {layer_name!r}"
            )
        layer_index = int(parts[1])
        cache = pipeline.kv_cache1[layer_index]
        end_index = int(cache["local_end_index"].item())
        result[layer_name] = {
            "k": cache["k"][:, :end_index].detach().clone(),
            "v": cache["v"][:, :end_index].detach().clone(),
        }
    return result


def commit_clean_block_with_kv_cache_capture(
    pipeline: CausalInferencePipeline,
    block_latents: torch.Tensor,
    conditional_dict: dict,
    current_start_frame: int,
    layer_names: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    commit_clean_block(pipeline, block_latents, conditional_dict, current_start_frame)
    return target_kv_cache_from_pipeline(pipeline, layer_names)


def block_agreement_delta(
    draft_latents: torch.Tensor,
    target_latents: torch.Tensor,
    metric: str,
) -> float:
    draft = draft_latents.float()
    target = target_latents.float()
    if metric == "mse":
        return float(torch.mean((draft - target) ** 2).item())
    if metric == "rmse":
        return float(torch.sqrt(torch.mean((draft - target) ** 2)).item())
    if metric == "l1":
        return float(torch.mean(torch.abs(draft - target)).item())
    if metric == "cosine":
        draft_flat = draft.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        similarity = torch.nn.functional.cosine_similarity(draft_flat, target_flat, dim=1)
        return float((1.0 - similarity).mean().item())
    raise ValueError(f"Unsupported agreement metric: {metric}")


def sigma_for_timestep(
    scheduler: Any,
    timestep: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
    clamp_min: float = 1e-5,
) -> torch.Tensor:
    timestep = timestep.to(device=device)
    scheduler.sigmas = scheduler.sigmas.to(device)
    scheduler.timesteps = scheduler.timesteps.to(device)
    timestep_id = torch.argmin(
        (scheduler.timesteps.unsqueeze(0) - timestep.flatten().unsqueeze(1)).abs(),
        dim=1,
    )
    sigma = scheduler.sigmas[timestep_id].reshape(*timestep.shape, 1, 1, 1)
    sigma = torch.where(timestep.reshape(*timestep.shape, 1, 1, 1) <= 0, torch.zeros_like(sigma), sigma)
    return sigma.to(dtype=dtype).clamp_min(clamp_min)


def draft_output_to_clean_latent(
    model_output: torch.Tensor,
    *,
    prediction_type: str,
    scheduler: Any,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "flow":
        sigma = sigma_for_timestep(scheduler, timestep, device=model_output.device, dtype=model_output.dtype, clamp_min=0.0)
        return noisy_latents - sigma * model_output
    if prediction_type == "clean_latent":
        return model_output
    raise ValueError("draft head prediction_type must be 'flow' or 'clean_latent'")


def draft_flow_step(
    model_output: torch.Tensor,
    *,
    prediction_type: str,
    scheduler: Any,
    noisy_latents: torch.Tensor,
    current_timestep: torch.Tensor,
    next_timestep: torch.Tensor,
    clean_prediction: torch.Tensor,
    next_noise: torch.Tensor,
) -> torch.Tensor:
    if prediction_type == "flow":
        current_sigma = sigma_for_timestep(
            scheduler,
            current_timestep,
            device=model_output.device,
            dtype=model_output.dtype,
            clamp_min=0.0,
        )
        next_sigma = sigma_for_timestep(
            scheduler,
            next_timestep,
            device=model_output.device,
            dtype=model_output.dtype,
            clamp_min=0.0,
        )
        return noisy_latents + (next_sigma - current_sigma) * model_output
    if prediction_type == "clean_latent":
        return scheduler.add_noise(
            clean_prediction.flatten(0, 1),
            next_noise.flatten(0, 1),
            next_timestep.flatten(0, 1),
        ).unflatten(0, clean_prediction.shape[:2])
    raise ValueError("draft head prediction_type must be 'flow' or 'clean_latent'")


@torch.no_grad()
def denoise_block_with_draft_head(
    draft_head: torch.nn.Module,
    block_latents: torch.Tensor,
    target_context: dict,
    layer_names: tuple[str, ...],
    block_index: int,
    num_blocks: int,
    scheduler: Any | None = None,
    denoising_step_list: list[int] | None = None,
    prediction_type: str = "clean_latent",
    context_latents: torch.Tensor | None = None,
    conditional_dict: dict | None = None,
    causal_kv_cache: list[dict] | None = None,
    causal_crossattn_cache: list[dict] | None = None,
    causal_current_start: int | None = None,
    causal_use_incremental_kv: bool = False,
    overhead_profile: dict[str, Any] | None = None,
    stochastic_generator: torch.Generator | None = None,
) -> torch.Tensor:
    if isinstance(draft_head, BidirectionalPromptAnchorDraftHead):
        if conditional_dict is None or "prompt_embeds" not in conditional_dict:
            raise ValueError("ar_bidirectional draft head requires conditional_dict with prompt_embeds")
        if scheduler is None or denoising_step_list is None:
            raise ValueError("ar_bidirectional draft head requires scheduler and denoising_step_list")
        current = block_latents
        prediction = block_latents
        batch_size, current_num_frames = block_latents.shape[:2]
        anchor_latents = context_latents
        if anchor_latents is not None:
            t_profile = sync_time() if overhead_profile is not None else 0.0
            anchor_latents = anchor_latents.to(device=block_latents.device, dtype=block_latents.dtype)
            if anchor_latents.shape[1] != current_num_frames:
                anchor_latents = anchor_latents[:, -current_num_frames:]
            record_profile_ms(overhead_profile, "context_prepare_ms", t_profile)
        t_profile = sync_time() if overhead_profile is not None else 0.0
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=block_latents.device, dtype=block_latents.dtype)
        record_profile_ms(overhead_profile, "prompt_prepare_ms", t_profile)
        step_profiles = overhead_profile.setdefault("steps", []) if overhead_profile is not None else None
        for step_index, current_timestep in enumerate(denoising_step_list):
            step_profile: dict[str, Any] | None = {"step_index": step_index, "timestep": int(current_timestep)} if step_profiles is not None else None
            t_profile = sync_time() if overhead_profile is not None else 0.0
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=block_latents.device,
                dtype=torch.int64,
            ) * current_timestep
            record_profile_ms(step_profile, "timestep_alloc_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            model_output = draft_head(
                anchor_latents=anchor_latents,
                future_noise=current,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
            )
            record_profile_ms(step_profile, "forward_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            prediction = draft_output_to_clean_latent(
                model_output,
                prediction_type=prediction_type,
                scheduler=scheduler,
                noisy_latents=current,
                timestep=timestep,
            )
            record_profile_ms(step_profile, "clean_convert_ms", t_profile)
            if step_index < len(denoising_step_list) - 1:
                next_timestep = denoising_step_list[step_index + 1]
                t_profile = sync_time() if overhead_profile is not None else 0.0
                next_timestep_tensor = next_timestep * torch.ones(
                    [batch_size, current_num_frames],
                    device=block_latents.device,
                    dtype=torch.long,
                )
                record_profile_ms(step_profile, "next_timestep_alloc_ms", t_profile)
                t_profile = sync_time() if overhead_profile is not None else 0.0
                next_noise = torch.randn_like(prediction, generator=stochastic_generator)
                record_profile_ms(step_profile, "next_noise_ms", t_profile)
                t_profile = sync_time() if overhead_profile is not None else 0.0
                current = draft_flow_step(
                    model_output,
                    prediction_type=prediction_type,
                    scheduler=scheduler,
                    noisy_latents=current,
                    current_timestep=timestep,
                    next_timestep=next_timestep_tensor,
                    clean_prediction=prediction,
                    next_noise=next_noise,
                )
                record_profile_ms(step_profile, "flow_step_ms", t_profile)
            if step_profiles is not None and step_profile is not None:
                step_profile["total_ms"] = sum(
                    float(value) for key, value in step_profile.items() if key.endswith("_ms")
                )
                step_profiles.append(step_profile)
        return prediction
    if isinstance(draft_head, CausalWanARDraftHead):
        if conditional_dict is None or "prompt_embeds" not in conditional_dict:
            raise ValueError("causal_wan_ar draft head requires conditional_dict with prompt_embeds")
        if scheduler is None or denoising_step_list is None:
            raise ValueError("causal_wan_ar draft head requires scheduler and denoising_step_list")
        current = block_latents
        prediction = block_latents
        batch_size, current_num_frames = block_latents.shape[:2]
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=block_latents.device, dtype=block_latents.dtype)
        clean_prefix_latents = None
        if context_latents is not None and context_latents.shape[1] > 0:
            t_profile = sync_time() if overhead_profile is not None else 0.0
            clean_prefix_latents = context_latents.to(device=block_latents.device, dtype=block_latents.dtype)
            record_profile_ms(overhead_profile, "context_prepare_ms", t_profile)
        t_profile = sync_time() if overhead_profile is not None else 0.0
        prompt_embeds = conditional_dict["prompt_embeds"].to(device=block_latents.device, dtype=block_latents.dtype)
        record_profile_ms(overhead_profile, "prompt_prepare_ms", t_profile)
        pad_to_frames = int(num_blocks) * int(current_num_frames)
        step_profiles = overhead_profile.setdefault("steps", []) if overhead_profile is not None else None
        for step_index, current_timestep in enumerate(denoising_step_list):
            step_profile = {"step_index": step_index, "timestep": int(current_timestep)} if step_profiles is not None else None
            t_profile = sync_time() if overhead_profile is not None else 0.0
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=block_latents.device,
                dtype=torch.int64,
            ) * current_timestep
            record_profile_ms(step_profile, "timestep_alloc_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            model_output = draft_head(
                noisy_latents=current,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                clean_prefix_latents=clean_prefix_latents,
                pad_to_frames=pad_to_frames,
                kv_cache=causal_kv_cache if causal_use_incremental_kv else None,
                crossattn_cache=causal_crossattn_cache if causal_use_incremental_kv else None,
                current_start=causal_current_start if causal_use_incremental_kv else None,
            )
            record_profile_ms(step_profile, "forward_ms", t_profile)
            t_profile = sync_time() if overhead_profile is not None else 0.0
            prediction = draft_output_to_clean_latent(
                model_output,
                prediction_type=prediction_type,
                scheduler=scheduler,
                noisy_latents=current,
                timestep=timestep,
            )
            record_profile_ms(step_profile, "clean_convert_ms", t_profile)
            if step_index < len(denoising_step_list) - 1:
                next_timestep = denoising_step_list[step_index + 1]
                t_profile = sync_time() if overhead_profile is not None else 0.0
                next_timestep_tensor = next_timestep * torch.ones(
                    [batch_size, current_num_frames],
                    device=block_latents.device,
                    dtype=torch.long,
                )
                record_profile_ms(step_profile, "next_timestep_alloc_ms", t_profile)
                t_profile = sync_time() if overhead_profile is not None else 0.0
                next_noise = torch.randn_like(prediction, generator=stochastic_generator)
                record_profile_ms(step_profile, "next_noise_ms", t_profile)
                t_profile = sync_time() if overhead_profile is not None else 0.0
                current = draft_flow_step(
                    model_output,
                    prediction_type=prediction_type,
                    scheduler=scheduler,
                    noisy_latents=current,
                    current_timestep=timestep,
                    next_timestep=next_timestep_tensor,
                    clean_prediction=prediction,
                    next_noise=next_noise,
                )
                record_profile_ms(step_profile, "flow_step_ms", t_profile)
            if step_profiles is not None and step_profile is not None:
                step_profile["total_ms"] = sum(
                    float(value) for key, value in step_profile.items() if key.endswith("_ms")
                )
                step_profiles.append(step_profile)
        return prediction
    if isinstance(draft_head, KVCacheInjectedLatentDraftHead):
        kv_cache = {
            name: {
                "k": cache["k"].to(device=block_latents.device),
                "v": cache["v"].to(device=block_latents.device),
            }
            for name, cache in target_context.items()
        }
        if scheduler is not None and denoising_step_list is not None:
            current = block_latents
            prediction = block_latents
            batch_size, current_num_frames = block_latents.shape[:2]
            for index, current_timestep in enumerate(denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=block_latents.device,
                    dtype=torch.int64,
                ) * current_timestep
                model_output = draft_head(
                    current,
                    kv_cache,
                    block_index=block_index,
                    num_blocks=num_blocks,
                    timestep=timestep,
                )
                prediction = draft_output_to_clean_latent(
                    model_output,
                    prediction_type=prediction_type,
                    scheduler=scheduler,
                    noisy_latents=current,
                    timestep=timestep,
                )
                if index < len(denoising_step_list) - 1:
                    next_timestep = denoising_step_list[index + 1]
                    next_timestep_tensor = next_timestep * torch.ones(
                        [batch_size, current_num_frames],
                        device=block_latents.device,
                        dtype=torch.long,
                    )
                    current = draft_flow_step(
                        model_output,
                        prediction_type=prediction_type,
                        scheduler=scheduler,
                        noisy_latents=current,
                        current_timestep=timestep,
                        next_timestep=next_timestep_tensor,
                        clean_prediction=prediction,
                        next_noise=torch.randn_like(prediction, generator=stochastic_generator),
                    )
            return prediction
        return draft_head(block_latents, kv_cache, block_index=block_index, num_blocks=num_blocks)
    if isinstance(draft_head, (KVInjectedLatentDraftHead, WanDFlashLatentDraftHead)):
        features = {
            name: value.to(device=block_latents.device)
            for name, value in target_context.items()
        }
        if scheduler is not None and denoising_step_list is not None:
            current = block_latents
            prediction = block_latents
            batch_size, current_num_frames = block_latents.shape[:2]
            for step_index, current_timestep in enumerate(denoising_step_list):
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=block_latents.device,
                    dtype=torch.int64,
                ) * current_timestep
                model_output = draft_head(
                    current,
                    features,
                    block_index=block_index,
                    num_blocks=num_blocks,
                    timestep=timestep,
                )
                prediction = draft_output_to_clean_latent(
                    model_output,
                    prediction_type=prediction_type,
                    scheduler=scheduler,
                    noisy_latents=current,
                    timestep=timestep,
                )
                if step_index < len(denoising_step_list) - 1:
                    next_timestep = denoising_step_list[step_index + 1]
                    next_timestep_tensor = next_timestep * torch.ones(
                        [batch_size, current_num_frames],
                        device=block_latents.device,
                        dtype=torch.long,
                    )
                    current = draft_flow_step(
                        model_output,
                        prediction_type=prediction_type,
                        scheduler=scheduler,
                        noisy_latents=current,
                        current_timestep=timestep,
                        next_timestep=next_timestep_tensor,
                        clean_prediction=prediction,
                        next_noise=torch.randn_like(prediction, generator=stochastic_generator),
                    )
            return prediction
        return draft_head(block_latents, features, block_index=block_index, num_blocks=num_blocks)
    feature_vector = pool_target_features(target_context, layer_names=layer_names).to(
        device=block_latents.device,
        dtype=block_latents.dtype,
    )
    return draft_head(
        block_latents,
        feature_vector,
        block_index=block_index,
        num_blocks=num_blocks,
    )


class Router:
    def __init__(
        self,
        mode: str,
        threshold: float,
        device: str,
        reward_model_name: str,
        reward_download_root: str | None,
        verifier_checkpoint_path: str | None,
        verifier_device: str,
    ):
        self.mode = mode
        self.threshold = threshold
        self.device = device
        self.model = None
        self.verifier_device = torch.device(verifier_device)
        if mode == "image_reward":
            # ImageReward imports this helper from its older transformers location.
            import transformers.modeling_utils as modeling_utils
            import transformers.pytorch_utils as pytorch_utils

            for helper_name in (
                "apply_chunking_to_forward",
                "find_pruneable_heads_and_indices",
                "prune_linear_layer",
            ):
                if not hasattr(modeling_utils, helper_name):
                    setattr(modeling_utils, helper_name, getattr(pytorch_utils, helper_name))

            try:
                import ImageReward as RM
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "ImageReward is not installed. Install requirements.txt or run: "
                    "pip install image-reward"
                ) from exc
            except ImportError as exc:
                raise RuntimeError(
                    "ImageReward import failed. Use transformers<5 and install requirements.txt."
                ) from exc
            self.model = RM.load(reward_model_name, download_root=reward_download_root).to(device)
            self.model.eval()
        elif mode == "latent_verifier":
            if verifier_checkpoint_path is None:
                raise ValueError("--verifier_checkpoint_path is required for router_mode=latent_verifier")
            # Older verifier checkpoints accidentally pickled argparse's func callback
            # as __main__.train_verifier. Provide a harmless placeholder for loading.
            for callback_name in ("train_verifier", "collect_pairs"):
                if not hasattr(sys.modules["__main__"], callback_name):
                    setattr(sys.modules["__main__"], callback_name, lambda *args, **kwargs: None)
            checkpoint = torch.load(verifier_checkpoint_path, map_location="cpu", weights_only=False)
            self.model = LatentBlockVerifier(
                checkpoint["input_dim"],
                hidden_dim=checkpoint.get("hidden_dim", 256),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.verifier_device).eval().requires_grad_(False)

    def score(self, frames: torch.Tensor, prompt: str) -> float:
        # frames: [T, C, H, W], normalized to [0, 1]
        if self.mode == "heuristic":
            # Deterministic smoke-test proxy: prefer frames with non-degenerate contrast.
            return float(frames.float().std().item())

        import torchvision.transforms.functional as TF

        images = [TF.to_pil_image(frame.cpu().clamp(0, 1)) for frame in frames]
        image_tensor = torch.stack([self.model.preprocess(image) for image in images]).to(self.device)

        with torch.inference_mode():
            text_input = self.model.blip.tokenizer(
                prompt,
                padding="max_length",
                truncation=True,
                max_length=35,
                return_tensors="pt",
            ).to(self.device)
            prompt_ids = text_input.input_ids.repeat(image_tensor.shape[0], 1)
            prompt_attention_mask = text_input.attention_mask.repeat(image_tensor.shape[0], 1)

            image_embeds = self.model.blip.visual_encoder(image_tensor)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)
            text_output = self.model.blip.text_encoder(
                prompt_ids,
                attention_mask=prompt_attention_mask,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            txt_features = text_output.last_hidden_state[:, 0, :].float()
            rewards = self.model.mlp(txt_features)
            rewards = (rewards - self.model.mean) / self.model.std
        return float(rewards.min().item())

    def score_latents(
        self,
        block_latents: torch.Tensor,
        context_latents: torch.Tensor | None,
        block_index: int,
        num_blocks: int,
    ) -> float:
        if self.mode != "latent_verifier":
            raise RuntimeError(f"score_latents is only valid for latent_verifier, got {self.mode}")
        with torch.inference_mode():
            features = verifier_features(block_latents, context_latents, block_index, num_blocks)
            logits = self.model(features.to(self.verifier_device))
        return float(logits[0].item())

    def accept(self, score: float) -> bool:
        return score >= self.threshold


def run_mode(
    *,
    mode: str,
    prompt: str,
    prompt_index: int | None,
    config: Any,
    draft_pipeline: CausalInferencePipeline | None,
    target_pipeline: CausalInferencePipeline,
    router: Router,
    noise: torch.Tensor,
    output_dir: Path,
    output_stem: str,
    fps: int,
    force_target_first_block: bool,
    reuse_scoring_decodes_for_output: bool,
    agreement_metric: str,
    target_regen_pairs: list[dict] | None,
    store_target_regen_context: bool,
    draft_head_writer: DraftHeadDatasetWriter | None,
    draft_head_model: torch.nn.Module | None,
    draft_head_capture_layers: tuple[str, ...],
    draft_head_context_source: str,
    draft_head_feature_storage: str,
    draft_head_prediction_type: str,
    draft_head_log_target_delta: bool,
    draft_head_oracle_context: bool,
    draft_head_inference_mode: str,
    profile_overheads: bool,
    output_decode_mode: str,
    stochastic_generator: torch.Generator | None = None,
) -> dict:
    batch_size, num_frames = noise.shape[:2]
    current_num_frames = target_pipeline.num_frame_per_block
    num_blocks = num_frames // current_num_frames
    device = noise.device
    dtype = noise.dtype

    run_overhead_profile: dict[str, Any] | None = {"blocks": []} if profile_overheads else None
    t_profile = sync_time() if run_overhead_profile is not None else 0.0
    reset_pipeline_cache(target_pipeline, batch_size, dtype, device, total_frames=num_frames)
    record_profile_ms(run_overhead_profile, "reset_target_cache_ms", t_profile)
    if draft_pipeline is not None:
        t_profile = sync_time() if run_overhead_profile is not None else 0.0
        reset_pipeline_cache(draft_pipeline, batch_size, dtype, device, total_frames=num_frames)
        record_profile_ms(run_overhead_profile, "reset_draft_cache_ms", t_profile)

    t_profile = sync_time() if run_overhead_profile is not None else 0.0
    target_cond = target_pipeline.text_encoder([prompt])
    record_profile_ms(run_overhead_profile, "target_text_encode_ms", t_profile)
    t_profile = sync_time() if run_overhead_profile is not None else 0.0
    draft_cond = target_cond if draft_pipeline is None or draft_pipeline.text_encoder is target_pipeline.text_encoder else draft_pipeline.text_encoder([prompt])
    record_profile_ms(run_overhead_profile, "draft_text_encode_ms", t_profile)
    output = torch.zeros_like(noise)
    profiles: list[BlockProfile] = []
    accepted = 0
    use_streamed_output = (
        output_decode_mode == "chunk_streaming"
        or (reuse_scoring_decodes_for_output and mode in ("sdvg", "target_regen", "draft_head"))
    )
    output_chunks: list[torch.Tensor] = []
    latest_target_context: dict | None = None
    draft_head_kv_cache: list[dict] | None = None
    draft_head_crossattn_cache: list[dict] | None = None
    use_draft_head_incremental_kv = (
        mode == "draft_head"
        and draft_head_inference_mode == "incremental_kv"
        and isinstance(draft_head_model, CausalWanARDraftHead)
    )
    if use_draft_head_incremental_kv:
        t_profile = sync_time() if run_overhead_profile is not None else 0.0
        draft_head_kv_cache, draft_head_crossattn_cache = initialize_causal_generator_caches(
            draft_head_model.generator,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            total_frames=num_frames,
            frame_seq_length=target_pipeline.frame_seq_length,
        )
        record_profile_ms(run_overhead_profile, "reset_draft_head_cache_ms", t_profile)
    if use_streamed_output:
        target_pipeline.vae.model.clear_cache()

    total_start = sync_time()
    for block_index in tqdm(range(num_blocks), desc=f"{mode} blocks"):
        start = block_index * current_num_frames
        end = start + current_num_frames
        block_noise = noise[:, start:end]
        profile = BlockProfile(block_index=block_index, source="target", accepted=False, score=None)
        block_overhead_profile: dict[str, Any] | None = (
            {"block_index": block_index, "mode": mode} if run_overhead_profile is not None else None
        )

        use_target = mode == "target_only" or (
            mode == "sdvg" and force_target_first_block and block_index == 0
        ) or (
            mode == "draft_head" and block_index == 0
        )
        draft_latents = None
        draft_video = None
        vae_cache_before_draft = None

        if mode == "target_regen":
            if draft_pipeline is None:
                raise ValueError("target_regen mode requires a draft pipeline")
            context_latents = output[:, :start] if start > 0 else None
            draft_cache_before = snapshot_current_block_cache(draft_pipeline, current_num_frames)
            target_cache_before = snapshot_current_block_cache(target_pipeline, current_num_frames)

            t0 = sync_time()
            draft_latents = denoise_block(
                draft_pipeline,
                block_noise,
                draft_cond,
                start,
                stochastic_generator=stochastic_generator,
            )
            profile.draft_ms = (sync_time() - t0) * 1000.0
            restore_current_block_cache(draft_pipeline, draft_cache_before)

            target_denoise_profile = (
                {"type": type(target_pipeline.generator.model).__name__, "num_steps": len(target_pipeline.denoising_step_list)}
                if block_overhead_profile is not None
                else None
            )
            t0 = sync_time()
            target_latents = denoise_block(
                target_pipeline,
                block_noise,
                target_cond,
                start,
                overhead_profile=target_denoise_profile,
                stochastic_generator=stochastic_generator,
            )
            profile.target_ms = (sync_time() - t0) * 1000.0
            if block_overhead_profile is not None:
                block_overhead_profile["target_denoise"] = target_denoise_profile
            restore_current_block_cache(target_pipeline, target_cache_before)

            t0 = sync_time()
            delta = block_agreement_delta(draft_latents, target_latents, agreement_metric)
            profile.score = delta
            profile.agreement_delta = delta
            profile.score_ms = (sync_time() - t0) * 1000.0

            t0 = sync_time()
            accept = delta < router.threshold
            profile.route_ms = (sync_time() - t0) * 1000.0

            chosen_latents = draft_latents if accept else target_latents
            output[:, start:end] = chosen_latents
            profile.source = "draft" if accept else "target"
            profile.accepted = accept
            if accept:
                accepted += 1

            if target_regen_pairs is not None:
                record = {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "block_index": block_index,
                    "agreement_metric": agreement_metric,
                    "delta": delta,
                    "accepted": accept,
                    "threshold": router.threshold,
                    "draft_latents": draft_latents.detach().cpu(),
                    "target_latents": target_latents.detach().cpu(),
                }
                if store_target_regen_context:
                    record["context_latents"] = (
                        context_latents.detach().cpu() if context_latents is not None else None
                    )
                target_regen_pairs.append(record)

            if draft_head_writer is not None and latest_target_context is not None:
                draft_head_writer.add(
                    make_draft_head_record(
                        prompt=prompt,
                        prompt_index=prompt_index,
                        block_index=block_index,
                        block_noise=block_noise,
                        target_latents=target_latents,
                        target_features=latest_target_context if draft_head_context_source == "target_features" else {},
                        target_kv_cache=latest_target_context if draft_head_context_source == "kv_cache" else None,
                        context_latents=context_latents if store_target_regen_context else None,
                        draft_latents=draft_latents,
                        delta=delta,
                        target_feature_storage=draft_head_feature_storage,
                    )
                )

            t0 = sync_time()
            if draft_head_capture_layers and draft_head_context_source == "target_features":
                latest_target_context = commit_clean_block_with_feature_capture(
                    target_pipeline,
                    chosen_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            elif draft_head_capture_layers and draft_head_context_source == "kv_cache":
                latest_target_context = commit_clean_block_with_kv_cache_capture(
                    target_pipeline,
                    chosen_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            else:
                target_commit_profile = {} if block_overhead_profile is not None else None
                commit_clean_block(
                    target_pipeline,
                    chosen_latents,
                    target_cond,
                    start,
                    overhead_profile=target_commit_profile,
                )
                if block_overhead_profile is not None:
                    block_overhead_profile["target_context_commit"] = target_commit_profile
            if draft_pipeline is not None:
                commit_clean_block(draft_pipeline, chosen_latents, draft_cond, start)
            profile.commit_ms = (sync_time() - t0) * 1000.0

            if use_streamed_output:
                t0 = sync_time()
                block_video = normalize_video(target_pipeline.vae.decode_to_pixel(chosen_latents, use_cache=True))
                profile.output_decode_ms += (sync_time() - t0) * 1000.0
                output_chunks.append(block_video.detach().cpu())

        elif mode == "draft_head" and not use_target:
            if draft_head_model is None:
                raise ValueError("draft_head mode requires --draft_head_checkpoint_path")
            if latest_target_context is None and not isinstance(draft_head_model, (BidirectionalPromptAnchorDraftHead, CausalWanARDraftHead)):
                raise RuntimeError("draft_head mode has no target context; block 0 must be target-generated")
            if use_draft_head_incremental_kv and (draft_head_kv_cache is None or draft_head_crossattn_cache is None):
                raise RuntimeError("incremental KV draft-head mode was requested but drafter caches are not initialized")

            t_profile = sync_time() if block_overhead_profile is not None else 0.0
            context_latents = None if use_draft_head_incremental_kv else (output[:, :start] if start > 0 else None)
            record_profile_ms(block_overhead_profile, "context_slice_ms", t_profile)
            t0 = sync_time()
            draft_head_overhead_profile = (
                {
                    "type": type(draft_head_model).__name__,
                    "prediction_type": draft_head_prediction_type,
                    "inference_mode": "incremental_kv" if use_draft_head_incremental_kv else "prefix",
                    "num_steps": len(target_pipeline.denoising_step_list),
                }
                if block_overhead_profile is not None
                else None
            )
            draft_head_latents = denoise_block_with_draft_head(
                draft_head_model,
                block_noise,
                latest_target_context or {},
                draft_head_capture_layers,
                block_index,
                num_blocks,
                scheduler=target_pipeline.scheduler,
                denoising_step_list=list(target_pipeline.denoising_step_list),
                prediction_type=draft_head_prediction_type,
                context_latents=context_latents,
                conditional_dict=target_cond,
                causal_kv_cache=draft_head_kv_cache if use_draft_head_incremental_kv else target_pipeline.kv_cache1,
                causal_crossattn_cache=draft_head_crossattn_cache if use_draft_head_incremental_kv else target_pipeline.crossattn_cache,
                causal_current_start=start * target_pipeline.frame_seq_length,
                causal_use_incremental_kv=use_draft_head_incremental_kv,
                overhead_profile=draft_head_overhead_profile,
                stochastic_generator=stochastic_generator,
            )
            profile.draft_ms = (sync_time() - t0) * 1000.0
            if block_overhead_profile is not None:
                block_overhead_profile["draft_head"] = draft_head_overhead_profile

            target_latents = None
            if draft_head_log_target_delta:
                t_profile = sync_time() if block_overhead_profile is not None else 0.0
                target_cache_before = snapshot_current_block_cache(target_pipeline, current_num_frames)
                record_profile_ms(block_overhead_profile, "target_cache_snapshot_ms", t_profile)
                target_denoise_profile = (
                    {"type": type(target_pipeline.generator.model).__name__, "num_steps": len(target_pipeline.denoising_step_list)}
                    if block_overhead_profile is not None
                    else None
                )
                t0 = sync_time()
                target_latents = denoise_block(
                    target_pipeline,
                    block_noise,
                    target_cond,
                    start,
                    overhead_profile=target_denoise_profile,
                    stochastic_generator=stochastic_generator,
                )
                profile.target_ms = (sync_time() - t0) * 1000.0
                if block_overhead_profile is not None:
                    block_overhead_profile["target_denoise"] = target_denoise_profile
                t_profile = sync_time() if block_overhead_profile is not None else 0.0
                restore_current_block_cache(target_pipeline, target_cache_before)
                record_profile_ms(block_overhead_profile, "target_cache_restore_ms", t_profile)
                profile.score = block_agreement_delta(draft_head_latents, target_latents, agreement_metric)
                profile.agreement_delta = profile.score
                profile.score_ms = 0.0

            t_profile = sync_time() if block_overhead_profile is not None else 0.0
            output[:, start:end] = draft_head_latents
            record_profile_ms(block_overhead_profile, "output_assign_ms", t_profile)
            profile.source = "draft_head"
            profile.accepted = True
            accepted += 1

            t0 = sync_time()
            commit_latents = target_latents if draft_head_oracle_context and target_latents is not None else draft_head_latents
            if draft_head_oracle_context and target_latents is None:
                target_cache_before = snapshot_current_block_cache(target_pipeline, current_num_frames)
                target_latents = denoise_block(
                    target_pipeline,
                    block_noise,
                    target_cond,
                    start,
                    stochastic_generator=stochastic_generator,
                )
                restore_current_block_cache(target_pipeline, target_cache_before)
                commit_latents = target_latents
            if draft_head_context_source == "target_features":
                latest_target_context = commit_clean_block_with_feature_capture(
                    target_pipeline,
                    commit_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            elif draft_head_context_source == "kv_cache":
                latest_target_context = commit_clean_block_with_kv_cache_capture(
                    target_pipeline,
                    commit_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            else:
                target_commit_profile = {} if block_overhead_profile is not None else None
                commit_clean_block(
                    target_pipeline,
                    commit_latents,
                    target_cond,
                    start,
                    overhead_profile=target_commit_profile,
                )
                if block_overhead_profile is not None:
                    block_overhead_profile["target_context_commit"] = target_commit_profile
            profile.commit_ms = (sync_time() - t0) * 1000.0
            if use_draft_head_incremental_kv:
                t_profile = sync_time() if block_overhead_profile is not None else 0.0
                assert draft_head_kv_cache is not None and draft_head_crossattn_cache is not None
                draft_head_commit_profile = {} if block_overhead_profile is not None else None
                commit_causal_wan_ar_draft_head_block(
                    draft_head_model,
                    commit_latents,
                    target_cond["prompt_embeds"],
                    current_start_frame=start,
                    frame_seq_length=target_pipeline.frame_seq_length,
                    context_noise=int(target_pipeline.args.context_noise),
                    kv_cache=draft_head_kv_cache,
                    crossattn_cache=draft_head_crossattn_cache,
                    overhead_profile=draft_head_commit_profile,
                )
                if block_overhead_profile is not None:
                    block_overhead_profile["draft_head_context_commit"] = draft_head_commit_profile
                record_profile_ms(block_overhead_profile, "draft_head_context_commit_ms", t_profile)

            if use_streamed_output:
                t0 = sync_time()
                block_video = normalize_video(target_pipeline.vae.decode_to_pixel(draft_head_latents, use_cache=True))
                profile.output_decode_ms += (sync_time() - t0) * 1000.0
                output_chunks.append(block_video.detach().cpu())

        elif mode in ("draft_only", "sdvg") and draft_pipeline is not None and not use_target:
            t0 = sync_time()
            draft_latents = denoise_block(
                draft_pipeline,
                block_noise,
                draft_cond,
                start,
                stochastic_generator=stochastic_generator,
            )
            profile.draft_ms = (sync_time() - t0) * 1000.0
            commit_clean_block(draft_pipeline, draft_latents, draft_cond, start)

        if mode == "draft_only":
            output[:, start:end] = draft_latents
            profile.source = "draft"
            profile.accepted = True
            accepted += 1
            if use_streamed_output:
                t0 = sync_time()
                block_video = normalize_video(target_pipeline.vae.decode_to_pixel(draft_latents, use_cache=True))
                profile.output_decode_ms += (sync_time() - t0) * 1000.0
                output_chunks.append(block_video.detach().cpu())
        elif mode == "sdvg" and not use_target:
            t0 = sync_time()
            if router.mode == "latent_verifier":
                context_latents = output[:, :start] if start > 0 else None
                score = router.score_latents(draft_latents, context_latents, block_index, num_blocks)
            else:
                t_decode = sync_time()
                if use_streamed_output:
                    vae_cache_before_draft = clone_vae_cache(target_pipeline.vae)
                    draft_video = target_pipeline.vae.decode_to_pixel(draft_latents, use_cache=True)
                else:
                    draft_video = draft_pipeline.vae.decode_to_pixel(draft_latents, use_cache=False)
                draft_video = normalize_video(draft_video)
                profile.decode_ms = (sync_time() - t_decode) * 1000.0
                frames = draft_video[0]
                score = router.score(frames, prompt)
            profile.score = score
            profile.score_ms = (sync_time() - t0) * 1000.0

            t0 = sync_time()
            accept = router.accept(score)
            profile.route_ms = (sync_time() - t0) * 1000.0

            if accept:
                output[:, start:end] = draft_latents
                profile.source = "draft"
                profile.accepted = True
                accepted += 1
                if use_streamed_output:
                    if draft_video is None:
                        t0 = sync_time()
                        draft_video = normalize_video(target_pipeline.vae.decode_to_pixel(draft_latents, use_cache=True))
                        profile.output_decode_ms += (sync_time() - t0) * 1000.0
                    output_chunks.append(draft_video.detach().cpu())
                t0 = sync_time()
                commit_clean_block(target_pipeline, draft_latents, target_cond, start)
                profile.commit_ms = (sync_time() - t0) * 1000.0
            else:
                if use_streamed_output and vae_cache_before_draft is not None:
                    restore_vae_cache(target_pipeline.vae, vae_cache_before_draft)
                use_target = True

        if use_target:
            target_denoise_profile = (
                {"type": type(target_pipeline.generator.model).__name__, "num_steps": len(target_pipeline.denoising_step_list)}
                if block_overhead_profile is not None
                else None
            )
            t0 = sync_time()
            target_latents = denoise_block(
                target_pipeline,
                block_noise,
                target_cond,
                start,
                overhead_profile=target_denoise_profile,
                stochastic_generator=stochastic_generator,
            )
            profile.target_ms = (sync_time() - t0) * 1000.0
            if block_overhead_profile is not None:
                block_overhead_profile["target_denoise"] = target_denoise_profile
            output[:, start:end] = target_latents
            if mode == "target_only" and draft_head_writer is not None and latest_target_context is not None:
                context_latents = output[:, :start] if start > 0 else None
                draft_head_writer.add(
                    make_draft_head_record(
                        prompt=prompt,
                        prompt_index=prompt_index,
                        block_index=block_index,
                        block_noise=block_noise,
                        target_latents=target_latents,
                        target_features=latest_target_context if draft_head_context_source == "target_features" else {},
                        target_kv_cache=latest_target_context if draft_head_context_source == "kv_cache" else None,
                        context_latents=context_latents if store_target_regen_context else None,
                        draft_latents=None,
                        delta=None,
                        target_feature_storage=draft_head_feature_storage,
                    )
                )
            t0 = sync_time()
            if draft_head_capture_layers and draft_head_context_source == "target_features":
                latest_target_context = commit_clean_block_with_feature_capture(
                    target_pipeline,
                    target_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            elif draft_head_capture_layers and draft_head_context_source == "kv_cache":
                latest_target_context = commit_clean_block_with_kv_cache_capture(
                    target_pipeline,
                    target_latents,
                    target_cond,
                    start,
                    draft_head_capture_layers,
                )
            else:
                target_commit_profile = {} if block_overhead_profile is not None else None
                commit_clean_block(
                    target_pipeline,
                    target_latents,
                    target_cond,
                    start,
                    overhead_profile=target_commit_profile,
                )
                if block_overhead_profile is not None:
                    block_overhead_profile["target_context_commit"] = target_commit_profile
            profile.commit_ms += (sync_time() - t0) * 1000.0
            if use_draft_head_incremental_kv:
                t_profile = sync_time() if block_overhead_profile is not None else 0.0
                assert isinstance(draft_head_model, CausalWanARDraftHead)
                assert draft_head_kv_cache is not None and draft_head_crossattn_cache is not None
                draft_head_commit_profile = {} if block_overhead_profile is not None else None
                commit_causal_wan_ar_draft_head_block(
                    draft_head_model,
                    target_latents,
                    target_cond["prompt_embeds"],
                    current_start_frame=start,
                    frame_seq_length=target_pipeline.frame_seq_length,
                    context_noise=int(target_pipeline.args.context_noise),
                    kv_cache=draft_head_kv_cache,
                    crossattn_cache=draft_head_crossattn_cache,
                    overhead_profile=draft_head_commit_profile,
                )
                if block_overhead_profile is not None:
                    block_overhead_profile["draft_head_context_commit"] = draft_head_commit_profile
                record_profile_ms(block_overhead_profile, "draft_head_context_commit_ms", t_profile)
            if use_streamed_output:
                t0 = sync_time()
                target_video = normalize_video(target_pipeline.vae.decode_to_pixel(target_latents, use_cache=True))
                profile.output_decode_ms += (sync_time() - t0) * 1000.0
                output_chunks.append(target_video.detach().cpu())

        if block_overhead_profile is not None:
            profile.overhead_profile = block_overhead_profile
            run_overhead_profile["blocks"].append(block_overhead_profile)
        profiles.append(profile)

    generation_ms = (sync_time() - total_start) * 1000.0
    if use_streamed_output:
        video = torch.cat(output_chunks, dim=1)
        vae_decode_ms = 0.0
        target_pipeline.vae.model.clear_cache()
    elif output_decode_mode == "chunk_independent":
        t0 = sync_time()
        video = decode_latents_by_chunk(
            target_pipeline.vae,
            output,
            current_num_frames,
            use_cache=False,
        )
        vae_decode_ms = (sync_time() - t0) * 1000.0
        if run_overhead_profile is not None:
            run_overhead_profile["final_vae_decode_ms"] = vae_decode_ms
    else:
        t0 = sync_time()
        video = normalize_video(target_pipeline.vae.decode_to_pixel(output, use_cache=False))
        vae_decode_ms = (sync_time() - t0) * 1000.0
        if run_overhead_profile is not None:
            run_overhead_profile["final_vae_decode_ms"] = vae_decode_ms

    video_path = output_dir / f"{output_stem}_{mode}.mp4"
    t_profile = sync_time() if run_overhead_profile is not None else 0.0
    write_video(video_path, 255.0 * rearrange(video, "b t c h w -> b t h w c")[0], fps=fps)
    record_profile_ms(run_overhead_profile, "write_video_ms", t_profile)

    profile_dicts = [asdict(p) for p in profiles]
    summary = {
        "mode": mode,
        "prompt_index": prompt_index,
        "prompt": prompt,
        "video_path": str(video_path),
        "num_blocks": num_blocks,
        "accepted_blocks": accepted,
        "accept_rate": accepted / max(1, num_blocks - int((force_target_first_block and mode == "sdvg") or mode == "draft_head")),
        "generation_ms": generation_ms,
        "vae_decode_ms": vae_decode_ms,
        "total_ms": generation_ms + vae_decode_ms,
        "draft_ms": sum(p.draft_ms for p in profiles),
        "target_ms": sum(p.target_ms for p in profiles),
        "routing_ms": sum(p.route_ms for p in profiles),
        "draft_decode_ms": sum(p.decode_ms for p in profiles),
        "score_ms": sum(p.score_ms for p in profiles),
        "commit_ms": sum(p.commit_ms for p in profiles),
        "output_decode_ms": sum(p.output_decode_ms for p in profiles),
        "output_decode_mode": output_decode_mode,
        "streamed_output": use_streamed_output,
        "reuse_scoring_decodes_for_output": reuse_scoring_decodes_for_output,
        "blocks": profile_dicts,
    }
    if run_overhead_profile is not None:
        summary["overhead_profile"] = run_overhead_profile
    return summary


def split_indices(num_records: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    if val_fraction <= 0:
        return sorted(indices), []
    val_count = max(1, int(round(num_records * val_fraction)))
    val_count = min(val_count, num_records - 1)
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def load_manifest_split_prompts(
    manifest_path: str | Path,
    *,
    video_dataset_index: int | None,
    video_prompt_index: int | None,
    video_split: str,
    video_split_index: int,
    val_fraction: float,
    seed: int,
    max_prompts: int,
) -> list[tuple[int | None, str]]:
    dataset = DraftHeadRecordDataset(manifest_path)
    if video_prompt_index is not None:
        for dataset_index in range(len(dataset)):
            record = dataset[dataset_index]
            if int(record["prompt_index"]) == int(video_prompt_index):
                return [(int(record["prompt_index"]), str(record["prompt"]))]
        raise ValueError(f"video_prompt_index={video_prompt_index} not found in {manifest_path}")

    if video_dataset_index is not None:
        if not 0 <= video_dataset_index < len(dataset):
            raise ValueError(f"video_dataset_index={video_dataset_index} out of range for dataset length {len(dataset)}")
        record = dataset[video_dataset_index]
        prompt_index = record.get("prompt_index")
        return [(None if prompt_index is None else int(prompt_index), str(record["prompt"]))]

    if video_split == "all":
        indices = list(range(len(dataset)))
    else:
        train_indices, val_indices = split_indices(len(dataset), val_fraction, seed)
        indices = train_indices if video_split == "train" else val_indices
        if not indices:
            raise ValueError(f"Requested split {video_split!r} is empty")
    if not 0 <= video_split_index < len(indices):
        raise ValueError(
            f"video_split_index={video_split_index} out of range for {video_split} split length {len(indices)}"
        )

    prompts: list[tuple[int | None, str]] = []
    seen_prompt_indices: set[int] = set()
    seen_prompts: set[str] = set()
    for dataset_index in indices[video_split_index:]:
        record = dataset[dataset_index]
        prompt_text = str(record["prompt"])
        raw_prompt_index = record.get("prompt_index")
        prompt_index = None if raw_prompt_index is None else int(raw_prompt_index)
        if prompt_index is not None:
            if prompt_index in seen_prompt_indices:
                continue
            seen_prompt_indices.add(prompt_index)
        elif prompt_text in seen_prompts:
            continue
        seen_prompts.add(prompt_text)
        prompts.append((prompt_index, prompt_text))
        if max_prompts > 0 and len(prompts) >= max_prompts:
            break
    if not prompts:
        raise ValueError(f"No prompts selected from {manifest_path} split={video_split}")
    return prompts


def load_prompts(prompt: str, prompt_file: str | None, start_index: int, max_prompts: int) -> list[tuple[int | None, str]]:
    if prompt_file is None:
        return [(None, prompt)]

    with open(prompt_file, encoding="utf-8") as f:
        prompts = [(idx, line.strip()) for idx, line in enumerate(f) if line.strip()]
    prompts = prompts[start_index:]
    if max_prompts > 0:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts selected from {prompt_file}")
    return prompts


def parse_timestep_list(value: str) -> list[int]:
    timesteps = [int(item) for item in value.replace(",", " ").split() if item.strip()]
    if not timesteps:
        raise ValueError("--denoising_step_list must contain at least one timestep")
    return timesteps


def add_aggregate_metrics(summaries: list[dict]) -> dict:
    by_mode: dict[str, list[dict]] = {}
    for summary in summaries:
        by_mode.setdefault(summary["mode"], []).append(summary)

    aggregate = {}
    for mode, items in by_mode.items():
        aggregate[mode] = {
            "num_prompts": len(items),
            "avg_total_ms": sum(x["total_ms"] for x in items) / len(items),
            "avg_generation_ms": sum(x["generation_ms"] for x in items) / len(items),
            "avg_vae_decode_ms": sum(x["vae_decode_ms"] for x in items) / len(items),
            "avg_accept_rate": sum(x["accept_rate"] for x in items) / len(items),
            "avg_draft_ms": sum(x["draft_ms"] for x in items) / len(items),
            "avg_target_ms": sum(x["target_ms"] for x in items) / len(items),
            "avg_routing_ms": sum(x["routing_ms"] for x in items) / len(items),
            "avg_draft_decode_ms": sum(x["draft_decode_ms"] for x in items) / len(items),
            "avg_score_ms": sum(x["score_ms"] for x in items) / len(items),
            "avg_commit_ms": sum(x["commit_ms"] for x in items) / len(items),
            "avg_output_decode_ms": sum(x.get("output_decode_ms", 0.0) for x in items) / len(items),
        }

    if "target_only" in aggregate:
        for candidate_mode in ("sdvg", "target_regen", "draft_head"):
            if candidate_mode in aggregate:
                aggregate[candidate_mode]["speedup_vs_target"] = (
                    aggregate["target_only"]["avg_total_ms"] / aggregate[candidate_mode]["avg_total_ms"]
                )
    return aggregate


def mode_needs_draft_pipeline(mode: str, compare_mode: str = "sdvg") -> bool:
    if mode in ("draft_only", "sdvg", "target_regen"):
        return True
    if mode == "compare":
        return compare_mode in ("sdvg", "target_regen")
    return False


def main():
    parser = argparse.ArgumentParser(description="SDVG inference/profiling scaffold for Self-Forcing.")
    parser.add_argument(
        "router_mode_pos",
        nargs="?",
        choices=["heuristic", "image_reward", "latent_verifier"],
        help="Router to use for SDVG acceptance. Example: python sdvg_inference.py image_reward ...",
    )
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--draft_model_name", default="Wan2.1-T2V-1.3B")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--draft_checkpoint_path", default="/mnt/lanxiangh/models/Self-Forcing/checkpoints/self_forcing_dmd.pt")
    parser.add_argument("--target_checkpoint_path", default="/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors")
    parser.add_argument("--prompt", default="A hyperrealistic close-up of ocean waves shimmering at sunset.")
    parser.add_argument("--prompt_file", default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=0)
    parser.add_argument(
        "--video_manifest_path",
        default=None,
        help=(
            "Optional draft-head/teacher-trajectory manifest to select prompts by dataset split. "
            "Mutually exclusive with --prompt_file."
        ),
    )
    parser.add_argument("--video_dataset_index", type=int, default=None)
    parser.add_argument("--video_prompt_index", type=int, default=None)
    parser.add_argument("--video_split", choices=["all", "train", "val"], default="all")
    parser.add_argument(
        "--video_split_index",
        type=int,
        default=0,
        help="Start offset within --video_split when selecting prompts from --video_manifest_path.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--output_dir", default="outputs/sdvg")
    parser.add_argument(
        "--mode",
        choices=["target_only", "draft_only", "sdvg", "target_regen", "draft_head", "compare"],
        default="compare",
    )
    parser.add_argument(
        "--router-mode",
        dest="router_mode_flag",
        choices=["heuristic", "image_reward", "latent_verifier"],
        default=None,
        help="Router to use. Kept for backward compatibility; overrides positional router_mode.",
    )
    parser.add_argument("--tau", type=float, default=-0.7)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--denoising_step_list", default="1000 750 500 250 0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--no_force_target_first_block", action="store_true")
    parser.add_argument("--reward_device", default="cuda:0")
    parser.add_argument("--reward_model_name", default="ImageReward-v1.0")
    parser.add_argument("--reward_download_root", default=None)
    parser.add_argument("--verifier_checkpoint_path", default=None)
    parser.add_argument("--verifier_device", default="cuda:0")
    parser.add_argument(
        "--reuse_scoring_decodes_for_output",
        action="store_true",
        help="For SDVG/target_regen, reuse streamed block VAE decodes as output and skip final full-sequence VAE decode.",
    )
    parser.add_argument(
        "--output_decode_mode",
        choices=["full", "chunk_streaming", "chunk_independent"],
        default="full",
        help=(
            "How to decode final latents. 'full' decodes the whole sequence at once; "
            "'chunk_streaming' decodes committed chunks in order with VAE cache; "
            "'chunk_independent' decodes each chunk separately without VAE cache."
        ),
    )
    parser.add_argument(
        "--also_save_target_video",
        action="store_true",
        help=(
            "When running a non-compare candidate mode, also run target_only on the same prompt/noise "
            "and save the target video in the same output directory."
        ),
    )
    parser.add_argument(
        "--agreement_metric",
        choices=["mse", "rmse", "l1", "cosine"],
        default="rmse",
        help="Latent agreement distance for target_regen mode.",
    )
    parser.add_argument(
        "--target_regen_pairs_path",
        default=None,
        help="Optional .pt path to save draft/target latent block pairs from target_regen mode.",
    )
    parser.add_argument(
        "--store_target_regen_context",
        action="store_true",
        help="Also store committed context latents in --target_regen_pairs_path records. This can be large.",
    )
    parser.add_argument(
        "--draft_head_dataset_dir",
        default=None,
        help="Optional directory to save target-conditioned draft-head supervision shards from target_regen mode.",
    )
    parser.add_argument(
        "--draft_head_checkpoint_path",
        default=None,
        help="Optional LatentDraftHead checkpoint for --mode draft_head or --compare_mode draft_head.",
    )
    parser.add_argument(
        "--draft_head_capture_layers",
        nargs="+",
        default=None,
        help=(
            "Named target generator modules to capture during committed target-context forwards. "
            "Example: blocks.8 blocks.16 blocks.24"
        ),
    )
    parser.add_argument("--draft_head_shard_size", type=int, default=128)
    parser.add_argument(
        "--draft_head_feature_storage",
        choices=["pooled", "full"],
        default="pooled",
        help="Store pooled per-layer vectors by default; full activations are extremely large.",
    )
    parser.add_argument(
        "--draft_head_context_source",
        choices=["target_features", "kv_cache"],
        default="target_features",
        help=(
            "Context captured for draft-head records/inference. "
            "'target_features' is the older hidden-feature fuser path; "
            "'kv_cache' stores/reuses real target self-attention K/V tensors."
        ),
    )
    parser.add_argument(
        "--draft_head_log_target_delta",
        action="store_true",
        help="In draft_head mode, also run target regeneration to log old/corrected latent deltas. Expensive diagnostic only.",
    )
    parser.add_argument(
        "--draft_head_oracle_context",
        action="store_true",
        help=(
            "Diagnostic only: after each draft_head block, commit target-regenerated latents "
            "for the next block's target context/features while keeping draft-head latents in the output."
        ),
    )
    parser.add_argument(
        "--draft_head_inference_mode",
        choices=["prefix", "incremental_kv"],
        default="prefix",
        help=(
            "CausalWanARDraftHead rollout strategy. 'prefix' reprocesses previous clean latents; "
            "'incremental_kv' keeps a separate drafter KV cache and only processes the current block."
        ),
    )
    parser.add_argument(
        "--profile_overheads",
        action="store_true",
        help=(
            "Add detailed timing buckets to profile.json. This synchronizes CUDA more often, "
            "so use only for diagnostics."
        ),
    )
    parser.add_argument(
        "--compare_mode",
        choices=["sdvg", "target_regen", "draft_head"],
        default="sdvg",
        help="Candidate mode to pair with target_only when --mode compare is used.",
    )
    args = parser.parse_args()
    args.router_mode = args.router_mode_flag or args.router_mode_pos or "heuristic"
    delattr(args, "router_mode_flag")
    delattr(args, "router_mode_pos")

    os.chdir(Path(__file__).parent)
    ensure_wan_symlinks(args.model_root)
    set_seed(args.seed)
    torch.set_grad_enabled(False)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config_path)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    config.denoising_step_list = parse_timestep_list(args.denoising_step_list)
    config.warp_denoising_step = False
    config.num_frame_per_block = 3

    text_encoder = WanTextEncoder().to(device=device, dtype=dtype).eval().requires_grad_(False)
    vae = WanVAEWrapper().to(device=device, dtype=dtype).eval().requires_grad_(False)

    if args.mode == "draft_only":
        target_pipeline = None
    else:
        print("Loading target:", args.target_model_name, args.target_checkpoint_path)
        target_pipeline = build_pipeline(
            config,
            args.target_model_name,
            args.target_checkpoint_path,
            device,
            dtype,
            text_encoder=text_encoder,
            vae=vae,
            use_ema=False,
        )

    needs_draft = mode_needs_draft_pipeline(args.mode, args.compare_mode)
    draft_pipeline = None
    if needs_draft:
        print("Loading drafter:", args.draft_model_name, args.draft_checkpoint_path)
        draft_pipeline = build_pipeline(
            config,
            args.draft_model_name,
            args.draft_checkpoint_path,
            device,
            dtype,
            text_encoder=text_encoder,
            vae=vae,
            use_ema=args.use_ema,
        )

    uses_draft_head = args.mode == "draft_head" or (args.mode == "compare" and args.compare_mode == "draft_head")
    draft_head_model = None
    draft_head_capture_layers = tuple(args.draft_head_capture_layers or ())
    draft_head_context_source = args.draft_head_context_source
    draft_head_prediction_type = "clean_latent"
    if uses_draft_head:
        if args.draft_head_checkpoint_path is None:
            raise ValueError("--draft_head_checkpoint_path is required for draft_head mode")

    def ensure_draft_head_loaded() -> None:
        nonlocal draft_head_model
        nonlocal draft_head_capture_layers
        nonlocal draft_head_context_source
        nonlocal draft_head_prediction_type
        if draft_head_model is not None:
            return
        print("Loading draft head:", args.draft_head_checkpoint_path)
        draft_head_model, draft_head_metadata = load_draft_head_checkpoint(args.draft_head_checkpoint_path)
        draft_head_prediction_type = str(draft_head_metadata.get("metadata", {}).get("prediction_type", "clean_latent"))
        print("Draft head prediction_type:", draft_head_prediction_type)
        no_capture_draft_head = isinstance(draft_head_model, (BidirectionalPromptAnchorDraftHead, CausalWanARDraftHead))
        if no_capture_draft_head:
            draft_head_capture_layers = ()
        elif isinstance(draft_head_model, KVCacheInjectedLatentDraftHead):
            draft_head_context_source = "kv_cache"
        elif args.draft_head_context_source == "kv_cache":
            raise ValueError("--draft_head_context_source kv_cache requires a kv_cache_attention draft-head checkpoint")
        checkpoint_layers = tuple(draft_head_metadata["layer_names"])
        if draft_head_capture_layers and draft_head_capture_layers != checkpoint_layers:
            raise ValueError(
                "--draft_head_capture_layers must match checkpoint layer_names "
                f"{checkpoint_layers}, got {draft_head_capture_layers}"
            )
        if not no_capture_draft_head:
            draft_head_capture_layers = checkpoint_layers
        draft_head_model.to(device=device, dtype=dtype).eval().requires_grad_(False)

    def unload_draft_head_model() -> None:
        nonlocal draft_head_model
        draft_head_model = None
        cleanup_cuda_runtime_state()

    router = Router(
        args.router_mode,
        args.tau,
        args.reward_device,
        args.reward_model_name,
        args.reward_download_root,
        args.verifier_checkpoint_path,
        args.verifier_device,
    )
    if args.video_manifest_path and args.prompt_file:
        raise ValueError("--prompt_file and --video_manifest_path are mutually exclusive")
    if args.video_manifest_path and args.start_index != 0:
        raise ValueError("--start_index is only supported with --prompt_file; use --video_split_index for manifests")
    if args.video_dataset_index is not None and args.video_dataset_index < 0:
        raise ValueError("--video_dataset_index must be >= 0")
    if args.video_prompt_index is not None and args.video_prompt_index < 0:
        raise ValueError("--video_prompt_index must be >= 0")
    if args.video_split_index < 0:
        raise ValueError("--video_split_index must be >= 0")
    if args.val_fraction < 0:
        raise ValueError("--val_fraction must be >= 0")

    if args.video_manifest_path:
        prompts = load_manifest_split_prompts(
            args.video_manifest_path,
            video_dataset_index=args.video_dataset_index,
            video_prompt_index=args.video_prompt_index,
            video_split=args.video_split,
            video_split_index=args.video_split_index,
            val_fraction=args.val_fraction,
            seed=args.seed,
            max_prompts=args.max_prompts,
        )
    else:
        prompts = load_prompts(args.prompt, args.prompt_file, args.start_index, args.max_prompts)

    if args.also_save_target_video and args.mode == "draft_only":
        raise ValueError("--also_save_target_video is not supported with --mode draft_only")
    if args.mode == "compare":
        modes = ["target_only", args.compare_mode]
    elif args.also_save_target_video and args.mode != "target_only":
        modes = ["target_only", args.mode]
    else:
        modes = [args.mode]
    summaries = []
    target_regen_pairs: list[dict] | None = [] if args.target_regen_pairs_path else None
    draft_head_writer = None
    if args.draft_head_dataset_dir is not None:
        if uses_draft_head:
            ensure_draft_head_loaded()
        if not draft_head_capture_layers:
            raise ValueError("--draft_head_capture_layers is required with --draft_head_dataset_dir")
        if args.draft_head_context_source == "kv_cache":
            print("Capturing draft-head training records from real target KV cache.")
        draft_head_writer = DraftHeadDatasetWriter(args.draft_head_dataset_dir, shard_size=args.draft_head_shard_size)
    for prompt_index, prompt_text in tqdm(prompts, desc="prompts"):
        # Use a per-prompt generator so target-only and SDVG compare on identical noise.
        generator = torch.Generator(device=device).manual_seed(args.seed + (prompt_index or 0))
        noise = torch.randn(
            [1, args.num_blocks * 3, 16, 60, 104],
            device=device,
            dtype=dtype,
            generator=generator,
        )
        output_stem = f"prompt_{prompt_index:04d}" if prompt_index is not None else "prompt_single"
        for mode in modes:
            if mode == "draft_head":
                ensure_draft_head_loaded()
            mode_target_pipeline = draft_pipeline if mode == "draft_only" else target_pipeline
            stochastic_generator = torch.Generator(device=device).manual_seed(
                args.seed + (prompt_index or 0) + 1_000_000
            )
            summary = run_mode(
                mode=mode,
                prompt=prompt_text,
                prompt_index=prompt_index,
                config=config,
                draft_pipeline=draft_pipeline,
                target_pipeline=mode_target_pipeline,
                router=router,
                noise=noise.clone(),
                output_dir=output_dir,
                output_stem=output_stem,
                fps=args.fps,
                force_target_first_block=not args.no_force_target_first_block,
                reuse_scoring_decodes_for_output=args.reuse_scoring_decodes_for_output,
                agreement_metric=args.agreement_metric,
                target_regen_pairs=target_regen_pairs if mode == "target_regen" else None,
                store_target_regen_context=args.store_target_regen_context,
                draft_head_writer=draft_head_writer if mode in ("target_regen", "target_only") else None,
                draft_head_model=draft_head_model if mode == "draft_head" else None,
                draft_head_capture_layers=draft_head_capture_layers if mode in ("target_regen", "target_only", "draft_head") else (),
                draft_head_context_source=draft_head_context_source if mode == "draft_head" else args.draft_head_context_source,
                draft_head_feature_storage=args.draft_head_feature_storage,
                draft_head_prediction_type=draft_head_prediction_type,
                draft_head_log_target_delta=args.draft_head_log_target_delta if mode == "draft_head" else False,
                draft_head_oracle_context=args.draft_head_oracle_context if mode == "draft_head" else False,
                draft_head_inference_mode=args.draft_head_inference_mode,
                profile_overheads=args.profile_overheads,
                output_decode_mode=args.output_decode_mode,
                stochastic_generator=stochastic_generator,
            )
            summaries.append(summary)
            cleanup_cuda_runtime_state(target_pipeline, draft_pipeline)
            if args.mode == "compare" and mode == "draft_head" and draft_head_writer is None:
                unload_draft_head_model()
        del noise
        cleanup_cuda_runtime_state(target_pipeline, draft_pipeline)

    aggregate = add_aggregate_metrics(summaries)
    if "sdvg" in aggregate and "speedup_vs_target" in aggregate["sdvg"]:
        print(f"Average SDVG speedup vs target: {aggregate['sdvg']['speedup_vs_target']:.3f}x")

    profile_path = output_dir / "profile.json"
    profile_path.write_text(json.dumps({"args": vars(args), "aggregate": aggregate, "runs": summaries}, indent=2))
    print(f"Wrote profile: {profile_path}")
    if args.target_regen_pairs_path and target_regen_pairs is not None:
        pairs_path = Path(args.target_regen_pairs_path)
        if not pairs_path.is_absolute():
            pairs_path = output_dir / pairs_path
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "pairs": target_regen_pairs,
                "args": vars(args),
                "aggregate": aggregate,
            },
            pairs_path,
        )
        print(f"Wrote target-regeneration pairs: {pairs_path}")
    if draft_head_writer is not None:
        manifest_path = draft_head_writer.close()
        print(f"Wrote draft-head supervision manifest: {manifest_path}")
    for summary in summaries:
        print(f"{summary['mode']}: {summary['total_ms'] / 1000:.2f}s -> {summary['video_path']}")


if __name__ == "__main__":
    main()
