#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fsdp_utils import fsdp_rank0_state_dict, rank0_save_with_state_dict, unwrap_model, wrap_model_for_training
from sdvg_draft_head import (
    DraftHeadRecordDataset,
    CausalWanARDraftHead,
    KVCacheInjectedLatentDraftHead,
    KVInjectedLatentDraftHead,
    LatentDraftHead,
    WanDFlashLatentDraftHead,
    collate_draft_head_records,
    infer_target_feature_dims,
    initialize_attention_head_from_target_blocks,
    initialize_wan_dflash_head_from_target_blocks,
    pool_target_features,
    predict_draft_head_batch as predict_sdvg_draft_head_batch,
    save_draft_head_checkpoint,
)
from train_bidirectional_draft_head import (
    BidirectionalPromptAnchorDraftHead,
    initialize_bidirectional_wan_head_from_target_blocks,
)
from utils.scheduler import FlowMatchScheduler


def is_ar_bidirectional_head(model: torch.nn.Module) -> bool:
    inner = model.module if hasattr(model, "module") else model
    return isinstance(inner, BidirectionalPromptAnchorDraftHead)


def is_causal_wan_ar_head(model: torch.nn.Module) -> bool:
    inner = model.module if hasattr(model, "module") else model
    return isinstance(inner, CausalWanARDraftHead)


def causal_wan_frame_seq_length(model: torch.nn.Module, latents: torch.Tensor) -> int:
    inner = unwrap_model(model)
    _, _, _, height, width = latents.shape
    patch = getattr(inner.generator.model, "patch_size", (1, 2, 2))
    _, patch_h, patch_w = (int(value) for value in patch)
    return (int(height) // patch_h) * (int(width) // patch_w)


def predict_draft_head_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    num_blocks: int,
    *,
    input_key: str = "block_noise",
) -> torch.Tensor:
    """Dispatch AR draft-head batches, including the new bidirectional-style head."""
    if is_causal_wan_ar_head(model):
        if "prompt_embeds" not in batch:
            raise ValueError("--head_type causal_wan_ar requires prompt_embeds in the batch")
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        context_latents = batch.get("context_latents")
        if context_latents is not None:
            context_latents = context_latents.to(device=device, dtype=dtype)
        return model(
            noisy_latents=batch[input_key].to(device=device, dtype=dtype),
            prompt_embeds=batch["prompt_embeds"].to(device=device, dtype=dtype),
            timestep=batch["timestep"].to(device=device),
            clean_prefix_latents=context_latents,
            pad_to_frames=num_blocks * int(batch[input_key].shape[1]),
        )
    if not is_ar_bidirectional_head(model):
        return predict_sdvg_draft_head_batch(model, batch, num_blocks, input_key=input_key)
    if "prompt_embeds" not in batch:
        raise ValueError("--head_type ar_bidirectional requires prompt_embeds in the batch")
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    context_latents = batch.get("context_latents")
    if context_latents is not None:
        context_latents = context_latents.to(device=device, dtype=dtype)
        future_frames = int(batch[input_key].shape[1])
        if context_latents.shape[1] != future_frames:
            # The bidirectional-style head conditions on one anchor chunk. AR
            # records store the full prefix, so use the latest causal context
            # block while keeping the full-prefix format available for future
            # cache-injection experiments.
            context_latents = context_latents[:, -future_frames:]
    return model(
        anchor_latents=context_latents,
        future_noise=batch[input_key].to(device=device, dtype=dtype),
        prompt_embeds=batch["prompt_embeds"].to(device=device, dtype=dtype),
        timestep=batch.get("timestep"),
    )


def ar_bidirectional_model_config(model: BidirectionalPromptAnchorDraftHead) -> dict[str, Any]:
    return {
        "latent_channels": model.latent_channels,
        "hidden_channels": model.hidden_channels,
        "prompt_dim": model.prompt_dim,
        "num_layers": model.num_layers,
        "num_heads": model.num_heads,
        "patch_size": model.patch_size,
        "ffn_dim": model.ffn_dim,
        "freq_dim": model.freq_dim,
        "temporal_mixer_layers": model.temporal_mixer_layers,
        "temporal_mixer_ffn_dim": model.temporal_mixer_ffn_dim,
        "max_frames": model.max_frames,
        "gradient_checkpointing": model.gradient_checkpointing,
        "eps": model.eps,
    }


def save_ar_draft_head_checkpoint(
    model: torch.nn.Module,
    path: str | Path,
    *,
    layer_names: tuple[str, ...],
    num_blocks: int,
    metadata: dict[str, Any] | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    inner = model.module if hasattr(model, "module") else model
    if not isinstance(inner, BidirectionalPromptAnchorDraftHead):
        return save_draft_head_checkpoint(
            model,
            path,
            layer_names=layer_names,
            num_blocks=num_blocks,
            metadata=metadata,
            state_dict=state_dict,
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "sdvg_latent_draft_head_v1",
            "head_type": "ar_bidirectional",
            "model_state_dict": inner.state_dict() if state_dict is None else state_dict,
            "model_config": ar_bidirectional_model_config(inner),
            "layer_names": list(layer_names),
            "num_blocks": int(num_blocks),
            "metadata": metadata or {},
        },
        path,
    )
    return path


@torch.no_grad()
def attach_prompt_embeds_if_needed(
    batch: dict[str, Any],
    *,
    model: torch.nn.Module,
    text_encoder: torch.nn.Module | None,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    if not (is_ar_bidirectional_head(model) or is_causal_wan_ar_head(model)) or "prompt_embeds" in batch:
        return batch
    if text_encoder is None:
        raise ValueError("--head_type ar_bidirectional requires a text encoder or cached prompt_embeds")
    encoded = text_encoder(text_prompts=batch["prompts"])
    enriched = dict(batch)
    enriched["prompt_embeds"] = encoded["prompt_embeds"].to(device=device, dtype=dtype)
    return enriched


def parse_layer_names(value: list[str] | None, sample: dict[str, Any]) -> tuple[str, ...]:
    if value:
        return tuple(value)
    if sample.get("target_kv_cache") is not None:
        return tuple(sorted(sample["target_kv_cache"]))
    if sample.get("target_features"):
        return tuple(sorted(sample["target_features"]))
    return ("__ar_bidirectional__",)


def split_indices(num_records: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    if val_fraction <= 0:
        return sorted(indices), []
    val_count = max(1, int(round(num_records * val_fraction)))
    val_count = min(val_count, num_records - 1)
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def infer_model_dims(sample: dict[str, Any], layer_names: tuple[str, ...]) -> tuple[int, int]:
    block_noise = sample["block_noise"]
    if block_noise.ndim != 5:
        raise ValueError(f"Expected block_noise shape [B, T, C, H, W], got {tuple(block_noise.shape)}")
    feature_dim = (
        pool_target_features(sample["target_features"], layer_names=layer_names).shape[-1]
        if sample.get("target_features")
        else 0
    )
    return int(block_noise.shape[2]), int(feature_dim)


def make_scheduler(timestep_shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=timestep_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def add_scheduled_noise(
    batch: dict[str, Any],
    scheduler: FlowMatchScheduler,
    denoising_step_list: list[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    target_latents = batch["target_latents"].to(device=device, dtype=dtype)
    batch_size, frames = target_latents.shape[:2]
    step_index = torch.randint(
        0,
        len(denoising_step_list),
        (batch_size,),
        device=device,
    )
    timestep = torch.tensor(denoising_step_list, device=device, dtype=torch.long).index_select(0, step_index)
    timestep = timestep[:, None].repeat(1, frames)
    noise = torch.randn_like(target_latents)
    noisy_latents = scheduler.add_noise(
        target_latents.flatten(0, 1),
        noise.flatten(0, 1),
        timestep.flatten(0, 1),
    ).unflatten(0, target_latents.shape[:2])

    scheduled = dict(batch)
    scheduled["scheduled_latents"] = noisy_latents
    scheduled["scheduled_noise"] = noise
    scheduled["timestep"] = timestep
    return scheduled


def sigma_for_timestep(
    scheduler: FlowMatchScheduler,
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


def clean_latent_to_flow_prediction(
    scheduler: FlowMatchScheduler,
    clean_latent_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma_for_timestep(
        scheduler,
        timestep,
        device=clean_latent_prediction.device,
        dtype=clean_latent_prediction.dtype,
    )
    return (noisy_latents - clean_latent_prediction) / sigma


def flow_prediction_to_clean_latent(
    scheduler: FlowMatchScheduler,
    flow_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = sigma_for_timestep(
        scheduler,
        timestep,
        device=flow_prediction.device,
        dtype=flow_prediction.dtype,
        clamp_min=0.0,
    )
    return noisy_latents - sigma * flow_prediction


def flow_prediction_step(
    scheduler: FlowMatchScheduler,
    flow_prediction: torch.Tensor,
    noisy_latents: torch.Tensor,
    current_timestep: torch.Tensor,
    next_timestep: torch.Tensor,
) -> torch.Tensor:
    current_sigma = sigma_for_timestep(
        scheduler,
        current_timestep,
        device=flow_prediction.device,
        dtype=flow_prediction.dtype,
        clamp_min=0.0,
    )
    next_sigma = sigma_for_timestep(
        scheduler,
        next_timestep,
        device=flow_prediction.device,
        dtype=flow_prediction.dtype,
        clamp_min=0.0,
    )
    return noisy_latents + (next_sigma - current_sigma) * flow_prediction


def tensor_finite_summary(name: str, tensor: torch.Tensor) -> str:
    finite = torch.isfinite(tensor)
    if finite.any():
        finite_values = tensor.detach()[finite].float()
        return (
            f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={int(finite.sum().item())}/{tensor.numel()} "
            f"min={float(finite_values.min().cpu().item()):.6g} "
            f"max={float(finite_values.max().cpu().item()):.6g} "
            f"mean={float(finite_values.mean().cpu().item()):.6g}"
        )
    return f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} finite=0/{tensor.numel()}"


def require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(tensor_finite_summary(name, tensor))


def require_model_parameters_finite(model: torch.nn.Module, *, context: str) -> None:
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise FloatingPointError(tensor_finite_summary(f"{context} parameter {name}", parameter))


def batch_scalar(batch: dict[str, Any], key: str, default: Any = "unknown") -> Any:
    value = batch.get(key)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.flatten()[0].detach().cpu().item()
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return default if value is None else value


class DraftHeadDMDLoss:
    """Teacher-score distribution matching term for draft-head clean-latent predictions.

    This uses the DMD generator-loss trick: compute a detached teacher score
    gradient, then apply it to the draft head's predicted clean latent.
    """

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str,
        model_root: str,
        config_path: str,
        device: torch.device,
        dtype: torch.dtype,
        guidance_scale: float,
        min_timestep: int,
        max_timestep: int,
        timestep_shift: float,
        negative_prompt: str,
    ):
        if not checkpoint_path:
            raise ValueError("--dmd_checkpoint_path is required when --dmd_loss_weight > 0")
        from sdvg_inference import ensure_wan_symlinks, load_checkpoint_into_generator, load_config
        from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder

        ensure_wan_symlinks(model_root)
        config = load_config(config_path)
        self.score_model = WanDiffusionWrapper(
            model_name=model_name,
            **getattr(config, "model_kwargs", {}),
            is_causal=True,
        )
        load_checkpoint_into_generator(self.score_model, checkpoint_path, use_ema=False)
        self.score_model = self.score_model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
        self.scheduler = self.score_model.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.device = device
        self.dtype = dtype
        self.guidance_scale = guidance_scale
        self.min_timestep = min_timestep
        self.max_timestep = max_timestep
        self.timestep_shift = timestep_shift
        self.negative_prompt = negative_prompt
        self._negative_cache: dict[int, dict[str, torch.Tensor]] = {}

    def _sample_timestep(self, batch_size: int, frames: int) -> torch.Tensor:
        timestep = torch.randint(
            self.min_timestep,
            self.max_timestep + 1,
            (batch_size, 1),
            device=self.device,
            dtype=torch.float32,
        ).repeat(1, frames)
        if self.timestep_shift > 1:
            timestep = self.timestep_shift * (timestep / 1000.0) / (
                1 + (self.timestep_shift - 1) * (timestep / 1000.0)
            ) * 1000.0
        return timestep.clamp(self.min_timestep, self.max_timestep)

    def _negative_condition(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size not in self._negative_cache:
            encoded = self.text_encoder(text_prompts=[self.negative_prompt] * batch_size)
            self._negative_cache[batch_size] = {key: value.detach() for key, value in encoded.items()}
        return self._negative_cache[batch_size]

    def __call__(self, prediction: torch.Tensor, prompts: list[str]) -> tuple[torch.Tensor, dict[str, float]]:
        batch_size, frames = prediction.shape[:2]
        if len(prompts) != batch_size:
            raise ValueError(f"Expected {batch_size} prompts for DMD loss, got {len(prompts)}")

        with torch.no_grad():
            timestep = self._sample_timestep(batch_size, frames)
            noise = torch.randn_like(prediction)
            noisy_prediction = self.scheduler.add_noise(
                prediction.detach().flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, prediction.shape[:2])
            noisy_prediction = noisy_prediction.to(dtype=self.dtype)
            conditional_dict = self.text_encoder(text_prompts=prompts)
            _, pred_real_cond = self.score_model(
                noisy_image_or_video=noisy_prediction,
                conditional_dict=conditional_dict,
                timestep=timestep,
            )
            pred_real = pred_real_cond
            if self.guidance_scale != 0.0:
                _, pred_real_uncond = self.score_model(
                    noisy_image_or_video=noisy_prediction,
                    conditional_dict=self._negative_condition(batch_size),
                    timestep=timestep,
                )
                pred_real = pred_real_cond + (pred_real_cond - pred_real_uncond) * self.guidance_scale

            grad = prediction.detach() - pred_real
            normalizer = (prediction.detach() - pred_real).abs().mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = torch.nan_to_num(grad / normalizer.clamp_min(1e-6))

        dmd_loss = 0.5 * F.mse_loss(
            prediction.float(),
            (prediction.float() - grad.float()).detach(),
            reduction="mean",
        )
        return dmd_loss, {
            "dmd_timestep": float(timestep.mean().detach().cpu().item()),
            "dmd_gradient_norm": float(grad.abs().mean().detach().cpu().item()),
        }


def compute_draft_head_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    num_blocks: int,
    input_key: str,
    scheduler: FlowMatchScheduler,
    prediction_type: str,
    loss_type: str,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    dmd_loss_weight: float,
    dmd_every: int,
    global_step: int,
    dmd_loss: DraftHeadDMDLoss | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    target_latents = batch["target_latents"].to(device=device, dtype=dtype)
    model_output = predict_draft_head_batch(model, batch, num_blocks, input_key=input_key)
    noisy_latents = batch[input_key].to(device=device, dtype=dtype)
    timestep = batch["timestep"].to(device=device)
    if prediction_type == "flow":
        prediction = flow_prediction_to_clean_latent(scheduler, model_output, noisy_latents, timestep)
        flow_prediction = model_output
    elif prediction_type == "clean_latent":
        prediction = model_output
        flow_prediction = clean_latent_to_flow_prediction(scheduler, prediction, noisy_latents, timestep)
    else:
        raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")

    components: dict[str, torch.Tensor] = {}
    clean_latent_loss = F.mse_loss(prediction.float(), target_latents.float())
    if loss_type in ("clean_latent", "clean_latent_flow"):
        components["clean_latent_loss"] = clean_latent_loss * clean_latent_loss_weight
    if loss_type in ("flow", "clean_latent_flow"):
        noise = batch["scheduled_noise"].to(device=device, dtype=dtype)
        flow_target = noise - target_latents
        flow_diff = (flow_prediction.float() - flow_target.float()).square()
        if prediction_type == "clean_latent":
            # Clean-latent heads cannot recover the sampled noise at sigma=0, so
            # the flow-conversion loss is undefined for the final clean step.
            valid_flow = (timestep > 0).reshape(*timestep.shape, 1, 1, 1).to(device=device, dtype=flow_diff.dtype)
            flow_denominator = valid_flow.expand_as(flow_diff).sum().clamp_min(1.0)
            flow_loss = (flow_diff * valid_flow).sum() / flow_denominator
        else:
            flow_loss = flow_diff.mean()
        components["flow_loss"] = flow_loss * flow_loss_weight
    if dmd_loss_weight > 0 and dmd_loss is not None and dmd_every > 0 and global_step % dmd_every == 0:
        dmd_component, dmd_metrics = dmd_loss(prediction, batch["prompts"])
        components["dmd_loss"] = dmd_component * dmd_loss_weight
    else:
        dmd_metrics = {}

    if not components:
        raise ValueError("At least one enabled loss component is required")
    total_loss = sum(components.values())
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(clean_latent_loss.detach().cpu().item()),
    }
    for name, value in components.items():
        metrics[name] = float(value.detach().cpu().item())
    metrics.update(dmd_metrics)
    return total_loss, metrics


def parse_unroll_step_weights(weights: list[float] | None, num_steps: int) -> list[float]:
    if weights is None:
        return [min(1.0, 0.25 + 0.25 * index) for index in range(num_steps)]
    if len(weights) != num_steps:
        raise ValueError(f"--unroll_step_weights must have {num_steps} values, got {len(weights)}")
    if any(weight < 0 for weight in weights):
        raise ValueError("--unroll_step_weights values must be non-negative")
    return [float(weight) for weight in weights]


def _timestep_batch(
    timestep: int,
    batch_size: int,
    frames: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.full(
        (batch_size, frames),
        int(timestep),
        device=device,
        dtype=torch.long,
    )


def compute_unrolled_draft_head_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    num_blocks: int,
    scheduler: FlowMatchScheduler,
    denoising_step_list: list[int],
    step_weights: list[float],
    prediction_type: str,
    loss_type: str,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    dmd_loss_weight: float,
    dmd_every: int,
    global_step: int,
    dmd_loss: DraftHeadDMDLoss | None,
    noise_mode: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(denoising_step_list) != len(step_weights):
        raise ValueError("denoising_step_list and step_weights must have the same length")
    if noise_mode not in ("fixed", "fresh"):
        raise ValueError("--unroll_noise_mode must be 'fixed' or 'fresh'")

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    target_latents = batch["target_latents"].to(device=device, dtype=dtype)
    current = batch["block_noise"].to(device=device, dtype=dtype)
    initial_noise = current
    batch_size, frames = target_latents.shape[:2]

    components: dict[str, torch.Tensor] = {}
    clean_losses = []
    flow_losses = []
    prediction = current
    current_noise = initial_noise
    clean_weight_denominator = max(sum(step_weights), 1e-8)
    flow_weight_denominator = max(
        sum(
            weight
            for weight, timestep in zip(step_weights, denoising_step_list, strict=True)
            if timestep > 0
        ),
        1e-8,
    )

    for index, current_timestep in enumerate(denoising_step_list):
        timestep = _timestep_batch(current_timestep, batch_size, frames, device=device)
        step_batch = dict(batch)
        step_batch["scheduled_latents"] = current
        step_batch["scheduled_noise"] = current_noise
        step_batch["timestep"] = timestep
        model_output = predict_draft_head_batch(model, step_batch, num_blocks, input_key="scheduled_latents")
        if prediction_type == "flow":
            prediction = flow_prediction_to_clean_latent(scheduler, model_output, current, timestep)
            flow_prediction = model_output
        elif prediction_type == "clean_latent":
            prediction = model_output
            flow_prediction = clean_latent_to_flow_prediction(scheduler, prediction, current, timestep)
        else:
            raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")

        step_weight = step_weights[index]
        clean_loss = F.mse_loss(prediction.float(), target_latents.float())
        clean_losses.append(clean_loss.detach())
        if loss_type in ("clean_latent", "clean_latent_flow") and step_weight > 0:
            normalized_step_weight = step_weight / clean_weight_denominator
            components[f"clean_latent_loss_step_{index}"] = clean_loss * clean_latent_loss_weight * normalized_step_weight

        if loss_type in ("flow", "clean_latent_flow") and current_timestep > 0 and step_weight > 0:
            if prediction_type == "flow":
                flow_target = clean_latent_to_flow_prediction(scheduler, target_latents, current, timestep)
            else:
                flow_target = current_noise - target_latents
            flow_loss = F.mse_loss(flow_prediction.float(), flow_target.float())
            flow_losses.append(flow_loss.detach())
            normalized_step_weight = step_weight / flow_weight_denominator
            components[f"flow_loss_step_{index}"] = flow_loss * flow_loss_weight * normalized_step_weight

        if index < len(denoising_step_list) - 1:
            next_timestep = denoising_step_list[index + 1]
            if noise_mode == "fresh":
                next_noise = torch.randn_like(target_latents)
            else:
                next_noise = initial_noise
            next_timestep_tensor = _timestep_batch(next_timestep, batch_size, frames, device=device)
            if prediction_type == "flow":
                current = flow_prediction_step(
                    scheduler,
                    flow_prediction.detach(),
                    current,
                    timestep,
                    next_timestep_tensor,
                )
            else:
                current = scheduler.add_noise(
                    prediction.detach().flatten(0, 1),
                    next_noise.flatten(0, 1),
                    next_timestep_tensor.flatten(0, 1),
                ).unflatten(0, prediction.shape[:2])
            current_noise = next_noise

    final_clean_loss = F.mse_loss(prediction.float(), target_latents.float())
    if dmd_loss_weight > 0 and dmd_loss is not None and dmd_every > 0 and global_step % dmd_every == 0:
        dmd_component, dmd_metrics = dmd_loss(prediction, batch["prompts"])
        components["dmd_loss"] = dmd_component * dmd_loss_weight
    else:
        dmd_metrics = {}

    if not components:
        raise ValueError("At least one enabled loss component is required")
    total_loss = sum(components.values())
    clean_stack = torch.stack(clean_losses) if clean_losses else final_clean_loss.detach().reshape(1)
    flow_stack = torch.stack(flow_losses) if flow_losses else torch.zeros(1, device=device, dtype=final_clean_loss.dtype)
    clean_component = sum(value for name, value in components.items() if name.startswith("clean_latent_loss_step_"))
    flow_component = sum(value for name, value in components.items() if name.startswith("flow_loss_step_"))
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(final_clean_loss.detach().cpu().item()),
        "unrolled_clean_latent_mse": float(clean_stack.mean().detach().cpu().item()),
        "clean_latent_loss": float(clean_component.detach().cpu().item()) if isinstance(clean_component, torch.Tensor) else 0.0,
        "flow_loss": float(flow_component.detach().cpu().item()) if isinstance(flow_component, torch.Tensor) else 0.0,
        "unrolled_flow_mse": float(flow_stack.mean().detach().cpu().item()),
    }
    metrics.update(dmd_metrics)
    return total_loss, metrics


def compute_teacher_trajectory_draft_head_losses(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    num_blocks: int,
    scheduler: FlowMatchScheduler,
    prediction_type: str,
    loss_type: str,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    step_indices: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if "teacher_trajectory_noisy_latents" not in batch or "teacher_trajectory_latents" not in batch:
        raise ValueError("teacher_trajectory mode requires teacher_trajectory_noisy_latents and teacher_trajectory_latents")
    if "teacher_trajectory_timesteps" not in batch:
        raise ValueError("teacher_trajectory mode requires teacher_trajectory_timesteps")
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    noisy_states = batch["teacher_trajectory_noisy_latents"].to(device=device, dtype=dtype)
    clean_targets = batch["teacher_trajectory_latents"].to(device=device, dtype=dtype)
    timesteps = batch["teacher_trajectory_timesteps"].to(device=device)
    require_finite("teacher_trajectory_noisy_latents", noisy_states)
    require_finite("teacher_trajectory_latents", clean_targets)
    if noisy_states.ndim != 6 or clean_targets.ndim != 6:
        raise ValueError("teacher trajectory tensors must have shape [B, S, T, C, H, W]")
    if noisy_states.shape != clean_targets.shape:
        raise ValueError("teacher trajectory noisy/clean tensor shape mismatch")
    if timesteps.ndim == 1:
        timesteps = timesteps.unsqueeze(0).expand(noisy_states.shape[0], -1)
    if timesteps.shape[:2] != noisy_states.shape[:2]:
        raise ValueError("teacher trajectory timesteps must have shape [B, S] or [S]")

    batch_size, num_steps, frames = noisy_states.shape[:3]
    if step_indices is None:
        active_step_indices = list(range(num_steps))
    else:
        active_step_indices = [int(index) for index in step_indices]
        if not active_step_indices:
            raise ValueError("--teacher_trajectory_step_indices must not be empty")
        invalid_indices = [index for index in active_step_indices if index < 0 or index >= num_steps]
        if invalid_indices:
            raise ValueError(
                f"teacher trajectory step indices {invalid_indices} are out of range for {num_steps} steps"
            )

    components: list[torch.Tensor] = []
    flow_losses = []
    clean_losses = []
    prefix_frames = int(batch["context_latents"].shape[1]) if batch.get("context_latents") is not None else 0
    for step_index in active_step_indices:
        state = noisy_states[:, step_index]
        clean_target = clean_targets[:, step_index]
        timestep = timesteps[:, step_index].round().long().reshape(batch_size, 1).expand(batch_size, frames)
        debug_context = (
            f"step={step_index} timestep={int(timestep.flatten()[0].detach().cpu().item())} "
            f"prompt_index={batch_scalar(batch, 'prompt_index')} "
            f"block_index={batch_scalar(batch, 'block_index')} "
            f"prefix_frames={prefix_frames}"
        )
        step_batch = dict(batch)
        step_batch["scheduled_latents"] = state
        step_batch["timestep"] = timestep
        model_output = predict_draft_head_batch(model, step_batch, num_blocks, input_key="scheduled_latents")
        require_finite(f"teacher_trajectory_model_output ({debug_context})", model_output)
        if prediction_type == "flow":
            flow_prediction = model_output
            clean_prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, state, timestep)
        elif prediction_type == "clean_latent":
            clean_prediction = model_output
            flow_prediction = clean_latent_to_flow_prediction(scheduler, clean_prediction, state, timestep)
        else:
            raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")

        if loss_type in ("flow", "clean_latent_flow"):
            flow_target = clean_latent_to_flow_prediction(scheduler, clean_target, state, timestep)
            require_finite(f"teacher_trajectory_flow_target ({debug_context})", flow_target)
            flow_loss = F.mse_loss(flow_prediction.float(), flow_target.float())
            require_finite(f"teacher_trajectory_flow_loss ({debug_context})", flow_loss)
            flow_losses.append(flow_loss.detach())
            if flow_loss_weight > 0:
                components.append(flow_loss * flow_loss_weight / max(len(active_step_indices), 1))
        if loss_type in ("clean_latent", "clean_latent_flow"):
            clean_loss = F.mse_loss(clean_prediction.float(), clean_target.float())
            require_finite(f"teacher_trajectory_clean_loss ({debug_context})", clean_loss)
            clean_losses.append(clean_loss.detach())
            if clean_latent_loss_weight > 0:
                components.append(clean_loss * clean_latent_loss_weight / max(len(active_step_indices), 1))

    if not components:
        raise ValueError("At least one teacher trajectory loss component must be enabled")
    total_loss = sum(components)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(torch.stack(clean_losses).mean().detach().cpu().item()) if clean_losses else 0.0,
        "teacher_trajectory_flow_mse": float(torch.stack(flow_losses).mean().detach().cpu().item()) if flow_losses else 0.0,
    }
    metrics["causal_prefix_frames"] = float(prefix_frames)
    metrics["teacher_trajectory_num_active_steps"] = float(len(active_step_indices))
    return total_loss, metrics


def predict_unrolled_draft_head_batch(
    model: torch.nn.Module,
    batch: dict[str, Any],
    *,
    num_blocks: int,
    scheduler: FlowMatchScheduler,
    denoising_step_list: list[int],
    prediction_type: str,
    noise_mode: str,
) -> torch.Tensor:
    if noise_mode not in ("fixed", "fresh"):
        raise ValueError("--unroll_noise_mode must be 'fixed' or 'fresh'")
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    current = batch["block_noise"].to(device=device, dtype=dtype)
    initial_noise = current
    batch_size, frames = current.shape[:2]
    prediction = current
    for index, current_timestep in enumerate(denoising_step_list):
        timestep = _timestep_batch(current_timestep, batch_size, frames, device=device)
        step_batch = dict(batch)
        step_batch["scheduled_latents"] = current
        step_batch["timestep"] = timestep
        model_output = predict_draft_head_batch(model, step_batch, num_blocks, input_key="scheduled_latents")
        if prediction_type == "flow":
            prediction = flow_prediction_to_clean_latent(scheduler, model_output, current, timestep)
        elif prediction_type == "clean_latent":
            prediction = model_output
        else:
            raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")
        if index < len(denoising_step_list) - 1:
            next_timestep = denoising_step_list[index + 1]
            next_noise = torch.randn_like(current) if noise_mode == "fresh" else initial_noise
            next_timestep_tensor = _timestep_batch(next_timestep, batch_size, frames, device=device)
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
    return prediction


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    num_blocks: int,
    device: torch.device,
    input_key: str,
    scheduler: FlowMatchScheduler | None = None,
    denoising_step_list: list[int] | None = None,
    prediction_type: str = "clean_latent",
    training_mode: str = "one_step",
    unroll_noise_mode: str = "fixed",
    prompt_text_encoder: torch.nn.Module | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_abs = 0.0
    total_values = 0
    dtype = next(model.parameters()).dtype
    for batch in loader:
        batch = attach_prompt_embeds_if_needed(
            batch,
            model=model,
            text_encoder=prompt_text_encoder,
            device=device,
            dtype=dtype,
        )
        if training_mode == "unrolled":
            if scheduler is None or denoising_step_list is None:
                raise ValueError("Unrolled evaluation requires scheduler and denoising_step_list")
            prediction = predict_unrolled_draft_head_batch(
                model,
                batch,
                num_blocks=num_blocks,
                scheduler=scheduler,
                denoising_step_list=denoising_step_list,
                prediction_type=prediction_type,
                noise_mode=unroll_noise_mode,
            )
        elif scheduler is not None and denoising_step_list is not None:
            batch = add_scheduled_noise(
                batch,
                scheduler,
                denoising_step_list,
                device=device,
                dtype=dtype,
            )
            model_output = predict_draft_head_batch(model, batch, num_blocks, input_key=input_key)
            if prediction_type == "flow":
                prediction = flow_prediction_to_clean_latent(
                    scheduler,
                    model_output,
                    batch[input_key].to(device=device, dtype=dtype),
                    batch["timestep"].to(device=device),
                )
            elif prediction_type == "clean_latent":
                prediction = model_output
            else:
                raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")
        else:
            prediction = predict_draft_head_batch(model, batch, num_blocks, input_key=input_key)
        target_latents = batch["target_latents"].to(device=device, dtype=dtype)
        diff = prediction.float() - target_latents.float()
        total_loss += float(diff.square().sum().item())
        total_abs += float(diff.abs().sum().item())
        total_values += diff.numel()

    if total_values == 0:
        return {}
    mse = total_loss / total_values
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "l1": total_abs / total_values,
    }


def serializable_args(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in result.items():
        if isinstance(value, Path):
            result[key] = str(value)
    return result


def load_draft_head_state_strict(
    model: torch.nn.Module,
    checkpoint_path: str,
    *,
    layer_names: tuple[str, ...],
    num_blocks: int,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != "sdvg_latent_draft_head_v1":
        raise ValueError(f"Unsupported draft-head checkpoint format: {payload.get('format')}")
    checkpoint_layer_names = tuple(payload["layer_names"])
    if checkpoint_layer_names != layer_names:
        raise ValueError(
            f"Checkpoint layer_names={checkpoint_layer_names} do not match current layer_names={layer_names}"
        )
    checkpoint_num_blocks = int(payload["num_blocks"])
    if checkpoint_num_blocks != num_blocks:
        raise ValueError(f"Checkpoint num_blocks={checkpoint_num_blocks} does not match current num_blocks={num_blocks}")

    inner_model = unwrap_model(model)
    if isinstance(inner_model, KVInjectedLatentDraftHead):
        expected_head_type = "kv_injected_attention"
    elif isinstance(inner_model, KVCacheInjectedLatentDraftHead):
        expected_head_type = "kv_cache_attention"
    elif isinstance(inner_model, WanDFlashLatentDraftHead):
        expected_head_type = "wan_dflash_attention"
    elif isinstance(inner_model, BidirectionalPromptAnchorDraftHead):
        expected_head_type = "ar_bidirectional"
    elif isinstance(inner_model, CausalWanARDraftHead):
        expected_head_type = "causal_wan_ar"
    else:
        expected_head_type = "conv"
    checkpoint_head_type = payload.get("head_type", "conv")
    if checkpoint_head_type != expected_head_type:
        raise ValueError(f"Checkpoint head_type={checkpoint_head_type} does not match current head_type={expected_head_type}")

    state_dict = payload["model_state_dict"]
    # strict=True guarantees every parameter and buffer is loaded and no extras are ignored.
    inner_model.load_state_dict(state_dict, strict=True)
    return {
        "path": str(Path(checkpoint_path).resolve()),
        "head_type": checkpoint_head_type,
        "layer_names": list(checkpoint_layer_names),
        "num_blocks": checkpoint_num_blocks,
        "model_config": payload.get("model_config", {}),
        "metadata": payload.get("metadata", {}),
        "num_tensors": len(state_dict),
    }


def distributed_info() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return is_distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_main(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def set_wan_dflash_copied_modules_trainable(model: torch.nn.Module, trainable: bool) -> dict[str, int]:
    inner_model = unwrap_model(model)
    if not isinstance(inner_model, WanDFlashLatentDraftHead):
        return {"modules": 0, "parameters": 0}
    copied_modules = [
        inner_model.patch_embedding,
        inner_model.time_embedding,
        inner_model.time_projection,
        inner_model.blocks,
        inner_model.head,
    ]
    parameter_count = 0
    for module in copied_modules:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
            parameter_count += parameter.numel()
    return {"modules": len(copied_modules), "parameters": parameter_count}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a target-conditioned latent draft head.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", default="outputs/sdvg/draft_head/draft_head.pt")
    parser.add_argument("--layer_names", nargs="+", default=None)
    parser.add_argument(
        "--head_type",
        choices=["conv", "attention", "kv_cache_attention", "wan_dflash_attention", "ar_bidirectional", "causal_wan_ar"],
        default="attention",
    )
    parser.add_argument("--num_blocks", type=int, default=9)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--max_context_tokens", type=int, default=512)
    parser.add_argument("--latent_pool", nargs=3, type=int, default=(1, 4, 4))
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--ffn_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--denoising_step_list", nargs="+", type=int, default=[1000, 750, 500, 250, 0])
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--training_mode", choices=["one_step", "unrolled", "teacher_trajectory"], default="one_step")
    parser.add_argument("--teacher_trajectory_step_indices", nargs="+", type=int, default=None)
    parser.add_argument("--unroll_step_weights", nargs="+", type=float, default=None)
    parser.add_argument("--unroll_noise_mode", choices=["fixed", "fresh"], default="fixed")
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument(
        "--freeze_copied_wan_epochs",
        type=int,
        default=0,
        help="For wan_dflash_attention, freeze target-copied Wan modules for the first N epochs.",
    )
    parser.add_argument("--parallel_strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--fsdp_min_num_params", type=int, default=100_000_000)
    parser.add_argument("--fsdp_mixed_precision", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--prediction_type", choices=["flow", "clean_latent"], default="flow")
    parser.add_argument(
        "--loss_type",
        choices=["clean_latent", "flow", "clean_latent_flow"],
        default="clean_latent_flow",
    )
    parser.add_argument("--clean_latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--flow_loss_weight", type=float, default=1.0)
    parser.add_argument("--dmd_loss_weight", type=float, default=0.0)
    parser.add_argument("--dmd_every", type=int, default=1)
    parser.add_argument("--dmd_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--dmd_checkpoint_path", default=None)
    parser.add_argument("--dmd_guidance_scale", type=float, default=3.0)
    parser.add_argument("--dmd_min_timestep", type=int, default=20)
    parser.add_argument("--dmd_max_timestep", type=int, default=980)
    parser.add_argument(
        "--negative_prompt",
        default="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    )
    parser.add_argument("--init_target_blocks", nargs="+", type=int, default=None)
    parser.add_argument("--init_target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--init_target_checkpoint_path", default=None)
    parser.add_argument("--init_draft_head_checkpoint_path", default=None)
    parser.add_argument("--causal_wan_local_attn_size", type=int, default=-1)
    parser.add_argument("--causal_wan_sink_size", type=int, default=0)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_records", type=int, default=0)
    parser.add_argument(
        "--shuffle_records",
        action="store_true",
        help="Shuffle individual records. Avoid this for full-feature datasets because it repeatedly reloads large shards.",
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.num_blocks <= 0:
        raise ValueError("--num_blocks must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.clean_latent_loss_weight < 0 or args.flow_loss_weight < 0 or args.dmd_loss_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if args.head_type == "kv_cache_attention" and args.batch_size != 1:
        raise ValueError("--head_type kv_cache_attention currently requires per-rank --batch_size 1 for online KV replay")
    if args.freeze_copied_wan_epochs < 0:
        raise ValueError("--freeze_copied_wan_epochs must be non-negative")
    if args.freeze_copied_wan_epochs > 0 and args.head_type != "wan_dflash_attention":
        raise ValueError("--freeze_copied_wan_epochs is only supported for --head_type wan_dflash_attention")
    if args.parallel_strategy == "fsdp" and args.freeze_copied_wan_epochs > 0:
        raise ValueError("--parallel_strategy fsdp is not compatible with --freeze_copied_wan_epochs")
    if args.dmd_every <= 0:
        raise ValueError("--dmd_every must be positive")
    if args.dmd_min_timestep < 0 or args.dmd_max_timestep <= args.dmd_min_timestep:
        raise ValueError("--dmd_max_timestep must be greater than --dmd_min_timestep")
    unroll_step_weights = parse_unroll_step_weights(args.unroll_step_weights, len(args.denoising_step_list))

    is_distributed, rank, local_rank, world_size = distributed_info()
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(local_rank)

    torch.manual_seed(args.seed)
    dataset = DraftHeadRecordDataset(args.manifest_path)
    if len(dataset) == 0:
        raise ValueError(f"No draft-head records found in {args.manifest_path}")
    if args.max_records > 0:
        dataset = Subset(dataset, list(range(min(args.max_records, len(dataset)))))

    sample = dataset[0]
    layer_names = parse_layer_names(args.layer_names, sample)
    latent_channels, feature_dim = infer_model_dims(sample, layer_names)
    layer_feature_dims = (
        infer_target_feature_dims(sample["target_features"], layer_names)
        if args.head_type in ("attention", "wan_dflash_attention")
        else {}
    )
    train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    include_target_features = args.head_type in ("attention", "wan_dflash_attention")
    collate = lambda records: collate_draft_head_records(
        records,
        layer_names=layer_names,
        include_target_features=include_target_features,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=args.shuffle_records,
            drop_last=False,
        )
        if is_distributed
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(args.shuffle_records and train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate,
    )
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
        if val_dataset is not None
        else None
    )

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() and not args.cpu else "cpu")
    scheduler = make_scheduler(args.timestep_shift)
    if args.head_type == "attention":
        model = KVInjectedLatentDraftHead(
            latent_channels=latent_channels,
            layer_feature_dims=layer_feature_dims,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_context_tokens=args.max_context_tokens,
            latent_pool=tuple(args.latent_pool),
            ffn_mult=args.ffn_mult,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(device)
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    elif args.head_type == "kv_cache_attention":
        model = KVCacheInjectedLatentDraftHead(
            latent_channels=latent_channels,
            kv_layer_names=layer_names,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            max_context_tokens=args.max_context_tokens,
            latent_pool=tuple(args.latent_pool),
            ffn_mult=args.ffn_mult,
            ffn_dim=args.ffn_dim,
            dropout=args.dropout,
        ).to(device)
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    elif args.head_type == "wan_dflash_attention":
        model = WanDFlashLatentDraftHead(
            latent_channels=latent_channels,
            layer_feature_dims=layer_feature_dims,
            hidden_channels=args.hidden_channels,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim or args.hidden_channels * args.ffn_mult,
            max_context_tokens=args.max_context_tokens,
            per_block_context=True,
            mean_init_context_fuser=True,
            dropout=args.dropout,
        ).to(device)
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    elif args.head_type == "ar_bidirectional":
        model = BidirectionalPromptAnchorDraftHead(
            latent_channels=latent_channels,
            hidden_channels=args.hidden_channels,
            prompt_dim=4096,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim or args.hidden_channels * args.ffn_mult,
            temporal_mixer_layers=0,
            max_frames=27,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    elif args.head_type == "causal_wan_ar":
        model = CausalWanARDraftHead(
            model_name=args.init_target_model_name,
            timestep_shift=args.timestep_shift,
            local_attn_size=args.causal_wan_local_attn_size,
            sink_size=args.causal_wan_sink_size,
        ).to(device)
        if args.gradient_checkpointing:
            model.enable_gradient_checkpointing()
    else:
        if args.gradient_checkpointing:
            raise ValueError("--gradient_checkpointing is only supported for attention draft-head types")
        model = LatentDraftHead(
            latent_channels=latent_channels,
            feature_dim=feature_dim,
            hidden_channels=args.hidden_channels,
            num_res_blocks=args.num_res_blocks,
        ).to(device)

    init_report = None
    init_draft_head_report = None
    if args.init_draft_head_checkpoint_path:
        if is_main_process(rank):
            init_draft_head_report = load_draft_head_state_strict(
                model,
                args.init_draft_head_checkpoint_path,
                layer_names=layer_names,
                num_blocks=args.num_blocks,
                device=device,
            )
        if is_distributed:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)
    elif args.head_type == "causal_wan_ar" and args.init_target_checkpoint_path:
        if is_main_process(rank):
            from sdvg_inference import load_checkpoint_into_generator

            inner = unwrap_model(model)
            load_checkpoint_into_generator(inner.generator, args.init_target_checkpoint_path, use_ema=True)
            init_report = {
                "type": "causal_wan_ar_full_checkpoint",
                "model_name": args.init_target_model_name,
                "checkpoint_path": args.init_target_checkpoint_path,
            }
        if is_distributed:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)
    elif args.init_target_blocks:
        if args.head_type not in ("attention", "wan_dflash_attention", "ar_bidirectional"):
            raise ValueError("--init_target_blocks is only supported for --head_type attention/wan_dflash_attention/ar_bidirectional")
        if args.init_target_checkpoint_path is None:
            raise ValueError("--init_target_checkpoint_path is required with --init_target_blocks")
        if is_main_process(rank):
            from sdvg_inference import build_pipeline, ensure_wan_symlinks, load_config
            from utils.wan_wrapper import WanTextEncoder, WanVAEWrapper

            ensure_wan_symlinks(args.model_root)
            config = load_config(args.config_path)
            text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            vae = WanVAEWrapper().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            target_pipeline = build_pipeline(
                config,
                args.init_target_model_name,
                args.init_target_checkpoint_path,
                device,
                torch.bfloat16,
                text_encoder=text_encoder,
                vae=vae,
                use_ema=False,
            )
            if args.head_type == "attention":
                init_report = initialize_attention_head_from_target_blocks(
                    model,
                    target_pipeline.generator.model,
                    tuple(args.init_target_blocks),
                )
            elif args.head_type == "wan_dflash_attention":
                init_report = initialize_wan_dflash_head_from_target_blocks(
                    model,
                    target_pipeline.generator.model,
                    tuple(args.init_target_blocks),
                )
            else:
                init_report = initialize_bidirectional_wan_head_from_target_blocks(
                    model,
                    target_pipeline.generator.model,
                    tuple(args.init_target_blocks),
                )
            del target_pipeline, text_encoder, vae
        if is_distributed:
            for tensor in model.state_dict().values():
                dist.broadcast(tensor, src=0)
    prompt_text_encoder = None
    if args.head_type == "ar_bidirectional":
        from utils.wan_wrapper import WanTextEncoder

        prompt_text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    raw_model = model
    model = wrap_model_for_training(
        model,
        strategy=args.parallel_strategy if is_distributed else "none",
        is_distributed=is_distributed,
        local_rank=local_rank,
        find_unused_parameters=args.freeze_copied_wan_epochs > 0,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_mixed_precision=args.fsdp_mixed_precision,
    )
    freeze_report = None
    copied_wan_frozen = False
    if args.freeze_copied_wan_epochs > 0:
        freeze_report = set_wan_dflash_copied_modules_trainable(model, trainable=False)
        copied_wan_frozen = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    dmd_loss = None
    if args.dmd_loss_weight > 0:
        dmd_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        dmd_loss = DraftHeadDMDLoss(
            model_name=args.dmd_model_name,
            checkpoint_path=args.dmd_checkpoint_path,
            model_root=args.model_root,
            config_path=args.config_path,
            device=device,
            dtype=dmd_dtype,
            guidance_scale=args.dmd_guidance_scale,
            min_timestep=args.dmd_min_timestep,
            max_timestep=args.dmd_max_timestep,
            timestep_shift=args.timestep_shift,
            negative_prompt=args.negative_prompt,
        )

    print_main(
        rank,
        "Training draft head "
        f"records={len(dataset)} train={len(train_dataset)} val={len(val_indices)} "
        f"layers={list(layer_names)} latent_channels={latent_channels} feature_dim={feature_dim} "
        f"head_type={args.head_type} input_source=scheduled_latents training_mode={args.training_mode} "
        f"denoising_steps={args.denoising_step_list} prediction_type={args.prediction_type} loss_type={args.loss_type} "
        f"dmd_weight={args.dmd_loss_weight} world_size={world_size} "
        f"parallel={args.parallel_strategy if is_distributed else 'none'}"
    )
    if init_report is not None:
        print_main(rank, f"Target initialization report: {init_report}")
    if init_draft_head_report is not None:
        print_main(rank, f"Draft-head checkpoint initialization report: {init_draft_head_report}")
    if freeze_report is not None:
        print_main(
            rank,
            "Freeze copied Wan modules: "
            f"epochs={args.freeze_copied_wan_epochs} modules={freeze_report['modules']} "
            f"parameters={freeze_report['parameters']}",
        )

    history = []
    step_history = []
    global_step = 0
    output_path = Path(args.output_path)
    run_dir = output_path.parent
    metrics_path = run_dir / "metrics.json"
    for epoch in range(1, args.epochs + 1):
        should_freeze_copied_wan = args.freeze_copied_wan_epochs > 0 and epoch <= args.freeze_copied_wan_epochs
        if should_freeze_copied_wan != copied_wan_frozen:
            freeze_report = set_wan_dflash_copied_modules_trainable(model, trainable=not should_freeze_copied_wan)
            copied_wan_frozen = should_freeze_copied_wan
            print_main(
                rank,
                "Copied Wan modules "
                f"{'frozen' if copied_wan_frozen else 'unfrozen'} at epoch {epoch}: {freeze_report}",
            )
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        total_loss = 0.0
        total_clean_latent_mse = 0.0
        total_batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, disable=not is_main_process(rank))
        for batch in progress:
            step_start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch = attach_prompt_embeds_if_needed(
                batch,
                model=raw_model,
                text_encoder=prompt_text_encoder,
                device=device,
                dtype=next(raw_model.parameters()).dtype,
            )
            with amp_context(device, args.amp_dtype):
                if args.training_mode == "unrolled":
                    loss_tensor, loss_metrics = compute_unrolled_draft_head_losses(
                        model,
                        batch,
                        num_blocks=args.num_blocks,
                        scheduler=scheduler,
                        denoising_step_list=args.denoising_step_list,
                        step_weights=unroll_step_weights,
                        prediction_type=args.prediction_type,
                        loss_type=args.loss_type,
                        clean_latent_loss_weight=args.clean_latent_loss_weight,
                        flow_loss_weight=args.flow_loss_weight,
                        dmd_loss_weight=args.dmd_loss_weight,
                        dmd_every=args.dmd_every,
                        global_step=global_step + 1,
                        dmd_loss=dmd_loss,
                        noise_mode=args.unroll_noise_mode,
                    )
                elif args.training_mode == "teacher_trajectory":
                    loss_tensor, loss_metrics = compute_teacher_trajectory_draft_head_losses(
                        model,
                        batch,
                        num_blocks=args.num_blocks,
                        scheduler=scheduler,
                        prediction_type=args.prediction_type,
                        loss_type=args.loss_type,
                        clean_latent_loss_weight=args.clean_latent_loss_weight,
                        flow_loss_weight=args.flow_loss_weight,
                        step_indices=args.teacher_trajectory_step_indices,
                    )
                else:
                    scheduled_batch = add_scheduled_noise(
                        batch,
                        scheduler,
                        args.denoising_step_list,
                        device=device,
                        dtype=next(model.parameters()).dtype,
                    )
                    loss_tensor, loss_metrics = compute_draft_head_losses(
                        model,
                        scheduled_batch,
                        num_blocks=args.num_blocks,
                        input_key="scheduled_latents",
                        scheduler=scheduler,
                        prediction_type=args.prediction_type,
                        loss_type=args.loss_type,
                        clean_latent_loss_weight=args.clean_latent_loss_weight,
                        flow_loss_weight=args.flow_loss_weight,
                        dmd_loss_weight=args.dmd_loss_weight,
                        dmd_every=args.dmd_every,
                        global_step=global_step + 1,
                        dmd_loss=dmd_loss,
                    )
            require_finite("loss_tensor", loss_tensor)
            loss_tensor.backward()
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                require_finite("grad_norm", grad_norm)
            optimizer.step()
            require_model_parameters_finite(model, context=f"after_optimizer_step_{global_step + 1}")
            loss = loss_metrics["loss"]
            step_seconds = time.perf_counter() - step_start
            total_loss += loss
            total_clean_latent_mse += loss_metrics["clean_latent_mse"]
            total_batches += 1
            global_step += 1
            running_loss = total_loss / max(1, total_batches)
            running_clean_latent_mse = total_clean_latent_mse / max(1, total_batches)
            running_clean_latent_rmse = running_clean_latent_mse ** 0.5
            if is_main_process(rank):
                progress.set_postfix(
                    loss=loss,
                    avg=running_loss,
                    rmse=running_clean_latent_rmse,
                    sec=step_seconds,
                )
            if is_main_process(rank) and args.log_every > 0 and global_step % args.log_every == 0:
                step_metrics = {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": total_batches,
                    "running_loss": running_loss,
                    "running_clean_latent_mse": running_clean_latent_mse,
                    "running_clean_latent_rmse": running_clean_latent_rmse,
                    "step_seconds": step_seconds,
                }
                step_metrics.update(loss_metrics)
                step_history.append(step_metrics)
                print(
                    " ".join(
                        f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}"
                        for key, value in step_metrics.items()
                    ),
                    flush=True,
                )
                partial_metadata = {
                    "train_args": serializable_args(args),
                    "manifest_path": str(Path(args.manifest_path).resolve()),
                    "num_records": len(dataset),
                    "train_records": len(train_dataset),
                    "val_records": len(val_indices),
                    "history": history,
                    "step_history": step_history[-2000:],
                    "head_type": args.head_type,
                    "input_source": "scheduled_latents",
                    "denoising_step_list": args.denoising_step_list,
                    "timestep_shift": args.timestep_shift,
                    "prediction_type": args.prediction_type,
                    "loss_type": args.loss_type,
                    "training_mode": args.training_mode,
                    "unroll_step_weights": unroll_step_weights,
                    "unroll_noise_mode": args.unroll_noise_mode,
                    "amp_dtype": args.amp_dtype,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                    "copied_wan_frozen": copied_wan_frozen,
                    "clean_latent_loss_weight": args.clean_latent_loss_weight,
                    "flow_loss_weight": args.flow_loss_weight,
                    "dmd_loss_weight": args.dmd_loss_weight,
                    "target_init_report": init_report,
                    "init_draft_head_report": init_draft_head_report,
                    "status": "running",
                }
                metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_path.write_text(json.dumps(partial_metadata, indent=2), encoding="utf-8")

        train_loss = total_loss / max(1, total_batches)
        train_clean_latent_mse = total_clean_latent_mse / max(1, total_batches)
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_clean_latent_mse": train_clean_latent_mse,
            "train_clean_latent_rmse": train_clean_latent_mse ** 0.5,
        }
        # In DDP, do validation after all training is finished. Otherwise rank 0
        # spends many minutes reading full-feature validation shards while other
        # ranks wait in a collective and hit the NCCL watchdog timeout.
        if (not is_distributed) and is_main_process(rank) and val_loader is not None:
            val_metrics = evaluate(
                raw_model,
                val_loader,
                num_blocks=args.num_blocks,
                device=device,
                input_key="scheduled_latents",
                scheduler=scheduler,
                denoising_step_list=args.denoising_step_list,
                prediction_type=args.prediction_type,
                training_mode=args.training_mode,
                unroll_noise_mode=args.unroll_noise_mode,
                prompt_text_encoder=prompt_text_encoder,
            )
            metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
        if is_main_process(rank):
            history.append(metrics)
            print(" ".join(f"{key}={value:.6f}" if isinstance(value, float) else f"{key}={value}" for key, value in metrics.items()))
            epoch_metadata = {
                "train_args": serializable_args(args),
                "manifest_path": str(Path(args.manifest_path).resolve()),
                "num_records": len(dataset),
                "train_records": len(train_dataset),
                "val_records": len(val_indices),
                "history": history,
                "step_history": step_history[-2000:],
                "head_type": args.head_type,
                "input_source": "scheduled_latents",
                "denoising_step_list": args.denoising_step_list,
                "timestep_shift": args.timestep_shift,
                "prediction_type": args.prediction_type,
                "loss_type": args.loss_type,
                "training_mode": args.training_mode,
                "unroll_step_weights": unroll_step_weights,
                "unroll_noise_mode": args.unroll_noise_mode,
                "amp_dtype": args.amp_dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                "copied_wan_frozen": copied_wan_frozen,
                "clean_latent_loss_weight": args.clean_latent_loss_weight,
                "flow_loss_weight": args.flow_loss_weight,
                "dmd_loss_weight": args.dmd_loss_weight,
                "target_init_report": init_report,
                "init_draft_head_report": init_draft_head_report,
                "status": "running",
                "last_epoch": epoch,
            }
            epoch_path = run_dir / f"epoch_{epoch:04d}.pt"
            latest_path = run_dir / "latest.pt"
            if args.parallel_strategy != "fsdp" or not is_distributed:
                save_ar_draft_head_checkpoint(
                    raw_model,
                    epoch_path,
                    layer_names=layer_names,
                    num_blocks=args.num_blocks,
                    metadata=epoch_metadata,
                )
                save_ar_draft_head_checkpoint(
                    raw_model,
                    latest_path,
                    layer_names=layer_names,
                    num_blocks=args.num_blocks,
                    metadata=epoch_metadata,
                )
                metrics_path.write_text(json.dumps(epoch_metadata, indent=2), encoding="utf-8")
                print(f"Wrote epoch checkpoint: {epoch_path}")
        if args.parallel_strategy == "fsdp" and is_distributed:
            epoch_metadata = {
                "train_args": serializable_args(args),
                "manifest_path": str(Path(args.manifest_path).resolve()),
                "num_records": len(dataset),
                "train_records": len(train_dataset),
                "val_records": len(val_indices),
                "history": history,
                "step_history": step_history[-2000:],
                "head_type": args.head_type,
                "input_source": "scheduled_latents",
                "denoising_step_list": args.denoising_step_list,
                "timestep_shift": args.timestep_shift,
                "prediction_type": args.prediction_type,
                "loss_type": args.loss_type,
                "training_mode": args.training_mode,
                "unroll_step_weights": unroll_step_weights,
                "unroll_noise_mode": args.unroll_noise_mode,
                "amp_dtype": args.amp_dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                "copied_wan_frozen": copied_wan_frozen,
                "clean_latent_loss_weight": args.clean_latent_loss_weight,
                "flow_loss_weight": args.flow_loss_weight,
                "dmd_loss_weight": args.dmd_loss_weight,
                "target_init_report": init_report,
                "init_draft_head_report": init_draft_head_report,
                "status": "running",
                "last_epoch": epoch,
            }
            epoch_path = run_dir / f"epoch_{epoch:04d}.pt"
            latest_path = run_dir / "latest.pt"
            state_dict = fsdp_rank0_state_dict(model)
            saved_epoch = None
            if is_main_process(rank):
                saved_epoch = save_ar_draft_head_checkpoint(
                    raw_model,
                    epoch_path,
                    layer_names=layer_names,
                    num_blocks=args.num_blocks,
                    metadata=epoch_metadata,
                    state_dict=state_dict,
                )
                save_ar_draft_head_checkpoint(
                    raw_model,
                    latest_path,
                    layer_names=layer_names,
                    num_blocks=args.num_blocks,
                    metadata=epoch_metadata,
                    state_dict=state_dict,
                )
                metrics_path.write_text(json.dumps(epoch_metadata, indent=2), encoding="utf-8")
                print(f"Wrote epoch checkpoint: {saved_epoch}")

    final_val_metrics = None
    if is_distributed and args.parallel_strategy != "fsdp" and is_main_process(rank) and val_loader is not None:
        final_val_metrics = evaluate(
            raw_model,
            val_loader,
            num_blocks=args.num_blocks,
            device=device,
            input_key="scheduled_latents",
            scheduler=scheduler,
            denoising_step_list=args.denoising_step_list,
            prediction_type=args.prediction_type,
            training_mode=args.training_mode,
            unroll_noise_mode=args.unroll_noise_mode,
            prompt_text_encoder=prompt_text_encoder,
        )
        print(
            " ".join(
                f"final_val_{key}={value:.6f}" if isinstance(value, float) else f"final_val_{key}={value}"
                for key, value in final_val_metrics.items()
            )
        )
    if not is_main_process(rank) and not (args.parallel_strategy == "fsdp" and is_distributed):
        if is_distributed:
            dist.destroy_process_group()
        return
    metadata = {
        "train_args": serializable_args(args),
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "num_records": len(dataset),
        "train_records": len(train_dataset),
        "val_records": len(val_indices),
        "history": history,
        "step_history": step_history,
        "head_type": args.head_type,
        "input_source": "scheduled_latents",
        "denoising_step_list": args.denoising_step_list,
        "timestep_shift": args.timestep_shift,
        "prediction_type": args.prediction_type,
        "loss_type": args.loss_type,
        "training_mode": args.training_mode,
        "unroll_step_weights": unroll_step_weights,
        "unroll_noise_mode": args.unroll_noise_mode,
        "amp_dtype": args.amp_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
        "clean_latent_loss_weight": args.clean_latent_loss_weight,
        "flow_loss_weight": args.flow_loss_weight,
        "dmd_loss_weight": args.dmd_loss_weight,
        "target_init_report": init_report,
        "init_draft_head_report": init_draft_head_report,
        "final_val": final_val_metrics,
        "status": "completed",
    }
    if args.parallel_strategy == "fsdp" and is_distributed:
        saved_output_path = rank0_save_with_state_dict(
            model=model,
            unwrapped_model=raw_model,
            path=output_path,
            save_fn=save_ar_draft_head_checkpoint,
            save_kwargs={"layer_names": layer_names, "num_blocks": args.num_blocks, "metadata": metadata},
        )
    elif is_main_process(rank):
        saved_output_path = save_ar_draft_head_checkpoint(
            raw_model,
            output_path,
            layer_names=layer_names,
            num_blocks=args.num_blocks,
            metadata=metadata,
        )
    else:
        saved_output_path = None
    if is_main_process(rank):
        metrics_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Wrote draft-head checkpoint: {saved_output_path}")
        print(f"Wrote training metrics: {metrics_path}")
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
