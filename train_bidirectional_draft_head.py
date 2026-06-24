#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from contextlib import nullcontext
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fsdp_utils import rank0_save_with_state_dict, unwrap_model, wrap_model_for_training
from utils.scheduler import FlowMatchScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.rcm_rf import rf_to_sigma, rf_to_trig_time, sample_lognormal_rf_times, sample_shifted_uniform_rf_times
from sdvg_draft_head import (
    DraftCausalHead,
    WanDFlashDraftBlock,
    causal_rope_apply,
    rope_params,
    sinusoidal_embedding_1d,
)
from wan.modules.model import (
    Head as WanHead,
    WanAttentionBlock,
    rope_params as wan_rope_params,
    sinusoidal_embedding_1d as wan_sinusoidal_embedding_1d,
)


def split_indices(num_records: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(num_records))
    random.Random(seed).shuffle(indices)
    if val_fraction <= 0:
        return sorted(indices), []
    val_count = max(1, int(round(num_records * val_fraction)))
    val_count = min(val_count, num_records - 1)
    return sorted(indices[val_count:]), sorted(indices[:val_count])


def amp_context(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    return torch.amp.autocast(device_type=device.type, dtype=dtype)


def configure_attention_backend(backend: str) -> None:
    if backend == "auto" or not torch.cuda.is_available():
        return
    cuda_backends = torch.backends.cuda
    if backend == "no_cudnn":
        if hasattr(cuda_backends, "enable_cudnn_sdp"):
            cuda_backends.enable_cudnn_sdp(False)
        return
    if backend != "math":
        raise ValueError("--attention_backend must be 'auto', 'no_cudnn', or 'math'")

    if hasattr(cuda_backends, "enable_flash_sdp"):
        cuda_backends.enable_flash_sdp(False)
    if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
        cuda_backends.enable_mem_efficient_sdp(False)
    if hasattr(cuda_backends, "enable_cudnn_sdp"):
        cuda_backends.enable_cudnn_sdp(False)
    if hasattr(cuda_backends, "enable_math_sdp"):
        cuda_backends.enable_math_sdp(True)


def make_scheduler(timestep_shift: float) -> FlowMatchScheduler:
    scheduler = FlowMatchScheduler(
        shift=timestep_shift,
        sigma_min=0.0,
        extra_one_step=True,
    )
    scheduler.set_timesteps(1000, training=True)
    return scheduler


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


def spatial_detail_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Match latent-space spatial gradients to discourage overly smooth outputs."""
    pred_h = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_h = target[..., 1:, :] - target[..., :-1, :]
    pred_w = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_w = target[..., :, 1:] - target[..., :, :-1]
    return 0.5 * (F.mse_loss(pred_h.float(), target_h.float()) + F.mse_loss(pred_w.float(), target_w.float()))


def timestep_batch(
    timestep: int,
    batch_size: int,
    frames: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.full((batch_size, frames), int(timestep), device=device, dtype=torch.long)


def make_descending_timestep_list(num_steps: int) -> list[int]:
    if num_steps < 2:
        raise ValueError("--dense_schedule_steps must be >= 2")
    return torch.linspace(1000, 0, steps=num_steps).round().long().tolist()


def resolve_rollout_steps(
    *,
    rollout_steps: int,
    denoising_step_list: list[int],
) -> int:
    if rollout_steps > 0:
        if rollout_steps < 2:
            raise ValueError("--rollout_steps must be >= 2")
        return int(rollout_steps)
    if not denoising_step_list:
        raise ValueError("denoising_step_list must not be empty")
    return len(denoising_step_list)


def _parse_rollout_time_token(token: str, *, sigma_max: float) -> float:
    token = token.strip()
    if not token:
        raise ValueError("empty rollout schedule token")
    if token == "sigma_max":
        if sigma_max <= 0:
            raise ValueError("--rollout_sigma_max must be positive")
        return sigma_max / (sigma_max + 1.0)
    if "/" in token:
        numerator, denominator = token.split("/", 1)
        return float(numerator) / float(denominator)
    return float(token)


def parse_rollout_timestep_schedule(schedule: str, *, sigma_max: float) -> list[int]:
    """Parse rCM-style RF times or raw timesteps into descending 0..1000 timesteps."""
    tokens = [token for token in schedule.replace(",", " ").split() if token]
    if len(tokens) < 2:
        raise ValueError("--rollout_schedule must contain at least two time points")
    values = [_parse_rollout_time_token(token, sigma_max=sigma_max) for token in tokens]
    timesteps = [int(round(value * 1000.0)) if abs(value) <= 1.0 else int(round(value)) for value in values]
    timesteps = [max(0, min(1000, timestep)) for timestep in timesteps]
    if any(a < b for a, b in zip(timesteps, timesteps[1:])):
        raise ValueError("--rollout_schedule must be descending from noisy to clean")
    if timesteps[-1] != 0:
        raise ValueError("--rollout_schedule must end at 0")
    return timesteps


def default_rcm_rollout_schedule(*, rollout_steps: int, sigma_max: float) -> list[int]:
    if rollout_steps == 5:
        return parse_rollout_timestep_schedule("sigma_max 15/16 5/6 5/8 0", sigma_max=sigma_max)
    if rollout_steps == 4:
        return parse_rollout_timestep_schedule("sigma_max 5/6 5/8 0", sigma_max=sigma_max)
    if rollout_steps == 3:
        return parse_rollout_timestep_schedule("sigma_max 5/8 0", sigma_max=sigma_max)
    if rollout_steps == 2:
        return parse_rollout_timestep_schedule("sigma_max 0", sigma_max=sigma_max)
    raise ValueError("--rollout_solver rcm supports rollout_steps in {2,3,4,5} unless --rollout_schedule is set")


def resolve_rollout_timestep_list(
    *,
    rollout_solver: str,
    rollout_steps: int,
    rollout_schedule: str,
    rollout_sigma_max: float,
    denoising_step_list: list[int],
) -> list[int]:
    if rollout_schedule:
        return parse_rollout_timestep_schedule(rollout_schedule, sigma_max=rollout_sigma_max)
    if rollout_solver == "rcm":
        steps = resolve_rollout_steps(rollout_steps=rollout_steps, denoising_step_list=denoising_step_list)
        return default_rcm_rollout_schedule(rollout_steps=steps, sigma_max=rollout_sigma_max)
    return denoising_step_list


def sample_random_timesteps(
    *,
    batch_size: int,
    candidate_timesteps: list[int],
    sampling: str,
    logit_normal_mean: float,
    logit_normal_std: float,
    device: torch.device,
) -> torch.Tensor:
    if sampling == "uniform_schedule":
        return torch.tensor(candidate_timesteps, device=device, dtype=torch.long)[
            torch.randint(len(candidate_timesteps), (batch_size,), device=device)
        ]
    if sampling == "logit_normal":
        if logit_normal_std <= 0:
            raise ValueError("--logit_normal_std must be > 0")
        # Wan-style flow matching samples continuous t from a logit-normal
        # distribution. In this codebase, timestep 1000 is the noisiest end and
        # timestep 0 is clean, so scale t in [0, 1] to [1, 1000].
        t = torch.sigmoid(torch.randn(batch_size, device=device) * logit_normal_std + logit_normal_mean)
        return (t * 1000.0).round().long().clamp_(1, 1000)
    raise ValueError("--random_timestep_sampling must be 'uniform_schedule' or 'logit_normal'")


def parse_unroll_step_weights(weights: list[float] | None, num_steps: int) -> list[float]:
    if weights is None:
        return [min(1.0, 0.25 + 0.25 * index) for index in range(num_steps)]
    if len(weights) != num_steps:
        raise ValueError(f"--unroll_step_weights must have {num_steps} values, got {len(weights)}")
    if any(weight < 0 for weight in weights):
        raise ValueError("--unroll_step_weights values must be non-negative")
    return [float(weight) for weight in weights]


def shifted_uniform_timesteps(
    *,
    batch_size: int,
    frames: int,
    shift: float,
    min_timestep: float,
    max_timestep: float,
    device: torch.device,
) -> torch.Tensor:
    """Sample rCM-style D timesteps from uniform RF time followed by timeshift."""
    if shift <= 0:
        raise ValueError("DMD timestep shift must be positive")
    if max_timestep <= min_timestep:
        raise ValueError("DMD max timestep must be greater than min timestep")
    unit = torch.rand(batch_size, 1, device=device, dtype=torch.float32)
    shifted = shift * unit / (1 + (shift - 1) * unit)
    timestep = (shifted * 1000.0).clamp(min_timestep, max_timestep)
    return timestep.expand(batch_size, frames)


def rcm_rf_training_timesteps(
    *,
    batch_size: int,
    frames: int,
    distribution: str,
    shift: float,
    lognormal_mean: float,
    lognormal_std: float,
    min_timestep: float,
    max_timestep: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample DMD score times in RF domain and return Wan timesteps plus rCM diagnostics."""
    if max_timestep <= min_timestep:
        raise ValueError("DMD max timestep must be greater than min timestep")
    shape = (batch_size, 1)
    if distribution == "shifted_uniform":
        rf_time = sample_shifted_uniform_rf_times(shape=shape, shift=shift, device=device, dtype=torch.float32)
    elif distribution == "rcm_lognormal":
        rf_time = sample_lognormal_rf_times(
            shape=shape,
            mean=lognormal_mean,
            std=lognormal_std,
            device=device,
            dtype=torch.float32,
        )
    else:
        raise ValueError("--dmd_time_distribution must be 'shifted_uniform' or 'rcm_lognormal'")
    timestep = (rf_time * 1000.0).clamp(min_timestep, max_timestep)
    rf_time = (timestep / 1000.0).clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
    trig_time = rf_to_trig_time(rf_time)
    sigma = rf_to_sigma(rf_time)
    return (
        timestep.expand(batch_size, frames),
        rf_time.expand(batch_size, frames),
        trig_time.expand(batch_size, frames),
        sigma.expand(batch_size, frames),
    )


def rcm_dmd_gradient_target(
    generator_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
    *,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the detached rCM DMD target and normalized fake-minus-teacher gradient."""
    if generator_x0.shape != fake_score_x0.shape or generator_x0.shape != teacher_x0.shape:
        raise ValueError("DMD generator, fake-score, and teacher predictions must share shape")
    reduce_dims = tuple(range(1, generator_x0.ndim))
    weight_factor = (generator_x0.detach().double() - teacher_x0.detach().double()).abs().mean(
        dim=reduce_dims,
        keepdim=True,
    ).clamp_min(eps)
    grad = (fake_score_x0.detach().double() - teacher_x0.detach().double()) / weight_factor
    grad = torch.nan_to_num(grad).to(dtype=generator_x0.dtype)
    return (generator_x0 - grad).detach(), grad


def rcm_dmd_surrogate_loss(
    generator_x0: torch.Tensor,
    fake_score_x0: torch.Tensor,
    teacher_x0: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """rCM student DMD loss: apply the fake-score minus teacher score to G(x)."""
    target, grad = rcm_dmd_gradient_target(generator_x0, fake_score_x0, teacher_x0)
    per_element = (generator_x0.float() - target.float()).square()
    nan_sample = torch.isnan(per_element).flatten(start_dim=1).any(dim=1)
    if nan_sample.any():
        per_element = per_element.clone()
        per_element[nan_sample] = 0
    loss = per_element.flatten(start_dim=1).sum(dim=1).mean()
    return loss, {
        "dmd_gradient_norm": float(grad.detach().abs().mean().cpu().item()),
    }


def rcm_fake_score_loss(
    fake_score_x0: torch.Tensor,
    generator_x0: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    """Train the fake-score net to predict x0 on student-generated samples."""
    if fake_score_x0.shape != generator_x0.shape:
        raise ValueError("fake-score prediction and generator target must share shape")
    while sigma.ndim < fake_score_x0.ndim:
        sigma = sigma.unsqueeze(-1)
    per_element = (fake_score_x0.float() - generator_x0.detach().float()).square() / sigma.float().square().clamp_min(1e-6)
    nan_sample = torch.isnan(per_element).flatten(start_dim=1).any(dim=1)
    if nan_sample.any():
        per_element = per_element.clone()
        per_element[nan_sample] = 0
    return per_element.flatten(start_dim=1).mean(dim=1).mean()


class RCMStyleDraftHeadDMD:
    """rCM-style DMD owner with a frozen bidirectional teacher and trainable fake-score net."""

    def __init__(
        self,
        *,
        teacher_model_name: str,
        fake_score_model_name: str,
        teacher_checkpoint_path: str | None,
        fake_score_checkpoint_path: str | None,
        model_root: str,
        config_path: str,
        text_encoder: nn.Module,
        device: torch.device,
        fake_score_device: torch.device,
        dtype: torch.dtype,
        guidance_scale: float,
        timestep_shift: float,
        min_timestep: float,
        max_timestep: float,
        time_distribution: str,
        lognormal_mean: float,
        lognormal_std: float,
        fake_score_weighting: str,
        target_convention: str,
        consistency_objective: str,
        consistency_loss_weight: float,
        consistency_fd_epsilon: float,
        score_scope: str,
        fake_score_lr: float,
        fake_score_weight_decay: float,
        negative_prompt: str,
    ):
        if score_scope not in ("future", "full"):
            raise ValueError("--dmd_score_scope must be 'future' or 'full'")
        from sdvg_inference import ensure_wan_symlinks, load_checkpoint_into_generator, load_config
        from utils.wan_wrapper import WanDiffusionWrapper

        ensure_wan_symlinks(model_root)
        config = load_config(config_path)
        model_kwargs = dict(getattr(config, "model_kwargs", {}))
        model_kwargs.pop("model_name", None)
        self.teacher_model_name = teacher_model_name
        self.fake_score_model_name = fake_score_model_name
        self.teacher = WanDiffusionWrapper(model_name=teacher_model_name, **model_kwargs, is_causal=False)
        self.fake_score = WanDiffusionWrapper(model_name=fake_score_model_name, **model_kwargs, is_causal=False)
        if teacher_checkpoint_path and not teacher_checkpoint_path.endswith(".index.json"):
            load_checkpoint_into_generator(self.teacher, teacher_checkpoint_path, use_ema=False)
        self.teacher = self.teacher.to(device=device, dtype=dtype).eval().requires_grad_(False)
        if fake_score_checkpoint_path:
            load_checkpoint_into_generator(self.fake_score, fake_score_checkpoint_path, use_ema=False)
            self.fake_score_init = f"checkpoint:{fake_score_checkpoint_path}"
        elif fake_score_model_name == teacher_model_name:
            self.fake_score.load_state_dict(self.teacher.state_dict(), strict=True)
            self.fake_score_init = "teacher_state"
        else:
            # Cross-size ablations cannot copy teacher weights; keep the fake-score
            # model's own pretrained initialization from WanDiffusionWrapper.
            self.fake_score_init = "pretrained_model"
        self.fake_score = self.fake_score.to(device=fake_score_device, dtype=dtype).train().requires_grad_(True)
        self.raw_fake_score = self.fake_score
        self.fake_score_is_wrapped = False
        self.fake_score_init_device = str(fake_score_device)
        self.fake_score_lr = float(fake_score_lr)
        self.fake_score_weight_decay = float(fake_score_weight_decay)
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler = self.teacher.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.text_encoder = text_encoder
        self.device = device
        self.dtype = dtype
        self.guidance_scale = float(guidance_scale)
        self.timestep_shift = float(timestep_shift)
        self.min_timestep = float(min_timestep)
        self.max_timestep = float(max_timestep)
        self.time_distribution = str(time_distribution)
        self.lognormal_mean = float(lognormal_mean)
        self.lognormal_std = float(lognormal_std)
        self.fake_score_weighting = str(fake_score_weighting)
        self.target_convention = str(target_convention)
        if self.target_convention not in ("wan_rf_x0", "rcm_trig_x0"):
            raise ValueError("--dmd_target_convention must be 'wan_rf_x0' or 'rcm_trig_x0'")
        if consistency_objective not in ("none", "rcm_fd", "rcm_jvp"):
            raise ValueError("--dmd_consistency_objective must be 'none', 'rcm_fd', or 'rcm_jvp'")
        if consistency_loss_weight < 0:
            raise ValueError("--dmd_consistency_loss_weight must be non-negative")
        if consistency_fd_epsilon <= 0:
            raise ValueError("--dmd_consistency_fd_epsilon must be positive")
        self.consistency_objective = str(consistency_objective)
        self.consistency_loss_weight = float(consistency_loss_weight)
        self.consistency_fd_epsilon = float(consistency_fd_epsilon)
        self.score_scope = score_scope
        self.negative_prompt = negative_prompt
        self._negative_cache: dict[int, torch.Tensor] = {}

    def enable_fake_score_gradient_checkpointing(self) -> None:
        if hasattr(self.raw_fake_score, "enable_gradient_checkpointing"):
            self.raw_fake_score.enable_gradient_checkpointing()

    def wrap_fake_score(
        self,
        *,
        strategy: str,
        is_distributed: bool,
        local_rank: int,
        fsdp_min_num_params: int,
        fsdp_mixed_precision: str,
    ) -> None:
        self.fake_score = wrap_model_for_training(
            self.fake_score,
            strategy=strategy,
            is_distributed=is_distributed,
            local_rank=local_rank,
            fsdp_min_num_params=fsdp_min_num_params,
            fsdp_mixed_precision=fsdp_mixed_precision,
        )
        self.fake_score_is_wrapped = self.fake_score is not self.raw_fake_score

    def create_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.fake_score.parameters(),
            lr=self.fake_score_lr,
            weight_decay=self.fake_score_weight_decay,
        )

    @staticmethod
    def is_student_phase(step: int, *, warmup_steps: int, student_update_freq: int) -> bool:
        if student_update_freq <= 0:
            raise ValueError("--dmd_student_update_freq must be positive")
        return step <= warmup_steps or (step - warmup_steps) % student_update_freq == 0

    def _negative_prompt_embeds(self, batch_size: int) -> torch.Tensor:
        if batch_size not in self._negative_cache:
            with torch.no_grad():
                encoded = self.text_encoder([self.negative_prompt] * batch_size)["prompt_embeds"]
            self._negative_cache[batch_size] = encoded.detach().to(device=self.device, dtype=self.dtype)
        return self._negative_cache[batch_size]

    def _score_clean_and_slice(
        self,
        generator_x0: torch.Tensor,
        anchor: torch.Tensor | None,
    ) -> tuple[torch.Tensor, slice]:
        if self.score_scope == "full" and anchor is not None:
            anchor = anchor.detach().to(device=generator_x0.device, dtype=generator_x0.dtype)
            return torch.cat([anchor, generator_x0], dim=1), slice(anchor.shape[1], None)
        return generator_x0, slice(None)

    def _sample_noisy_score_input(
        self,
        clean_x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, frames = clean_x0.shape[:2]
        if self.time_distribution == "shifted_uniform":
            timestep = shifted_uniform_timesteps(
                batch_size=batch_size,
                frames=frames,
                shift=self.timestep_shift,
                min_timestep=self.min_timestep,
                max_timestep=self.max_timestep,
                device=clean_x0.device,
            )
            rf_time = (timestep / 1000.0).clamp(0.0, 1.0 - torch.finfo(torch.float32).eps)
            trig_time = rf_to_trig_time(rf_time)
            explicit_weight_scale = None
        elif self.time_distribution == "rcm_lognormal":
            timestep, rf_time, trig_time, rf_sigma = rcm_rf_training_timesteps(
                batch_size=batch_size,
                frames=frames,
                distribution=self.time_distribution,
                shift=self.timestep_shift,
                lognormal_mean=self.lognormal_mean,
                lognormal_std=self.lognormal_std,
                min_timestep=self.min_timestep,
                max_timestep=self.max_timestep,
                device=clean_x0.device,
            )
            if self.fake_score_weighting == "scheduler_sigma":
                explicit_weight_scale = rf_sigma
            elif self.fake_score_weighting == "rcm_trig_sin":
                explicit_weight_scale = torch.sin(trig_time).clamp_min(1e-6)
            else:
                raise ValueError("--dmd_fake_score_weighting must be 'scheduler_sigma' or 'rcm_trig_sin'")
        else:
            raise ValueError("--dmd_time_distribution must be 'shifted_uniform' or 'rcm_lognormal'")
        noise = torch.randn_like(clean_x0)
        if self.time_distribution == "rcm_lognormal":
            rf = rf_time.reshape(*rf_time.shape, 1, 1, 1).to(device=clean_x0.device, dtype=clean_x0.dtype)
            noisy = (1.0 - rf) * clean_x0.detach() + rf * noise
        else:
            noisy = self.scheduler.add_noise(
                clean_x0.detach().flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1),
            ).unflatten(0, clean_x0.shape[:2])
        sigma = sigma_for_timestep(
            self.scheduler,
            timestep,
            device=clean_x0.device,
            dtype=clean_x0.dtype,
        )
        if explicit_weight_scale is not None:
            sigma = explicit_weight_scale.reshape(*explicit_weight_scale.shape, 1, 1, 1).to(device=clean_x0.device, dtype=clean_x0.dtype)
        return noisy.to(dtype=self.dtype), timestep, sigma, rf_time, trig_time

    def _predict_rcm_trig_x0(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        prompt_embeds: torch.Tensor,
        trig_time: torch.Tensor,
    ) -> torch.Tensor:
        c_skip = 1.0 / (torch.cos(trig_time.double()) + torch.sin(trig_time.double()))
        c_out = -torch.sin(trig_time.double()) / (torch.cos(trig_time.double()) + torch.sin(trig_time.double()))
        c_in = c_skip
        c_noise = (torch.sin(trig_time.double()) / (torch.cos(trig_time.double()) + torch.sin(trig_time.double()))) * 1000.0
        while c_in.ndim < noisy.ndim:
            c_in = c_in.unsqueeze(-1)
            c_skip = c_skip.unsqueeze(-1)
            c_out = c_out.unsqueeze(-1)
        timestep = c_noise.to(device=noisy.device, dtype=torch.float32)
        net_output, _ = model(
            noisy_image_or_video=(noisy.double() * c_in).to(device=noisy.device, dtype=self.dtype),
            conditional_dict={"prompt_embeds": prompt_embeds.to(device=noisy.device, dtype=self.dtype)},
            timestep=timestep,
        )
        x0 = c_skip.to(device=noisy.device) * noisy.double() + c_out.to(device=noisy.device) * net_output.double()
        return x0.to(dtype=noisy.dtype)

    def _predict_x0(
        self,
        model: nn.Module,
        noisy: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        trig_time: torch.Tensor,
    ) -> torch.Tensor:
        if self.target_convention == "rcm_trig_x0":
            return self._predict_rcm_trig_x0(model, noisy.to(device=self.device, dtype=self.dtype), prompt_embeds, trig_time)
        _, pred_x0 = model(
            noisy_image_or_video=noisy.to(device=self.device, dtype=self.dtype),
            conditional_dict={"prompt_embeds": prompt_embeds.to(device=self.device, dtype=self.dtype)},
            timestep=timestep.to(device=self.device),
        )
        return pred_x0

    def _teacher_cfg_x0(
        self,
        noisy: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        trig_time: torch.Tensor,
    ) -> torch.Tensor:
        cond_x0 = self._predict_x0(self.teacher, noisy, prompt_embeds, timestep, trig_time)
        if self.guidance_scale <= 1.0:
            return cond_x0
        uncond_x0 = self._predict_x0(self.teacher, noisy, self._negative_prompt_embeds(noisy.shape[0]), timestep, trig_time)
        return uncond_x0 + self.guidance_scale * (cond_x0 - uncond_x0)

    def _predict_flow(self, model: nn.Module, noisy: torch.Tensor, prompt_embeds: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        flow, _x0 = model(
            noisy_image_or_video=noisy.to(device=self.device, dtype=self.dtype),
            conditional_dict={"prompt_embeds": prompt_embeds.to(device=self.device, dtype=self.dtype)},
            timestep=timestep.to(device=self.device),
        )
        return flow

    def _teacher_cfg_flow(self, noisy: torch.Tensor, prompt_embeds: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        cond_flow = self._predict_flow(self.teacher, noisy, prompt_embeds, timestep)
        if self.guidance_scale <= 1.0:
            return cond_flow
        uncond_flow = self._predict_flow(self.teacher, noisy, self._negative_prompt_embeds(noisy.shape[0]), timestep)
        return uncond_flow + self.guidance_scale * (cond_flow - uncond_flow)

    @staticmethod
    def _student_future_flow(
        student_model: nn.Module,
        *,
        anchor: torch.Tensor | None,
        noisy_future: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return student_model(
            anchor_latents=anchor,
            future_noise=noisy_future,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
        )

    def consistency_loss(
        self,
        student_model: nn.Module,
        generator_x0: torch.Tensor,
        *,
        anchor: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.consistency_objective == "none" or self.consistency_loss_weight <= 0:
            return generator_x0.new_zeros(()), {}
        clean_score_x0, future_slice = self._score_clean_and_slice(generator_x0.detach(), anchor)
        noisy, timestep, _sigma, rf_time, _trig_time = self._sample_noisy_score_input(clean_score_x0)
        with torch.no_grad():
            teacher_flow = self._teacher_cfg_flow(noisy, prompt_embeds, timestep)[:, future_slice]
        noisy_future = noisy[:, future_slice].detach()
        timestep_future = timestep[:, future_slice].detach()
        student_anchor = anchor.detach().to(device=noisy_future.device, dtype=noisy_future.dtype) if anchor is not None else None
        prompt_embeds = prompt_embeds.to(device=noisy_future.device, dtype=noisy_future.dtype)

        with torch.no_grad():
            if self.consistency_objective == "rcm_fd":
                h = float(self.consistency_fd_epsilon)
                base_flow = self._student_future_flow(
                    student_model,
                    anchor=student_anchor,
                    noisy_future=noisy_future,
                    prompt_embeds=prompt_embeds,
                    timestep=timestep_future,
                )
                next_noisy = noisy_future + h * teacher_flow.to(dtype=noisy_future.dtype)
                next_timestep = (timestep_future + h * 1000.0).clamp(0.0, 1000.0)
                next_flow = self._student_future_flow(
                    student_model,
                    anchor=student_anchor,
                    noisy_future=next_noisy,
                    prompt_embeds=prompt_embeds,
                    timestep=next_timestep,
                )
                tangent_flow = (next_flow.float() - base_flow.float()) / h
            elif self.consistency_objective == "rcm_jvp":
                def student_fn(noisy_arg: torch.Tensor, timestep_arg: torch.Tensor) -> torch.Tensor:
                    return self._student_future_flow(
                        student_model,
                        anchor=student_anchor,
                        noisy_future=noisy_arg,
                        prompt_embeds=prompt_embeds,
                        timestep=timestep_arg,
                    )

                _base_flow, tangent_flow = torch.func.jvp(
                    student_fn,
                    (noisy_future, timestep_future),
                    (teacher_flow.to(dtype=noisy_future.dtype), torch.ones_like(timestep_future) * 1000.0),
                )
                tangent_flow = tangent_flow.float()
            else:
                raise ValueError("--dmd_consistency_objective must be 'none', 'rcm_fd', or 'rcm_jvp'")

        student_flow = self._student_future_flow(
            student_model,
            anchor=student_anchor,
            noisy_future=noisy_future,
            prompt_embeds=prompt_embeds,
            timestep=timestep_future,
        )
        student_flow_sg = student_flow.detach()
        rf = rf_time[:, future_slice].reshape(*rf_time[:, future_slice].shape, 1, 1, 1).to(
            device=student_flow.device,
            dtype=student_flow.dtype,
        )
        grad = -(student_flow_sg.float() - teacher_flow.float()) - rf.float() * tangent_flow.float()
        reduce_dims = tuple(range(1, grad.ndim))
        grad = grad.double() / (grad.double().norm(p=2, dim=reduce_dims, keepdim=True) + 0.1)
        grad = torch.nan_to_num(grad).to(dtype=student_flow.dtype)
        per_element = (student_flow.float() - student_flow_sg.float() - grad.float()).square()
        nan_sample = torch.isnan(per_element).flatten(start_dim=1).any(dim=1)
        if nan_sample.any():
            per_element = per_element.clone()
            per_element[nan_sample] = 0
        loss = per_element.flatten(start_dim=1).sum(dim=1).mean()
        return loss, {
            "dmd_consistency_loss_raw": float(loss.detach().cpu().item()),
            "dmd_consistency_rf_time": float(rf_time.detach().mean().cpu().item()),
            "dmd_consistency_gradient_norm": float(grad.detach().abs().mean().cpu().item()),
        }

    def student_loss(
        self,
        generator_x0: torch.Tensor,
        *,
        anchor: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        clean_score_x0, future_slice = self._score_clean_and_slice(generator_x0, anchor)
        noisy, timestep, _sigma, rf_time, trig_time = self._sample_noisy_score_input(clean_score_x0)
        self.teacher.eval()
        self.fake_score.eval()
        with torch.no_grad():
            fake_x0 = self._predict_x0(self.fake_score, noisy, prompt_embeds, timestep, trig_time)
            teacher_x0 = self._teacher_cfg_x0(noisy, prompt_embeds, timestep, trig_time)
        loss, metrics = rcm_dmd_surrogate_loss(generator_x0, fake_x0[:, future_slice], teacher_x0[:, future_slice])
        metrics["dmd_timestep"] = float(timestep.detach().mean().cpu().item())
        metrics["dmd_rf_time"] = float(rf_time.detach().mean().cpu().item())
        metrics["dmd_trig_time"] = float(trig_time.detach().mean().cpu().item())
        metrics[f"dmd_target_convention_{self.target_convention}"] = 1.0
        return loss, metrics

    def fake_score_loss(
        self,
        generator_x0: torch.Tensor,
        *,
        anchor: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        clean_score_x0, _future_slice = self._score_clean_and_slice(generator_x0.detach(), anchor)
        noisy, timestep, sigma, rf_time, trig_time = self._sample_noisy_score_input(clean_score_x0)
        self.fake_score.train()
        fake_x0 = self._predict_x0(self.fake_score, noisy, prompt_embeds.detach(), timestep, trig_time)
        loss = rcm_fake_score_loss(fake_x0, clean_score_x0.detach(), sigma)
        return loss, {
            "dmd_fake_score_loss": float(loss.detach().cpu().item()),
            "dmd_fake_score_timestep": float(timestep.detach().mean().cpu().item()),
            "dmd_fake_score_rf_time": float(rf_time.detach().mean().cpu().item()),
            "dmd_fake_score_trig_time": float(trig_time.detach().mean().cpu().item()),
        }

    def average_fake_score_gradients(self, world_size: int) -> None:
        if world_size <= 1 or self.fake_score_is_wrapped:
            return
        for parameter in self.fake_score.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
                parameter.grad.div_(world_size)


def _index_bidirectional_shard(
    shard_path: str,
    expected_records: int,
    num_blocks: int,
) -> tuple[int, list[tuple[int, int, str, str, int]]]:
    path = Path(shard_path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload["records"]
    if len(records) != expected_records:
        raise ValueError(f"Shard {path} expected {expected_records} records, found {len(records)}")
    entries = []
    for offset, record in enumerate(records):
        prompt_index = record.get("prompt_index")
        block_index = int(record.get("block_index", -1))
        if prompt_index is None or int(prompt_index) < 0:
            continue
        if block_index <= 0 or block_index >= num_blocks:
            continue
        entries.append((int(prompt_index), block_index, record["prompt"], str(path), offset))
    return len(records), entries


class BidirectionalPromptAnchorDataset(Dataset):
    """Group per-chunk draft-head records into one future-sequence example.

    Each item concatenates records for chunks 1..num_blocks-1 into one future sequence.
    Chunk 0 anchors are generated online by the target model during training.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        num_blocks: int = 9,
        cache_dir: str | Path | None = "/mnt/lanxiangh/data/cache/specgen",
        use_cache: bool = True,
        index_workers: int = 8,
        cache_wait_seconds: int = 3600,
    ):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.format = self.manifest.get("format")
        if self.format == "bidirectional_wan_full_video_v1":
            self.num_blocks = int(num_blocks)
            self._cached_shard_path = None
            self._cached_records = None
            self.index: list[tuple[Path, int]] = []
            for shard in self.manifest["shards"]:
                shard_path = self.manifest_path.parent / shard["path"]
                self.index.extend((shard_path, offset) for offset in range(int(shard["num_records"])))
            if len(self.index) != int(self.manifest["num_records"]):
                raise ValueError(f"Manifest expected {self.manifest['num_records']} records, indexed {len(self.index)}")
            self.examples = []
            return
        if self.format != "sdvg_draft_head_v1":
            raise ValueError(f"Unsupported draft-head dataset format: {self.manifest.get('format')}")
        self.num_blocks = int(num_blocks)
        if self.num_blocks <= 1:
            raise ValueError("num_blocks must be greater than 1")
        self._cached_shard_path: Path | None = None
        self._cached_records: list[dict[str, Any]] | None = None
        self.cache_path = self._cache_path(cache_dir) if use_cache and cache_dir else None
        self.index_workers = max(1, int(index_workers))

        if self.cache_path is not None and self._load_index_cache():
            return
        if self.cache_path is not None and int(os.environ.get("RANK", "0")) != 0:
            self._wait_for_index_cache(cache_wait_seconds)
            return

        grouped: dict[int, dict[int, tuple[Path, int]]] = {}
        prompt_text: dict[int, str] = {}
        total_records = 0
        show_progress = int(os.environ.get("RANK", "0")) == 0
        shard_jobs = [
            (str(self.manifest_path.parent / shard["path"]), int(shard["num_records"]), self.num_blocks)
            for shard in self.manifest["shards"]
        ]
        if show_progress:
            print(
                f"Building bidirectional dataset index with workers={self.index_workers} "
                f"shards={len(shard_jobs)} cache={self.cache_path}",
                flush=True,
            )
        progress = tqdm(
            total=len(shard_jobs),
            desc="building bidirectional dataset index",
            unit="shard",
            disable=not show_progress,
        )
        with ProcessPoolExecutor(max_workers=self.index_workers) as executor:
            futures = [
                executor.submit(_index_bidirectional_shard, shard_path, expected_records, num_blocks)
                for shard_path, expected_records, num_blocks in shard_jobs
            ]
            for future in as_completed(futures):
                shard_records, entries = future.result()
                total_records += shard_records
                for prompt_index, block_index, prompt, shard_path, offset in entries:
                    grouped.setdefault(prompt_index, {})[block_index] = (Path(shard_path), offset)
                    prompt_text[prompt_index] = prompt
                if show_progress:
                    progress.update(1)
                    progress.set_postfix(records=total_records, prompts=len(grouped))
        progress.close()
        expected_blocks = set(range(1, self.num_blocks))
        self.examples: list[tuple[int, str, dict[int, tuple[Path, int]]]] = []
        for prompt_index, blocks in sorted(grouped.items()):
            if set(blocks) == expected_blocks:
                self.examples.append((prompt_index, prompt_text[prompt_index], blocks))
        if not self.examples:
            raise ValueError(
                f"No complete prompt groups found in {self.manifest_path}; expected blocks {sorted(expected_blocks)}"
            )
        if len(self.examples) * (self.num_blocks - 1) > total_records:
            raise RuntimeError("Internal grouping error: grouped examples exceed total record count")
        self._write_index_cache(total_records)

    def _cache_fingerprint(self) -> dict[str, Any]:
        manifest_stat = self.manifest_path.stat()
        return {
            "manifest_path": str(self.manifest_path.resolve()),
            "manifest_size": manifest_stat.st_size,
            "manifest_mtime_ns": manifest_stat.st_mtime_ns,
            "num_records": int(self.manifest["num_records"]),
            "num_shards": len(self.manifest["shards"]),
            "num_blocks": self.num_blocks,
        }

    def _cache_path(self, cache_dir: str | Path | None) -> Path | None:
        if cache_dir is None:
            return None
        fingerprint = self._cache_fingerprint()
        key = hashlib.sha1(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()[:20]
        return Path(cache_dir) / "bidirectional_prompt_anchor" / f"{key}.json"

    def _load_index_cache(self) -> bool:
        assert self.cache_path is not None
        if not self.cache_path.exists():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            if payload.get("format") != "bidirectional_prompt_anchor_index_v1":
                return False
            if payload.get("fingerprint") != self._cache_fingerprint():
                return False
            self.examples = [
                (
                    int(example["prompt_index"]),
                    str(example["prompt"]),
                    {
                        int(block_index): (Path(pointer["shard_path"]), int(pointer["offset"]))
                        for block_index, pointer in example["blocks"].items()
                    },
                )
                for example in payload["examples"]
            ]
            if not self.examples:
                return False
            print(f"Loaded bidirectional dataset index cache: {self.cache_path} examples={len(self.examples)}", flush=True)
            return True
        except Exception as exc:
            print(f"Ignoring invalid bidirectional dataset index cache {self.cache_path}: {exc}", flush=True)
            return False

    def _write_index_cache(self, total_records: int) -> None:
        if self.cache_path is None:
            return
        payload = {
            "format": "bidirectional_prompt_anchor_index_v1",
            "fingerprint": self._cache_fingerprint(),
            "total_records_scanned": int(total_records),
            "examples": [
                {
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    "blocks": {
                        str(block_index): {"shard_path": str(pointer[0]), "offset": int(pointer[1])}
                        for block_index, pointer in blocks.items()
                    },
                }
                for prompt_index, prompt, blocks in self.examples
            ],
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.cache_path.with_suffix(f".{os.getpid()}.tmp")
        tmp_path.write_text(json.dumps(payload), encoding="utf-8")
        tmp_path.replace(self.cache_path)
        print(f"Wrote bidirectional dataset index cache: {self.cache_path} examples={len(self.examples)}", flush=True)

    def _wait_for_index_cache(self, timeout_seconds: int) -> None:
        assert self.cache_path is not None
        deadline = time.time() + timeout_seconds
        print(f"Waiting for bidirectional dataset index cache from rank 0: {self.cache_path}", flush=True)
        while time.time() < deadline:
            if self._load_index_cache():
                return
            time.sleep(5)
        raise TimeoutError(f"Timed out waiting for bidirectional dataset index cache: {self.cache_path}")

    def __len__(self) -> int:
        if self.format == "bidirectional_wan_full_video_v1":
            return len(self.index)
        return len(self.examples)

    def _load_record(self, pointer: tuple[Path, int]) -> dict[str, Any]:
        shard_path, offset = pointer
        if self._cached_shard_path != shard_path or self._cached_records is None:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._cached_records = payload["records"]
            self._cached_shard_path = shard_path
        return self._cached_records[offset]

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.format == "bidirectional_wan_full_video_v1":
            record = self._load_record(self.index[index])
            noise = record["noise"].contiguous()
            target_latents = record["target_latents"].contiguous()
            if noise.shape[1] != self.num_blocks * 3:
                raise ValueError(f"Expected {self.num_blocks * 3} latent frames, got {noise.shape[1]}")
            return {
                "dataset_index": int(index),
                "prompt": record["prompt"],
                "prompt_index": int(record.get("prompt_index", -1)),
                "anchor_latents": target_latents[:, :3].contiguous(),
                "future_noise": noise[:, 3:].contiguous(),
                "future_target_latents": target_latents[:, 3:].contiguous(),
                "full_noise": noise.contiguous(),
                "full_target_latents": target_latents.contiguous(),
            }
        prompt_index, prompt, blocks = self.examples[index]
        ordered = [self._load_record(blocks[block_index]) for block_index in range(1, self.num_blocks)]
        future_noise = torch.cat([record["block_noise"] for record in ordered], dim=1).contiguous()
        future_target_latents = torch.cat([record["target_latents"] for record in ordered], dim=1).contiguous()
        return {
            "dataset_index": int(index),
            "prompt": prompt,
            "prompt_index": prompt_index,
            "future_noise": future_noise,
            "future_target_latents": future_target_latents,
        }


def collate_bidirectional_examples(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")
    batch = {
        "prompts": [record["prompt"] for record in records],
        "dataset_index": torch.tensor([int(record["dataset_index"]) for record in records], dtype=torch.long),
        "prompt_index": torch.tensor([int(record["prompt_index"]) for record in records], dtype=torch.long),
        "future_noise": torch.cat([record["future_noise"] for record in records], dim=0),
        "future_target_latents": torch.cat([record["future_target_latents"] for record in records], dim=0),
    }
    if "anchor_latents" in records[0]:
        batch["anchor_latents"] = torch.cat([record["anchor_latents"] for record in records], dim=0)
    if "full_noise" in records[0]:
        batch["full_noise"] = torch.cat([record["full_noise"] for record in records], dim=0)
    if "full_target_latents" in records[0]:
        batch["full_target_latents"] = torch.cat([record["full_target_latents"] for record in records], dim=0)
    return batch


class OnlineTargetAnchorGenerator:
    """Regenerate target chunk 0 anchors from prompt and deterministic initial noise."""

    def __init__(
        self,
        *,
        model_name: str,
        checkpoint_path: str,
        model_root: str,
        config_path: str,
        num_blocks: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        text_encoder: nn.Module,
    ):
        from pipeline import CausalInferencePipeline
        from sdvg_inference import (
            denoise_block,
            ensure_wan_symlinks,
            load_checkpoint_into_generator,
            load_config,
            reset_pipeline_cache,
        )
        from utils.wan_wrapper import WanDiffusionWrapper

        ensure_wan_symlinks(model_root)
        config = load_config(config_path)
        config.denoising_step_list = [1000, 750, 500, 250, 0]
        config.warp_denoising_step = False
        config.num_frame_per_block = 3
        generator = WanDiffusionWrapper(
            model_name=model_name,
            **getattr(config, "model_kwargs", {}),
            is_causal=True,
        )
        load_checkpoint_into_generator(generator, checkpoint_path, use_ema=False)
        generator = generator.to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.pipeline = CausalInferencePipeline(
            config,
            device=device,
            generator=generator,
            text_encoder=text_encoder,
            vae=torch.nn.Identity(),
        ).to(dtype=dtype)
        self.pipeline.generator.to(device=device)
        self.num_blocks = int(num_blocks)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype
        self._denoise_block = denoise_block
        self._reset_pipeline_cache = reset_pipeline_cache

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        future_noise = batch["future_noise"]
        batch_size, future_frames, channels, height, width = future_noise.shape
        if batch_size != 1:
            raise ValueError("online_target anchor generation currently requires per-rank batch_size=1")
        if future_frames % (self.num_blocks - 1) != 0:
            raise ValueError("future_noise frame count must be divisible by num_blocks - 1")
        frames = future_frames // (self.num_blocks - 1)
        total_frames = self.num_blocks * frames
        prompt_index = int(batch["prompt_index"][0].item())
        generator = torch.Generator(device=self.device).manual_seed(self.seed + prompt_index)
        chunk0_noise = torch.randn(
            [1, frames, channels, height, width],
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self._reset_pipeline_cache(
            self.pipeline,
            batch_size=batch_size,
            dtype=self.dtype,
            device=self.device,
            total_frames=total_frames,
        )
        conditional_dict = self.pipeline.text_encoder(batch["prompts"])
        fork_devices = [self.device.index] if self.device.type == "cuda" and self.device.index is not None else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(self.seed + prompt_index)
            if self.device.type == "cuda":
                torch.cuda.manual_seed(self.seed + prompt_index)
            anchor = self._denoise_block(self.pipeline, chunk0_noise, conditional_dict, 0).detach()
        return anchor, conditional_dict["prompt_embeds"].detach()


class OnlineTeacherTrajectoryCache:
    """Generate and cache compact non-causal Wan teacher trajectory targets."""

    def __init__(
        self,
        *,
        cache_dir: str | Path,
        manifest_path: str | Path,
        model_name: str,
        model_root: str,
        config_path: str,
        num_blocks: int,
        sampling_steps: int,
        sample_solver: str,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
        text_encoder: nn.Module,
        trajectory_scope: str = "future",
        trajectory_shift: float | None = None,
        trajectory_dataset_key: str | None = None,
    ):
        from pipeline.bidirectional_diffusion_inference import BidirectionalDiffusionInferencePipeline
        from sdvg_inference import ensure_wan_symlinks, load_config

        ensure_wan_symlinks(model_root)
        if sampling_steps < 2:
            raise ValueError("--teacher_trajectory_steps must be >= 2")
        if trajectory_scope not in ("future", "full"):
            raise ValueError("--anchor_conditioning none requires full teacher trajectories")
        self.trajectory_scope = trajectory_scope
        self.cache_dir = Path(cache_dir)
        self.trajectory_shift = trajectory_shift
        self.trajectory_dataset_key = trajectory_dataset_key or self._manifest_dataset_key(manifest_path)
        manifest_key = hashlib.sha1(str(Path(manifest_path).resolve()).encode("utf-8")).hexdigest()[:12]
        cache_version = "bidirectional_teacher_trajectory_full_v1" if trajectory_scope == "full" else "bidirectional_teacher_trajectory_v1"
        cache_leaf = self._cache_leaf(sampling_steps, sample_solver, seed, trajectory_shift)
        self.cache_roots = self._candidate_cache_roots(cache_version, manifest_key, cache_leaf)
        self.cache_root = self.cache_roots[0]
        self.cache_root.parent.mkdir(parents=True, exist_ok=True)
        if not self.cache_root.exists():
            self.cache_root.mkdir()
        config = load_config(config_path)
        model_kwargs = dict(getattr(config, "model_kwargs", {}))
        model_kwargs["model_name"] = model_name
        config.model_kwargs = model_kwargs
        config.num_train_timestep = getattr(config, "num_train_timestep", 1000)
        config.negative_prompt = getattr(
            config,
            "negative_prompt",
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        )
        config.guidance_scale = getattr(config, "guidance_scale", 5.0)
        self.pipeline = BidirectionalDiffusionInferencePipeline(
            config,
            device=device,
            text_encoder=text_encoder,
            vae=torch.nn.Identity(),
        ).to(device=device, dtype=dtype).eval().requires_grad_(False)
        self.pipeline.sampling_steps = int(sampling_steps)
        self.pipeline.sample_solver = sample_solver
        if trajectory_shift is not None:
            self.pipeline.shift = float(trajectory_shift)
        self.num_blocks = int(num_blocks)
        self.seed = int(seed)
        self.device = device
        self.dtype = dtype

    @staticmethod
    def _format_shift_for_path(shift: float) -> str:
        shift_text = f"{shift:.1f}" if float(shift).is_integer() else f"{shift:g}"
        return shift_text.replace(".", "p").replace("-", "m")

    @classmethod
    def _cache_leaf(
        cls,
        sampling_steps: int,
        sample_solver: str,
        seed: int,
        trajectory_shift: float | None,
    ) -> str:
        # Historical shift-8 caches were written without shift in the path.
        if trajectory_shift is None or math.isclose(float(trajectory_shift), 8.0):
            return f"steps{sampling_steps}_{sample_solver}_seed{seed}"
        shift_tag = cls._format_shift_for_path(float(trajectory_shift))
        return f"steps{sampling_steps}_{sample_solver}_shift{shift_tag}_seed{seed}"

    @staticmethod
    def _slug_token(value: Any) -> str:
        text = str(value).strip().lower()
        replacements = {
            "wan2.1-t2v-14b": "wan21_t2v_14b",
            "wan2.1-t2v-1.3b": "wan21_t2v_13b",
            ".": "p",
            "-": "_",
            "/": "_",
            " ": "_",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return "_".join(part for part in text.split("_") if part)

    @classmethod
    def _manifest_dataset_key(cls, manifest_path: str | Path) -> str | None:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except Exception:
            return None
        metadata = manifest.get("metadata") or {}
        model = metadata.get("target_model_name")
        sampling_steps = metadata.get("sampling_steps")
        sample_solver = metadata.get("sample_solver")
        guidance_scale = metadata.get("guidance_scale")
        if model is None or sampling_steps is None or sample_solver is None:
            return None
        model_slug = cls._slug_token(model)
        solver_slug = cls._slug_token(sample_solver)
        key = f"{model_slug}_{int(sampling_steps)}step_{solver_slug}"
        if guidance_scale is not None:
            key += f"_gs{cls._format_shift_for_path(float(guidance_scale))}"
        return f"{key}_clean_latents"

    def _scope_key(self) -> str:
        return "full_video_noanchor" if self.trajectory_scope == "full" else "chunk0_conditioned_future"

    def _candidate_cache_roots(self, cache_version: str, manifest_key: str, cache_leaf: str) -> list[Path]:
        roots = []
        if self.trajectory_dataset_key:
            roots.append(self.cache_dir / self.trajectory_dataset_key / self._scope_key() / cache_leaf)
        roots.append(self.cache_dir / cache_version / manifest_key / cache_leaf)
        if self.trajectory_scope == "full":
            # Some older full-video caches were produced before scope-specific
            # directories existed and live under the future-only version name.
            roots.append(self.cache_dir / "bidirectional_teacher_trajectory_v1" / manifest_key / cache_leaf)
        else:
            roots.append(
                self.cache_dir
                / "bidirectional_teacher_trajectory_conditioned_on_50steps_latent_v1"
                / manifest_key
                / cache_leaf
            )
        deduped: list[Path] = []
        for root in roots:
            if root not in deduped:
                deduped.append(root)
        return deduped

    def _path(self, dataset_index: int, prompt_index: int) -> Path:
        prompt_part = f"prompt{prompt_index:06d}" if prompt_index >= 0 else "prompt_unknown"
        return self.cache_root / f"idx{dataset_index:06d}_{prompt_part}.pt"

    def _candidate_paths(self, dataset_index: int, prompt_index: int) -> list[Path]:
        prompt_part = f"prompt{prompt_index:06d}" if prompt_index >= 0 else "prompt_unknown"
        return [root / f"idx{dataset_index:06d}_{prompt_part}.pt" for root in self.cache_roots]

    def _payload_matches_scope(self, payload: dict[str, Any]) -> bool:
        if self.trajectory_scope == "full":
            return all(key in payload for key in ("latents", "flows", "target_latents_full"))
        return all(key in payload for key in ("future_latents", "future_flows", "target_latents"))

    def unload_pipeline(self) -> None:
        self.pipeline = None
        torch.cuda.empty_cache()

    def precompute(self, dataset: Dataset, indices: list[int]) -> None:
        progress = tqdm(indices, desc="cache teacher trajectories", unit="sample")
        for index in progress:
            sample = dataset[index]
            batch = collate_bidirectional_examples([sample])
            self(batch)

    @torch.no_grad()
    def __call__(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if int(batch["future_noise"].shape[0]) != 1:
            raise ValueError("teacher trajectory cache currently requires per-rank batch_size=1")
        dataset_index = int(batch["dataset_index"][0].item())
        prompt_index = int(batch["prompt_index"][0].item())
        path = self._path(dataset_index, prompt_index)
        for candidate_path in self._candidate_paths(dataset_index, prompt_index):
            if candidate_path.exists():
                payload = torch.load(candidate_path, map_location="cpu", weights_only=False)
                if self._payload_matches_scope(payload):
                    return payload
        if self.pipeline is None:
            raise FileNotFoundError(f"Teacher trajectory cache missing after precompute: {path}")

        future_noise = batch["future_noise"]
        _, future_frames, channels, height, width = future_noise.shape
        if future_frames % (self.num_blocks - 1) != 0:
            raise ValueError("future_noise frame count must be divisible by num_blocks - 1")
        frames = future_frames // (self.num_blocks - 1)
        total_frames = self.num_blocks * frames
        seed_index = prompt_index if prompt_index >= 0 else dataset_index
        generator = torch.Generator(device=self.device).manual_seed(self.seed + seed_index)
        noise = torch.randn(
            [1, total_frames, channels, height, width],
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        _video, latents, trajectory = self.pipeline.inference(
            noise=noise,
            text_prompts=batch["prompts"],
            return_latents=True,
            decode_video=False,
            return_trajectory=True,
        )
        payload = {
            "format": "bidirectional_teacher_trajectory_v1",
            "dataset_index": dataset_index,
            "prompt_index": prompt_index,
            "timesteps": trajectory["timesteps"].to(dtype=torch.float32).contiguous(),
            "latents": trajectory["latents"].to(dtype=torch.bfloat16).contiguous(),
            "flows": trajectory["flows"].to(dtype=torch.bfloat16).contiguous(),
            "target_latents_full": latents.detach().cpu().to(dtype=torch.bfloat16).contiguous(),
            # Store future-only tensors to keep cache compact. Chunk-0 anchor and
            # final future target are already in the base Option-B dataset.
            "future_latents": trajectory["latents"][:, :, 3:].to(dtype=torch.bfloat16).contiguous(),
            "future_flows": trajectory["flows"][:, :, 3:].to(dtype=torch.bfloat16).contiguous(),
            "target_latents": latents.detach().cpu()[:, 3:].to(dtype=torch.bfloat16).contiguous(),
        }
        tmp_path = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(payload, tmp_path)
        tmp_path.replace(path)
        torch.cuda.empty_cache()
        return payload


class BidirectionalPromptAnchorDraftHead(nn.Module):
    """Wan-token future predictor conditioned on prompt and chunk-0 anchor.

    The head predicts all future chunks as one sequence-level call, but processes
    each 3-frame future chunk with Wan-DFlash-style blocks using the shared
    target chunk-0 anchor and prompt tokens as context.
    """

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        hidden_channels: int = 5120,
        prompt_dim: int = 4096,
        num_layers: int = 6,
        num_heads: int = 40,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        temporal_mixer_layers: int = 0,
        temporal_mixer_ffn_dim: int = 2048,
        max_frames: int = 27,
        gradient_checkpointing: bool = False,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.prompt_dim = int(prompt_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.temporal_mixer_layers = int(temporal_mixer_layers)
        self.temporal_mixer_ffn_dim = int(temporal_mixer_ffn_dim)
        self.max_frames = int(max_frames)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.eps = float(eps)

        self.patch_embedding = nn.Conv3d(latent_channels, hidden_channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.prompt_proj = nn.Linear(prompt_dim, hidden_channels, bias=False)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels * 6))
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, max_frames, hidden_channels))
        self.temporal_mixer = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_channels,
                    nhead=num_heads,
                    dim_feedforward=temporal_mixer_ffn_dim,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(self.temporal_mixer_layers)
            ]
        )
        self.temporal_inject_norm = nn.LayerNorm(hidden_channels, eps=eps)
        self.temporal_inject_scale = nn.Parameter(torch.tensor([0.1]))
        self.blocks = nn.ModuleList(
            [
                WanDFlashDraftBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=0.0,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = DraftCausalHead(hidden_channels, latent_channels, self.patch_size, eps=eps)
        head_dim = hidden_channels // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )
        if self.temporal_mixer_layers > 0:
            nn.init.normal_(self.temporal_pos_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _select_prompt_tokens(prompt_embeds: torch.Tensor) -> torch.Tensor:
        mask = prompt_embeds.float().abs().sum(dim=-1) > 0
        if mask.any():
            max_len = int(mask.sum(dim=1).max().item())
            return prompt_embeds[:, :max_len]
        return prompt_embeds[:, :1]

    def _timestep_for_frames(
        self,
        timestep: torch.Tensor | int | float | None,
        batch_size: int,
        frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if timestep is None:
            timestep = torch.zeros((batch_size, frames), device=device, dtype=dtype)
        elif isinstance(timestep, (int, float)):
            timestep = torch.full((batch_size, frames), float(timestep), device=device, dtype=dtype)
        else:
            timestep = timestep.to(device=device, dtype=dtype)
            if timestep.ndim == 1:
                timestep = timestep[:, None].expand(batch_size, frames)
        return timestep

    def _patchify_with_time(
        self,
        latents: torch.Tensor,
        timestep: torch.Tensor | int | float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, frames = latents.shape[:2]
        dtype = self.patch_embedding.weight.dtype
        x = latents.to(dtype=dtype).permute(0, 2, 1, 3, 4)
        embedded = self.patch_embedding(x)
        grid_sizes = torch.tensor(
            [embedded.shape[-3:]] * batch_size,
            dtype=torch.long,
            device=latents.device,
        )
        tokens = embedded.flatten(2).transpose(1, 2)
        frame_timesteps = self._timestep_for_frames(timestep, batch_size, frames, latents.device, dtype)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, frame_timesteps.flatten()).type_as(tokens))
        e_head = e.unflatten(0, (batch_size, frames))
        e_block = self.time_projection(e).unflatten(1, (6, self.hidden_channels)).unflatten(0, (batch_size, frames))
        frame_seq_len = tokens.shape[1] // frames
        e_chunks = e_block.chunk(6, dim=2)
        tokens = tokens.unflatten(1, (frames, frame_seq_len)) * (1 + e_chunks[1]) + e_chunks[0]
        return tokens.flatten(1, 2), grid_sizes, e_head

    def _unpatchify(self, tokens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        c = self.latent_channels
        output = []
        for sample, grid in zip(tokens, grid_sizes.tolist(), strict=True):
            sample = sample[: math.prod(grid)].view(*grid, *self.patch_size, c)
            sample = torch.einsum("fhwpqrc->cfphqwr", sample)
            sample = sample.reshape(c, *[i * j for i, j in zip(grid, self.patch_size, strict=True)])
            output.append(sample)
        return torch.stack(output).permute(0, 2, 1, 3, 4)

    def _encode_context(
        self,
        anchor_latents: torch.Tensor | None,
        prompt_embeds: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        batch_size: int,
        latent_height: int,
        latent_width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self.prompt_proj(self._select_prompt_tokens(prompt_embeds).to(device=device, dtype=dtype))
        if anchor_latents is None:
            context_grid_sizes = torch.tensor(
                [[0, latent_height // self.patch_size[1], latent_width // self.patch_size[2]]] * batch_size,
                dtype=torch.long,
                device=device,
            )
            return prompt_tokens, context_grid_sizes
        anchor_tokens, _anchor_grid, _ = self._patchify_with_time(anchor_latents, timestep=0)
        context_grid_sizes = torch.tensor(
            [
                [
                    anchor_latents.shape[1] // self.patch_size[0],
                    latent_height // self.patch_size[1],
                    latent_width // self.patch_size[2],
                ]
            ]
            * batch_size,
            dtype=torch.long,
            device=device,
        )
        return torch.cat([anchor_tokens, prompt_tokens], dim=1), context_grid_sizes

    def _mix_future_tokens(self, tokens: torch.Tensor, future_frames: int) -> torch.Tensor:
        if self.temporal_mixer_layers <= 0:
            return tokens
        batch_size, token_count, hidden = tokens.shape
        if future_frames > self.max_frames:
            raise ValueError(f"future_frames={future_frames} exceeds max_frames={self.max_frames}")
        if token_count % future_frames != 0:
            raise ValueError("future token count must be divisible by frame count")
        frame_seq_len = token_count // future_frames
        tokens_by_frame = tokens.unflatten(1, (future_frames, frame_seq_len))
        frame_tokens = tokens_by_frame.mean(dim=2) + self.temporal_pos_embedding[:, :future_frames].to(
            device=tokens.device,
            dtype=tokens.dtype,
        )
        for layer in self.temporal_mixer:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                frame_tokens = checkpoint.checkpoint(layer, frame_tokens, use_reentrant=False)
            else:
                frame_tokens = layer(frame_tokens)
        frame_tokens = self.temporal_inject_norm(frame_tokens)
        tokens_by_frame = tokens_by_frame + self.temporal_inject_scale.to(dtype=tokens.dtype) * frame_tokens.unsqueeze(2)
        return tokens_by_frame.flatten(1, 2)

    def forward(
        self,
        *,
        anchor_latents: torch.Tensor | None,
        future_noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor | int | float | None = None,
    ) -> torch.Tensor:
        if future_noise.ndim != 5:
            raise ValueError("future_noise must have shape [B, T, C, H, W]")
        if anchor_latents is not None and anchor_latents.ndim != 5:
            raise ValueError("anchor_latents must have shape [B, T, C, H, W]")
        if anchor_latents is not None and anchor_latents.shape[0] != future_noise.shape[0]:
            raise ValueError("anchor_latents and future_noise batch sizes must match")
        if anchor_latents is not None and anchor_latents.shape[2:] != future_noise.shape[2:]:
            raise ValueError("anchor_latents and future_noise latent dimensions must match")
        batch_size, future_frames, channels, height, width = future_noise.shape
        anchor_frames = int(anchor_latents.shape[1]) if anchor_latents is not None else 3
        future_frames = future_noise.shape[1]
        total_frames = anchor_frames + future_frames if anchor_latents is not None else future_frames
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if total_frames > self.max_frames:
            raise ValueError(f"total_frames={total_frames} exceeds max_frames={self.max_frames}")

        context_tokens, context_grid_sizes = self._encode_context(
            anchor_latents,
            prompt_embeds,
            device=future_noise.device,
            dtype=self.patch_embedding.weight.dtype,
            batch_size=batch_size,
            latent_height=height,
            latent_width=width,
        )
        chunk_frames = anchor_frames
        if future_frames % chunk_frames != 0:
            raise ValueError("future_noise frames must be a multiple of anchor chunk frames")
        future_tokens, future_grid_sizes, future_e_head = self._patchify_with_time(future_noise, timestep=timestep)
        future_tokens = self._mix_future_tokens(future_tokens, future_frames)
        full_grid = future_grid_sizes[0].tolist()
        freqs = self.freqs.to(device=future_noise.device)
        if anchor_latents is None:
            tokens = future_tokens
            current_start_frame = 0
            for block in self.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    tokens = checkpoint.checkpoint(
                        block,
                        tokens,
                        context_tokens,
                        future_grid_sizes,
                        freqs,
                        current_start_frame,
                        0,
                        context_grid_sizes,
                        use_reentrant=False,
                    )
                else:
                    tokens = block(tokens, context_tokens, future_grid_sizes, freqs, current_start_frame, 0, context_grid_sizes)
            head_out = self.head(tokens, future_e_head.unsqueeze(2))
            return self._unpatchify(head_out, future_grid_sizes).to(dtype=future_noise.dtype)

        chunk_grid_sizes = torch.tensor(
            [[chunk_frames // self.patch_size[0], full_grid[1], full_grid[2]]] * batch_size,
            dtype=torch.long,
            device=future_noise.device,
        )
        frame_seq_len = future_tokens.shape[1] // future_frames
        future_tokens_by_frame = future_tokens.unflatten(1, (future_frames, frame_seq_len))
        outputs = []
        for chunk_offset in range(0, future_frames, chunk_frames):
            block_id = 1 + chunk_offset // chunk_frames
            tokens = future_tokens_by_frame[:, chunk_offset:chunk_offset + chunk_frames].flatten(1, 2)
            e_head = future_e_head[:, chunk_offset:chunk_offset + chunk_frames]
            current_start_frame = block_id * chunk_frames
            for block in self.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    tokens = checkpoint.checkpoint(
                        block,
                        tokens,
                        context_tokens,
                        chunk_grid_sizes,
                        freqs,
                        current_start_frame,
                        0,
                        context_grid_sizes,
                        use_reentrant=False,
                    )
                else:
                    tokens = block(tokens, context_tokens, chunk_grid_sizes, freqs, current_start_frame, 0, context_grid_sizes)
            head_out = self.head(tokens, e_head.unsqueeze(2))
            outputs.append(self._unpatchify(head_out, chunk_grid_sizes).to(dtype=future_noise.dtype))
        return torch.cat(outputs, dim=1)


class WanFullVideoDraftHead(nn.Module):
    """Shallow Wan-compatible full-video flow head for no-anchor diagnostics."""

    def __init__(
        self,
        *,
        latent_channels: int = 16,
        hidden_channels: int = 5120,
        prompt_dim: int = 4096,
        num_layers: int = 6,
        num_heads: int = 40,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        max_frames: int = 27,
        gradient_checkpointing: bool = False,
        eps: float = 1e-6,
        text_len: int = 512,
        model_type: str = "t2v",
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
    ):
        super().__init__()
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.prompt_dim = int(prompt_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.ffn_dim = int(ffn_dim)
        self.freq_dim = int(freq_dim)
        self.max_frames = int(max_frames)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.eps = float(eps)
        self.text_len = int(text_len)
        self.model_type = model_type
        self.qk_norm = bool(qk_norm)
        self.cross_attn_norm = bool(cross_attn_norm)

        self.patch_embedding = nn.Conv3d(latent_channels, hidden_channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(prompt_dim, hidden_channels),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels * 6))
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    "t2v_cross_attn",
                    hidden_channels,
                    ffn_dim,
                    num_heads,
                    (-1, -1),
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = WanHead(hidden_channels, latent_channels, self.patch_size, eps)
        head_dim = hidden_channels // num_heads
        self.freqs = torch.cat(
            [
                wan_rope_params(1024, head_dim - 4 * (head_dim // 6)),
                wan_rope_params(1024, 2 * (head_dim // 6)),
                wan_rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )

    @staticmethod
    def _select_prompt_tokens(prompt_embeds: torch.Tensor) -> torch.Tensor:
        mask = prompt_embeds.float().abs().sum(dim=-1) > 0
        if mask.any():
            max_len = int(mask.sum(dim=1).max().item())
            return prompt_embeds[:, :max_len]
        return prompt_embeds[:, :1]

    def _time_for_batch(self, timestep: torch.Tensor | int | float | None, batch_size: int, device: torch.device) -> torch.Tensor:
        if timestep is None:
            return torch.zeros(batch_size, device=device)
        if isinstance(timestep, (int, float)):
            return torch.full((batch_size,), float(timestep), device=device)
        timestep = timestep.to(device=device)
        if timestep.ndim == 2:
            return timestep[:, 0].float()
        return timestep.float().reshape(batch_size)

    def _encode_context(self, prompt_embeds: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        prompt_tokens = self._select_prompt_tokens(prompt_embeds).to(device=tokens.device, dtype=tokens.dtype)
        prompt_tokens = self.text_embedding(prompt_tokens)
        if prompt_tokens.shape[1] > self.text_len:
            return prompt_tokens[:, : self.text_len]
        if prompt_tokens.shape[1] == self.text_len:
            return prompt_tokens
        padding = prompt_tokens.new_zeros(prompt_tokens.shape[0], self.text_len - prompt_tokens.shape[1], prompt_tokens.shape[2])
        return torch.cat([prompt_tokens, padding], dim=1)

    def _unpatchify(self, tokens: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        c = self.latent_channels
        output = []
        for sample, grid in zip(tokens, grid_sizes.tolist(), strict=True):
            sample = sample[: math.prod(grid)].view(*grid, *self.patch_size, c)
            sample = torch.einsum("fhwpqrc->cfphqwr", sample)
            sample = sample.reshape(c, *[i * j for i, j in zip(grid, self.patch_size, strict=True)])
            output.append(sample)
        return torch.stack(output).permute(0, 2, 1, 3, 4)

    def forward(
        self,
        *,
        anchor_latents: torch.Tensor | None,
        future_noise: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor | int | float | None = None,
    ) -> torch.Tensor:
        if anchor_latents is not None:
            raise ValueError("WanFullVideoDraftHead is no-anchor only")
        if future_noise.ndim != 5:
            raise ValueError("future_noise must have shape [B, T, C, H, W]")
        batch_size, frames, channels, height, width = future_noise.shape
        if channels != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {channels}")
        if frames > self.max_frames:
            raise ValueError(f"frames={frames} exceeds max_frames={self.max_frames}")

        dtype = self.patch_embedding.weight.dtype
        x = self.patch_embedding(future_noise.to(dtype=dtype).permute(0, 2, 1, 3, 4))
        grid_sizes = torch.tensor([x.shape[-3:]] * batch_size, dtype=torch.long, device=future_noise.device)
        tokens = x.flatten(2).transpose(1, 2)
        seq_lens = torch.full((batch_size,), tokens.shape[1], dtype=torch.long, device=future_noise.device)
        t = self._time_for_batch(timestep, batch_size, future_noise.device)
        e = self.time_embedding(wan_sinusoidal_embedding_1d(self.freq_dim, t).type_as(tokens))
        e0 = self.time_projection(e).unflatten(1, (6, self.hidden_channels))
        context = self._encode_context(prompt_embeds, tokens)
        freqs = self.freqs.to(device=future_noise.device)

        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint.checkpoint(
                    block,
                    tokens,
                    e0,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    None,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, e0, seq_lens, grid_sizes, freqs, context, None)
        head_out = self.head(tokens, e)
        return self._unpatchify(head_out, grid_sizes).to(dtype=future_noise.dtype)


def initialize_bidirectional_wan_head_from_target_blocks(
    model: nn.Module,
    target_model: nn.Module,
    source_block_indices: tuple[int, ...],
) -> dict[str, int]:
    copied: dict[str, int] = {
        "patch_embedding": 0,
        "time_embedding": 0,
        "time_projection": 0,
        "head": 0,
        "attention": 0,
        "norm": 0,
        "ffn": 0,
        "skipped": 0,
    }

    def _copy_module_if_compatible(dst: nn.Module, src: nn.Module) -> bool:
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        if src_state.keys() != dst_state.keys():
            return False
        if any(src_state[name].shape != dst_state[name].shape for name in src_state):
            return False
        dst.load_state_dict(src_state, strict=True)
        return True

    with torch.no_grad():
        if hasattr(target_model, "patch_embedding") and target_model.patch_embedding.weight.shape == model.patch_embedding.weight.shape:
            model.patch_embedding.weight.copy_(target_model.patch_embedding.weight)
            if target_model.patch_embedding.bias is not None and model.patch_embedding.bias is not None:
                model.patch_embedding.bias.copy_(target_model.patch_embedding.bias)
            copied["patch_embedding"] = 1
        if isinstance(model, WanFullVideoDraftHead) and hasattr(target_model, "text_embedding"):
            model.text_embedding.load_state_dict(target_model.text_embedding.state_dict(), strict=True)
        if hasattr(target_model, "time_embedding"):
            try:
                model.time_embedding.load_state_dict(target_model.time_embedding.state_dict(), strict=True)
                copied["time_embedding"] = 1
            except RuntimeError:
                copied["skipped"] += 1
        if hasattr(target_model, "time_projection"):
            source_state = target_model.time_projection.state_dict()
            target_state = model.time_projection.state_dict()
            if all(name in source_state and source_state[name].shape == value.shape for name, value in target_state.items()):
                model.time_projection.load_state_dict(source_state, strict=True)
                copied["time_projection"] = 1
        if hasattr(target_model, "head") and hasattr(target_model.head, "head"):
            if target_model.head.head.weight.shape == model.head.head.weight.shape:
                model.head.load_state_dict(target_model.head.state_dict(), strict=True)
                copied["head"] = 1
        if hasattr(target_model, "freqs") and target_model.freqs.shape == model.freqs.shape:
            model.freqs = target_model.freqs.detach().cpu().clone()

    modules = dict(target_model.named_modules())
    if isinstance(model, WanFullVideoDraftHead):
        for draft_block, source_index in zip(model.blocks, source_block_indices, strict=False):
            source = modules.get(f"blocks.{source_index}")
            if source is None:
                copied["skipped"] += 1
                continue
            try:
                draft_block.load_state_dict(source.state_dict(), strict=True)
                copied["attention"] += 1
                copied["norm"] += 3
                copied["ffn"] += 1
            except RuntimeError:
                copied["skipped"] += 1
        return copied

    for draft_block, source_index in zip(model.blocks, source_block_indices, strict=False):
        source = modules.get(f"blocks.{source_index}")
        if source is None or not hasattr(source, "self_attn"):
            copied["skipped"] += 1
            continue
        with torch.no_grad():
            if hasattr(source, "norm1") and _copy_module_if_compatible(draft_block.norm1, source.norm1):
                copied["norm"] += 1
            if hasattr(source, "norm2") and _copy_module_if_compatible(draft_block.norm2, source.norm2):
                copied["norm"] += 1
            source_attn = source.self_attn
            if all(
                hasattr(source_attn, name)
                and getattr(source_attn, name).weight.shape == getattr(draft_block.self_attn, name).weight.shape
                for name in ("q", "k", "v", "o")
            ):
                for name in ("q", "k", "v", "o", "norm_q", "norm_k"):
                    getattr(draft_block.self_attn, name).load_state_dict(
                        getattr(source_attn, name).state_dict(),
                        strict=True,
                    )
                copied["attention"] += 1
            if hasattr(source, "ffn"):
                source_linears = [module for module in source.ffn if isinstance(module, nn.Linear)]
                draft_linears = [module for module in draft_block.ffn if isinstance(module, nn.Linear)]
                if len(source_linears) == len(draft_linears) == 2 and all(
                    src.weight.shape == dst.weight.shape
                    for src, dst in zip(source_linears, draft_linears, strict=True)
                ):
                    for source_linear, draft_linear in zip(source_linears, draft_linears, strict=True):
                        draft_linear.weight.copy_(source_linear.weight)
                        draft_linear.bias.copy_(source_linear.bias)
                    copied["ffn"] += 1
    return copied


def sequence_losses(
    student_model: nn.Module,
    model_output: torch.Tensor,
    target: torch.Tensor,
    anchor: torch.Tensor,
    *,
    noisy_latents: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
    scheduler: FlowMatchScheduler,
    prediction_type: str,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    detail_loss_weight: float,
    temporal_delta_weight: float,
    boundary_weight: float,
    dmd: RCMStyleDraftHeadDMD | None = None,
    dmd_loss_weight: float = 0.0,
    prompt_embeds: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if prediction_type == "flow":
        flow_prediction = model_output
        clean_prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, noisy_latents, timestep)
    elif prediction_type == "clean_latent":
        clean_prediction = model_output
        flow_prediction = clean_latent_to_flow_prediction(scheduler, clean_prediction, noisy_latents, timestep)
    else:
        raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")

    clean_mse = F.mse_loss(clean_prediction.float(), target.float())
    loss = clean_mse * clean_latent_loss_weight
    metrics = {"clean_latent_mse": float(clean_mse.detach().cpu().item())}
    if flow_loss_weight > 0:
        flow_target = noise - target
        flow_mse = F.mse_loss(flow_prediction.float(), flow_target.float())
        loss = loss + flow_mse * flow_loss_weight
        metrics["flow_mse"] = float(flow_mse.detach().cpu().item())
    if detail_loss_weight > 0:
        detail_loss = spatial_detail_loss(clean_prediction, target)
        loss = loss + detail_loss * detail_loss_weight
        metrics["detail_loss"] = float(detail_loss.detach().cpu().item())
    if temporal_delta_weight > 0:
        pred_delta = clean_prediction[:, 1:].float() - clean_prediction[:, :-1].float()
        target_delta = target[:, 1:].float() - target[:, :-1].float()
        temporal_delta_loss = F.mse_loss(pred_delta, target_delta)
        loss = loss + temporal_delta_loss * temporal_delta_weight
        metrics["temporal_delta_loss"] = float(temporal_delta_loss.detach().cpu().item())
    if boundary_weight > 0:
        pred_boundary = clean_prediction[:, 0].float() - anchor[:, -1].float()
        target_boundary = target[:, 0].float() - anchor[:, -1].float()
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        loss = loss + boundary_loss * boundary_weight
        metrics["boundary_loss"] = float(boundary_loss.detach().cpu().item())
    if dmd is not None and dmd_loss_weight > 0:
        if prompt_embeds is None:
            raise ValueError("prompt_embeds is required when DMD is enabled")
        dmd_component, dmd_metrics = dmd.student_loss(clean_prediction, anchor=anchor, prompt_embeds=prompt_embeds)
        loss = loss + dmd_component * dmd_loss_weight
        metrics["dmd_loss"] = float((dmd_component * dmd_loss_weight).detach().cpu().item())
        metrics.update(dmd_metrics)
        consistency_component, consistency_metrics = dmd.consistency_loss(
            student_model,
            clean_prediction,
            anchor=anchor,
            prompt_embeds=prompt_embeds,
        )
        if consistency_metrics:
            loss = loss + consistency_component * dmd.consistency_loss_weight
            metrics["dmd_consistency_loss"] = float(
                (consistency_component * dmd.consistency_loss_weight).detach().cpu().item()
            )
            metrics.update(consistency_metrics)
    metrics["loss"] = float(loss.detach().cpu().item())
    return loss, metrics


def compute_bidirectional_losses(
    model: nn.Module,
    *,
    anchor: torch.Tensor,
    initial_noise: torch.Tensor,
    target: torch.Tensor,
    prompt_embeds: torch.Tensor,
    scheduler: FlowMatchScheduler,
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
    dmd: RCMStyleDraftHeadDMD | None = None,
    dmd_loss_weight: float = 0.0,
    rollout_solver: str = "euler",
    rollout_steps: int = 0,
    rollout_timestep_list: list[int] | None = None,
    rollout_solver_shift: float = 8.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if training_mode not in ("one_step", "unrolled", "random_timestep"):
        raise ValueError("--training_mode must be 'one_step', 'unrolled', or 'random_timestep'")
    if prediction_type not in ("flow", "clean_latent"):
        raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")
    if not denoising_step_list:
        raise ValueError("denoising_step_list must not be empty")
    if unroll_noise_mode not in ("fixed", "fresh"):
        raise ValueError("--unroll_noise_mode must be 'fixed' or 'fresh'")
    batch_size, frames = target.shape[:2]

    if training_mode == "one_step":
        timestep = timestep_batch(denoising_step_list[0], batch_size, frames, device=target.device)
        model_output = model(anchor_latents=anchor, future_noise=initial_noise, prompt_embeds=prompt_embeds, timestep=timestep)
        return sequence_losses(
            model,
            model_output,
            target,
            anchor,
            noisy_latents=initial_noise,
            noise=initial_noise,
            timestep=timestep,
            scheduler=scheduler,
            prediction_type=prediction_type,
            clean_latent_loss_weight=clean_latent_loss_weight,
            flow_loss_weight=flow_loss_weight,
            detail_loss_weight=detail_loss_weight,
            temporal_delta_weight=temporal_delta_weight,
            boundary_weight=boundary_weight,
            dmd=dmd,
            dmd_loss_weight=dmd_loss_weight,
            prompt_embeds=prompt_embeds,
        )

    if training_mode == "random_timestep":
        candidate_timesteps = [int(timestep) for timestep in denoising_step_list if int(timestep) > 0]
        if not candidate_timesteps:
            raise ValueError("random_timestep requires at least one positive timestep")
        sampled_timesteps = sample_random_timesteps(
            batch_size=batch_size,
            candidate_timesteps=candidate_timesteps,
            sampling=random_timestep_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            device=target.device,
        )
        timestep = sampled_timesteps[:, None].expand(batch_size, frames)
        noisy_latents = scheduler.add_noise(
            target.flatten(0, 1),
            initial_noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, target.shape[:2])
        model_output = model(anchor_latents=anchor, future_noise=noisy_latents, prompt_embeds=prompt_embeds, timestep=timestep)
        loss, metrics = sequence_losses(
            model,
            model_output,
            target,
            anchor,
            noisy_latents=noisy_latents,
            noise=initial_noise,
            timestep=timestep,
            scheduler=scheduler,
            prediction_type=prediction_type,
            clean_latent_loss_weight=clean_latent_loss_weight,
            flow_loss_weight=flow_loss_weight,
            detail_loss_weight=detail_loss_weight,
            temporal_delta_weight=temporal_delta_weight,
            boundary_weight=boundary_weight,
            dmd=dmd,
            dmd_loss_weight=dmd_loss_weight,
            prompt_embeds=prompt_embeds,
        )
        metrics["sampled_timestep_mean"] = float(sampled_timesteps.float().mean().detach().cpu().item())
        metrics["sampled_timestep_min"] = float(sampled_timesteps.float().min().detach().cpu().item())
        metrics["sampled_timestep_max"] = float(sampled_timesteps.float().max().detach().cpu().item())
        return loss, metrics

    if len(denoising_step_list) != len(unroll_step_weights):
        raise ValueError("denoising_step_list and unroll_step_weights must have the same length")
    if rollout_solver in ("unipc", "rcm"):
        if prediction_type != "flow":
            raise ValueError(f"--rollout_solver {rollout_solver} requires --prediction_type flow")
        if rollout_solver == "unipc":
            actual_rollout_steps = resolve_rollout_steps(
                rollout_steps=rollout_steps,
                denoising_step_list=denoising_step_list,
            )
            prediction, used_timesteps = unipc_bidirectional_clean_rollout(
                model,
                anchor=anchor,
                initial_latents=initial_noise,
                prompt_embeds=prompt_embeds,
                num_steps=actual_rollout_steps,
                shift=rollout_solver_shift,
            )
        else:
            used_timesteps = list(rollout_timestep_list or denoising_step_list)
            prediction = timestep_list_bidirectional_clean_rollout(
                model,
                anchor=anchor,
                initial_latents=initial_noise,
                prompt_embeds=prompt_embeds,
                scheduler=scheduler,
                timestep_list=used_timesteps,
            )
        components: list[torch.Tensor] = []
        final_clean_loss = F.mse_loss(prediction.float(), target.float())
        if clean_latent_loss_weight > 0:
            components.append(final_clean_loss * clean_latent_loss_weight)
        if detail_loss_weight > 0:
            detail_loss = spatial_detail_loss(prediction, target)
            components.append(detail_loss * detail_loss_weight)
        else:
            detail_loss = None
        if temporal_delta_weight > 0:
            pred_delta = prediction[:, 1:].float() - prediction[:, :-1].float()
            target_delta = target[:, 1:].float() - target[:, :-1].float()
            temporal_delta_loss = F.mse_loss(pred_delta, target_delta)
            components.append(temporal_delta_loss * temporal_delta_weight)
        else:
            temporal_delta_loss = None
        if boundary_weight > 0:
            pred_boundary = prediction[:, 0].float() - anchor[:, -1].float()
            target_boundary = target[:, 0].float() - anchor[:, -1].float()
            boundary_loss = F.mse_loss(pred_boundary, target_boundary)
            components.append(boundary_loss * boundary_weight)
        else:
            boundary_loss = None
        dmd_metrics: dict[str, float] = {}
        if dmd is not None and dmd_loss_weight > 0:
            dmd_component, dmd_metrics = dmd.student_loss(prediction, anchor=anchor, prompt_embeds=prompt_embeds)
            components.append(dmd_component * dmd_loss_weight)
            consistency_component, consistency_metrics = dmd.consistency_loss(
                model,
                prediction,
                anchor=anchor,
                prompt_embeds=prompt_embeds,
            )
            if consistency_metrics:
                components.append(consistency_component * dmd.consistency_loss_weight)
                dmd_metrics["dmd_consistency_loss"] = float(
                    (consistency_component * dmd.consistency_loss_weight).detach().cpu().item()
                )
                dmd_metrics.update(consistency_metrics)
        if flow_loss_weight > 0:
            raise ValueError(f"--rollout_solver {rollout_solver} does not support flow supervision; use DMD/clean/detail losses or rollout_solver=euler")
        if not components:
            raise ValueError("At least one loss component must be enabled")
        total_loss = sum(components)
        metrics = {
            "loss": float(total_loss.detach().cpu().item()),
            "clean_latent_mse": float(final_clean_loss.detach().cpu().item()),
            f"rollout_{rollout_solver}_steps": float(len(used_timesteps)),
        }
        if detail_loss is not None:
            metrics["unrolled_detail_loss"] = float(detail_loss.detach().cpu().item())
        if temporal_delta_loss is not None:
            metrics["temporal_delta_loss"] = float(temporal_delta_loss.detach().cpu().item())
        if boundary_loss is not None:
            metrics["boundary_loss"] = float(boundary_loss.detach().cpu().item())
        if dmd_metrics:
            metrics["dmd_loss"] = float((dmd_component * dmd_loss_weight).detach().cpu().item())
            metrics.update(dmd_metrics)
        return total_loss, metrics
    if rollout_solver != "euler":
        raise ValueError("--rollout_solver must be 'euler', 'unipc', or 'rcm'")
    clean_weight_denominator = max(sum(unroll_step_weights), 1e-8)
    flow_weight_denominator = max(
        sum(
            weight
            for weight, timestep in zip(unroll_step_weights, denoising_step_list, strict=True)
            if timestep > 0
        ),
        1e-8,
    )
    current = initial_noise
    current_noise = initial_noise
    components: list[torch.Tensor] = []
    clean_losses = []
    flow_losses = []
    detail_losses = []
    prediction = current
    for index, current_timestep in enumerate(denoising_step_list):
        timestep = timestep_batch(current_timestep, batch_size, frames, device=target.device)
        model_output = model(anchor_latents=anchor, future_noise=current, prompt_embeds=prompt_embeds, timestep=timestep)
        if prediction_type == "flow":
            prediction = flow_prediction_to_clean_latent(scheduler, model_output, current, timestep)
            flow_prediction = model_output
        else:
            prediction = model_output
            flow_prediction = clean_latent_to_flow_prediction(scheduler, prediction, current, timestep)
        step_weight = unroll_step_weights[index]
        clean_loss = F.mse_loss(prediction.float(), target.float())
        clean_losses.append(clean_loss.detach())
        if clean_latent_loss_weight > 0 and step_weight > 0:
            components.append(clean_loss * clean_latent_loss_weight * (step_weight / clean_weight_denominator))
        if flow_loss_weight > 0 and current_timestep > 0 and step_weight > 0:
            if prediction_type == "flow":
                flow_target = clean_latent_to_flow_prediction(scheduler, target, current, timestep)
            else:
                flow_target = current_noise - target
            flow_loss = F.mse_loss(flow_prediction.float(), flow_target.float())
            flow_losses.append(flow_loss.detach())
            components.append(flow_loss * flow_loss_weight * (step_weight / flow_weight_denominator))
        if detail_loss_weight > 0 and step_weight > 0:
            detail_loss = spatial_detail_loss(prediction, target)
            detail_losses.append(detail_loss.detach())
            components.append(detail_loss * detail_loss_weight * (step_weight / clean_weight_denominator))
        if index < len(denoising_step_list) - 1:
            next_timestep = denoising_step_list[index + 1]
            next_noise = torch.randn_like(target) if unroll_noise_mode == "fresh" else initial_noise
            next_timestep_tensor = timestep_batch(next_timestep, batch_size, frames, device=target.device)
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

    final_clean_loss = F.mse_loss(prediction.float(), target.float())
    if temporal_delta_weight > 0:
        pred_delta = prediction[:, 1:].float() - prediction[:, :-1].float()
        target_delta = target[:, 1:].float() - target[:, :-1].float()
        temporal_delta_loss = F.mse_loss(pred_delta, target_delta)
        components.append(temporal_delta_loss * temporal_delta_weight)
    else:
        temporal_delta_loss = None
    if boundary_weight > 0:
        pred_boundary = prediction[:, 0].float() - anchor[:, -1].float()
        target_boundary = target[:, 0].float() - anchor[:, -1].float()
        boundary_loss = F.mse_loss(pred_boundary, target_boundary)
        components.append(boundary_loss * boundary_weight)
    else:
        boundary_loss = None
    dmd_metrics: dict[str, float] = {}
    if dmd is not None and dmd_loss_weight > 0:
        dmd_component, dmd_metrics = dmd.student_loss(prediction, anchor=anchor, prompt_embeds=prompt_embeds)
        components.append(dmd_component * dmd_loss_weight)
        consistency_component, consistency_metrics = dmd.consistency_loss(
            model,
            prediction,
            anchor=anchor,
            prompt_embeds=prompt_embeds,
        )
        if consistency_metrics:
            components.append(consistency_component * dmd.consistency_loss_weight)
            dmd_metrics["dmd_consistency_loss"] = float(
                (consistency_component * dmd.consistency_loss_weight).detach().cpu().item()
            )
            dmd_metrics.update(consistency_metrics)
    if not components:
        raise ValueError("At least one loss component must be enabled")
    total_loss = sum(components)
    clean_stack = torch.stack(clean_losses) if clean_losses else final_clean_loss.detach().reshape(1)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(final_clean_loss.detach().cpu().item()),
        "unrolled_clean_latent_mse": float(clean_stack.mean().detach().cpu().item()),
    }
    if flow_losses:
        metrics["unrolled_flow_mse"] = float(torch.stack(flow_losses).mean().detach().cpu().item())
    if detail_losses:
        metrics["unrolled_detail_loss"] = float(torch.stack(detail_losses).mean().detach().cpu().item())
    if temporal_delta_loss is not None:
        metrics["temporal_delta_loss"] = float(temporal_delta_loss.detach().cpu().item())
    if boundary_loss is not None:
        metrics["boundary_loss"] = float(boundary_loss.detach().cpu().item())
    if dmd_metrics:
        metrics["dmd_loss"] = float((dmd_component * dmd_loss_weight).detach().cpu().item())
        metrics.update(dmd_metrics)
    return total_loss, metrics


def unipc_bidirectional_clean_rollout(
    model: nn.Module,
    *,
    anchor: torch.Tensor | None,
    initial_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int,
    shift: float,
) -> tuple[torch.Tensor, list[int]]:
    if num_steps < 2:
        raise ValueError("UniPC rollout requires at least 2 steps")
    sample_scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    sample_scheduler.set_timesteps(num_steps, device=initial_latents.device, shift=shift)
    current = initial_latents
    used_timesteps: list[int] = []
    for t in sample_scheduler.timesteps:
        timestep = t * torch.ones(current.shape[:2], device=current.device, dtype=torch.float32)
        flow_prediction = model(
            anchor_latents=anchor,
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


def timestep_list_bidirectional_clean_rollout(
    model: nn.Module,
    *,
    anchor: torch.Tensor | None,
    initial_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    scheduler: FlowMatchScheduler,
    timestep_list: list[int],
) -> torch.Tensor:
    if len(timestep_list) < 2:
        raise ValueError("rollout timestep list must contain at least two time points")
    batch_size, frames = initial_latents.shape[:2]
    current = initial_latents
    prediction = current
    for index, current_timestep in enumerate(timestep_list):
        timestep = timestep_batch(current_timestep, batch_size, frames, device=initial_latents.device)
        flow_prediction = model(anchor_latents=anchor, future_noise=current, prompt_embeds=prompt_embeds, timestep=timestep)
        prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, current, timestep)
        if index < len(timestep_list) - 1:
            next_timestep_tensor = timestep_batch(timestep_list[index + 1], batch_size, frames, device=initial_latents.device)
            current = flow_prediction_step(
                scheduler,
                flow_prediction,
                current,
                timestep,
                next_timestep_tensor,
            )
    return prediction


def predict_bidirectional_clean_rollout(
    model: nn.Module,
    *,
    anchor: torch.Tensor,
    initial_noise: torch.Tensor,
    target: torch.Tensor,
    prompt_embeds: torch.Tensor,
    scheduler: FlowMatchScheduler,
    training_mode: str,
    denoising_step_list: list[int],
    prediction_type: str,
    random_timestep_sampling: str,
    logit_normal_mean: float,
    logit_normal_std: float,
    unroll_noise_mode: str,
    rollout_solver: str = "euler",
    rollout_steps: int = 0,
    rollout_timestep_list: list[int] | None = None,
    rollout_solver_shift: float = 8.0,
) -> torch.Tensor:
    """Generate the clean draft-head prediction used as G(x) for fake-score updates."""
    if training_mode not in ("one_step", "unrolled", "random_timestep"):
        raise ValueError("--training_mode must be 'one_step', 'unrolled', or 'random_timestep'")
    if prediction_type not in ("flow", "clean_latent"):
        raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")
    if unroll_noise_mode not in ("fixed", "fresh"):
        raise ValueError("--unroll_noise_mode must be 'fixed' or 'fresh'")
    batch_size, frames = target.shape[:2]
    if training_mode == "one_step":
        timestep = timestep_batch(denoising_step_list[0], batch_size, frames, device=target.device)
        model_output = model(anchor_latents=anchor, future_noise=initial_noise, prompt_embeds=prompt_embeds, timestep=timestep)
        if prediction_type == "flow":
            return flow_prediction_to_clean_latent(scheduler, model_output, initial_noise, timestep)
        return model_output

    if training_mode == "random_timestep":
        candidate_timesteps = [int(timestep) for timestep in denoising_step_list if int(timestep) > 0]
        if not candidate_timesteps:
            raise ValueError("random_timestep requires at least one positive timestep")
        sampled_timesteps = sample_random_timesteps(
            batch_size=batch_size,
            candidate_timesteps=candidate_timesteps,
            sampling=random_timestep_sampling,
            logit_normal_mean=logit_normal_mean,
            logit_normal_std=logit_normal_std,
            device=target.device,
        )
        timestep = sampled_timesteps[:, None].expand(batch_size, frames)
        noisy_latents = scheduler.add_noise(
            target.flatten(0, 1),
            initial_noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, target.shape[:2])
        model_output = model(anchor_latents=anchor, future_noise=noisy_latents, prompt_embeds=prompt_embeds, timestep=timestep)
        if prediction_type == "flow":
            return flow_prediction_to_clean_latent(scheduler, model_output, noisy_latents, timestep)
        return model_output

    if rollout_solver == "unipc":
        if prediction_type != "flow":
            raise ValueError("--rollout_solver unipc requires --prediction_type flow")
        actual_rollout_steps = resolve_rollout_steps(
            rollout_steps=rollout_steps,
            denoising_step_list=denoising_step_list,
        )
        prediction, _used_timesteps = unipc_bidirectional_clean_rollout(
            model,
            anchor=anchor,
            initial_latents=initial_noise,
            prompt_embeds=prompt_embeds,
            num_steps=actual_rollout_steps,
            shift=rollout_solver_shift,
        )
        return prediction
    if rollout_solver == "rcm":
        if prediction_type != "flow":
            raise ValueError("--rollout_solver rcm requires --prediction_type flow")
        return timestep_list_bidirectional_clean_rollout(
            model,
            anchor=anchor,
            initial_latents=initial_noise,
            prompt_embeds=prompt_embeds,
            scheduler=scheduler,
            timestep_list=rollout_timestep_list or denoising_step_list,
        )
    if rollout_solver != "euler":
        raise ValueError("--rollout_solver must be 'euler', 'unipc', or 'rcm'")

    current = initial_noise
    current_noise = initial_noise
    prediction = current
    for index, current_timestep in enumerate(denoising_step_list):
        timestep = timestep_batch(current_timestep, batch_size, frames, device=target.device)
        model_output = model(anchor_latents=anchor, future_noise=current, prompt_embeds=prompt_embeds, timestep=timestep)
        if prediction_type == "flow":
            prediction = flow_prediction_to_clean_latent(scheduler, model_output, current, timestep)
            flow_prediction = model_output
        else:
            prediction = model_output
            flow_prediction = clean_latent_to_flow_prediction(scheduler, prediction, current, timestep)
        if index < len(denoising_step_list) - 1:
            next_timestep = denoising_step_list[index + 1]
            next_noise = torch.randn_like(target) if unroll_noise_mode == "fresh" else initial_noise
            next_timestep_tensor = timestep_batch(next_timestep, batch_size, frames, device=target.device)
            if prediction_type == "flow":
                current = flow_prediction_step(
                    scheduler,
                    flow_prediction,
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
            current_noise = next_noise
    _ = current_noise
    return prediction


def compute_teacher_trajectory_losses(
    model: nn.Module,
    *,
    anchor: torch.Tensor | None,
    prompt_embeds: torch.Tensor,
    trajectory: dict[str, torch.Tensor],
    scheduler: FlowMatchScheduler,
    clean_latent_loss_weight: float,
    flow_loss_weight: float,
    detail_loss_weight: float,
    anchor_conditioning: str = "clean",
) -> tuple[torch.Tensor, dict[str, float]]:
    if anchor_conditioning not in ("clean", "none"):
        raise ValueError("--anchor_conditioning must be 'clean' or 'none'")
    device = prompt_embeds.device
    dtype = prompt_embeds.dtype
    timesteps = trajectory["timesteps"].to(device=device)
    if anchor_conditioning == "none":
        future_latents = trajectory["latents"].to(device=device, dtype=dtype)
        future_flows = trajectory["flows"].to(device=device, dtype=dtype)
        target = trajectory.get("target_latents_full")
    else:
        if anchor is None:
            raise ValueError("anchor_conditioning='clean' requires anchor latents")
        future_latents = trajectory["future_latents"].to(device=device, dtype=dtype)
        future_flows = trajectory["future_flows"].to(device=device, dtype=dtype)
        target = trajectory.get("target_latents")
    target = target.to(device=device, dtype=dtype) if target is not None else None
    if future_latents.ndim != 6 or future_flows.ndim != 6:
        raise ValueError("teacher trajectory tensors must have shape [B, S, T, C, H, W]")
    if future_latents.shape != future_flows.shape:
        raise ValueError("teacher trajectory future_latents/future_flows shape mismatch")

    batch_size, num_steps, future_frames = future_latents.shape[:3]
    components: list[torch.Tensor] = []
    flow_losses = []
    clean_losses = []
    detail_losses = []
    for step_index in range(num_steps):
        timestep_value = timesteps[step_index].round().long()
        timestep = timestep_value.reshape(1, 1).expand(batch_size, future_frames)
        state = future_latents[:, step_index]
        target_flow = future_flows[:, step_index]
        flow_prediction = model(
            anchor_latents=anchor if anchor_conditioning == "clean" else None,
            future_noise=state,
            prompt_embeds=prompt_embeds,
            timestep=timestep,
        )
        flow_loss = F.mse_loss(flow_prediction.float(), target_flow.float())
        flow_losses.append(flow_loss.detach())
        if flow_loss_weight > 0:
            components.append(flow_loss * flow_loss_weight / max(num_steps, 1))
        if target is not None and (clean_latent_loss_weight > 0 or detail_loss_weight > 0):
            clean_prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, state, timestep)
            clean_loss = F.mse_loss(clean_prediction.float(), target.float())
            clean_losses.append(clean_loss.detach())
            if clean_latent_loss_weight > 0:
                components.append(clean_loss * clean_latent_loss_weight / max(num_steps, 1))
            if detail_loss_weight > 0:
                detail_loss = spatial_detail_loss(clean_prediction, target)
                detail_losses.append(detail_loss.detach())
                components.append(detail_loss * detail_loss_weight / max(num_steps, 1))
    if not components:
        raise ValueError("At least one teacher trajectory loss component must be enabled")
    total_loss = sum(components)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "teacher_trajectory_flow_mse": float(torch.stack(flow_losses).mean().detach().cpu().item()),
        "clean_latent_mse": float(torch.stack(clean_losses).mean().detach().cpu().item()) if clean_losses else 0.0,
    }
    if detail_losses:
        metrics["teacher_trajectory_detail_loss"] = float(torch.stack(detail_losses).mean().detach().cpu().item())
    return total_loss, metrics


def save_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    inner = unwrap_model(model)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = {
        "model_class": type(inner).__name__,
        "latent_channels": inner.latent_channels,
        "hidden_channels": inner.hidden_channels,
        "prompt_dim": inner.prompt_dim,
        "num_layers": inner.num_layers,
        "num_heads": inner.num_heads,
        "patch_size": inner.patch_size,
        "ffn_dim": inner.ffn_dim,
        "freq_dim": inner.freq_dim,
        "max_frames": inner.max_frames,
        "gradient_checkpointing": inner.gradient_checkpointing,
    }
    if isinstance(inner, BidirectionalPromptAnchorDraftHead):
        model_config.update(
            {
                "temporal_mixer_layers": inner.temporal_mixer_layers,
                "temporal_mixer_ffn_dim": inner.temporal_mixer_ffn_dim,
            }
        )
    elif isinstance(inner, WanFullVideoDraftHead):
        model_config.update(
            {
                "text_len": inner.text_len,
                "model_type": inner.model_type,
                "qk_norm": inner.qk_norm,
                "cross_attn_norm": inner.cross_attn_norm,
            }
        )
    payload = {
        "format": "bidirectional_prompt_anchor_draft_head_v1",
        "model_state_dict": inner.state_dict() if state_dict is None else state_dict,
        "model_config": model_config,
        "train_args": vars(args),
        "metadata": metadata,
    }
    torch.save(payload, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train bidirectional prompt+anchor latent draft head.")
    parser.add_argument("--manifest_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--dataset_cache_dir", default="/mnt/lanxiangh/data/cache/specgen")
    parser.add_argument("--disable_dataset_cache", action="store_true")
    parser.add_argument("--dataset_index_workers", type=int, default=8)
    parser.add_argument("--dataset_cache_wait_seconds", type=int, default=3600)
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--target_model_name", default="Wan2.1-T2V-14B")
    parser.add_argument("--target_checkpoint_path", default="/mnt/lanxiangh/models/realtime-video/checkpoints/krea-realtime-video-14b.safetensors")
    parser.add_argument("--init_model_name", default=None)
    parser.add_argument("--init_draft_head_checkpoint_path", default="")
    parser.add_argument("--anchor_noise_seed", type=int, default=42)
    parser.add_argument("--num_blocks", type=int, default=9)
    parser.add_argument("--hidden_channels", type=int, default=5120)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=40)
    parser.add_argument("--ffn_dim", type=int, default=13824)
    parser.add_argument("--temporal_mixer_layers", type=int, default=2)
    parser.add_argument("--temporal_mixer_ffn_dim", type=int, default=2048)
    parser.add_argument("--init_target_blocks", nargs="+", type=int, default=[0, 8, 16, 24, 32, 39])
    parser.add_argument("--prompt_dim", type=int, default=4096)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--denoising_step_list", nargs="+", type=int, default=[1000, 750, 500, 250, 0])
    parser.add_argument("--dense_schedule_steps", type=int, default=0)
    parser.add_argument("--timestep_shift", type=float, default=5.0)
    parser.add_argument("--prediction_type", choices=["flow", "clean_latent"], default="flow")
    parser.add_argument("--training_mode", choices=["one_step", "unrolled", "random_timestep", "teacher_trajectory"], default="unrolled")
    parser.add_argument("--anchor_conditioning", choices=["clean", "none"], default="clean")
    parser.add_argument("--random_timestep_sampling", choices=["uniform_schedule", "logit_normal"], default="uniform_schedule")
    parser.add_argument("--logit_normal_mean", type=float, default=0.0)
    parser.add_argument("--logit_normal_std", type=float, default=1.0)
    parser.add_argument("--unroll_step_weights", nargs="+", type=float, default=None)
    parser.add_argument("--unroll_noise_mode", choices=["fixed", "fresh"], default="fixed")
    parser.add_argument("--rollout_solver", choices=["euler", "unipc", "rcm"], default="euler")
    parser.add_argument("--rollout_steps", type=int, default=0)
    parser.add_argument("--rollout_schedule", default="")
    parser.add_argument("--rollout_sigma_max", type=float, default=1600.0)
    parser.add_argument("--rollout_solver_shift", type=float, default=8.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--overfit_num_examples", type=int, default=0)
    parser.add_argument("--overfit_start_index", type=int, default=0)
    parser.add_argument("--clean_latent_loss_weight", type=float, default=1.0)
    parser.add_argument("--flow_loss_weight", type=float, default=0.25)
    parser.add_argument("--detail_loss_weight", type=float, default=0.0)
    parser.add_argument("--temporal_delta_weight", type=float, default=0.0)
    parser.add_argument("--boundary_weight", type=float, default=0.0)
    parser.add_argument("--dmd_loss_weight", type=float, default=0.0)
    parser.add_argument("--dmd_fake_score_loss_weight", type=float, default=1.0)
    parser.add_argument("--dmd_warmup_steps", type=int, default=0)
    parser.add_argument("--dmd_student_update_freq", type=int, default=5)
    parser.add_argument("--dmd_fake_score_lr", type=float, default=1e-7)
    parser.add_argument("--dmd_fake_score_weight_decay", type=float, default=0.01)
    parser.add_argument("--dmd_fake_score_gradient_checkpointing", action="store_true")
    parser.add_argument("--dmd_model_name", default=None, help="Deprecated alias: set both DMD teacher and fake-score model names.")
    parser.add_argument("--dmd_teacher_model_name", default=None)
    parser.add_argument("--dmd_fake_score_model_name", default=None)
    parser.add_argument("--dmd_teacher_checkpoint_path", default="")
    parser.add_argument("--dmd_fake_score_checkpoint_path", default="")
    parser.add_argument("--dmd_guidance_scale", type=float, default=5.0)
    parser.add_argument("--dmd_timestep_shift", type=float, default=5.0)
    parser.add_argument("--dmd_min_timestep", type=float, default=20.0)
    parser.add_argument("--dmd_max_timestep", type=float, default=980.0)
    parser.add_argument("--dmd_time_distribution", choices=["shifted_uniform", "rcm_lognormal"], default="shifted_uniform")
    parser.add_argument("--dmd_lognormal_mean", type=float, default=0.0)
    parser.add_argument("--dmd_lognormal_std", type=float, default=1.6)
    parser.add_argument("--dmd_fake_score_weighting", choices=["scheduler_sigma", "rcm_trig_sin"], default="scheduler_sigma")
    parser.add_argument("--dmd_target_convention", choices=["wan_rf_x0", "rcm_trig_x0"], default="wan_rf_x0")
    parser.add_argument("--dmd_consistency_objective", choices=["none", "rcm_fd", "rcm_jvp"], default="none")
    parser.add_argument("--dmd_consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--dmd_consistency_fd_epsilon", type=float, default=1e-4)
    parser.add_argument("--dmd_score_scope", choices=["future", "full"], default="full")
    parser.add_argument("--teacher_trajectory_cache_dir", default="/mnt/lanxiangh/data/ff_exec/teacher_trajectory_cache")
    parser.add_argument("--teacher_trajectory_steps", type=int, default=5)
    parser.add_argument("--teacher_trajectory_solver", choices=["unipc", "dpm++"], default="unipc")
    parser.add_argument("--teacher_trajectory_shift", type=float, default=None)
    parser.add_argument("--teacher_trajectory_dataset_key", default=None)
    parser.add_argument("--amp_dtype", choices=["none", "bf16", "fp16"], default="bf16")
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--parallel_strategy", choices=["ddp", "fsdp"], default="ddp")
    parser.add_argument("--fsdp_min_num_params", type=int, default=100_000_000)
    parser.add_argument("--fsdp_mixed_precision", choices=["none", "bf16", "fp16"], default="none")
    parser.add_argument("--attention_backend", choices=["auto", "no_cudnn", "math"], default="auto")
    args = parser.parse_args()
    configure_attention_backend(args.attention_backend)
    if args.dmd_loss_weight < 0 or args.dmd_fake_score_loss_weight < 0:
        raise ValueError("DMD loss weights must be non-negative")
    if args.dmd_warmup_steps < 0:
        raise ValueError("--dmd_warmup_steps must be non-negative")
    if args.dmd_student_update_freq <= 0:
        raise ValueError("--dmd_student_update_freq must be positive")
    if args.dmd_max_timestep <= args.dmd_min_timestep:
        raise ValueError("--dmd_max_timestep must be greater than --dmd_min_timestep")
    if args.dmd_lognormal_std <= 0:
        raise ValueError("--dmd_lognormal_std must be positive")
    if args.dmd_consistency_loss_weight < 0:
        raise ValueError("--dmd_consistency_loss_weight must be non-negative")
    if args.dmd_consistency_fd_epsilon <= 0:
        raise ValueError("--dmd_consistency_fd_epsilon must be positive")
    if args.dense_schedule_steps:
        args.denoising_step_list = make_descending_timestep_list(args.dense_schedule_steps)
    if args.anchor_conditioning == "none" and args.training_mode != "teacher_trajectory" and args.dmd_loss_weight <= 0:
        raise ValueError("--anchor_conditioning none outside teacher_trajectory currently requires --dmd_loss_weight > 0")
    if args.dmd_loss_weight > 0 and args.training_mode == "teacher_trajectory":
        raise ValueError("rCM-style DMD is currently wired for one_step/unrolled/random_timestep, not teacher_trajectory")
    if args.rollout_solver in ("unipc", "rcm") and args.prediction_type != "flow":
        raise ValueError(f"--rollout_solver {args.rollout_solver} requires --prediction_type flow")
    if args.rollout_steps < 0:
        raise ValueError("--rollout_steps must be non-negative")
    if args.rollout_schedule and args.rollout_solver != "rcm":
        raise ValueError("--rollout_schedule is only used with --rollout_solver rcm")
    if args.rollout_solver == "unipc" and args.rollout_steps == 0:
        args.rollout_steps = len(args.denoising_step_list)
    if args.rollout_solver == "rcm" and args.rollout_steps == 0:
        args.rollout_steps = 5
    rollout_timestep_list = resolve_rollout_timestep_list(
        rollout_solver=args.rollout_solver,
        rollout_steps=args.rollout_steps,
        rollout_schedule=args.rollout_schedule,
        rollout_sigma_max=args.rollout_sigma_max,
        denoising_step_list=args.denoising_step_list,
    )

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0
    is_distributed = world_size > 1
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    def log_stage(message: str) -> None:
        if is_main:
            print(f"[startup rank0 {time.strftime('%F %T')}] {message}", flush=True)

    # WanTextEncoder expects relative wan_models paths.
    os.chdir(Path(__file__).parent)
    wan_models = Path(args.model_root) / "wan_models"
    local_wan_models = Path("wan_models")
    if not local_wan_models.exists():
        local_wan_models.symlink_to(wan_models, target_is_directory=True)

    startup_t0 = time.perf_counter()
    log_stage("building/loading bidirectional dataset index")
    dataset = BidirectionalPromptAnchorDataset(
        args.manifest_path,
        num_blocks=args.num_blocks,
        cache_dir=args.dataset_cache_dir,
        use_cache=not args.disable_dataset_cache,
        index_workers=args.dataset_index_workers,
        cache_wait_seconds=args.dataset_cache_wait_seconds,
    )
    log_stage(f"dataset index ready in {time.perf_counter() - startup_t0:.1f}s examples={len(dataset)}")
    sample_t0 = time.perf_counter()
    log_stage("loading one sample to infer latent shape")
    sample = dataset[0]
    latent_channels = int(sample["future_noise"].shape[2])
    frames_per_block = int(sample["future_noise"].shape[1] // (args.num_blocks - 1))
    max_frames = int(args.num_blocks * frames_per_block)
    log_stage(f"sample shape ready in {time.perf_counter() - sample_t0:.1f}s latent_channels={latent_channels} frames={max_frames}")
    if args.batch_size != 1:
        raise ValueError("online target anchor generation currently requires per-rank --batch_size 1")
    unroll_step_weights = parse_unroll_step_weights(args.unroll_step_weights, len(args.denoising_step_list))
    scheduler = make_scheduler(args.timestep_shift)
    if args.overfit_num_examples > 0:
        if args.overfit_start_index < 0:
            raise ValueError("--overfit_start_index must be >= 0")
        end_index = min(len(dataset), args.overfit_start_index + args.overfit_num_examples)
        train_indices = list(range(args.overfit_start_index, end_index))
        if not train_indices:
            raise ValueError(
                f"Requested empty overfit subset start={args.overfit_start_index} "
                f"num_examples={args.overfit_num_examples} dataset_len={len(dataset)}"
            )
        val_indices = []
        log_stage(f"using overfit subset indices={train_indices}")
    else:
        train_indices, val_indices = split_indices(len(dataset), args.val_fraction, args.seed)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices) if val_indices else None

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_bidirectional_examples,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_bidirectional_examples,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    from utils.wan_wrapper import WanTextEncoder

    text_t0 = time.perf_counter()
    log_stage("loading Wan text encoder")
    text_encoder = WanTextEncoder().to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    log_stage(f"Wan text encoder ready in {time.perf_counter() - text_t0:.1f}s")
    use_stored_anchor = dataset.format == "bidirectional_wan_full_video_v1"
    teacher_trajectory_cache = None
    if args.training_mode == "teacher_trajectory":
        if not use_stored_anchor:
            raise ValueError("--training_mode teacher_trajectory currently requires Option-B full-video dataset with stored anchors")
        trajectory_t0 = time.perf_counter()
        log_stage(
            f"loading online teacher trajectory cache steps={args.teacher_trajectory_steps} "
            f"solver={args.teacher_trajectory_solver} shift={args.teacher_trajectory_shift or 'legacy'} "
            f"dataset_key={args.teacher_trajectory_dataset_key or 'auto'} "
            f"dir={args.teacher_trajectory_cache_dir}"
        )
        teacher_trajectory_cache = OnlineTeacherTrajectoryCache(
            cache_dir=args.teacher_trajectory_cache_dir,
            manifest_path=args.manifest_path,
            model_name=args.target_model_name,
            model_root=args.model_root,
            config_path=args.config_path,
            num_blocks=args.num_blocks,
            sampling_steps=args.teacher_trajectory_steps,
            sample_solver=args.teacher_trajectory_solver,
            seed=args.anchor_noise_seed,
            device=device,
            dtype=torch.bfloat16,
            text_encoder=text_encoder,
            trajectory_scope="full" if args.anchor_conditioning == "none" else "future",
            trajectory_shift=args.teacher_trajectory_shift,
            trajectory_dataset_key=args.teacher_trajectory_dataset_key,
        )
        log_stage(f"teacher trajectory cache ready in {time.perf_counter() - trajectory_t0:.1f}s")
        if is_main:
            teacher_trajectory_cache.precompute(dataset, train_indices)
        if is_distributed:
            dist.barrier()
        teacher_trajectory_cache.unload_pipeline()
        log_stage("teacher trajectory cache precomputed and teacher pipeline unloaded")
    anchor_generator = None
    if not use_stored_anchor:
        anchor_t0 = time.perf_counter()
        log_stage("loading online target Wan anchor generator")
        anchor_generator = OnlineTargetAnchorGenerator(
            model_name=args.target_model_name,
            checkpoint_path=args.target_checkpoint_path,
            model_root=args.model_root,
            config_path=args.config_path,
            num_blocks=args.num_blocks,
            seed=args.anchor_noise_seed,
            device=device,
            dtype=torch.bfloat16,
            text_encoder=text_encoder,
        )
        log_stage(f"online target Wan anchor generator ready in {time.perf_counter() - anchor_t0:.1f}s")
    else:
        if args.anchor_conditioning == "none":
            log_stage("dataset has stored full-video/chunk-0 tensors; no-anchor model will not condition on chunk 0")
        else:
            log_stage("using stored full-video Wan chunk-0 anchors as model conditioning")
    head_t0 = time.perf_counter()
    if args.anchor_conditioning == "none":
        log_stage("building Wan-compatible full-video no-anchor draft head")
        model = WanFullVideoDraftHead(
            latent_channels=latent_channels,
            hidden_channels=args.hidden_channels,
            prompt_dim=args.prompt_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            max_frames=max_frames,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    else:
        log_stage("building bidirectional Wan draft head")
        model = BidirectionalPromptAnchorDraftHead(
            latent_channels=latent_channels,
            hidden_channels=args.hidden_channels,
            prompt_dim=args.prompt_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            ffn_dim=args.ffn_dim,
            temporal_mixer_layers=args.temporal_mixer_layers,
            temporal_mixer_ffn_dim=args.temporal_mixer_ffn_dim,
            max_frames=max_frames,
            gradient_checkpointing=args.gradient_checkpointing,
        ).to(device)
    log_stage(f"draft head ready in {time.perf_counter() - head_t0:.1f}s class={type(model).__name__}")
    init_report = None
    if args.init_draft_head_checkpoint_path:
        init_ckpt_t0 = time.perf_counter()
        log_stage(f"loading draft-head warm-start checkpoint {args.init_draft_head_checkpoint_path}")
        init_payload = torch.load(args.init_draft_head_checkpoint_path, map_location="cpu", weights_only=False)
        if init_payload.get("format") != "bidirectional_prompt_anchor_draft_head_v1":
            raise ValueError(f"Unsupported draft-head checkpoint format: {init_payload.get('format')}")
        init_model_config = dict(init_payload.get("model_config", {}))
        init_model_class = init_model_config.get("model_class", "BidirectionalPromptAnchorDraftHead")
        if init_model_class != type(model).__name__:
            raise ValueError(
                f"Warm-start checkpoint model_class={init_model_class} does not match current model class {type(model).__name__}"
            )
        model.load_state_dict(init_payload["model_state_dict"], strict=True)
        log_stage(f"draft-head warm-start loaded in {time.perf_counter() - init_ckpt_t0:.1f}s")
    if args.init_target_blocks and not args.init_draft_head_checkpoint_path:
        init_t0 = time.perf_counter()
        init_model_name = args.init_model_name or args.target_model_name
        log_stage(f"initializing draft head from {init_model_name} blocks {args.init_target_blocks}")
        if use_stored_anchor:
            from sdvg_inference import load_config
            from utils.wan_wrapper import WanDiffusionWrapper

            init_config = load_config(args.config_path)
            model_kwargs = dict(getattr(init_config, "model_kwargs", {}))
            model_kwargs.pop("model_name", None)
            target_init_model = WanDiffusionWrapper(
                model_name=init_model_name,
                **model_kwargs,
                is_causal=False,
            ).to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
            target_model_for_init = target_init_model.model
        else:
            assert anchor_generator is not None
            target_init_model = None
            target_model_for_init = anchor_generator.pipeline.generator.model
        init_report = initialize_bidirectional_wan_head_from_target_blocks(
            model,
            target_model_for_init,
            tuple(args.init_target_blocks),
        )
        if target_init_model is not None:
            del target_init_model
            torch.cuda.empty_cache()
        log_stage(f"target block initialization ready in {time.perf_counter() - init_t0:.1f}s report={init_report}")
        if is_main:
            print(f"Initialized bidirectional Wan head from target blocks {args.init_target_blocks}: {init_report}")
    elif args.init_target_blocks and args.init_draft_head_checkpoint_path:
        log_stage("skipping target block initialization because draft-head warm-start checkpoint was provided")
    raw_model = model
    wrap_t0 = time.perf_counter()
    log_stage(f"wrapping model with {args.parallel_strategy if is_distributed else 'none'}")
    model = wrap_model_for_training(
        model,
        strategy=args.parallel_strategy if is_distributed else "none",
        is_distributed=is_distributed,
        local_rank=device.index or 0,
        fsdp_min_num_params=args.fsdp_min_num_params,
        fsdp_mixed_precision=args.fsdp_mixed_precision,
    )
    log_stage(f"parallel wrapper ready in {time.perf_counter() - wrap_t0:.1f}s")
    optim_t0 = time.perf_counter()
    log_stage("creating AdamW optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    log_stage(f"optimizer ready in {time.perf_counter() - optim_t0:.1f}s")
    dmd: RCMStyleDraftHeadDMD | None = None
    if args.dmd_loss_weight > 0:
        from sdvg_inference import load_config

        dmd_t0 = time.perf_counter()
        log_stage("loading rCM-style DMD teacher and fake-score networks")
        dmd_config = load_config(args.config_path)
        negative_prompt = getattr(
            dmd_config,
            "negative_prompt",
            "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        )
        fake_score_init_device = torch.device("cpu") if is_distributed and args.parallel_strategy == "fsdp" else device
        dmd = RCMStyleDraftHeadDMD(
            teacher_model_name=args.dmd_teacher_model_name or args.dmd_model_name or args.target_model_name,
            fake_score_model_name=args.dmd_fake_score_model_name or args.dmd_model_name or args.target_model_name,
            teacher_checkpoint_path=args.dmd_teacher_checkpoint_path or None,
            fake_score_checkpoint_path=args.dmd_fake_score_checkpoint_path or None,
            model_root=args.model_root,
            config_path=args.config_path,
            text_encoder=text_encoder,
            device=device,
            fake_score_device=fake_score_init_device,
            dtype=torch.bfloat16 if args.amp_dtype == "bf16" and device.type == "cuda" else torch.float32,
            guidance_scale=args.dmd_guidance_scale,
            timestep_shift=args.dmd_timestep_shift,
            min_timestep=args.dmd_min_timestep,
            max_timestep=args.dmd_max_timestep,
            time_distribution=args.dmd_time_distribution,
            lognormal_mean=args.dmd_lognormal_mean,
            lognormal_std=args.dmd_lognormal_std,
            fake_score_weighting=args.dmd_fake_score_weighting,
            target_convention=args.dmd_target_convention,
            consistency_objective=args.dmd_consistency_objective,
            consistency_loss_weight=args.dmd_consistency_loss_weight,
            consistency_fd_epsilon=args.dmd_consistency_fd_epsilon,
            score_scope=args.dmd_score_scope,
            fake_score_lr=args.dmd_fake_score_lr,
            fake_score_weight_decay=args.dmd_fake_score_weight_decay,
            negative_prompt=negative_prompt,
        )
        if args.dmd_fake_score_gradient_checkpointing:
            dmd.enable_fake_score_gradient_checkpointing()
        dmd.wrap_fake_score(
            strategy=args.parallel_strategy if is_distributed else "none",
            is_distributed=is_distributed,
            local_rank=device.index or 0,
            fsdp_min_num_params=args.fsdp_min_num_params,
            fsdp_mixed_precision=args.fsdp_mixed_precision,
        )
        dmd.create_optimizer()
        log_stage(
            f"rCM-style DMD ready in {time.perf_counter() - dmd_t0:.1f}s "
            f"teacher={dmd.teacher_model_name} fake_score={dmd.fake_score_model_name} "
            f"fake_init={dmd.fake_score_init} weight={args.dmd_loss_weight} fake_lr={args.dmd_fake_score_lr} "
            f"fake_init_device={dmd.fake_score_init_device} fake_wrapped={dmd.fake_score_is_wrapped} "
            f"fake_gc={args.dmd_fake_score_gradient_checkpointing} "
            f"student_update_freq={args.dmd_student_update_freq} scope={args.dmd_score_scope}"
        )

    if is_main:
        print(
            f"Training bidirectional draft head prompts={len(dataset)} train={len(train_dataset)} "
            f"val={0 if val_dataset is None else len(val_dataset)} latent_channels={latent_channels} "
            f"frames={max_frames} hidden={args.hidden_channels} layers={args.num_layers} "
            f"heads={args.num_heads} ffn_dim={args.ffn_dim} "
            f"temporal_mixer_layers={args.temporal_mixer_layers} "
            f"temporal_mixer_ffn_dim={args.temporal_mixer_ffn_dim} world_size={world_size} "
            f"dmd_weight={args.dmd_loss_weight} rollout={args.rollout_solver}:{rollout_timestep_list} "
            f"parallel={args.parallel_strategy if is_distributed else 'none'}"
        )

    history = []
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_metric_sums: dict[str, float] = {}
        total_batches = 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}", disable=not is_main)
        for batch_idx, batch in enumerate(iterator, start=1):
            start_time = time.perf_counter()
            with torch.no_grad():
                if use_stored_anchor:
                    anchor = batch["anchor_latents"].to(device=device, dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32)
                    prompt_embeds = text_encoder(batch["prompts"])["prompt_embeds"].detach()
                else:
                    assert anchor_generator is not None
                    anchor, prompt_embeds = anchor_generator(batch)
                anchor = anchor.to(device=device, dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32)
                prompt_embeds = prompt_embeds.to(device=device, dtype=anchor.dtype)
                if args.anchor_conditioning == "none":
                    if "full_noise" not in batch or "full_target_latents" not in batch:
                        raise ValueError("--anchor_conditioning none requires a full-video Option-B manifest")
                    model_anchor = None
                    future_noise = batch["full_noise"].to(device=device, dtype=anchor.dtype)
                    target = batch["full_target_latents"].to(device=device, dtype=anchor.dtype)
                else:
                    model_anchor = anchor
                    future_noise = batch["future_noise"].to(device=device, dtype=anchor.dtype)
                    target = batch["future_target_latents"].to(device=device, dtype=anchor.dtype)
                trajectory = teacher_trajectory_cache(batch) if teacher_trajectory_cache is not None else None
            next_step = global_step + 1
            dmd_fake_phase = (
                dmd is not None
                and not RCMStyleDraftHeadDMD.is_student_phase(
                    next_step,
                    warmup_steps=args.dmd_warmup_steps,
                    student_update_freq=args.dmd_student_update_freq,
                )
            )
            if dmd_fake_phase:
                assert dmd is not None
                assert dmd.optimizer is not None
                dmd.optimizer.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad(), amp_context(device, args.amp_dtype):
                    generator_prediction = predict_bidirectional_clean_rollout(
                        model,
                        anchor=model_anchor,
                        initial_noise=future_noise,
                        target=target,
                        prompt_embeds=prompt_embeds,
                        scheduler=scheduler,
                        training_mode=args.training_mode,
                        denoising_step_list=args.denoising_step_list,
                        prediction_type=args.prediction_type,
                        random_timestep_sampling=args.random_timestep_sampling,
                        logit_normal_mean=args.logit_normal_mean,
                        logit_normal_std=args.logit_normal_std,
                        unroll_noise_mode=args.unroll_noise_mode,
                        rollout_solver=args.rollout_solver,
                        rollout_steps=args.rollout_steps,
                        rollout_timestep_list=rollout_timestep_list,
                        rollout_solver_shift=args.rollout_solver_shift,
                    )
                with amp_context(device, args.amp_dtype):
                    fake_loss, fake_metrics = dmd.fake_score_loss(
                        generator_prediction.detach(),
                        anchor=model_anchor,
                        prompt_embeds=prompt_embeds,
                    )
                    loss = fake_loss * args.dmd_fake_score_loss_weight
                loss.backward()
                dmd.average_fake_score_gradients(world_size)
                dmd.optimizer.step()
                model.train()
                clean_mse = F.mse_loss(generator_prediction.float(), target.float())
                metrics = {
                    "loss": float(loss.detach().cpu().item()),
                    "clean_latent_mse": float(clean_mse.detach().cpu().item()),
                    "dmd_fake_score_loss": float(loss.detach().cpu().item()),
                    "dmd_phase_fake_score": 1.0,
                }
                metrics.update(fake_metrics)
            else:
                optimizer.zero_grad(set_to_none=True)
                with amp_context(device, args.amp_dtype):
                    if args.training_mode == "teacher_trajectory":
                        assert trajectory is not None
                        loss, metrics = compute_teacher_trajectory_losses(
                            model,
                            anchor=model_anchor,
                            prompt_embeds=prompt_embeds,
                            trajectory=trajectory,
                            scheduler=scheduler,
                            clean_latent_loss_weight=args.clean_latent_loss_weight,
                            flow_loss_weight=args.flow_loss_weight,
                            detail_loss_weight=args.detail_loss_weight,
                            anchor_conditioning=args.anchor_conditioning,
                        )
                    else:
                        loss, metrics = compute_bidirectional_losses(
                            model,
                            anchor=model_anchor,
                            initial_noise=future_noise,
                            target=target,
                            prompt_embeds=prompt_embeds,
                            scheduler=scheduler,
                            training_mode=args.training_mode,
                            denoising_step_list=args.denoising_step_list,
                            prediction_type=args.prediction_type,
                            random_timestep_sampling=args.random_timestep_sampling,
                            logit_normal_mean=args.logit_normal_mean,
                            logit_normal_std=args.logit_normal_std,
                            unroll_step_weights=unroll_step_weights,
                            unroll_noise_mode=args.unroll_noise_mode,
                            clean_latent_loss_weight=args.clean_latent_loss_weight,
                            flow_loss_weight=args.flow_loss_weight,
                            detail_loss_weight=args.detail_loss_weight,
                            temporal_delta_weight=args.temporal_delta_weight,
                            boundary_weight=args.boundary_weight,
                            dmd=dmd,
                            dmd_loss_weight=args.dmd_loss_weight,
                            rollout_solver=args.rollout_solver,
                            rollout_steps=args.rollout_steps,
                            rollout_timestep_list=rollout_timestep_list,
                            rollout_solver_shift=args.rollout_solver_shift,
                        )
                loss.backward()
                optimizer.step()
                if dmd is not None:
                    metrics["dmd_phase_student"] = 1.0
            global_step += 1
            total_loss += metrics["loss"]
            total_mse += metrics["clean_latent_mse"]
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    total_metric_sums[key] = total_metric_sums.get(key, 0.0) + float(value)
            total_batches += 1
            if is_main and (batch_idx % args.log_every == 0):
                running_mse = total_mse / max(1, total_batches)
                extra_metrics = " ".join(
                    f"{key}={float(value):.6f}"
                    for key, value in metrics.items()
                    if key not in ("loss", "clean_latent_mse") and isinstance(value, (int, float))
                )
                iterator.set_postfix(
                    loss=metrics["loss"],
                    rmse=math.sqrt(running_mse),
                    sec=time.perf_counter() - start_time,
                )
                print(
                    f"step={global_step} epoch={epoch} batch={batch_idx} "
                    f"running_loss={total_loss / max(1, total_batches):.6f} "
                    f"running_clean_latent_rmse={math.sqrt(running_mse):.6f} "
                    f"loss={metrics['loss']:.6f} clean_latent_mse={metrics['clean_latent_mse']:.6f}"
                    f"{(' ' + extra_metrics) if extra_metrics else ''}"
                )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_batches),
            "train_clean_latent_mse": total_mse / max(1, total_batches),
            "train_clean_latent_rmse": math.sqrt(total_mse / max(1, total_batches)),
        }
        for key, value in sorted(total_metric_sums.items()):
            if key in ("loss", "clean_latent_mse"):
                continue
            epoch_metrics[f"train_{key}"] = value / max(1, total_batches)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_mse = 0.0
            val_metric_sums: dict[str, float] = {}
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if use_stored_anchor:
                        anchor = batch["anchor_latents"].to(
                            device=device,
                            dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32,
                        )
                        prompt_embeds = text_encoder(batch["prompts"])["prompt_embeds"].detach()
                    else:
                        assert anchor_generator is not None
                        anchor, prompt_embeds = anchor_generator(batch)
                    anchor = anchor.to(
                        device=device,
                        dtype=torch.bfloat16 if args.amp_dtype == "bf16" else torch.float32,
                    )
                    prompt_embeds = prompt_embeds.to(device=device, dtype=anchor.dtype)
                    if args.anchor_conditioning == "none":
                        if "full_noise" not in batch or "full_target_latents" not in batch:
                            raise ValueError("--anchor_conditioning none requires a full-video Option-B manifest")
                        model_anchor = None
                        future_noise = batch["full_noise"].to(device=device, dtype=anchor.dtype)
                        target = batch["full_target_latents"].to(device=device, dtype=anchor.dtype)
                    else:
                        model_anchor = anchor
                        future_noise = batch["future_noise"].to(device=device, dtype=anchor.dtype)
                        target = batch["future_target_latents"].to(device=device, dtype=anchor.dtype)
                    trajectory = teacher_trajectory_cache(batch) if teacher_trajectory_cache is not None else None
                    with amp_context(device, args.amp_dtype):
                        if args.training_mode == "teacher_trajectory":
                            assert trajectory is not None
                            _loss, metrics = compute_teacher_trajectory_losses(
                                model,
                                anchor=model_anchor,
                                prompt_embeds=prompt_embeds,
                                trajectory=trajectory,
                                scheduler=scheduler,
                                clean_latent_loss_weight=args.clean_latent_loss_weight,
                                flow_loss_weight=args.flow_loss_weight,
                                detail_loss_weight=args.detail_loss_weight,
                                anchor_conditioning=args.anchor_conditioning,
                            )
                        else:
                            _loss, metrics = compute_bidirectional_losses(
                                model,
                                anchor=model_anchor,
                                initial_noise=future_noise,
                                target=target,
                                prompt_embeds=prompt_embeds,
                                scheduler=scheduler,
                                training_mode=args.training_mode,
                                denoising_step_list=args.denoising_step_list,
                                prediction_type=args.prediction_type,
                                random_timestep_sampling=args.random_timestep_sampling,
                                logit_normal_mean=args.logit_normal_mean,
                                logit_normal_std=args.logit_normal_std,
                                unroll_step_weights=unroll_step_weights,
                                unroll_noise_mode=args.unroll_noise_mode,
                                clean_latent_loss_weight=args.clean_latent_loss_weight,
                                flow_loss_weight=args.flow_loss_weight,
                                detail_loss_weight=args.detail_loss_weight,
                                temporal_delta_weight=args.temporal_delta_weight,
                                boundary_weight=args.boundary_weight,
                                dmd=dmd,
                                dmd_loss_weight=args.dmd_loss_weight,
                                rollout_solver=args.rollout_solver,
                                rollout_steps=args.rollout_steps,
                                rollout_timestep_list=rollout_timestep_list,
                                rollout_solver_shift=args.rollout_solver_shift,
                            )
                    val_loss += metrics["loss"]
                    val_mse += metrics["clean_latent_mse"]
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value)
                    val_batches += 1
            epoch_metrics.update(
                {
                    "val_loss": val_loss / max(1, val_batches),
                    "val_clean_latent_mse": val_mse / max(1, val_batches),
                    "val_clean_latent_rmse": math.sqrt(val_mse / max(1, val_batches)),
                }
            )
            for key, value in sorted(val_metric_sums.items()):
                if key in ("loss", "clean_latent_mse"):
                    continue
                epoch_metrics[f"val_{key}"] = value / max(1, val_batches)
        history.append(epoch_metrics)
        if is_main:
            print(json.dumps(epoch_metrics, indent=2))
        epoch_path = Path(args.output_path).with_name(f"epoch_{epoch:04d}.pt")
        if args.parallel_strategy == "fsdp" and is_distributed:
            saved_path = rank0_save_with_state_dict(
                model=model,
                unwrapped_model=raw_model,
                path=epoch_path,
                save_fn=save_checkpoint,
                save_kwargs={"args": args, "metadata": {"history": history, "init_report": init_report}},
            )
            if is_main and saved_path is not None:
                print(f"Wrote epoch checkpoint: {saved_path}")
        elif is_main:
            save_checkpoint(model, epoch_path, args=args, metadata={"history": history, "init_report": init_report})
            print(f"Wrote epoch checkpoint: {epoch_path}")

    if args.parallel_strategy == "fsdp" and is_distributed:
        final_path = rank0_save_with_state_dict(
            model=model,
            unwrapped_model=raw_model,
            path=args.output_path,
            save_fn=save_checkpoint,
            save_kwargs={"args": args, "metadata": {"history": history, "init_report": init_report}},
        )
    elif is_main:
        final_path = save_checkpoint(model, args.output_path, args=args, metadata={"history": history, "init_report": init_report})
    else:
        final_path = None
    if is_main:
        metrics_path = Path(args.output_path).with_name("metrics.json")
        metrics_path.write_text(
            json.dumps(
                {
                    "train_args": vars(args),
                    "manifest_path": str(args.manifest_path),
                    "num_prompt_examples": len(dataset),
                    "train_examples": len(train_dataset),
                    "val_examples": 0 if val_dataset is None else len(val_dataset),
                    "history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Wrote bidirectional draft-head checkpoint: {final_path}")
        print(f"Wrote training metrics: {metrics_path}")
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
