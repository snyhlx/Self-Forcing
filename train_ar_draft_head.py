#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fsdp_utils import (
    fsdp_rank0_state_dict,
    is_fsdp_model,
    rank0_save_with_state_dict,
    unwrap_model,
    wrap_model_for_training,
)
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
    inner = unwrap_model(model)
    return isinstance(inner, BidirectionalPromptAnchorDraftHead)


def is_causal_wan_ar_head(model: torch.nn.Module) -> bool:
    inner = unwrap_model(model)
    return isinstance(inner, CausalWanARDraftHead)


def causal_wan_frame_seq_length(model: torch.nn.Module, latents: torch.Tensor) -> int:
    inner = unwrap_model(model)
    _, _, _, height, width = latents.shape
    patch = getattr(inner.generator.model, "patch_size", (1, 2, 2))
    _, patch_h, patch_w = (int(value) for value in patch)
    return (int(height) // patch_h) * (int(width) // patch_w)


def initialize_causal_wan_ar_caches(
    model: torch.nn.Module,
    *,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    total_frames: int,
    frame_seq_length: int,
) -> tuple[list[dict[str, torch.Tensor | bool]], list[dict[str, torch.Tensor | bool]]]:
    """Create standalone KV/cross-attention caches for CausalWanARDraftHead replay."""
    inner = unwrap_model(model)
    if not isinstance(inner, CausalWanARDraftHead):
        raise ValueError("incremental KV replay requires --head_type causal_wan_ar")
    generator = inner.generator
    num_heads = int(generator.model.num_heads)
    head_dim = int(generator.model.dim) // num_heads
    kv_cache_size = max(int(generator.seq_len), int(total_frames) * int(frame_seq_length))
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
            pad_to_frames=batch.get("causal_wan_pad_to_frames", num_blocks * int(batch[input_key].shape[1])),
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


def first_nonfinite_grad_summary(model: torch.nn.Module) -> str | None:
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        if grad is not None and not torch.isfinite(grad).all().item():
            return tensor_finite_summary(f"grad:{name}", grad)
    return None


def register_nonfinite_backward_debug_hooks(
    model: torch.nn.Module,
    *,
    rank: int,
    state: dict[str, Any],
) -> list[Any]:
    """Register lightweight hooks that report where backward first becomes non-finite."""
    handles = []
    inner = unwrap_model(model)

    def should_watch(name: str) -> bool:
        if not name.startswith("generator.model."):
            return False
        if ".blocks." in name:
            return name.endswith((".self_attn", ".cross_attn", ".ffn", ".norm1", ".norm2", ".norm3"))
        return name in ("generator.model.patch_embedding", "generator.model.head", "generator.model.time_embedding")

    def tensor_list_nonfinite_summary(label: str, tensors: tuple[Any, ...]) -> str | None:
        for index, tensor in enumerate(tensors):
            if torch.is_tensor(tensor) and not torch.isfinite(tensor).all().item():
                return tensor_finite_summary(f"{label}[{index}]", tensor)
        return None

    def tensor_list_all_finite(tensors: tuple[Any, ...]) -> bool:
        found_tensor = False
        for tensor in tensors:
            if torch.is_tensor(tensor):
                found_tensor = True
                if not torch.isfinite(tensor).all().item():
                    return False
        return found_tensor

    def make_hook(name: str):
        def hook(_module, grad_input, grad_output):
            if not state.get("enabled", False):
                return
            if int(state.get("reports", 0)) >= int(state.get("max_reports", 8)):
                return
            grad_output_finite = tensor_list_all_finite(grad_output)
            grad_input_summary = tensor_list_nonfinite_summary("grad_input", grad_input)
            if grad_output_finite and grad_input_summary is not None:
                summary = grad_input_summary
                direction = "introduced_nonfinite_grad_input"
            else:
                summary = tensor_list_nonfinite_summary("grad_output", grad_output)
                direction = "received_nonfinite_grad_output"
            if summary is None:
                return
            state["reports"] = int(state.get("reports", 0)) + 1
            print(
                "[nonfinite-backward] "
                f"rank={rank} global_step={state.get('global_step')} "
                f"module={name} direction={direction} {summary}",
                flush=True,
            )
        return hook

    for name, module in inner.named_modules():
        if should_watch(name):
            handles.append(module.register_full_backward_hook(make_hook(name)))
    return handles


def batch_scalar(batch: dict[str, Any], key: str, default: Any = "unknown") -> Any:
    value = batch.get(key)
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        return value.flatten()[0].detach().cpu().item()
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return default if value is None else value


def teacher_trajectory_active_step_indices(
    batch: dict[str, Any],
    step_indices: list[int] | None,
    *,
    sample_one: bool = False,
) -> list[int]:
    if "teacher_trajectory_noisy_latents" not in batch:
        raise ValueError("teacher_trajectory mode requires teacher_trajectory_noisy_latents")
    num_steps = int(batch["teacher_trajectory_noisy_latents"].shape[1])
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
    if sample_one and len(active_step_indices) > 1:
        device = batch["teacher_trajectory_noisy_latents"].device
        choice = torch.empty((), device=device, dtype=torch.long)
        if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
            choice.fill_(random.choice(active_step_indices))
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(choice, src=0)
        active_step_indices = [int(choice.detach().cpu().item())]
    return active_step_indices


def distributed_sample_teacher_trajectory_step(step_indices: list[int], device: torch.device) -> int:
    choice = torch.empty((), device=device, dtype=torch.long)
    if not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0:
        choice.fill_(random.choice(step_indices))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(choice, src=0)
    return int(choice.detach().cpu().item())


class StopGradientSelfConditioningDataset(Dataset):
    """Group adjacent AR teacher-trajectory records for short self-conditioning."""

    def __init__(
        self,
        base: Dataset,
        *,
        num_blocks: int,
        lookahead_chunks: int,
        anchor_policy: str,
        fixed_anchor_index: int,
    ) -> None:
        if lookahead_chunks <= 0:
            raise ValueError("--self_conditioning_lookahead_chunks must be positive")
        if anchor_policy not in ("random_valid", "early_bias", "late_bias", "fixed"):
            raise ValueError("--self_conditioning_anchor_policy must be random_valid, early_bias, late_bias, or fixed")
        self.base = base
        self.num_blocks = int(num_blocks)
        self.lookahead_chunks = int(lookahead_chunks)
        self.anchor_policy = anchor_policy
        self.fixed_anchor_index = int(fixed_anchor_index)
        self._records: list[dict[str, Any]] = [base[index] for index in range(len(base))]
        self._by_prompt_block: dict[tuple[int, str, int], int] = {}
        for index, record in enumerate(self._records):
            prompt_index = -1 if record.get("prompt_index") is None else int(record["prompt_index"])
            prompt = str(record.get("prompt", ""))
            block_index = int(record["block_index"])
            self._by_prompt_block[(prompt_index, prompt, block_index)] = index

        self.examples: list[tuple[tuple[int, str], int, list[int]]] = []
        for key_prompt_index, key_prompt, first_block in sorted(self._by_prompt_block):
            if first_block <= 0:
                continue
            anchor_index = first_block - 1
            if first_block + self.lookahead_chunks > self.num_blocks:
                continue
            record_indices = []
            for block_index in range(first_block, first_block + self.lookahead_chunks):
                pointer = self._by_prompt_block.get((key_prompt_index, key_prompt, block_index))
                if pointer is None:
                    break
                record_indices.append(pointer)
            if len(record_indices) == self.lookahead_chunks:
                self.examples.append(((key_prompt_index, key_prompt), anchor_index, record_indices))
        if not self.examples:
            raise ValueError(
                "No valid stop-gradient self-conditioning examples. "
                "Expected adjacent AR records with first future block >= 1."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def _sample_example_index(self, index: int) -> int:
        if self.anchor_policy == "fixed":
            valid = [
                example_index
                for example_index, (_prompt_key, anchor_index, _record_indices) in enumerate(self.examples)
                if anchor_index == self.fixed_anchor_index
            ]
            if not valid:
                raise ValueError(
                    f"No self-conditioning examples for fixed anchor index {self.fixed_anchor_index}"
                )
            return valid[index % len(valid)]
        if self.anchor_policy == "random_valid":
            return random.randrange(len(self.examples))
        anchors = [anchor_index for _prompt_key, anchor_index, _record_indices in self.examples]
        if self.anchor_policy == "early_bias":
            weights = [float(self.num_blocks - anchor_index) for anchor_index in anchors]
        else:
            weights = [float(anchor_index + 1) for anchor_index in anchors]
        return random.choices(range(len(self.examples)), weights=weights, k=1)[0]

    def __getitem__(self, index: int) -> dict[str, Any]:
        _prompt_key, anchor_index, record_indices = self.examples[self._sample_example_index(index)]
        future_records = [self._records[pointer] for pointer in record_indices]
        record = dict(future_records[0])
        record["self_conditioning_anchor_index"] = anchor_index
        record["self_conditioning_first_block_index"] = int(future_records[0]["block_index"])
        record["self_conditioning_clean_latents"] = torch.stack(
            [future_record["teacher_trajectory_latents"] for future_record in future_records],
            dim=0,
        )
        record["self_conditioning_noisy_latents"] = torch.stack(
            [future_record["teacher_trajectory_noisy_latents"] for future_record in future_records],
            dim=0,
        )
        record["self_conditioning_timesteps"] = torch.stack(
            [future_record["teacher_trajectory_timesteps"] for future_record in future_records],
            dim=0,
        )
        record["self_conditioning_target_latents"] = torch.cat(
            [future_record["target_latents"] for future_record in future_records],
            dim=0,
        )
        return record


def filter_teacher_trajectory_step_indices_for_loss(
    batch: dict[str, Any],
    step_indices: list[int],
    *,
    loss_type: str,
    device: torch.device | None = None,
) -> list[int]:
    if loss_type != "flow":
        return step_indices
    timesteps = batch.get("teacher_trajectory_timesteps")
    if timesteps is None:
        return step_indices
    if timesteps.ndim == 1:
        local_valid = [bool((timesteps[index] > 0).item()) for index in step_indices]
    else:
        local_valid = [bool((timesteps[:, index] > 0).any().item()) for index in step_indices]
    if dist.is_available() and dist.is_initialized():
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        valid_tensor = torch.tensor(local_valid, device=device, dtype=torch.int32)
        dist.all_reduce(valid_tensor, op=dist.ReduceOp.MIN)
        local_valid = [bool(value) for value in valid_tensor.detach().cpu().tolist()]
    return [index for index, is_valid in zip(step_indices, local_valid, strict=True) if is_valid]


def merge_loss_metric_lists(metric_lists: list[dict[str, float]], *, loss: float) -> dict[str, float]:
    if not metric_lists:
        return {"loss": loss, "clean_latent_mse": 0.0}
    merged: dict[str, float] = {"loss": loss}
    keys = sorted({key for metrics in metric_lists for key in metrics if key != "loss"})
    for key in keys:
        values = [metrics[key] for metrics in metric_lists if key in metrics]
        if values:
            merged[key] = float(sum(values) / len(values))
    if "clean_latent_mse" not in merged:
        merged["clean_latent_mse"] = 0.0
    return merged


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
    step_mode: str = "whole_graph",
    teacher_trajectory_objective: str = "prefix_flow",
    teacher_trajectory_prefix_loss_weight: float = 1.0,
    teacher_trajectory_incremental_kv_loss_weight: float = 1.0,
    incremental_kv_consistency_weight: float = 0.0,
    incremental_kv_context_noise: int = 0,
    debug_timing: bool = False,
    debug_nonfinite_backward: bool = False,
    debug_rank: int = 0,
    debug_global_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if step_mode not in ("whole_graph", "sample_one", "sequential_backward"):
        raise ValueError("--teacher_trajectory_step_mode must be whole_graph, sample_one, or sequential_backward")
    if teacher_trajectory_objective not in ("prefix_flow", "incremental_kv_flow", "hybrid_prefix_incremental_kv_flow"):
        raise ValueError(
            "--teacher_trajectory_objective must be prefix_flow, incremental_kv_flow, or "
            "hybrid_prefix_incremental_kv_flow"
        )
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
    active_step_indices = teacher_trajectory_active_step_indices(
        batch,
        step_indices,
        sample_one=False,
    )
    active_step_indices = filter_teacher_trajectory_step_indices_for_loss(
        batch,
        active_step_indices,
        loss_type=loss_type,
        device=noisy_states.device,
    )
    if step_mode == "sample_one" and len(active_step_indices) > 1:
        active_step_indices = [distributed_sample_teacher_trajectory_step(active_step_indices, noisy_states.device)]
    if not active_step_indices:
        raise ValueError("No teacher trajectory steps can contribute to the selected loss")

    components: list[torch.Tensor] = []
    flow_losses = []
    clean_losses = []
    consistency_losses = []
    committed_prefix_frame_counts = []
    prefix_frames = int(batch["context_latents"].shape[1]) if batch.get("context_latents") is not None else 0
    use_prefix_flow = teacher_trajectory_objective in ("prefix_flow", "hybrid_prefix_incremental_kv_flow")
    use_incremental_kv_flow = teacher_trajectory_objective in (
        "incremental_kv_flow",
        "hybrid_prefix_incremental_kv_flow",
    )
    use_hybrid_flow = teacher_trajectory_objective == "hybrid_prefix_incremental_kv_flow"
    use_incremental_kv_consistency = incremental_kv_consistency_weight > 0
    needs_incremental_kv = use_incremental_kv_flow or use_incremental_kv_consistency
    if use_incremental_kv_consistency and teacher_trajectory_objective != "prefix_flow":
        raise ValueError("--incremental_kv_consistency_weight is only used with prefix_flow objective")
    if needs_incremental_kv:
        if not is_causal_wan_ar_head(model):
            raise ValueError("incremental KV teacher trajectory objectives require --head_type causal_wan_ar")
        if batch.get("context_latents") is None:
            raise ValueError("incremental KV teacher trajectory objectives require context_latents")
        if use_incremental_kv_consistency and prefix_frames == 0:
            raise ValueError("incremental KV consistency requires context_latents with at least one prefix frame")
        if "prompt_embeds" not in batch:
            raise ValueError("incremental KV teacher trajectory objectives require prompt_embeds in the batch")
        frame_seq_length = causal_wan_frame_seq_length(model, noisy_states[:, 0])
        prefix_latents = batch["context_latents"].to(device=device, dtype=dtype)
        prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
        require_finite("teacher_trajectory_context_latents", prefix_latents)
        require_finite("teacher_trajectory_prompt_embeds", prompt_embeds)
    else:
        frame_seq_length = 0
        prefix_latents = None
        prompt_embeds = None

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
        valid_flow_timestep = bool((timestep > 0).any().detach().cpu().item())
        valid_consistency_timestep = valid_flow_timestep
        needs_flow_forward = loss_type in ("flow", "clean_latent_flow") and valid_flow_timestep
        needs_clean_forward = loss_type in ("clean_latent", "clean_latent_flow")
        needs_consistency_forward = use_incremental_kv_consistency and valid_consistency_timestep
        if not (needs_flow_forward or needs_clean_forward or needs_consistency_forward):
            continue
        model_output = None
        if use_prefix_flow or needs_consistency_forward:
            step_batch = dict(batch)
            step_batch["scheduled_latents"] = state
            step_batch["timestep"] = timestep
            model_output = predict_draft_head_batch(model, step_batch, num_blocks, input_key="scheduled_latents")
            require_finite(f"teacher_trajectory_model_output ({debug_context})", model_output)
        incremental_output = None
        if use_incremental_kv_flow or needs_consistency_forward:
            incremental_replay_start = time.perf_counter()
            assert prefix_latents is not None
            assert prompt_embeds is not None
            kv_cache, crossattn_cache = initialize_causal_wan_ar_caches(
                model,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                total_frames=prefix_frames + frames,
                frame_seq_length=frame_seq_length,
            )
            cache_init_seconds = debug_elapsed(incremental_replay_start, device) if debug_timing else 0.0
            inner_model = unwrap_model(model)
            generator_model = inner_model.generator.model if isinstance(inner_model, CausalWanARDraftHead) else None
            local_attn_size = int(getattr(generator_model, "local_attn_size", -1))
            if local_attn_size != -1:
                committed_prefix_frames = min(prefix_frames, max(local_attn_size - frames, 0))
            else:
                committed_prefix_frames = prefix_frames
            prefix_commit_start = max(0, prefix_frames - committed_prefix_frames)
            if prefix_commit_start > 0:
                skipped_tokens = prefix_commit_start * frame_seq_length
                for cache in kv_cache:
                    cache["global_end_index"].fill_(skipped_tokens)
                    cache["local_end_index"].zero_()
            committed_prefix_frame_counts.append(float(committed_prefix_frames))
            prefix_chunk_frames = frames
            prefix_commit_seconds = 0.0
            actual_prefix_starts = list(range(prefix_commit_start, prefix_frames, prefix_chunk_frames))
            num_prefix_commit_forwards = torch.tensor([len(actual_prefix_starts)], device=device, dtype=torch.int32)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(num_prefix_commit_forwards, op=dist.ReduceOp.MAX)
            max_prefix_commit_forwards = int(num_prefix_commit_forwards.detach().cpu().item())
            dummy_kv_cache = None
            dummy_crossattn_cache = None
            with torch.no_grad():
                for prefix_slot in range(max_prefix_commit_forwards):
                    prefix_commit_start_time = time.perf_counter()
                    is_real_prefix_commit = prefix_slot < len(actual_prefix_starts)
                    if is_real_prefix_commit:
                        prefix_start = actual_prefix_starts[prefix_slot]
                        prefix_chunk = prefix_latents[:, prefix_start:prefix_start + prefix_chunk_frames]
                        commit_kv_cache = kv_cache
                        commit_crossattn_cache = crossattn_cache
                        commit_current_start = prefix_start * frame_seq_length
                    else:
                        prefix_start = -1
                        prefix_chunk = torch.zeros_like(state)
                        if dummy_kv_cache is None or dummy_crossattn_cache is None:
                            dummy_kv_cache, dummy_crossattn_cache = initialize_causal_wan_ar_caches(
                                model,
                                batch_size=batch_size,
                                dtype=dtype,
                                device=device,
                                total_frames=frames,
                                frame_seq_length=frame_seq_length,
                            )
                        commit_kv_cache = dummy_kv_cache
                        commit_crossattn_cache = dummy_crossattn_cache
                        commit_current_start = 0
                    prefix_timestep = torch.full(
                        (batch_size, prefix_chunk.shape[1]),
                        int(incremental_kv_context_noise),
                        device=device,
                        dtype=torch.long,
                    )
                    prefix_output = model(
                        noisy_latents=prefix_chunk,
                        prompt_embeds=prompt_embeds,
                        timestep=prefix_timestep,
                        kv_cache=commit_kv_cache,
                        crossattn_cache=commit_crossattn_cache,
                        current_start=commit_current_start,
                    )
                    if is_real_prefix_commit:
                        require_finite(
                            f"teacher_trajectory_incremental_prefix_commit_output "
                            f"({debug_context} prefix_start={prefix_start})",
                            prefix_output,
                        )
                    if debug_timing:
                        prefix_chunk_seconds = debug_elapsed(prefix_commit_start_time, device)
                        prefix_commit_seconds += prefix_chunk_seconds
                        print_main(
                            debug_rank,
                            "[timing] "
                            f"global_step={debug_global_step} {debug_context} "
                            f"prefix_commit_start={prefix_start} "
                            f"prefix_slot={prefix_slot} "
                            f"real={int(is_real_prefix_commit)} "
                            f"chunk_frames={prefix_chunk.shape[1]} "
                            f"seconds={prefix_chunk_seconds:.3f}",
                            flush=True,
                        )
            current_forward_start = time.perf_counter()
            incremental_output = model(
                noisy_latents=state,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=prefix_frames * frame_seq_length,
            )
            require_finite(f"teacher_trajectory_incremental_output ({debug_context})", incremental_output)
            if debug_nonfinite_backward and incremental_output.requires_grad:
                def incremental_output_grad_hook(grad, context=debug_context):
                    if not torch.isfinite(grad).all().item():
                        print(
                            "[nonfinite-backward] "
                            f"rank={debug_rank} global_step={debug_global_step} "
                            f"tensor=incremental_output {context} {tensor_finite_summary('grad', grad)}",
                            flush=True,
                        )
                    return grad

                incremental_output.register_hook(incremental_output_grad_hook)
            if debug_timing:
                current_forward_seconds = debug_elapsed(current_forward_start, device)
                total_incremental_seconds = debug_elapsed(incremental_replay_start, device)
                print_main(
                    debug_rank,
                    "[timing] "
                    f"global_step={debug_global_step} {debug_context} "
                    f"local_attn_size={local_attn_size} "
                    f"committed_prefix_frames={committed_prefix_frames} "
                    f"prefix_commit_start={prefix_commit_start} "
                    f"cache_init_seconds={cache_init_seconds:.3f} "
                    f"prefix_commit_seconds={prefix_commit_seconds:.3f} "
                    f"current_forward_seconds={current_forward_seconds:.3f} "
                    f"incremental_replay_seconds={total_incremental_seconds:.3f}",
                    flush=True,
                )

        def add_supervised_components(
            output: torch.Tensor,
            *,
            loss_weight_multiplier: float,
            metric_prefix: str,
        ) -> None:
            if prediction_type == "flow":
                flow_prediction = output
                if debug_nonfinite_backward and flow_prediction.requires_grad:
                    def flow_prediction_grad_hook(grad, prefix=metric_prefix):
                        if not torch.isfinite(grad).all().item():
                            print(
                                "[nonfinite-backward] "
                                f"rank={debug_rank} global_step={debug_global_step} "
                                f"tensor={prefix}_flow_prediction {debug_context} "
                                f"{tensor_finite_summary('grad', grad)}",
                                flush=True,
                            )
                        return grad

                    flow_prediction.register_hook(flow_prediction_grad_hook)
                clean_prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, state, timestep)
            elif prediction_type == "clean_latent":
                clean_prediction = output
                flow_prediction = clean_latent_to_flow_prediction(scheduler, clean_prediction, state, timestep)
                if debug_nonfinite_backward and flow_prediction.requires_grad:
                    def converted_flow_prediction_grad_hook(grad, prefix=metric_prefix):
                        if not torch.isfinite(grad).all().item():
                            print(
                                "[nonfinite-backward] "
                                f"rank={debug_rank} global_step={debug_global_step} "
                                f"tensor={prefix}_converted_flow_prediction {debug_context} "
                                f"{tensor_finite_summary('grad', grad)}",
                                flush=True,
                            )
                        return grad

                    flow_prediction.register_hook(converted_flow_prediction_grad_hook)
            else:
                raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")
            if needs_flow_forward:
                flow_target = clean_latent_to_flow_prediction(scheduler, clean_target, state, timestep)
                require_finite(f"{metric_prefix}_flow_target ({debug_context})", flow_target)
                flow_diff = (flow_prediction.float() - flow_target.float()).square()
                valid_flow = (timestep > 0).reshape(*timestep.shape, 1, 1, 1).to(device=device, dtype=flow_diff.dtype)
                valid_flow_count = valid_flow.expand_as(flow_diff).sum()
                if valid_flow_count > 0:
                    flow_loss = (flow_diff * valid_flow).sum() / valid_flow_count.clamp_min(1.0)
                    require_finite(f"{metric_prefix}_flow_loss ({debug_context})", flow_loss)
                    flow_losses.append(flow_loss.detach())
                    if flow_loss_weight > 0 and loss_weight_multiplier > 0:
                        components.append(
                            flow_loss
                            * flow_loss_weight
                            * float(loss_weight_multiplier)
                            / max(len(active_step_indices), 1)
                        )
            if loss_type in ("clean_latent", "clean_latent_flow"):
                clean_loss = F.mse_loss(clean_prediction.float(), clean_target.float())
                require_finite(f"{metric_prefix}_clean_loss ({debug_context})", clean_loss)
                clean_losses.append(clean_loss.detach())
                if clean_latent_loss_weight > 0 and loss_weight_multiplier > 0:
                    components.append(
                        clean_loss
                        * clean_latent_loss_weight
                        * float(loss_weight_multiplier)
                        / max(len(active_step_indices), 1)
                    )

        if use_prefix_flow:
            assert model_output is not None
            prefix_loss_weight = teacher_trajectory_prefix_loss_weight if use_hybrid_flow else 1.0
            add_supervised_components(
                model_output,
                loss_weight_multiplier=prefix_loss_weight,
                metric_prefix="teacher_trajectory_prefix",
            )
        if use_incremental_kv_flow:
            assert incremental_output is not None
            incremental_loss_weight = teacher_trajectory_incremental_kv_loss_weight if use_hybrid_flow else 1.0
            add_supervised_components(
                incremental_output,
                loss_weight_multiplier=incremental_loss_weight,
                metric_prefix="teacher_trajectory_incremental_kv",
            )

        if needs_consistency_forward:
            assert incremental_output is not None
            assert model_output is not None
            consistency_loss = F.mse_loss(incremental_output.float(), model_output.detach().float())
            require_finite(f"teacher_trajectory_incremental_consistency_loss ({debug_context})", consistency_loss)
            consistency_losses.append(consistency_loss.detach())
            components.append(
                consistency_loss * incremental_kv_consistency_weight / max(len(active_step_indices), 1)
            )

    if not components:
        raise ValueError("At least one teacher trajectory loss component must be enabled")
    total_loss = sum(components)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(torch.stack(clean_losses).mean().detach().cpu().item()) if clean_losses else 0.0,
        "teacher_trajectory_flow_mse": float(torch.stack(flow_losses).mean().detach().cpu().item()) if flow_losses else 0.0,
    }
    metrics["causal_prefix_frames"] = float(prefix_frames)
    metrics["causal_committed_prefix_frames"] = (
        float(sum(committed_prefix_frame_counts) / len(committed_prefix_frame_counts))
        if committed_prefix_frame_counts
        else 0.0
    )
    metrics["teacher_trajectory_objective_incremental_kv_flow"] = float(use_incremental_kv_flow)
    metrics["teacher_trajectory_objective_hybrid_prefix_incremental_kv_flow"] = float(use_hybrid_flow)
    metrics["teacher_trajectory_prefix_loss_weight"] = float(
        teacher_trajectory_prefix_loss_weight if use_hybrid_flow else float(use_prefix_flow)
    )
    metrics["teacher_trajectory_incremental_kv_loss_weight"] = float(
        teacher_trajectory_incremental_kv_loss_weight if use_hybrid_flow else float(use_incremental_kv_flow)
    )
    metrics["teacher_trajectory_num_active_steps"] = float(len(active_step_indices))
    metrics["teacher_trajectory_num_contributing_steps"] = float(
        len(clean_losses) if loss_type == "clean_latent" else max(len(flow_losses), len(consistency_losses), len(clean_losses))
    )
    metrics["teacher_trajectory_step_mode_sample_one"] = float(step_mode == "sample_one")
    metrics["teacher_trajectory_step_mode_sequential"] = float(step_mode == "sequential_backward")
    if consistency_losses:
        metrics["incremental_kv_consistency_mse"] = float(
            torch.stack(consistency_losses).mean().detach().cpu().item()
        )
        metrics["incremental_kv_consistency_loss"] = (
            metrics["incremental_kv_consistency_mse"] * float(incremental_kv_consistency_weight)
        )
    return total_loss, metrics


def compute_stop_gradient_self_conditioning_losses(
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
    step_mode: str = "whole_graph",
    incremental_kv_context_noise: int = 0,
    self_conditioning_mix_ratio: float = 0.25,
    self_conditioning_consistency_weight: float = 0.0,
    self_conditioning_loss_on: str = "all_predicted_chunks",
    self_conditioning_intermediate_loss_weight: float = 0.5,
    debug_timing: bool = False,
    debug_rank: int = 0,
    debug_global_step: int = 0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if step_mode not in ("whole_graph", "sample_one", "sequential_backward"):
        raise ValueError("--teacher_trajectory_step_mode must be whole_graph, sample_one, or sequential_backward")
    if self_conditioning_loss_on not in ("all_predicted_chunks", "final_chunk_only"):
        raise ValueError("--self_conditioning_loss_on must be all_predicted_chunks or final_chunk_only")
    required_keys = (
        "self_conditioning_noisy_latents",
        "self_conditioning_clean_latents",
        "self_conditioning_timesteps",
        "self_conditioning_target_latents",
        "context_latents",
        "prompt_embeds",
    )
    missing = [key for key in required_keys if key not in batch or batch.get(key) is None]
    if missing:
        raise ValueError(f"stop-gradient self-conditioning requires grouped batch keys: {missing}")
    if not is_causal_wan_ar_head(model):
        raise ValueError("stop-gradient self-conditioning requires --head_type causal_wan_ar")

    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    noisy_states = batch["self_conditioning_noisy_latents"].to(device=device, dtype=dtype)
    clean_targets = batch["self_conditioning_clean_latents"].to(device=device, dtype=dtype)
    timesteps = batch["self_conditioning_timesteps"].to(device=device)
    target_latents = batch["self_conditioning_target_latents"].to(device=device, dtype=dtype)
    prefix_latents = batch["context_latents"].to(device=device, dtype=dtype)
    prompt_embeds = batch["prompt_embeds"].to(device=device, dtype=dtype)
    require_finite("self_conditioning_noisy_latents", noisy_states)
    require_finite("self_conditioning_clean_latents", clean_targets)
    require_finite("self_conditioning_target_latents", target_latents)
    require_finite("self_conditioning_context_latents", prefix_latents)
    require_finite("self_conditioning_prompt_embeds", prompt_embeds)
    if noisy_states.ndim != 7 or clean_targets.ndim != 7:
        raise ValueError("self-conditioning tensors must have shape [B, K, S, T, C, H, W]")
    if noisy_states.shape != clean_targets.shape:
        raise ValueError("self-conditioning noisy/clean tensor shape mismatch")
    batch_size, lookahead_chunks, num_steps, frames = noisy_states.shape[:4]
    if timesteps.shape[:3] != noisy_states.shape[:3]:
        raise ValueError("self-conditioning timesteps must have shape [B, K, S]")
    if target_latents.shape[:3] != (batch_size, lookahead_chunks, frames):
        raise ValueError("self-conditioning target latents must have shape [B, K, T, C, H, W]")

    active_step_indices = teacher_trajectory_active_step_indices(
        {"teacher_trajectory_noisy_latents": noisy_states[:, 0]},
        step_indices,
        sample_one=False,
    )
    active_step_indices = filter_teacher_trajectory_step_indices_for_loss(
        {"teacher_trajectory_timesteps": timesteps[:, 0]},
        active_step_indices,
        loss_type=loss_type,
        device=device,
    )
    if step_mode == "sample_one" and len(active_step_indices) > 1:
        active_step_indices = [distributed_sample_teacher_trajectory_step(active_step_indices, device)]
    if not active_step_indices:
        raise ValueError("No self-conditioning teacher trajectory steps can contribute to the selected loss")

    frame_seq_length = causal_wan_frame_seq_length(model, noisy_states[:, 0, 0])
    prefix_frames = int(prefix_latents.shape[1])
    inner_model = unwrap_model(model)
    generator_model = inner_model.generator.model if isinstance(inner_model, CausalWanARDraftHead) else None
    local_attn_size = int(getattr(generator_model, "local_attn_size", -1))
    if local_attn_size != -1:
        committed_prefix_frames = min(prefix_frames, max(local_attn_size - frames, 0))
    else:
        committed_prefix_frames = prefix_frames
    prefix_commit_start = max(0, prefix_frames - committed_prefix_frames)

    components: list[torch.Tensor] = []
    flow_losses = []
    clean_losses = []
    consistency_losses = []
    student_prefix_commits = 0
    target_prefix_commits = 0
    if self_conditioning_loss_on == "final_chunk_only":
        loss_weight_normalizer = 1.0
    else:
        loss_weight_normalizer = (
            float(max(lookahead_chunks - 1, 0)) * float(self_conditioning_intermediate_loss_weight)
            + 1.0
        )

    for step_index in active_step_indices:
        replay_start = time.perf_counter()
        kv_cache, crossattn_cache = initialize_causal_wan_ar_caches(
            model,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            total_frames=prefix_frames + lookahead_chunks * frames,
            frame_seq_length=frame_seq_length,
        )
        target_kv_cache = None
        target_crossattn_cache = None
        if self_conditioning_consistency_weight > 0:
            target_kv_cache, target_crossattn_cache = initialize_causal_wan_ar_caches(
                model,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                total_frames=prefix_frames + lookahead_chunks * frames,
                frame_seq_length=frame_seq_length,
            )
        if prefix_commit_start > 0:
            skipped_tokens = prefix_commit_start * frame_seq_length
            for cache in kv_cache:
                cache["global_end_index"].fill_(skipped_tokens)
                cache["local_end_index"].zero_()
            if target_kv_cache is not None:
                for cache in target_kv_cache:
                    cache["global_end_index"].fill_(skipped_tokens)
                    cache["local_end_index"].zero_()

        actual_prefix_starts = list(range(prefix_commit_start, prefix_frames, frames))
        num_prefix_commit_forwards = torch.tensor([len(actual_prefix_starts)], device=device, dtype=torch.int32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_prefix_commit_forwards, op=dist.ReduceOp.MAX)
        max_prefix_commit_forwards = int(num_prefix_commit_forwards.detach().cpu().item())
        dummy_kv_cache = None
        dummy_crossattn_cache = None
        dummy_target_kv_cache = None
        dummy_target_crossattn_cache = None
        with torch.no_grad():
            for prefix_slot in range(max_prefix_commit_forwards):
                is_real_prefix_commit = prefix_slot < len(actual_prefix_starts)
                if is_real_prefix_commit:
                    prefix_start = actual_prefix_starts[prefix_slot]
                    prefix_chunk = prefix_latents[:, prefix_start:prefix_start + frames]
                    commit_current_start = prefix_start * frame_seq_length
                    commit_targets = [(kv_cache, crossattn_cache)]
                    if target_kv_cache is not None and target_crossattn_cache is not None:
                        commit_targets.append((target_kv_cache, target_crossattn_cache))
                else:
                    prefix_start = -1
                    prefix_chunk = torch.zeros_like(noisy_states[:, 0, step_index])
                    commit_current_start = 0
                    if dummy_kv_cache is None or dummy_crossattn_cache is None:
                        dummy_kv_cache, dummy_crossattn_cache = initialize_causal_wan_ar_caches(
                            model,
                            batch_size=batch_size,
                            dtype=dtype,
                            device=device,
                            total_frames=frames,
                            frame_seq_length=frame_seq_length,
                        )
                    commit_targets = [(dummy_kv_cache, dummy_crossattn_cache)]
                    if target_kv_cache is not None and target_crossattn_cache is not None:
                        if dummy_target_kv_cache is None or dummy_target_crossattn_cache is None:
                            dummy_target_kv_cache, dummy_target_crossattn_cache = initialize_causal_wan_ar_caches(
                                model,
                                batch_size=batch_size,
                                dtype=dtype,
                                device=device,
                                total_frames=frames,
                                frame_seq_length=frame_seq_length,
                            )
                        commit_targets.append((dummy_target_kv_cache, dummy_target_crossattn_cache))
                prefix_timestep = torch.full(
                    (batch_size, prefix_chunk.shape[1]),
                    int(incremental_kv_context_noise),
                    device=device,
                    dtype=torch.long,
                )
                for commit_kv_cache, commit_crossattn_cache in commit_targets:
                    prefix_output = model(
                        noisy_latents=prefix_chunk,
                        prompt_embeds=prompt_embeds,
                        timestep=prefix_timestep,
                        kv_cache=commit_kv_cache,
                        crossattn_cache=commit_crossattn_cache,
                        current_start=commit_current_start,
                    )
                    if is_real_prefix_commit:
                        require_finite(
                            f"self_conditioning_prefix_commit_output "
                            f"(step={step_index} prefix_start={prefix_start})",
                            prefix_output,
                        )

        for chunk_offset in range(lookahead_chunks):
            state = noisy_states[:, chunk_offset, step_index]
            clean_target = clean_targets[:, chunk_offset, step_index]
            timestep = timesteps[:, chunk_offset, step_index].round().long().reshape(batch_size, 1).expand(
                batch_size,
                frames,
            )
            current_start = (prefix_frames + chunk_offset * frames) * frame_seq_length
            valid_flow_timestep = bool((timestep > 0).any().detach().cpu().item())
            needs_flow_forward = loss_type in ("flow", "clean_latent_flow") and valid_flow_timestep
            if not needs_flow_forward and loss_type not in ("clean_latent", "clean_latent_flow"):
                continue

            output = model(
                noisy_latents=state,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
            )
            require_finite(
                f"self_conditioning_output "
                f"(step={step_index} chunk_offset={chunk_offset} prefix_frames={prefix_frames})",
                output,
            )
            if prediction_type == "flow":
                flow_prediction = output
                clean_prediction = flow_prediction_to_clean_latent(scheduler, flow_prediction, state, timestep)
            elif prediction_type == "clean_latent":
                clean_prediction = output
                flow_prediction = clean_latent_to_flow_prediction(scheduler, clean_prediction, state, timestep)
            else:
                raise ValueError("--prediction_type must be 'flow' or 'clean_latent'")

            if self_conditioning_consistency_weight > 0:
                assert target_kv_cache is not None
                assert target_crossattn_cache is not None
                target_output = model(
                    noisy_latents=state,
                    prompt_embeds=prompt_embeds,
                    timestep=timestep,
                    kv_cache=target_kv_cache,
                    crossattn_cache=target_crossattn_cache,
                    current_start=current_start,
                )
                require_finite(
                    f"self_conditioning_target_prefix_output "
                    f"(step={step_index} chunk_offset={chunk_offset})",
                    target_output,
                )
                consistency_loss = F.mse_loss(output.float(), target_output.detach().float())
                require_finite(
                    f"self_conditioning_consistency_loss "
                    f"(step={step_index} chunk_offset={chunk_offset})",
                    consistency_loss,
                )
                consistency_losses.append(consistency_loss.detach())
                components.append(
                    consistency_loss
                    * float(self_conditioning_consistency_weight)
                    / max(len(active_step_indices), 1)
                    / max(lookahead_chunks, 1)
                )

            loss_weight = 1.0
            if self_conditioning_loss_on == "final_chunk_only" and chunk_offset != lookahead_chunks - 1:
                loss_weight = 0.0
            elif chunk_offset != lookahead_chunks - 1:
                loss_weight = float(self_conditioning_intermediate_loss_weight)

            if needs_flow_forward:
                flow_target = clean_latent_to_flow_prediction(scheduler, clean_target, state, timestep)
                require_finite(
                    f"self_conditioning_flow_target (step={step_index} chunk_offset={chunk_offset})",
                    flow_target,
                )
                flow_diff = (flow_prediction.float() - flow_target.float()).square()
                valid_flow = (timestep > 0).reshape(*timestep.shape, 1, 1, 1).to(device=device, dtype=flow_diff.dtype)
                valid_flow_count = valid_flow.expand_as(flow_diff).sum()
                if valid_flow_count > 0:
                    flow_loss = (flow_diff * valid_flow).sum() / valid_flow_count.clamp_min(1.0)
                    require_finite(
                        f"self_conditioning_flow_loss (step={step_index} chunk_offset={chunk_offset})",
                        flow_loss,
                    )
                    flow_losses.append(flow_loss.detach())
                    if flow_loss_weight > 0 and loss_weight > 0:
                        components.append(
                            flow_loss
                            * flow_loss_weight
                            * loss_weight
                            / max(len(active_step_indices), 1)
                            / max(loss_weight_normalizer, 1e-8)
                        )
            if loss_type in ("clean_latent", "clean_latent_flow"):
                clean_loss = F.mse_loss(clean_prediction.float(), clean_target.float())
                require_finite(
                    f"self_conditioning_clean_loss (step={step_index} chunk_offset={chunk_offset})",
                    clean_loss,
                )
                clean_losses.append(clean_loss.detach())
                if clean_latent_loss_weight > 0 and loss_weight > 0:
                    components.append(
                        clean_loss
                        * clean_latent_loss_weight
                        * loss_weight
                        / max(len(active_step_indices), 1)
                        / max(loss_weight_normalizer, 1e-8)
                    )

            if chunk_offset < lookahead_chunks - 1:
                use_student_prefix = random.random() < float(self_conditioning_mix_ratio)
                commit_latents = clean_prediction.detach() if use_student_prefix else target_latents[:, chunk_offset]
                if use_student_prefix:
                    student_prefix_commits += 1
                else:
                    target_prefix_commits += 1
                with torch.no_grad():
                    commit_timestep = torch.full(
                        (batch_size, frames),
                        int(incremental_kv_context_noise),
                        device=device,
                        dtype=torch.long,
                    )
                    commit_output = model(
                        noisy_latents=commit_latents,
                        prompt_embeds=prompt_embeds,
                        timestep=commit_timestep,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=current_start,
                    )
                    require_finite(
                        f"self_conditioning_detached_prefix_commit_output "
                        f"(step={step_index} chunk_offset={chunk_offset} student={int(use_student_prefix)})",
                        commit_output,
                    )
                    if target_kv_cache is not None and target_crossattn_cache is not None:
                        target_commit_output = model(
                            noisy_latents=target_latents[:, chunk_offset],
                            prompt_embeds=prompt_embeds,
                            timestep=commit_timestep,
                            kv_cache=target_kv_cache,
                            crossattn_cache=target_crossattn_cache,
                            current_start=current_start,
                        )
                        require_finite(
                            f"self_conditioning_target_prefix_commit_output "
                            f"(step={step_index} chunk_offset={chunk_offset})",
                            target_commit_output,
                        )

        if debug_timing:
            print_main(
                debug_rank,
                "[timing] "
                f"global_step={debug_global_step} "
                f"self_conditioning step={step_index} "
                f"anchor_index={batch_scalar(batch, 'self_conditioning_anchor_index')} "
                f"first_block={batch_scalar(batch, 'self_conditioning_first_block_index')} "
                f"lookahead_chunks={lookahead_chunks} "
                f"local_attn_size={local_attn_size} "
                f"prefix_frames={prefix_frames} "
                f"committed_prefix_frames={committed_prefix_frames} "
                f"seconds={debug_elapsed(replay_start, device):.3f}",
                flush=True,
            )

    if not components:
        raise ValueError("At least one self-conditioning loss component must be enabled")
    total_loss = sum(components)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "clean_latent_mse": float(torch.stack(clean_losses).mean().detach().cpu().item()) if clean_losses else 0.0,
        "teacher_trajectory_flow_mse": float(torch.stack(flow_losses).mean().detach().cpu().item()) if flow_losses else 0.0,
        "teacher_trajectory_objective_stop_gradient_self_conditioning_flow": 1.0,
        "self_conditioning_lookahead_chunks": float(lookahead_chunks),
        "self_conditioning_mix_ratio": float(self_conditioning_mix_ratio),
        "self_conditioning_student_prefix_commits": float(student_prefix_commits),
        "self_conditioning_target_prefix_commits": float(target_prefix_commits),
        "self_conditioning_anchor_index": float(batch_scalar(batch, "self_conditioning_anchor_index", 0)),
        "self_conditioning_first_block_index": float(batch_scalar(batch, "self_conditioning_first_block_index", 0)),
        "causal_prefix_frames": float(prefix_frames),
        "causal_committed_prefix_frames": float(committed_prefix_frames),
        "teacher_trajectory_num_active_steps": float(len(active_step_indices)),
        "teacher_trajectory_num_contributing_steps": float(max(len(flow_losses), len(clean_losses))),
        "teacher_trajectory_step_mode_sample_one": float(step_mode == "sample_one"),
        "teacher_trajectory_step_mode_sequential": float(step_mode == "sequential_backward"),
    }
    if consistency_losses:
        metrics["self_conditioning_consistency_mse"] = float(
            torch.stack(consistency_losses).mean().detach().cpu().item()
        )
        metrics["self_conditioning_consistency_loss"] = (
            metrics["self_conditioning_consistency_mse"] * float(self_conditioning_consistency_weight)
        )
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
        timeout_seconds = int(
            os.environ.get(
                "TORCH_DISTRIBUTED_TIMEOUT_SECONDS",
                os.environ.get("NCCL_TIMEOUT", "600"),
            )
        )
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout_seconds))
    return is_distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def print_main(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def sync_cuda_for_timing(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def debug_elapsed(start: float, device: torch.device) -> float:
    sync_cuda_for_timing(device)
    return time.perf_counter() - start


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
    parser.add_argument(
        "--teacher_trajectory_step_mode",
        choices=["whole_graph", "sample_one", "sequential_backward"],
        default="whole_graph",
        help=(
            "How teacher_trajectory step_indices are used. whole_graph backprops all selected steps together; "
            "sample_one randomly chooses one selected step per batch; sequential_backward backprops each selected step "
            "one at a time before a single optimizer step."
        ),
    )
    parser.add_argument(
        "--teacher_trajectory_objective",
        choices=[
            "prefix_flow",
            "incremental_kv_flow",
            "hybrid_prefix_incremental_kv_flow",
            "stop_gradient_self_conditioning_flow",
        ],
        default="prefix_flow",
        help=(
            "Teacher-trajectory objective. prefix_flow uses prefix teacher forcing; "
            "incremental_kv_flow commits prefix latents into a CausalWanAR KV cache and "
            "applies the supervised loss to the incremental current-chunk output; "
            "hybrid_prefix_incremental_kv_flow applies supervised losses to both paths; "
            "stop_gradient_self_conditioning_flow trains a short detached student-prefix rollout."
        ),
    )
    parser.add_argument(
        "--teacher_trajectory_prefix_loss_weight",
        type=float,
        default=1.0,
        help="Loss multiplier for the prefix-conditioned path when using hybrid_prefix_incremental_kv_flow.",
    )
    parser.add_argument(
        "--teacher_trajectory_incremental_kv_loss_weight",
        type=float,
        default=1.0,
        help="Loss multiplier for the incremental-KV path when using hybrid_prefix_incremental_kv_flow.",
    )
    parser.add_argument(
        "--incremental_kv_consistency_weight",
        type=float,
        default=0.0,
        help=(
            "Teacher-trajectory-only consistency weight between prefix teacher-forcing output and "
            "incremental KV replay output for causal_wan_ar."
        ),
    )
    parser.add_argument(
        "--incremental_kv_context_noise",
        type=int,
        default=0,
        help="Timestep used when committing clean prefix latents into the incremental drafter KV cache.",
    )
    parser.add_argument("--self_conditioning_lookahead_chunks", type=int, default=2)
    parser.add_argument(
        "--self_conditioning_anchor_policy",
        choices=["random_valid", "early_bias", "late_bias", "fixed"],
        default="random_valid",
    )
    parser.add_argument("--self_conditioning_fixed_anchor_index", type=int, default=0)
    parser.add_argument("--self_conditioning_mix_ratio", type=float, default=0.25)
    parser.add_argument("--self_conditioning_consistency_weight", type=float, default=0.0)
    parser.add_argument(
        "--self_conditioning_loss_on",
        choices=["all_predicted_chunks", "final_chunk_only"],
        default="all_predicted_chunks",
    )
    parser.add_argument("--self_conditioning_intermediate_loss_weight", type=float, default=0.5)
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
    parser.add_argument(
        "--causal_wan_prefix_padding",
        choices=["full", "none"],
        default="full",
        help="For causal_wan_ar prefix teacher forcing, pad to num_blocks*chunk_frames or use only prefix+current frames.",
    )
    parser.add_argument("--model_root", default="/mnt/lanxiangh/models")
    parser.add_argument("--config_path", default="configs/self_forcing_dmd.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--skip_nonfinite_grad",
        action="store_true",
        help="Skip optimizer steps with non-finite gradient norm instead of raising.",
    )
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--param_finite_check_every",
        type=int,
        default=1,
        help="Check model parameters for NaN/Inf every N optimizer steps; 0 disables.",
    )
    parser.add_argument(
        "--debug_timing_steps",
        type=int,
        default=0,
        help="Synchronize CUDA and print detailed timing for the first N global steps.",
    )
    parser.add_argument(
        "--debug_timing_all_ranks",
        action="store_true",
        help="Print debug timing from every distributed rank instead of rank 0 only.",
    )
    parser.add_argument(
        "--debug_nonfinite_backward",
        action="store_true",
        help="Print module/tensor hooks when backward first produces non-finite gradients.",
    )
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
    if (
        args.clean_latent_loss_weight < 0
        or args.flow_loss_weight < 0
        or args.dmd_loss_weight < 0
        or args.incremental_kv_consistency_weight < 0
        or args.teacher_trajectory_prefix_loss_weight < 0
        or args.teacher_trajectory_incremental_kv_loss_weight < 0
        or args.self_conditioning_consistency_weight < 0
        or args.self_conditioning_intermediate_loss_weight < 0
    ):
        raise ValueError("Loss weights must be non-negative")
    if args.self_conditioning_lookahead_chunks <= 0:
        raise ValueError("--self_conditioning_lookahead_chunks must be positive")
    if args.self_conditioning_lookahead_chunks >= args.num_blocks:
        raise ValueError("--self_conditioning_lookahead_chunks must be smaller than --num_blocks")
    if args.self_conditioning_mix_ratio < 0 or args.self_conditioning_mix_ratio > 1:
        raise ValueError("--self_conditioning_mix_ratio must be in [0, 1]")
    if args.incremental_kv_consistency_weight > 0:
        if args.training_mode != "teacher_trajectory":
            raise ValueError("--incremental_kv_consistency_weight is only supported with --training_mode teacher_trajectory")
        if args.head_type != "causal_wan_ar":
            raise ValueError("--incremental_kv_consistency_weight requires --head_type causal_wan_ar")
    if args.teacher_trajectory_objective in (
        "incremental_kv_flow",
        "hybrid_prefix_incremental_kv_flow",
        "stop_gradient_self_conditioning_flow",
    ):
        if args.training_mode != "teacher_trajectory":
            raise ValueError("--teacher_trajectory_objective incremental KV objectives require --training_mode teacher_trajectory")
        if args.head_type != "causal_wan_ar":
            raise ValueError("--teacher_trajectory_objective incremental KV objectives require --head_type causal_wan_ar")
        if args.incremental_kv_consistency_weight > 0:
            raise ValueError("--incremental_kv_consistency_weight is only used with prefix_flow objective")
    if (
        args.teacher_trajectory_objective == "stop_gradient_self_conditioning_flow"
        and args.teacher_trajectory_step_mode == "sequential_backward"
    ):
        raise ValueError("stop_gradient_self_conditioning_flow currently supports whole_graph or sample_one step mode")
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
    if args.teacher_trajectory_objective == "stop_gradient_self_conditioning_flow":
        dataset = StopGradientSelfConditioningDataset(
            dataset,
            num_blocks=args.num_blocks,
            lookahead_chunks=args.self_conditioning_lookahead_chunks,
            anchor_policy=args.self_conditioning_anchor_policy,
            fixed_anchor_index=args.self_conditioning_fixed_anchor_index,
        )

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
    debug_backward_state: dict[str, Any] = {
        "enabled": False,
        "global_step": 0,
        "reports": 0,
        "max_reports": 8,
    }
    debug_backward_handles = (
        register_nonfinite_backward_debug_hooks(raw_model, rank=rank, state=debug_backward_state)
        if args.debug_nonfinite_backward
        else []
    )
    if debug_backward_handles:
        print_main(rank, f"Registered {len(debug_backward_handles)} nonfinite backward debug hooks")
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
        f"teacher_trajectory_step_mode={args.teacher_trajectory_step_mode} "
        f"teacher_trajectory_objective={args.teacher_trajectory_objective} "
        f"teacher_trajectory_prefix_loss_weight={args.teacher_trajectory_prefix_loss_weight} "
        f"teacher_trajectory_incremental_kv_loss_weight={args.teacher_trajectory_incremental_kv_loss_weight} "
        f"self_conditioning_lookahead={args.self_conditioning_lookahead_chunks} "
        f"self_conditioning_anchor_policy={args.self_conditioning_anchor_policy} "
        f"self_conditioning_mix_ratio={args.self_conditioning_mix_ratio} "
        f"self_conditioning_consistency_weight={args.self_conditioning_consistency_weight} "
        f"causal_wan_prefix_padding={args.causal_wan_prefix_padding} "
        f"denoising_steps={args.denoising_step_list} prediction_type={args.prediction_type} loss_type={args.loss_type} "
        f"dmd_weight={args.dmd_loss_weight} incremental_kv_consistency_weight={args.incremental_kv_consistency_weight} "
        f"world_size={world_size} "
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
        skipped_batches = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False, disable=not is_main_process(rank))
        for batch in progress:
            step_start = time.perf_counter()
            debug_timing = args.debug_timing_steps > 0 and (global_step + skipped_batches) < args.debug_timing_steps
            debug_nonfinite_backward = (
                args.debug_nonfinite_backward
                and args.debug_timing_steps > 0
                and (global_step + skipped_batches) < args.debug_timing_steps
            )
            debug_backward_state["enabled"] = debug_nonfinite_backward
            debug_backward_state["global_step"] = global_step + 1
            debug_backward_state["reports"] = 0
            model.train()
            optimizer.zero_grad(set_to_none=True)
            batch_prepare_start = time.perf_counter()
            batch = attach_prompt_embeds_if_needed(
                batch,
                model=raw_model,
                text_encoder=prompt_text_encoder,
                device=device,
                dtype=next(raw_model.parameters()).dtype,
            )
            if args.head_type == "causal_wan_ar" and args.causal_wan_prefix_padding == "none":
                batch = dict(batch)
                batch["causal_wan_pad_to_frames"] = None
            if debug_timing:
                debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                debug_print(
                    f"[timing] global_step={global_step + 1} batch_prepare_seconds="
                    f"{debug_elapsed(batch_prepare_start, device):.3f} rank={rank}",
                    flush=True,
                )
            backward_already_done = False
            loss_forward_start = time.perf_counter()
            if args.training_mode == "teacher_trajectory" and args.teacher_trajectory_step_mode == "sequential_backward":
                active_step_indices = teacher_trajectory_active_step_indices(
                    batch,
                    args.teacher_trajectory_step_indices,
                    sample_one=False,
                )
                active_step_indices = filter_teacher_trajectory_step_indices_for_loss(
                    batch,
                    active_step_indices,
                    loss_type=args.loss_type,
                    device=next(model.parameters()).device,
                )
                if not active_step_indices:
                    raise ValueError("No teacher trajectory steps can contribute to the selected loss")
                step_metric_list = []
                scaled_loss_total = 0.0
                for teacher_step_index in active_step_indices:
                    with amp_context(device, args.amp_dtype):
                        step_loss_tensor, step_loss_metrics = compute_teacher_trajectory_draft_head_losses(
                            model,
                            batch,
                            num_blocks=args.num_blocks,
                            scheduler=scheduler,
                            prediction_type=args.prediction_type,
                            loss_type=args.loss_type,
                            clean_latent_loss_weight=args.clean_latent_loss_weight,
                            flow_loss_weight=args.flow_loss_weight,
                            step_indices=[teacher_step_index],
                            step_mode="sequential_backward",
                            teacher_trajectory_objective=args.teacher_trajectory_objective,
                            teacher_trajectory_prefix_loss_weight=args.teacher_trajectory_prefix_loss_weight,
                            teacher_trajectory_incremental_kv_loss_weight=(
                                args.teacher_trajectory_incremental_kv_loss_weight
                            ),
                            incremental_kv_consistency_weight=args.incremental_kv_consistency_weight,
                            incremental_kv_context_noise=args.incremental_kv_context_noise,
                            debug_timing=debug_timing,
                            debug_nonfinite_backward=debug_nonfinite_backward,
                            debug_rank=rank,
                            debug_global_step=global_step + 1,
                        )
                        scaled_step_loss = step_loss_tensor / max(len(active_step_indices), 1)
                    require_finite("loss_tensor", scaled_step_loss)
                    scaled_step_loss.backward()
                    scaled_loss_total += float(scaled_step_loss.detach().cpu().item())
                    step_metric_list.append(step_loss_metrics)
                loss_tensor = torch.tensor(scaled_loss_total, device=device, dtype=torch.float32)
                loss_metrics = merge_loss_metric_lists(step_metric_list, loss=scaled_loss_total)
                loss_metrics["teacher_trajectory_num_active_steps"] = float(len(active_step_indices))
                loss_metrics["teacher_trajectory_step_mode_sequential"] = 1.0
                backward_already_done = True
            else:
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
                        if args.teacher_trajectory_objective == "stop_gradient_self_conditioning_flow":
                            loss_tensor, loss_metrics = compute_stop_gradient_self_conditioning_losses(
                                model,
                                batch,
                                num_blocks=args.num_blocks,
                                scheduler=scheduler,
                                prediction_type=args.prediction_type,
                                loss_type=args.loss_type,
                                clean_latent_loss_weight=args.clean_latent_loss_weight,
                                flow_loss_weight=args.flow_loss_weight,
                                step_indices=args.teacher_trajectory_step_indices,
                                step_mode=args.teacher_trajectory_step_mode,
                                incremental_kv_context_noise=args.incremental_kv_context_noise,
                                self_conditioning_mix_ratio=args.self_conditioning_mix_ratio,
                                self_conditioning_consistency_weight=args.self_conditioning_consistency_weight,
                                self_conditioning_loss_on=args.self_conditioning_loss_on,
                                self_conditioning_intermediate_loss_weight=(
                                    args.self_conditioning_intermediate_loss_weight
                                ),
                                debug_timing=debug_timing,
                                debug_rank=rank,
                                debug_global_step=global_step + 1,
                            )
                        else:
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
                                step_mode=args.teacher_trajectory_step_mode,
                                teacher_trajectory_objective=args.teacher_trajectory_objective,
                                teacher_trajectory_prefix_loss_weight=args.teacher_trajectory_prefix_loss_weight,
                                teacher_trajectory_incremental_kv_loss_weight=(
                                    args.teacher_trajectory_incremental_kv_loss_weight
                                ),
                                incremental_kv_consistency_weight=args.incremental_kv_consistency_weight,
                                incremental_kv_context_noise=args.incremental_kv_context_noise,
                                debug_timing=debug_timing,
                                debug_nonfinite_backward=debug_nonfinite_backward,
                                debug_rank=rank,
                                debug_global_step=global_step + 1,
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
            if debug_timing:
                metric_summary = " ".join(
                    f"{key}={value:.6g}" if isinstance(value, float) else f"{key}={value}"
                    for key, value in sorted(loss_metrics.items())
                )
                debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                debug_print(
                    f"[timing] global_step={global_step + 1} loss_forward_seconds="
                    f"{debug_elapsed(loss_forward_start, device):.3f} "
                    f"rank={rank} loss={float(loss_tensor.detach().cpu().item()):.6g} {metric_summary}",
                    flush=True,
                )
            require_finite("loss_tensor", loss_tensor)
            if not backward_already_done:
                backward_start = time.perf_counter()
                loss_tensor.backward()
                if debug_timing:
                    debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                    debug_print(
                        f"[timing] global_step={global_step + 1} backward_seconds="
                        f"{debug_elapsed(backward_start, device):.3f} rank={rank}",
                        flush=True,
                    )
            if args.max_grad_norm > 0:
                grad_clip_start = time.perf_counter()
                grad_norm = None
                grad_clip_error = None
                try:
                    if is_fsdp_model(model):
                        grad_norm = model.clip_grad_norm_(args.max_grad_norm)
                        if not torch.isfinite(torch.as_tensor(grad_norm, device=device)):
                            raise RuntimeError(f"Non-finite FSDP grad norm: {grad_norm}")
                    else:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                except RuntimeError as exc:
                    grad_clip_error = exc
                    grad_norm = torch.tensor(float("nan"), device=device)
                if debug_timing:
                    debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                    debug_print(
                        f"[timing] global_step={global_step + 1} grad_clip_seconds="
                        f"{debug_elapsed(grad_clip_start, device):.3f} rank={rank} "
                        f"grad_norm={float(torch.as_tensor(grad_norm).detach().cpu().item())}",
                        flush=True,
                    )
                grad_norm_tensor = torch.as_tensor(grad_norm, device=device)
                grad_norm_is_finite = torch.isfinite(grad_norm_tensor).to(torch.int32)
                grad_norm_allreduce_start = time.perf_counter()
                if debug_timing:
                    debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                    debug_print(
                        f"[timing] global_step={global_step + 1} before_grad_finite_sync rank={rank} "
                        f"fsdp={int(is_fsdp_model(model))}",
                        flush=True,
                    )
                if dist.is_available() and dist.is_initialized() and not is_fsdp_model(model):
                    dist.all_reduce(grad_norm_is_finite, op=dist.ReduceOp.MIN)
                if debug_timing:
                    debug_print = print if args.debug_timing_all_ranks else lambda *a, **k: print_main(rank, *a, **k)
                    debug_print(
                        f"[timing] global_step={global_step + 1} grad_finite_allreduce_seconds="
                        f"{debug_elapsed(grad_norm_allreduce_start, device):.3f} "
                        f"rank={rank} all_ranks_grad_finite={int(grad_norm_is_finite.detach().cpu().item())} "
                        f"fsdp={int(is_fsdp_model(model))}",
                        flush=True,
                    )
                if not bool(grad_norm_is_finite.item()):
                    if not args.skip_nonfinite_grad:
                        if grad_clip_error is not None:
                            raise grad_clip_error
                        require_finite("grad_norm", grad_norm_tensor)
                    optimizer.zero_grad(set_to_none=True)
                    skipped_batches += 1
                    if is_main_process(rank):
                        print(
                            f"Skipping optimizer step {global_step + 1}: "
                            f"non-finite grad_norm={float(grad_norm_tensor.detach().cpu().item())}; "
                            "lower LR, lower local attention, or fewer timestep indices may be needed",
                            flush=True,
                    )
                    continue
            optimizer_start = time.perf_counter()
            if debug_timing:
                print_main(
                    rank,
                    f"[timing] global_step={global_step + 1} optimizer_step_start",
                    flush=True,
                )
            optimizer.step()
            if debug_timing:
                print_main(
                    rank,
                    f"[timing] global_step={global_step + 1} optimizer_step_seconds="
                    f"{debug_elapsed(optimizer_start, device):.3f}",
                    flush=True,
                )
            if args.param_finite_check_every > 0 and (global_step + 1) % args.param_finite_check_every == 0:
                finite_check_start = time.perf_counter()
                require_model_parameters_finite(model, context=f"after_optimizer_step_{global_step + 1}")
                if debug_timing:
                    print_main(
                        rank,
                        f"[timing] global_step={global_step + 1} finite_check_seconds="
                        f"{debug_elapsed(finite_check_start, device):.3f}",
                        flush=True,
                    )
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
                    "teacher_trajectory_step_mode": args.teacher_trajectory_step_mode,
                    "teacher_trajectory_objective": args.teacher_trajectory_objective,
                    "teacher_trajectory_prefix_loss_weight": args.teacher_trajectory_prefix_loss_weight,
                    "teacher_trajectory_incremental_kv_loss_weight": args.teacher_trajectory_incremental_kv_loss_weight,
                    "training_mode": args.training_mode,
                    "unroll_step_weights": unroll_step_weights,
                    "unroll_noise_mode": args.unroll_noise_mode,
                    "amp_dtype": args.amp_dtype,
                    "gradient_checkpointing": args.gradient_checkpointing,
                    "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                    "causal_wan_prefix_padding": args.causal_wan_prefix_padding,
                    "copied_wan_frozen": copied_wan_frozen,
                    "clean_latent_loss_weight": args.clean_latent_loss_weight,
                    "flow_loss_weight": args.flow_loss_weight,
                    "dmd_loss_weight": args.dmd_loss_weight,
                    "incremental_kv_consistency_weight": args.incremental_kv_consistency_weight,
                    "incremental_kv_context_noise": args.incremental_kv_context_noise,
                    "self_conditioning_lookahead_chunks": args.self_conditioning_lookahead_chunks,
                    "self_conditioning_anchor_policy": args.self_conditioning_anchor_policy,
                    "self_conditioning_fixed_anchor_index": args.self_conditioning_fixed_anchor_index,
                    "self_conditioning_mix_ratio": args.self_conditioning_mix_ratio,
                    "self_conditioning_consistency_weight": args.self_conditioning_consistency_weight,
                    "self_conditioning_loss_on": args.self_conditioning_loss_on,
                    "self_conditioning_intermediate_loss_weight": args.self_conditioning_intermediate_loss_weight,
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
                "teacher_trajectory_step_mode": args.teacher_trajectory_step_mode,
                "teacher_trajectory_objective": args.teacher_trajectory_objective,
                "teacher_trajectory_prefix_loss_weight": args.teacher_trajectory_prefix_loss_weight,
                "teacher_trajectory_incremental_kv_loss_weight": args.teacher_trajectory_incremental_kv_loss_weight,
                "training_mode": args.training_mode,
                "unroll_step_weights": unroll_step_weights,
                "unroll_noise_mode": args.unroll_noise_mode,
                "amp_dtype": args.amp_dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                "causal_wan_prefix_padding": args.causal_wan_prefix_padding,
                "copied_wan_frozen": copied_wan_frozen,
                "clean_latent_loss_weight": args.clean_latent_loss_weight,
                "flow_loss_weight": args.flow_loss_weight,
                "dmd_loss_weight": args.dmd_loss_weight,
                "incremental_kv_consistency_weight": args.incremental_kv_consistency_weight,
                "incremental_kv_context_noise": args.incremental_kv_context_noise,
                "self_conditioning_lookahead_chunks": args.self_conditioning_lookahead_chunks,
                "self_conditioning_anchor_policy": args.self_conditioning_anchor_policy,
                "self_conditioning_fixed_anchor_index": args.self_conditioning_fixed_anchor_index,
                "self_conditioning_mix_ratio": args.self_conditioning_mix_ratio,
                "self_conditioning_consistency_weight": args.self_conditioning_consistency_weight,
                "self_conditioning_loss_on": args.self_conditioning_loss_on,
                "self_conditioning_intermediate_loss_weight": args.self_conditioning_intermediate_loss_weight,
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
                "teacher_trajectory_step_mode": args.teacher_trajectory_step_mode,
                "teacher_trajectory_objective": args.teacher_trajectory_objective,
                "teacher_trajectory_prefix_loss_weight": args.teacher_trajectory_prefix_loss_weight,
                "teacher_trajectory_incremental_kv_loss_weight": args.teacher_trajectory_incremental_kv_loss_weight,
                "training_mode": args.training_mode,
                "unroll_step_weights": unroll_step_weights,
                "unroll_noise_mode": args.unroll_noise_mode,
                "amp_dtype": args.amp_dtype,
                "gradient_checkpointing": args.gradient_checkpointing,
                "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
                "causal_wan_prefix_padding": args.causal_wan_prefix_padding,
                "copied_wan_frozen": copied_wan_frozen,
                "clean_latent_loss_weight": args.clean_latent_loss_weight,
                "flow_loss_weight": args.flow_loss_weight,
                "dmd_loss_weight": args.dmd_loss_weight,
                "incremental_kv_consistency_weight": args.incremental_kv_consistency_weight,
                "incremental_kv_context_noise": args.incremental_kv_context_noise,
                "self_conditioning_lookahead_chunks": args.self_conditioning_lookahead_chunks,
                "self_conditioning_anchor_policy": args.self_conditioning_anchor_policy,
                "self_conditioning_fixed_anchor_index": args.self_conditioning_fixed_anchor_index,
                "self_conditioning_mix_ratio": args.self_conditioning_mix_ratio,
                "self_conditioning_consistency_weight": args.self_conditioning_consistency_weight,
                "self_conditioning_loss_on": args.self_conditioning_loss_on,
                "self_conditioning_intermediate_loss_weight": args.self_conditioning_intermediate_loss_weight,
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
        "teacher_trajectory_step_mode": args.teacher_trajectory_step_mode,
        "teacher_trajectory_objective": args.teacher_trajectory_objective,
        "teacher_trajectory_prefix_loss_weight": args.teacher_trajectory_prefix_loss_weight,
        "teacher_trajectory_incremental_kv_loss_weight": args.teacher_trajectory_incremental_kv_loss_weight,
        "training_mode": args.training_mode,
        "unroll_step_weights": unroll_step_weights,
        "unroll_noise_mode": args.unroll_noise_mode,
        "amp_dtype": args.amp_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "freeze_copied_wan_epochs": args.freeze_copied_wan_epochs,
        "causal_wan_prefix_padding": args.causal_wan_prefix_padding,
        "clean_latent_loss_weight": args.clean_latent_loss_weight,
        "flow_loss_weight": args.flow_loss_weight,
        "dmd_loss_weight": args.dmd_loss_weight,
        "incremental_kv_consistency_weight": args.incremental_kv_consistency_weight,
        "incremental_kv_context_noise": args.incremental_kv_context_noise,
        "self_conditioning_lookahead_chunks": args.self_conditioning_lookahead_chunks,
        "self_conditioning_anchor_policy": args.self_conditioning_anchor_policy,
        "self_conditioning_fixed_anchor_index": args.self_conditioning_fixed_anchor_index,
        "self_conditioning_mix_ratio": args.self_conditioning_mix_ratio,
        "self_conditioning_consistency_weight": args.self_conditioning_consistency_weight,
        "self_conditioning_loss_on": args.self_conditioning_loss_on,
        "self_conditioning_intermediate_loss_weight": args.self_conditioning_intermediate_loss_weight,
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
