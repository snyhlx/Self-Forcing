from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy


def is_fsdp_model(model: nn.Module) -> bool:
    return isinstance(model, FSDP)


def unwrap_model(model: nn.Module) -> nn.Module:
    if hasattr(model, "module"):
        return model.module
    return model


def wrap_model_for_training(
    model: nn.Module,
    *,
    strategy: str,
    is_distributed: bool,
    local_rank: int,
    find_unused_parameters: bool = False,
    fsdp_min_num_params: int = 100_000_000,
    fsdp_mixed_precision: str = "none",
) -> nn.Module:
    if strategy == "none":
        return model
    if strategy == "ddp":
        from torch.nn.parallel import DistributedDataParallel

        if not is_distributed:
            return model
        return DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused_parameters,
        )
    if strategy != "fsdp":
        raise ValueError(f"Unsupported parallel strategy: {strategy}")
    if not is_distributed:
        raise ValueError("FSDP requires torch.distributed; launch with torchrun and NUM_GPUS > 1")
    if find_unused_parameters:
        raise ValueError("FSDP does not support DDP-style find_unused_parameters")

    mixed_precision = None
    if fsdp_mixed_precision != "none":
        dtype = torch.bfloat16 if fsdp_mixed_precision == "bf16" else torch.float16
        mixed_precision = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    auto_wrap_policy = None
    if fsdp_min_num_params > 0:
        auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=fsdp_min_num_params)

    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mixed_precision,
        device_id=torch.device("cuda", local_rank),
        use_orig_params=True,
    )


def fsdp_rank0_state_dict(model: nn.Module) -> dict[str, torch.Tensor] | None:
    """Collect a full CPU state dict on rank 0 for FSDP; returns None on other ranks."""
    if not is_fsdp_model(model):
        return unwrap_model(model).state_dict()
    is_rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

    # FULL_STATE_DICT hooks can trip PyTorch's nested-FSDP root-state assertion
    # for some auto-wrapped/use_orig_params models. summon_full_params gathers
    # the same full parameters without walking the FSDP state_dict hook stack.
    with FSDP.summon_full_params(
        model,
        recurse=True,
        writeback=False,
        rank0_only=True,
        offload_to_cpu=True,
    ):
        if not is_rank0:
            return None
        state_dict = {
            key: value.detach().cpu().clone()
            for key, value in unwrap_model(model).state_dict().items()
        }
    return state_dict


def rank0_save_with_state_dict(
    *,
    model: nn.Module,
    unwrapped_model: nn.Module,
    path: str | Path,
    save_fn: Callable[..., Path],
    save_kwargs: dict[str, Any],
) -> Path | None:
    """Save in a checkpoint format owned by save_fn while supporting FSDP."""
    state_dict = fsdp_rank0_state_dict(model)
    is_rank0 = not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0
    if not is_rank0:
        return None
    return save_fn(unwrapped_model, path, state_dict=state_dict, **save_kwargs)


def maybe_no_sync(model: nn.Module):
    return nullcontext()
