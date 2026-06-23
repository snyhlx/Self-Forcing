from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset

from utils.wan_wrapper import WanDiffusionWrapper


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=position.device, dtype=torch.float32) / half
    )
    args = position.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, embedding.new_zeros(*embedding.shape[:-1], 1)], dim=-1)
    return embedding


def rope_params(max_seq_len: int, dim: int, theta: int = 10000) -> torch.Tensor:
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2, dtype=torch.float32).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def causal_rope_apply(x: torch.Tensor, grid_sizes: torch.Tensor, freqs: torch.Tensor, start_frame: int = 0) -> torch.Tensor:
    n, c = x.size(2), x.size(3) // 2
    split_freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    output = []
    for i, (frames, height, width) in enumerate(grid_sizes.tolist()):
        seq_len = frames * height * width
        if seq_len == 0:
            output.append(x[i])
            continue
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                split_freqs[0][start_frame:start_frame + frames].view(frames, 1, 1, -1).expand(frames, height, width, -1),
                split_freqs[1][:height].view(1, height, 1, -1).expand(frames, height, width, -1),
                split_freqs[2][:width].view(1, 1, width, -1).expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        output.append(x_i)
    return torch.stack(output).type_as(x)


class WanRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)).type_as(x) * self.weight


class WanLayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x).type_as(x)


class DraftCausalHead(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: tuple[int, int, int], eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, math.prod(patch_size) * out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])


@dataclass(frozen=True)
class FeatureCaptureConfig:
    layer_names: tuple[str, ...]
    detach: bool = True
    clone: bool = False


def first_tensor(value: Any) -> torch.Tensor:
    """Return the first tensor in a nested module output structure."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                continue
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"Could not find a tensor in output of type {type(value).__name__}")


class TargetFeatureCapture:
    """Context manager for capturing selected target-model layer features.

    This intentionally uses public PyTorch forward hooks and stores outputs by
    module name. It is small enough to test without loading Wan/Krea, while
    still matching the hook path we will use for selected transformer blocks.
    """

    def __init__(self, model: nn.Module, config: FeatureCaptureConfig):
        self.model = model
        self.config = config
        self.features: dict[str, list[torch.Tensor]] = {name: [] for name in config.layer_names}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> TargetFeatureCapture:
        modules = dict(self.model.named_modules())
        missing = [name for name in self.config.layer_names if name not in modules]
        if missing:
            available_preview = ", ".join(list(modules.keys())[:20])
            raise ValueError(
                f"Unknown feature capture layer(s): {missing}. "
                f"Available module names start with: {available_preview}"
            )

        for name in self.config.layer_names:
            self._handles.append(modules[name].register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            feature = first_tensor(output)
            if self.config.detach:
                feature = feature.detach()
            if self.config.clone:
                feature = feature.clone()
            self.features[name].append(feature)

        return hook

    def clear(self) -> None:
        for captured in self.features.values():
            captured.clear()

    def latest(self) -> dict[str, torch.Tensor]:
        missing = [name for name, values in self.features.items() if not values]
        if missing:
            raise RuntimeError(f"No captured feature for layer(s): {missing}")
        return {name: values[-1] for name, values in self.features.items()}


def _cpu_tensor(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu").contiguous()


def prepare_target_features_for_storage(
    target_features: dict[str, torch.Tensor],
    storage: str = "pooled",
) -> dict[str, torch.Tensor]:
    if storage == "full":
        return {
            name: _cpu_tensor(feature)
            for name, feature in sorted(target_features.items())
        }
    if storage == "pooled":
        pooled = {}
        for name, feature in sorted(target_features.items()):
            if feature.ndim < 2:
                raise ValueError(f"Feature {name} must have at least 2 dimensions, got {feature.ndim}")
            if feature.ndim == 2:
                pooled[name] = _cpu_tensor(feature)
            else:
                pooled[name] = _cpu_tensor(feature.float().reshape(feature.shape[0], -1, feature.shape[-1]).mean(dim=1))
        return pooled
    raise ValueError(f"Unsupported target feature storage mode: {storage}")


def prepare_target_kv_cache_for_storage(
    target_kv_cache: dict[str, dict[str, torch.Tensor]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        name: {
            "k": _cpu_tensor(cache["k"]),
            "v": _cpu_tensor(cache["v"]),
        }
        for name, cache in sorted(target_kv_cache.items())
    }


def make_draft_head_record(
    *,
    prompt: str,
    prompt_index: int | None,
    block_index: int,
    block_noise: torch.Tensor,
    target_latents: torch.Tensor,
    target_features: dict[str, torch.Tensor],
    target_kv_cache: dict[str, dict[str, torch.Tensor]] | None = None,
    context_latents: torch.Tensor | None = None,
    draft_latents: torch.Tensor | None = None,
    prompt_embeds: torch.Tensor | None = None,
    teacher_trajectory_latents: torch.Tensor | None = None,
    teacher_trajectory_noisy_latents: torch.Tensor | None = None,
    teacher_trajectory_timesteps: torch.Tensor | None = None,
    delta: float | None = None,
    target_feature_storage: str = "pooled",
) -> dict[str, Any]:
    if block_noise.shape != target_latents.shape:
        raise ValueError(
            f"block_noise and target_latents must have the same shape, "
            f"got {tuple(block_noise.shape)} and {tuple(target_latents.shape)}"
        )
    if not target_features and target_kv_cache is None and prompt_embeds is None:
        raise ValueError("target_features must not be empty unless target_kv_cache or prompt_embeds is provided")

    return {
        "prompt": prompt,
        "prompt_index": prompt_index,
        "block_index": int(block_index),
        "block_noise": _cpu_tensor(block_noise),
        "target_latents": _cpu_tensor(target_latents),
        "target_features": prepare_target_features_for_storage(target_features, target_feature_storage),
        "target_kv_cache": prepare_target_kv_cache_for_storage(target_kv_cache) if target_kv_cache is not None else None,
        "context_latents": _cpu_tensor(context_latents) if context_latents is not None else None,
        "draft_latents": _cpu_tensor(draft_latents) if draft_latents is not None else None,
        "prompt_embeds": _cpu_tensor(prompt_embeds) if prompt_embeds is not None else None,
        "teacher_trajectory_latents": _cpu_tensor(teacher_trajectory_latents) if teacher_trajectory_latents is not None else None,
        "teacher_trajectory_noisy_latents": _cpu_tensor(teacher_trajectory_noisy_latents) if teacher_trajectory_noisy_latents is not None else None,
        "teacher_trajectory_timesteps": _cpu_tensor(teacher_trajectory_timesteps) if teacher_trajectory_timesteps is not None else None,
        "delta": float(delta) if delta is not None else None,
    }


class DraftHeadDatasetWriter:
    """Write draft-head supervision records as small torch shards."""

    def __init__(self, output_dir: str | Path, shard_size: int = 128):
        if shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.records: list[dict[str, Any]] = []
        self.shards: list[dict[str, Any]] = []
        self.total_records = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        self.total_records += 1
        if len(self.records) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.records:
            return
        shard_index = len(self.shards)
        shard_name = f"shard_{shard_index:06d}.pt"
        shard_path = self.output_dir / shard_name
        torch.save({"records": self.records}, shard_path)
        self.shards.append({"path": shard_name, "num_records": len(self.records)})
        self.records = []

    def close(self) -> Path:
        self.flush()
        manifest = {
            "format": "sdvg_draft_head_v1",
            "num_records": self.total_records,
            "shard_size": self.shard_size,
            "shards": self.shards,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path


def load_draft_head_records(manifest_path: str | Path) -> list[dict[str, Any]]:
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "sdvg_draft_head_v1":
        raise ValueError(f"Unsupported draft-head dataset format: {manifest.get('format')}")

    records = []
    for shard in manifest["shards"]:
        payload = torch.load(manifest_path.parent / shard["path"], map_location="cpu", weights_only=False)
        shard_records = payload["records"]
        if len(shard_records) != shard["num_records"]:
            raise ValueError(
                f"Shard {shard['path']} expected {shard['num_records']} records, "
                f"found {len(shard_records)}"
            )
        records.extend(shard_records)
    if len(records) != manifest["num_records"]:
        raise ValueError(f"Manifest expected {manifest['num_records']} records, found {len(records)}")
    return records


class DraftHeadRecordDataset(Dataset):
    def __init__(self, manifest_path: str | Path):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("format") != "sdvg_draft_head_v1":
            raise ValueError(f"Unsupported draft-head dataset format: {self.manifest.get('format')}")
        self.index: list[tuple[Path, int]] = []
        for shard in self.manifest["shards"]:
            shard_path = self.manifest_path.parent / shard["path"]
            self.index.extend((shard_path, offset) for offset in range(shard["num_records"]))
        if len(self.index) != self.manifest["num_records"]:
            raise ValueError(f"Manifest expected {self.manifest['num_records']} records, indexed {len(self.index)}")
        self._cached_shard_path: Path | None = None
        self._cached_records: list[dict[str, Any]] | None = None

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        shard_path, record_offset = self.index[index]
        if self._cached_shard_path != shard_path or self._cached_records is None:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            self._cached_records = payload["records"]
            self._cached_shard_path = shard_path
        return self._cached_records[record_offset]


def pool_target_features(
    target_features: dict[str, torch.Tensor],
    layer_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    if not target_features:
        raise ValueError("target_features must not be empty")
    selected_names = layer_names or tuple(sorted(target_features))
    pooled = []
    batch_size = None
    for name in selected_names:
        if name not in target_features:
            raise ValueError(f"Missing target feature layer: {name}")
        feature = target_features[name].float()
        if feature.ndim < 2:
            raise ValueError(f"Feature {name} must have at least 2 dimensions, got {feature.ndim}")
        if batch_size is None:
            batch_size = feature.shape[0]
        elif feature.shape[0] != batch_size:
            raise ValueError("All target features must have the same batch size")

        if feature.ndim == 2:
            pooled.append(feature)
        else:
            pooled.append(feature.reshape(feature.shape[0], -1, feature.shape[-1]).mean(dim=1))
    return torch.cat(pooled, dim=-1)


def _select_token_subset(feature: torch.Tensor, max_tokens: int) -> torch.Tensor:
    if feature.ndim == 2:
        feature = feature.unsqueeze(1)
    elif feature.ndim > 3:
        feature = feature.reshape(feature.shape[0], -1, feature.shape[-1])
    if feature.ndim != 3:
        raise ValueError(f"Expected feature tokens with shape [B, N, D], got {tuple(feature.shape)}")
    if max_tokens <= 0 or feature.shape[1] <= max_tokens:
        return feature
    indices = torch.linspace(
        0,
        feature.shape[1] - 1,
        steps=max_tokens,
        device=feature.device,
    ).round().long()
    return feature.index_select(1, indices)


def infer_target_feature_dims(
    target_features: dict[str, torch.Tensor],
    layer_names: tuple[str, ...],
) -> dict[str, int]:
    dims = {}
    for name in layer_names:
        if name not in target_features:
            raise ValueError(f"Missing target feature layer: {name}")
        feature = target_features[name]
        if feature.ndim < 2:
            raise ValueError(f"Feature {name} must have at least 2 dimensions, got {feature.ndim}")
        dims[name] = int(feature.shape[-1])
    return dims


class TargetFeatureFuser(nn.Module):
    """Fuse full target-model token features into compact draft-context tokens."""

    def __init__(
        self,
        layer_feature_dims: dict[str, int],
        hidden_channels: int,
        max_context_tokens: int = 512,
    ):
        super().__init__()
        if not layer_feature_dims:
            raise ValueError("layer_feature_dims must not be empty")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        self.layer_names = tuple(sorted(layer_feature_dims))
        self.layer_feature_dims = {name: int(layer_feature_dims[name]) for name in self.layer_names}
        self.hidden_channels = hidden_channels
        self.max_context_tokens = max_context_tokens
        self.norms = nn.ModuleList(
            [nn.LayerNorm(self.layer_feature_dims[name]) for name in self.layer_names]
        )
        self.projections = nn.ModuleList(
            [nn.Linear(self.layer_feature_dims[name], hidden_channels) for name in self.layer_names]
        )
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_channels * len(self.layer_names)),
            nn.Linear(hidden_channels * len(self.layer_names), hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )

    def forward(
        self,
        target_features: dict[str, torch.Tensor],
        layer_names: tuple[str, ...] | None = None,
    ) -> torch.Tensor:
        selected_names = layer_names or self.layer_names
        if tuple(selected_names) != self.layer_names:
            raise ValueError(f"Expected layer_names={self.layer_names}, got {tuple(selected_names)}")

        projected = []
        token_count = None
        for index, name in enumerate(self.layer_names):
            if name not in target_features:
                raise ValueError(f"Missing target feature layer: {name}")
            feature = target_features[name].to(
                device=self.projections[index].weight.device,
                dtype=self.projections[index].weight.dtype,
            )
            feature = _select_token_subset(feature, self.max_context_tokens)
            if token_count is None:
                token_count = feature.shape[1]
            elif feature.shape[1] != token_count:
                raise ValueError("All selected target features must have the same token count")
            projected.append(self.projections[index](self.norms[index](feature)))
        return self.fuse(torch.cat(projected, dim=-1))


class KVInjectedDraftBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_mult: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_channels * ffn_mult
        self.self_norm = nn.LayerNorm(hidden_channels)
        self.self_attn = nn.MultiheadAttention(
            hidden_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(hidden_channels)
        self.context_attn = nn.MultiheadAttention(
            hidden_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, hidden_channels),
        )

    def forward(self, x: torch.Tensor, context_tokens: torch.Tensor) -> torch.Tensor:
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + y
        y, _ = self.context_attn(self.context_norm(x), context_tokens, context_tokens, need_weights=False)
        x = x + y
        return x + self.ffn(self.ffn_norm(x))


class KVInjectedLatentDraftHead(nn.Module):
    """Attention draft head with DFlash-style target-feature K/V conditioning."""

    def __init__(
        self,
        latent_channels: int,
        layer_feature_dims: dict[str, int],
        hidden_channels: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_context_tokens: int = 512,
        latent_pool: tuple[int, int, int] = (1, 4, 4),
        ffn_mult: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        self.latent_channels = latent_channels
        self.layer_feature_dims = {name: int(dim) for name, dim in sorted(layer_feature_dims.items())}
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_context_tokens = max_context_tokens
        self.latent_pool = tuple(int(x) for x in latent_pool)
        self.ffn_mult = ffn_mult
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_channels * ffn_mult
        self.dropout = dropout
        self.head_type = "kv_injected_attention"
        self.gradient_checkpointing = False

        self.feature_fuser = TargetFeatureFuser(
            self.layer_feature_dims,
            hidden_channels=hidden_channels,
            max_context_tokens=max_context_tokens,
        )
        self.latent_in = nn.Conv3d(latent_channels, hidden_channels, kernel_size=1)
        self.block_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.blocks = nn.ModuleList(
            [
                KVInjectedDraftBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.latent_out = nn.Conv3d(hidden_channels, latent_channels, kernel_size=1)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    @property
    def feature_dim(self) -> int:
        return sum(self.layer_feature_dims.values())

    def _block_index_tensor(
        self,
        block_index: torch.Tensor | int,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        if isinstance(block_index, int):
            return torch.full((batch_size, 1), float(block_index) / float(num_blocks), device=device, dtype=dtype)
        return block_index.to(device=device, dtype=dtype).reshape(batch_size, 1) / float(num_blocks)

    def _timestep_tensor(
        self,
        timestep: torch.Tensor | int | float | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if timestep is None:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if isinstance(timestep, (int, float)):
            return torch.full((batch_size, 1), float(timestep) / 1000.0, device=device, dtype=dtype)
        timestep = timestep.to(device=device, dtype=dtype)
        if timestep.ndim == 2:
            timestep = timestep[:, :1]
        return timestep.reshape(batch_size, 1) / 1000.0

    def forward(
        self,
        block_latents: torch.Tensor,
        target_features: dict[str, torch.Tensor],
        block_index: torch.Tensor | int,
        num_blocks: int,
        timestep: torch.Tensor | int | float | None = None,
    ) -> torch.Tensor:
        if block_latents.ndim != 5:
            raise ValueError(f"block_latents must have shape [B, T, C, H, W], got {tuple(block_latents.shape)}")
        if block_latents.shape[2] != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {block_latents.shape[2]}")

        batch_size, frames, _channels, height, width = block_latents.shape
        x = block_latents.permute(0, 2, 1, 3, 4)
        pooled = F.avg_pool3d(x, kernel_size=self.latent_pool, stride=self.latent_pool, ceil_mode=True)
        pooled_shape = pooled.shape[-3:]
        tokens = self.latent_in(pooled).flatten(2).transpose(1, 2)
        block_pos = self._block_index_tensor(
            block_index,
            batch_size,
            num_blocks,
            block_latents.device,
            tokens.dtype,
        )
        timestep_pos = self._timestep_tensor(
            timestep,
            batch_size,
            block_latents.device,
            tokens.dtype,
        )
        tokens = tokens + self.block_embed(block_pos).unsqueeze(1) + self.timestep_embed(timestep_pos).unsqueeze(1)
        context_tokens = self.feature_fuser(target_features).to(device=tokens.device, dtype=tokens.dtype)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint.checkpoint(block, tokens, context_tokens, use_reentrant=False)
            else:
                tokens = block(tokens, context_tokens)
        low_res = self.out_norm(tokens).transpose(1, 2).reshape(
            batch_size,
            self.hidden_channels,
            *pooled_shape,
        )
        prediction = self.latent_out(low_res)
        prediction = F.interpolate(prediction.float(), size=(frames, height, width), mode="trilinear", align_corners=False)
        prediction = prediction.to(dtype=block_latents.dtype).permute(0, 2, 1, 3, 4)
        return prediction


class KVCacheDraftBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_mult: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_channels * ffn_mult
        self.self_norm = nn.LayerNorm(hidden_channels)
        self.self_attn = nn.MultiheadAttention(
            hidden_channels,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.kv_query_norm = nn.LayerNorm(hidden_channels)
        self.kv_q = nn.Linear(hidden_channels, hidden_channels)
        self.kv_out = nn.Linear(hidden_channels, hidden_channels)
        self.ffn_norm = nn.LayerNorm(hidden_channels)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, self.ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, hidden_channels),
        )

    def forward(self, x: torch.Tensor, target_k: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
        y, _ = self.self_attn(self.self_norm(x), self.self_norm(x), self.self_norm(x), need_weights=False)
        x = x + y
        batch_size, num_tokens = x.shape[:2]
        q = self.kv_q(self.kv_query_norm(x)).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = target_k.transpose(1, 2)
        v = target_v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(batch_size, num_tokens, self.hidden_channels)
        x = x + self.kv_out(y)
        return x + self.ffn(self.ffn_norm(x))


class KVCacheInjectedLatentDraftHead(nn.Module):
    """Draft head that queries real target-model self-attention KV cache tensors."""

    def __init__(
        self,
        latent_channels: int,
        kv_layer_names: tuple[str, ...],
        hidden_channels: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        max_context_tokens: int = 512,
        latent_pool: tuple[int, int, int] = (1, 2, 2),
        ffn_mult: int = 4,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if latent_channels <= 0:
            raise ValueError("latent_channels must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if len(kv_layer_names) < num_layers:
            raise ValueError("kv_layer_names must have at least num_layers entries")
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.latent_channels = latent_channels
        self.kv_layer_names = tuple(kv_layer_names)
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_context_tokens = max_context_tokens
        self.latent_pool = tuple(int(x) for x in latent_pool)
        self.ffn_mult = ffn_mult
        self.ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_channels * ffn_mult
        self.dropout = dropout
        self.head_type = "kv_cache_attention"
        self.gradient_checkpointing = False

        self.latent_in = nn.Conv3d(latent_channels, hidden_channels, kernel_size=1)
        self.block_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.timestep_embed = nn.Sequential(
            nn.Linear(1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.blocks = nn.ModuleList(
            [
                KVCacheDraftBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    ffn_mult=ffn_mult,
                    ffn_dim=self.ffn_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.latent_out = nn.Conv3d(hidden_channels, latent_channels, kernel_size=1)

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    def _block_index_tensor(
        self,
        block_index: torch.Tensor | int,
        batch_size: int,
        num_blocks: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(block_index, int):
            return torch.full((batch_size, 1), float(block_index) / float(num_blocks), device=device, dtype=dtype)
        return block_index.to(device=device, dtype=dtype).reshape(batch_size, 1) / float(num_blocks)

    def _timestep_tensor(
        self,
        timestep: torch.Tensor | int | float | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if timestep is None:
            return torch.zeros((batch_size, 1), device=device, dtype=dtype)
        if isinstance(timestep, (int, float)):
            return torch.full((batch_size, 1), float(timestep) / 1000.0, device=device, dtype=dtype)
        timestep = timestep.to(device=device, dtype=dtype)
        if timestep.ndim == 2:
            timestep = timestep[:, :1]
        return timestep.reshape(batch_size, 1) / 1000.0

    def _select_kv_tokens(self, cache: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        k = cache["k"].to(device=device, dtype=dtype)
        v = cache["v"].to(device=device, dtype=dtype)
        if k.ndim != 4 or v.ndim != 4:
            raise ValueError(f"Expected target KV tensors [B, N, H, D], got k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape != v.shape:
            raise ValueError(f"Target K/V shapes must match, got k={tuple(k.shape)} v={tuple(v.shape)}")
        if k.shape[2] != self.num_heads or k.shape[3] != self.hidden_channels // self.num_heads:
            raise ValueError(
                "Target KV head shape does not match draft head: "
                f"kv={tuple(k.shape[2:])} expected={(self.num_heads, self.hidden_channels // self.num_heads)}"
            )
        if self.max_context_tokens > 0 and k.shape[1] > self.max_context_tokens:
            indices = torch.linspace(0, k.shape[1] - 1, steps=self.max_context_tokens, device=device).round().long()
            k = k.index_select(1, indices)
            v = v.index_select(1, indices)
        return k, v

    def forward(
        self,
        block_latents: torch.Tensor,
        target_kv_cache: dict[str, dict[str, torch.Tensor]],
        block_index: torch.Tensor | int,
        num_blocks: int,
        timestep: torch.Tensor | int | float | None = None,
    ) -> torch.Tensor:
        if block_latents.ndim != 5:
            raise ValueError(f"block_latents must have shape [B, T, C, H, W], got {tuple(block_latents.shape)}")
        batch_size, frames, _channels, height, width = block_latents.shape
        x = block_latents.permute(0, 2, 1, 3, 4)
        pooled = F.avg_pool3d(x, kernel_size=self.latent_pool, stride=self.latent_pool, ceil_mode=True)
        pooled_shape = pooled.shape[-3:]
        tokens = self.latent_in(pooled).flatten(2).transpose(1, 2)
        block_pos = self._block_index_tensor(block_index, batch_size, num_blocks, block_latents.device, tokens.dtype)
        timestep_pos = self._timestep_tensor(timestep, batch_size, block_latents.device, tokens.dtype)
        tokens = tokens + self.block_embed(block_pos).unsqueeze(1) + self.timestep_embed(timestep_pos).unsqueeze(1)
        for index, block in enumerate(self.blocks):
            layer_name = self.kv_layer_names[index]
            if layer_name not in target_kv_cache:
                raise ValueError(f"Missing target KV cache layer: {layer_name}")
            target_k, target_v = self._select_kv_tokens(target_kv_cache[layer_name], tokens.device, tokens.dtype)
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint.checkpoint(block, tokens, target_k, target_v, use_reentrant=False)
            else:
                tokens = block(tokens, target_k, target_v)
        low_res = self.out_norm(tokens).transpose(1, 2).reshape(batch_size, self.hidden_channels, *pooled_shape)
        prediction = self.latent_out(low_res)
        prediction = F.interpolate(prediction.float(), size=(frames, height, width), mode="trilinear", align_corners=False)
        return prediction.to(dtype=block_latents.dtype).permute(0, 2, 1, 3, 4)


class WanDFlashAttention(nn.Module):
    def __init__(self, hidden_channels: int, num_heads: int, dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        if hidden_channels % num_heads != 0:
            raise ValueError("hidden_channels must be divisible by num_heads")
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.head_dim = hidden_channels // num_heads
        self.dropout = dropout
        self.q = nn.Linear(hidden_channels, hidden_channels)
        self.k = nn.Linear(hidden_channels, hidden_channels)
        self.v = nn.Linear(hidden_channels, hidden_channels)
        self.o = nn.Linear(hidden_channels, hidden_channels)
        self.norm_q = WanRMSNorm(hidden_channels, eps=eps)
        self.norm_k = WanRMSNorm(hidden_channels, eps=eps)

    def forward(
        self,
        proposal: torch.Tensor,
        target_context: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        current_start_frame: int,
        context_start_frame: int,
        context_grid_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, proposal_tokens = proposal.shape[:2]
        context_tokens = target_context.shape[1]
        q = self.norm_q(self.q(proposal)).view(batch_size, proposal_tokens, self.num_heads, self.head_dim)
        proposal_k = self.norm_k(self.k(proposal)).view(batch_size, proposal_tokens, self.num_heads, self.head_dim)
        context_k = self.norm_k(self.k(target_context)).view(batch_size, context_tokens, self.num_heads, self.head_dim)
        proposal_v = self.v(proposal).view(batch_size, proposal_tokens, self.num_heads, self.head_dim)
        context_v = self.v(target_context).view(batch_size, context_tokens, self.num_heads, self.head_dim)

        q = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(proposal_v)
        proposal_k = causal_rope_apply(
            proposal_k,
            grid_sizes,
            freqs,
            start_frame=current_start_frame,
        ).type_as(proposal_v)
        context_k = causal_rope_apply(
            context_k,
            context_grid_sizes if context_grid_sizes is not None else grid_sizes,
            freqs,
            start_frame=context_start_frame,
        ).type_as(context_v)

        k = torch.cat([context_k, proposal_k], dim=1).transpose(1, 2)
        v = torch.cat([context_v, proposal_v], dim=1).transpose(1, 2)
        q = q.transpose(1, 2)
        output = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0)
        return self.o(output.transpose(1, 2).reshape(batch_size, proposal_tokens, self.hidden_channels))


class WanDFlashDraftBlock(nn.Module):
    def __init__(
        self,
        hidden_channels: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm1 = WanLayerNorm(hidden_channels, eps)
        self.self_attn = WanDFlashAttention(hidden_channels, num_heads, dropout=dropout, eps=eps)
        self.norm2 = WanLayerNorm(hidden_channels, eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channels, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_channels),
        )

    def forward(
        self,
        proposal: torch.Tensor,
        target_context: torch.Tensor,
        grid_sizes: torch.Tensor,
        freqs: torch.Tensor,
        current_start_frame: int,
        context_start_frame: int,
        context_grid_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        proposal = proposal + self.self_attn(
            self.norm1(proposal),
            target_context,
            grid_sizes,
            freqs,
            current_start_frame,
            context_start_frame,
            context_grid_sizes,
        )
        return proposal + self.ffn(self.norm2(proposal))


class WanDFlashLatentDraftHead(nn.Module):
    """Wan-token draft head with DFlash-style target-hidden conditioning."""

    def __init__(
        self,
        latent_channels: int,
        layer_feature_dims: dict[str, int],
        hidden_channels: int = 5120,
        num_layers: int = 3,
        num_heads: int = 40,
        patch_size: tuple[int, int, int] = (1, 2, 2),
        ffn_dim: int = 13824,
        freq_dim: int = 256,
        max_context_tokens: int = 4680,
        per_block_context: bool = True,
        mean_init_context_fuser: bool = True,
        dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        if not layer_feature_dims:
            raise ValueError("layer_feature_dims must not be empty")
        if any(dim != hidden_channels for dim in layer_feature_dims.values()):
            raise ValueError("WanDFlashLatentDraftHead expects target features in Wan hidden size")
        self.latent_channels = latent_channels
        self.layer_feature_dims = {name: int(dim) for name, dim in sorted(layer_feature_dims.items())}
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = tuple(int(x) for x in patch_size)
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.max_context_tokens = max_context_tokens
        self.per_block_context = per_block_context
        self.uses_per_block_context = per_block_context and len(layer_feature_dims) == num_layers
        self.mean_init_context_fuser = mean_init_context_fuser
        self.dropout = dropout
        self.eps = eps
        self.head_type = "wan_dflash_attention"
        self.gradient_checkpointing = False

        self.patch_embedding = nn.Conv3d(latent_channels, hidden_channels, kernel_size=self.patch_size, stride=self.patch_size)
        self.context_fc = nn.Linear(hidden_channels * len(self.layer_feature_dims), hidden_channels, bias=False)
        if self.mean_init_context_fuser:
            self._init_context_fuser_as_mean()
        if self.uses_per_block_context:
            self.context_fc.requires_grad_(False)
        self.context_norm = WanRMSNorm(hidden_channels, eps=eps)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, hidden_channels), nn.SiLU(), nn.Linear(hidden_channels, hidden_channels))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(hidden_channels, hidden_channels * 6))
        self.blocks = nn.ModuleList(
            [
                WanDFlashDraftBlock(
                    hidden_channels=hidden_channels,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
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

    def enable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = True

    @property
    def layer_names(self) -> tuple[str, ...]:
        return tuple(self.layer_feature_dims)

    def _init_context_fuser_as_mean(self) -> None:
        """Initialize concatenated target features as a simple layerwise average."""
        layer_count = len(self.layer_feature_dims)
        with torch.no_grad():
            self.context_fc.weight.zero_()
            eye = torch.eye(
                self.hidden_channels,
                device=self.context_fc.weight.device,
                dtype=self.context_fc.weight.dtype,
            )
            for index in range(layer_count):
                start = index * self.hidden_channels
                end = start + self.hidden_channels
                self.context_fc.weight[:, start:end].copy_(eye / float(layer_count))

    def _select_context_feature(
        self,
        target_features: dict[str, torch.Tensor],
        layer_name: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if layer_name not in target_features:
            raise ValueError(f"Missing target feature layer: {layer_name}")
        feature = target_features[layer_name].to(device=device, dtype=dtype)
        return _select_token_subset(feature, self.max_context_tokens)

    def _fuse_context(self, target_features: dict[str, torch.Tensor], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        features = []
        token_count = None
        for name in self.layer_names:
            feature = self._select_context_feature(target_features, name, device, dtype)
            if token_count is None:
                token_count = feature.shape[1]
            elif token_count != feature.shape[1]:
                raise ValueError("All selected target features must have the same token count")
            features.append(feature)
        return self.context_norm(self.context_fc(torch.cat(features, dim=-1)))

    def _per_block_contexts(
        self,
        target_features: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor] | None:
        if not self.uses_per_block_context:
            return None
        return [
            self.context_norm(self._select_context_feature(target_features, name, device, dtype))
            for name in self.layer_names
        ]

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
        block_latents: torch.Tensor,
        target_features: dict[str, torch.Tensor],
        block_index: torch.Tensor | int,
        num_blocks: int,
        timestep: torch.Tensor | int | float | None = None,
    ) -> torch.Tensor:
        if block_latents.ndim != 5:
            raise ValueError(f"block_latents must have shape [B, T, C, H, W], got {tuple(block_latents.shape)}")
        if block_latents.shape[2] != self.latent_channels:
            raise ValueError(f"Expected {self.latent_channels} latent channels, got {block_latents.shape[2]}")
        batch_size, frames = block_latents.shape[:2]
        dtype = self.patch_embedding.weight.dtype
        x = block_latents.to(dtype=dtype).permute(0, 2, 1, 3, 4)
        embedded = self.patch_embedding(x)
        grid_sizes = torch.tensor(
            [embedded.shape[-3:]] * batch_size,
            dtype=torch.long,
            device=block_latents.device,
        )
        tokens = embedded.flatten(2).transpose(1, 2)
        per_block_contexts = self._per_block_contexts(target_features, block_latents.device, dtype)
        context = None if per_block_contexts is not None else self._fuse_context(target_features, block_latents.device, dtype)

        frame_timesteps = self._timestep_for_frames(timestep, batch_size, frames, block_latents.device, dtype)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, frame_timesteps.flatten()).type_as(tokens))
        e_head = e.unflatten(0, (batch_size, frames))
        e_block = self.time_projection(e).unflatten(1, (6, self.hidden_channels)).unflatten(0, (batch_size, frames))
        frame_seq_len = tokens.shape[1] // frames
        e_chunks = e_block.chunk(6, dim=2)
        tokens = tokens.unflatten(1, (frames, frame_seq_len)) * (1 + e_chunks[1]) + e_chunks[0]
        tokens = tokens.flatten(1, 2)

        if isinstance(block_index, int):
            block_id = block_index
        else:
            block_id = int(block_index.reshape(-1)[0].detach().cpu().item())
        current_start_frame = block_id * frames
        context_start_frame = max(0, block_id - 1) * frames
        freqs = self.freqs.to(device=block_latents.device)
        for block_index, block in enumerate(self.blocks):
            block_context = per_block_contexts[block_index] if per_block_contexts is not None else context
            if block_context is None:
                raise RuntimeError("WanDFlashLatentDraftHead context was not initialized")
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = checkpoint.checkpoint(
                    block,
                    tokens,
                    block_context,
                    grid_sizes,
                    freqs,
                    current_start_frame,
                    context_start_frame,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, block_context, grid_sizes, freqs, current_start_frame, context_start_frame)

        head_out = self.head(tokens, e_head.unsqueeze(2))
        return self._unpatchify(head_out, grid_sizes).to(dtype=block_latents.dtype)


class ConditionalResBlock3D(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.cond = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond(cond).chunk(2, dim=-1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        residual = x
        x = self.conv1(self.act(self.norm1(x)))
        x = self.norm2(x) * (1.0 + scale) + shift
        x = self.conv2(self.act(x))
        return residual + x


class LatentDraftHead(nn.Module):
    """Small target-conditioned residual head for drafting latent blocks."""

    def __init__(
        self,
        latent_channels: int,
        feature_dim: int,
        hidden_channels: int = 64,
        num_res_blocks: int = 2,
    ):
        super().__init__()
        if latent_channels <= 0 or feature_dim <= 0 or hidden_channels <= 0:
            raise ValueError("latent_channels, feature_dim, and hidden_channels must be positive")
        if num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")

        self.latent_channels = latent_channels
        self.feature_dim = feature_dim
        self.hidden_channels = hidden_channels
        self.num_res_blocks = num_res_blocks
        self.cond_dim = hidden_channels

        self.cond_proj = nn.Sequential(
            nn.LayerNorm(feature_dim + 1),
            nn.Linear(feature_dim + 1, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.in_proj = nn.Conv3d(latent_channels, hidden_channels, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [ConditionalResBlock3D(hidden_channels, self.cond_dim) for _ in range(num_res_blocks)]
        )
        self.out_norm = nn.GroupNorm(8 if hidden_channels % 8 == 0 else 1, hidden_channels)
        self.out_proj = nn.Conv3d(hidden_channels, latent_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        block_noise: torch.Tensor,
        target_feature_vector: torch.Tensor,
        block_index: torch.Tensor | int,
        num_blocks: int,
    ) -> torch.Tensor:
        if block_noise.ndim != 5:
            raise ValueError(f"block_noise must have shape [B, T, C, H, W], got {tuple(block_noise.shape)}")
        if target_feature_vector.ndim != 2:
            raise ValueError("target_feature_vector must have shape [B, D]")
        if block_noise.shape[0] != target_feature_vector.shape[0]:
            raise ValueError("block_noise and target_feature_vector batch sizes must match")
        if block_noise.shape[2] != self.latent_channels:
            raise ValueError(
                f"Expected {self.latent_channels} latent channels, got {block_noise.shape[2]}"
            )
        if target_feature_vector.shape[1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {target_feature_vector.shape[1]}")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if isinstance(block_index, int):
            block_index = torch.full(
                (block_noise.shape[0], 1),
                float(block_index) / float(num_blocks),
                device=block_noise.device,
                dtype=target_feature_vector.dtype,
            )
        else:
            block_index = block_index.to(device=block_noise.device, dtype=target_feature_vector.dtype)
            block_index = block_index.reshape(block_noise.shape[0], 1) / float(num_blocks)

        cond = self.cond_proj(torch.cat([target_feature_vector, block_index], dim=-1))
        x = block_noise.permute(0, 2, 1, 3, 4)
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x, cond)
        residual = self.out_proj(torch.nn.functional.silu(self.out_norm(x)))
        residual = residual.permute(0, 2, 1, 3, 4)
        return block_noise + residual


class CausalWanARDraftHead(nn.Module):
    """Full causal Wan AR draft head initialized from Self-Forcing-style checkpoints.

    This is intentionally a full causal Wan model, not a DFlash/latent adapter.
    It supports:
    - training-time latest-block teacher forcing via clean_x/aug_t
    - inference-time accumulated KV/cross-attention cache via the regular
      CausalWanModel inference path
    """

    def __init__(
        self,
        model_name: str = "Wan2.1-T2V-1.3B",
        timestep_shift: float = 5.0,
        local_attn_size: int = -1,
        sink_size: int = 0,
    ):
        super().__init__()
        self.model_name = model_name
        self.timestep_shift = float(timestep_shift)
        self.local_attn_size = int(local_attn_size)
        self.sink_size = int(sink_size)
        self.generator = WanDiffusionWrapper(
            model_name=model_name,
            timestep_shift=timestep_shift,
            is_causal=True,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
        )

    @property
    def latent_channels(self) -> int:
        return int(getattr(self.generator.model, "in_dim", 16))

    def enable_gradient_checkpointing(self) -> None:
        self.generator.enable_gradient_checkpointing()

    def forward_prefix_teacher_forcing(
        self,
        *,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        clean_prefix_latents: torch.Tensor | None = None,
        pad_to_frames: int | None = None,
        return_clean: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if clean_prefix_latents is None or clean_prefix_latents.shape[1] == 0:
            return self(
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                return_clean=return_clean,
            )

        batch_size, current_frames = noisy_latents.shape[:2]
        prefix_frames = clean_prefix_latents.shape[1]
        full_latents = torch.cat([clean_prefix_latents, noisy_latents], dim=1)
        prefix_timestep = torch.zeros(
            [batch_size, prefix_frames],
            device=timestep.device,
            dtype=timestep.dtype,
        )
        full_timestep = torch.cat([prefix_timestep, timestep], dim=1)
        current_start = prefix_frames
        if pad_to_frames is not None and full_latents.shape[1] < pad_to_frames:
            pad_frames = int(pad_to_frames) - int(full_latents.shape[1])
            pad_latents = full_latents.new_zeros(
                [batch_size, pad_frames, *full_latents.shape[2:]]
            )
            pad_timestep = full_timestep.new_zeros([batch_size, pad_frames])
            full_latents = torch.cat([full_latents, pad_latents], dim=1)
            full_timestep = torch.cat([full_timestep, pad_timestep], dim=1)
        seq_len = self.generator._seq_len_for_latents(full_latents)

        flow_pred = self.generator.model(
            full_latents.permute(0, 2, 1, 3, 4),
            t=full_timestep,
            context=prompt_embeds,
            seq_len=seq_len,
        ).permute(0, 2, 1, 3, 4)[:, current_start:current_start + current_frames]

        clean_pred = self.generator._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_latents.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        if return_clean:
            return flow_pred, clean_pred
        return flow_pred

    def forward(
        self,
        *,
        noisy_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        timestep: torch.Tensor,
        clean_prefix_latents: torch.Tensor | None = None,
        pad_to_frames: int | None = None,
        clean_context_latents: torch.Tensor | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        cache_start: int | None = None,
        return_clean: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if clean_prefix_latents is not None:
            return self.forward_prefix_teacher_forcing(
                noisy_latents=noisy_latents,
                prompt_embeds=prompt_embeds,
                timestep=timestep,
                clean_prefix_latents=clean_prefix_latents,
                pad_to_frames=pad_to_frames,
                return_clean=return_clean,
            )
        conditional_dict = {"prompt_embeds": prompt_embeds}
        # Causal inference mutates KV caches in-place. Gradient checkpointing
        # can replay these forwards during backward, so keep cached forwards
        # non-checkpointed even if checkpointing is enabled elsewhere.
        wan_model = self.generator.model
        previous_gradient_checkpointing = getattr(wan_model, "gradient_checkpointing", False)
        if kv_cache is not None:
            wan_model.gradient_checkpointing = False
        try:
            flow_pred, clean_pred = self.generator(
                noisy_image_or_video=noisy_latents,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                clean_x=clean_context_latents,
                aug_t=torch.zeros_like(timestep) if clean_context_latents is not None else None,
                cache_start=cache_start,
            )
        finally:
            wan_model.gradient_checkpointing = previous_gradient_checkpointing
        if return_clean:
            return flow_pred, clean_pred
        return flow_pred


def collate_draft_head_records(
    records: list[dict[str, Any]],
    layer_names: tuple[str, ...] | None = None,
    include_target_features: bool = False,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("records must not be empty")

    block_noise = torch.cat([record["block_noise"] for record in records], dim=0)
    target_latents = torch.cat([record["target_latents"] for record in records], dim=0)
    has_target_features = all(bool(record.get("target_features")) for record in records)
    if has_target_features:
        feature_vectors = torch.cat(
            [pool_target_features(record["target_features"], layer_names=layer_names) for record in records],
            dim=0,
        )
    else:
        feature_vectors = torch.empty(block_noise.shape[0], 0, dtype=block_noise.dtype)
    block_index = torch.tensor(
        [int(record["block_index"]) for record in records],
        dtype=torch.long,
    )
    prompt_index = torch.tensor(
        [-1 if record.get("prompt_index") is None else int(record["prompt_index"]) for record in records],
        dtype=torch.long,
    )
    batch = {
        "prompts": [record["prompt"] for record in records],
        "prompt_index": prompt_index,
        "block_noise": block_noise,
        "target_latents": target_latents,
        "target_feature_vector": feature_vectors,
        "block_index": block_index,
    }
    if include_target_features:
        if not has_target_features:
            raise ValueError("include_target_features=True requires target_features in every record")
        selected_names = layer_names or tuple(sorted(records[0]["target_features"]))
        batch["target_features"] = {
            name: torch.cat([record["target_features"][name] for record in records], dim=0)
            for name in selected_names
        }
    if all(record.get("target_kv_cache") is not None for record in records):
        selected_names = layer_names or tuple(sorted(records[0]["target_kv_cache"]))
        batch["target_kv_cache"] = {
            name: {
                "k": torch.cat([record["target_kv_cache"][name]["k"] for record in records], dim=0),
                "v": torch.cat([record["target_kv_cache"][name]["v"] for record in records], dim=0),
            }
            for name in selected_names
        }
    if all(record.get("draft_latents") is not None for record in records):
        batch["draft_latents"] = torch.cat([record["draft_latents"] for record in records], dim=0)
    if all(record.get("context_latents") is not None for record in records):
        batch["context_latents"] = torch.cat([record["context_latents"] for record in records], dim=0)
    if all(record.get("prompt_embeds") is not None for record in records):
        batch["prompt_embeds"] = torch.cat([record["prompt_embeds"] for record in records], dim=0)
    if all(record.get("teacher_trajectory_latents") is not None for record in records):
        batch["teacher_trajectory_latents"] = torch.stack(
            [record["teacher_trajectory_latents"] for record in records],
            dim=0,
        )
    if all(record.get("teacher_trajectory_noisy_latents") is not None for record in records):
        batch["teacher_trajectory_noisy_latents"] = torch.stack(
            [record["teacher_trajectory_noisy_latents"] for record in records],
            dim=0,
        )
    if all(record.get("teacher_trajectory_timesteps") is not None for record in records):
        batch["teacher_trajectory_timesteps"] = torch.stack(
            [record["teacher_trajectory_timesteps"] for record in records],
            dim=0,
        )
    return batch


def predict_draft_head_batch(
    model: nn.Module,
    batch: dict[str, Any],
    num_blocks: int,
    *,
    input_key: str = "block_noise",
) -> torch.Tensor:
    inner_model = model.module if hasattr(model, "module") else model
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    block_latents = batch[input_key].to(device=device, dtype=dtype)
    block_index = batch["block_index"].to(device)
    if isinstance(inner_model, KVInjectedLatentDraftHead):
        target_features = {
            name: value.to(device=device, dtype=dtype)
            for name, value in batch["target_features"].items()
        }
        return model(
            block_latents,
            target_features,
            block_index=block_index,
            num_blocks=num_blocks,
            timestep=batch.get("timestep"),
        )
    if isinstance(inner_model, WanDFlashLatentDraftHead):
        target_features = {
            name: value.to(device=device, dtype=dtype)
            for name, value in batch["target_features"].items()
        }
        return model(
            block_latents,
            target_features,
            block_index=block_index,
            num_blocks=num_blocks,
            timestep=batch.get("timestep"),
        )
    if isinstance(inner_model, KVCacheInjectedLatentDraftHead):
        target_kv_cache = {
            name: {
                "k": cache["k"].to(device=device, dtype=dtype),
                "v": cache["v"].to(device=device, dtype=dtype),
            }
            for name, cache in batch["target_kv_cache"].items()
        }
        return model(
            block_latents,
            target_kv_cache,
            block_index=block_index,
            num_blocks=num_blocks,
            timestep=batch.get("timestep"),
        )

    target_feature_vector = batch["target_feature_vector"].to(device=device, dtype=dtype)
    return model(
        block_latents,
        target_feature_vector,
        block_index=block_index,
        num_blocks=num_blocks,
    )


def train_draft_head_step(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    num_blocks: int,
    input_key: str = "block_noise",
) -> float:
    model.train()
    parameter = next(model.parameters())
    device = parameter.device
    dtype = parameter.dtype
    target_latents = batch["target_latents"].to(device=device, dtype=dtype)

    optimizer.zero_grad(set_to_none=True)
    prediction = predict_draft_head_batch(model, batch, num_blocks, input_key=input_key)
    loss = torch.nn.functional.mse_loss(prediction.float(), target_latents.float())
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def initialize_attention_head_from_target_blocks(
    model: KVInjectedLatentDraftHead,
    target_model: nn.Module,
    source_block_indices: tuple[int, ...],
) -> dict[str, int]:
    """Best-effort initialization from Wan target blocks when dimensions match."""
    modules = dict(target_model.named_modules())
    copied: dict[str, int] = {"attention": 0, "ffn": 0, "skipped": 0}
    for draft_block, source_index in zip(model.blocks, source_block_indices, strict=False):
        source = modules.get(f"blocks.{source_index}")
        if source is None or not hasattr(source, "self_attn"):
            copied["skipped"] += 1
            continue
        source_attn = source.self_attn
        if (
            hasattr(source_attn, "q")
            and source_attn.q.weight.shape == draft_block.self_attn.in_proj_weight[: model.hidden_channels].shape
        ):
            with torch.no_grad():
                hidden = model.hidden_channels
                draft_block.self_attn.in_proj_weight[:hidden].copy_(source_attn.q.weight)
                draft_block.self_attn.in_proj_weight[hidden: 2 * hidden].copy_(source_attn.k.weight)
                draft_block.self_attn.in_proj_weight[2 * hidden:].copy_(source_attn.v.weight)
                draft_block.self_attn.in_proj_bias[:hidden].copy_(source_attn.q.bias)
                draft_block.self_attn.in_proj_bias[hidden: 2 * hidden].copy_(source_attn.k.bias)
                draft_block.self_attn.in_proj_bias[2 * hidden:].copy_(source_attn.v.bias)
                draft_block.self_attn.out_proj.weight.copy_(source_attn.o.weight)
                draft_block.self_attn.out_proj.bias.copy_(source_attn.o.bias)
                draft_block.context_attn.in_proj_weight.copy_(draft_block.self_attn.in_proj_weight)
                draft_block.context_attn.in_proj_bias.copy_(draft_block.self_attn.in_proj_bias)
                draft_block.context_attn.out_proj.weight.copy_(draft_block.self_attn.out_proj.weight)
                draft_block.context_attn.out_proj.bias.copy_(draft_block.self_attn.out_proj.bias)
            copied["attention"] += 1
        else:
            copied["skipped"] += 1
        if hasattr(source, "ffn"):
            source_linears = [module for module in source.ffn if isinstance(module, nn.Linear)]
            draft_linears = [module for module in draft_block.ffn if isinstance(module, nn.Linear)]
            if len(source_linears) == 2 and all(s.weight.shape == d.weight.shape for s, d in zip(source_linears, draft_linears, strict=True)):
                with torch.no_grad():
                    for source_linear, draft_linear in zip(source_linears, draft_linears, strict=True):
                        draft_linear.weight.copy_(source_linear.weight)
                        draft_linear.bias.copy_(source_linear.bias)
                copied["ffn"] += 1
    return copied


def initialize_wan_dflash_head_from_target_blocks(
    model: WanDFlashLatentDraftHead,
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
    with torch.no_grad():
        if hasattr(target_model, "patch_embedding") and target_model.patch_embedding.weight.shape == model.patch_embedding.weight.shape:
            model.patch_embedding.weight.copy_(target_model.patch_embedding.weight)
            if target_model.patch_embedding.bias is not None and model.patch_embedding.bias is not None:
                model.patch_embedding.bias.copy_(target_model.patch_embedding.bias)
            copied["patch_embedding"] = 1
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
    def _copy_module_if_compatible(dst: nn.Module, src: nn.Module) -> bool:
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        if src_state.keys() != dst_state.keys():
            return False
        if any(src_state[name].shape != dst_state[name].shape for name in src_state):
            return False
        dst.load_state_dict(src_state, strict=True)
        return True

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


def save_draft_head_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    layer_names: tuple[str, ...],
    num_blocks: int,
    metadata: dict[str, Any] | None = None,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> Path:
    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if not layer_names:
        raise ValueError("layer_names must not be empty")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(model, KVInjectedLatentDraftHead):
        head_type = "kv_injected_attention"
        model_config = {
            "latent_channels": model.latent_channels,
            "layer_feature_dims": model.layer_feature_dims,
            "hidden_channels": model.hidden_channels,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "max_context_tokens": model.max_context_tokens,
            "latent_pool": model.latent_pool,
            "ffn_mult": model.ffn_mult,
            "ffn_dim": model.ffn_dim,
            "dropout": model.dropout,
        }
    elif isinstance(model, KVCacheInjectedLatentDraftHead):
        head_type = "kv_cache_attention"
        model_config = {
            "latent_channels": model.latent_channels,
            "kv_layer_names": model.kv_layer_names,
            "hidden_channels": model.hidden_channels,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "max_context_tokens": model.max_context_tokens,
            "latent_pool": model.latent_pool,
            "ffn_mult": model.ffn_mult,
            "ffn_dim": model.ffn_dim,
            "dropout": model.dropout,
        }
    elif isinstance(model, WanDFlashLatentDraftHead):
        head_type = "wan_dflash_attention"
        model_config = {
            "latent_channels": model.latent_channels,
            "layer_feature_dims": model.layer_feature_dims,
            "hidden_channels": model.hidden_channels,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "patch_size": model.patch_size,
            "ffn_dim": model.ffn_dim,
            "freq_dim": model.freq_dim,
            "max_context_tokens": model.max_context_tokens,
            "per_block_context": model.per_block_context,
            "mean_init_context_fuser": model.mean_init_context_fuser,
            "dropout": model.dropout,
            "eps": model.eps,
        }
    elif isinstance(model, CausalWanARDraftHead):
        head_type = "causal_wan_ar"
        model_config = {
            "model_name": model.model_name,
            "timestep_shift": model.timestep_shift,
            "local_attn_size": model.local_attn_size,
            "sink_size": model.sink_size,
        }
    elif isinstance(model, LatentDraftHead):
        head_type = "conv"
        model_config = {
            "latent_channels": model.latent_channels,
            "feature_dim": model.feature_dim,
            "hidden_channels": model.hidden_channels,
            "num_res_blocks": model.num_res_blocks,
        }
    else:
        raise TypeError(f"Unsupported draft-head model type: {type(model).__name__}")
    payload = {
        "format": "sdvg_latent_draft_head_v1",
        "head_type": head_type,
        "model_state_dict": model.state_dict() if state_dict is None else state_dict,
        "model_config": model_config,
        "layer_names": list(layer_names),
        "num_blocks": int(num_blocks),
        "metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_draft_head_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if payload.get("format") != "sdvg_latent_draft_head_v1":
        raise ValueError(f"Unsupported draft-head checkpoint format: {payload.get('format')}")

    model_config = dict(payload["model_config"])
    head_type = payload.get("head_type", "conv")
    if head_type == "kv_injected_attention":
        if "latent_pool" in model_config:
            model_config["latent_pool"] = tuple(model_config["latent_pool"])
        model_config.pop("prediction_type", None)
        model = KVInjectedLatentDraftHead(**model_config)
    elif head_type == "kv_cache_attention":
        if "latent_pool" in model_config:
            model_config["latent_pool"] = tuple(model_config["latent_pool"])
        if "kv_layer_names" in model_config:
            model_config["kv_layer_names"] = tuple(model_config["kv_layer_names"])
        model = KVCacheInjectedLatentDraftHead(**model_config)
    elif head_type == "wan_dflash_attention":
        if "patch_size" in model_config:
            model_config["patch_size"] = tuple(model_config["patch_size"])
        model_config.setdefault("per_block_context", False)
        model_config.setdefault("mean_init_context_fuser", False)
        model = WanDFlashLatentDraftHead(**model_config)
    elif head_type == "ar_bidirectional":
        from train_bidirectional_draft_head import BidirectionalPromptAnchorDraftHead

        if "patch_size" in model_config:
            model_config["patch_size"] = tuple(model_config["patch_size"])
        model = BidirectionalPromptAnchorDraftHead(**model_config)
    elif head_type == "causal_wan_ar":
        model = CausalWanARDraftHead(**model_config)
    elif head_type == "conv":
        model = LatentDraftHead(**model_config)
    else:
        raise ValueError(f"Unsupported draft-head type: {head_type}")
    model.load_state_dict(payload["model_state_dict"])
    metadata = {
        "head_type": head_type,
        "layer_names": tuple(payload["layer_names"]),
        "num_blocks": int(payload["num_blocks"]),
        "model_config": model_config,
        "metadata": payload.get("metadata", {}),
    }
    return model, metadata
