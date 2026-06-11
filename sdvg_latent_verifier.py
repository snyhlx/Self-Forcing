from __future__ import annotations

import torch
from torch import nn


def pooled_latent_stats(latents: torch.Tensor) -> torch.Tensor:
    """Return per-channel mean/std/min/max for [B, F, C, H, W] latents."""
    if latents.ndim != 5:
        raise ValueError(f"Expected [B, F, C, H, W] latents, got shape {tuple(latents.shape)}")
    x = latents.float().permute(0, 2, 1, 3, 4).flatten(2)
    return torch.cat(
        [
            x.mean(dim=-1),
            x.std(dim=-1, unbiased=False),
            x.amin(dim=-1),
            x.amax(dim=-1),
        ],
        dim=1,
    )


def verifier_features(
    block_latents: torch.Tensor,
    context_latents: torch.Tensor | None,
    block_index: int,
    num_blocks: int,
) -> torch.Tensor:
    block_features = pooled_latent_stats(block_latents)
    if context_latents is None or context_latents.shape[1] == 0:
        context_features = torch.zeros_like(block_features)
        context_ratio = torch.zeros([block_features.shape[0], 1], device=block_features.device)
    else:
        context_features = pooled_latent_stats(context_latents)
        context_ratio = torch.full(
            [block_features.shape[0], 1],
            float(context_latents.shape[1]) / max(1, num_blocks),
            device=block_features.device,
        )

    block_ratio = torch.full(
        [block_features.shape[0], 1],
        float(block_index) / max(1, num_blocks - 1),
        device=block_features.device,
    )
    return torch.cat([block_features, context_features, block_ratio, context_ratio], dim=1)


class LatentBlockVerifier(nn.Module):
    """Small MLP classifier for overfitting target-vs-draft latent block features."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
