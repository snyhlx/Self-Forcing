from __future__ import annotations

import math

import torch


def rcm_trig_angles(num_steps: int, sigma_max: float = 80.0) -> torch.Tensor:
    """Return the original rCM trig-time schedule, including the final clean step."""
    if num_steps < 1 or num_steps > 4:
        raise ValueError("rCM inference supports num_steps in [1, 4]")
    mid_t = [1.5, 1.4, 1.0][: num_steps - 1]
    return torch.tensor([math.atan(float(sigma_max)), *mid_t, 0.0], dtype=torch.float64)


def trig_to_rf_time(trig_t: torch.Tensor) -> torch.Tensor:
    return torch.sin(trig_t) / (torch.cos(trig_t) + torch.sin(trig_t))


def rcm_rf_times(
    num_steps: int,
    *,
    sigma_max: float = 80.0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    return trig_to_rf_time(rcm_trig_angles(num_steps, sigma_max=sigma_max)).to(device=device)


def rcm_timestep_list(num_steps: int, *, sigma_max: float = 80.0) -> list[int]:
    times = rcm_rf_times(num_steps, sigma_max=sigma_max, device="cpu")
    return [int(round(float(value) * 1000.0)) for value in times]


def rcm_stochastic_step(
    sample: torch.Tensor,
    flow_prediction: torch.Tensor,
    *,
    t_cur: torch.Tensor,
    t_next: torch.Tensor,
    next_noise: torch.Tensor,
) -> torch.Tensor:
    while t_cur.ndim < sample.ndim:
        t_cur = t_cur.unsqueeze(-1)
    while t_next.ndim < sample.ndim:
        t_next = t_next.unsqueeze(-1)
    return (1.0 - t_next) * (sample - t_cur * flow_prediction) + t_next * next_noise
