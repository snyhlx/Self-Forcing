import unittest
import sys
import types

import torch
from torch import nn


def _install_lightweight_training_import_stubs():
    """The DMD math helpers are lightweight; the trainer's Wan classes are not."""
    module_names = ["sdvg_draft_head", "wan", "wan.modules", "wan.modules.model"]
    originals = {name: sys.modules.get(name) for name in module_names}
    sdvg_stub = types.ModuleType("sdvg_draft_head")
    sdvg_stub.DraftCausalHead = nn.Identity
    sdvg_stub.WanDFlashDraftBlock = nn.Identity
    sdvg_stub.causal_rope_apply = lambda *args, **kwargs: args[0]
    sdvg_stub.rope_params = lambda *args, **kwargs: torch.empty(0)
    sdvg_stub.sinusoidal_embedding_1d = lambda dim, timesteps: torch.zeros(timesteps.numel(), dim)
    sys.modules.setdefault("sdvg_draft_head", sdvg_stub)

    wan_stub = types.ModuleType("wan")
    wan_modules_stub = types.ModuleType("wan.modules")
    wan_model_stub = types.ModuleType("wan.modules.model")
    wan_model_stub.Head = nn.Identity
    wan_model_stub.WanAttentionBlock = nn.Identity
    wan_model_stub.rope_params = lambda *args, **kwargs: torch.empty(0)
    wan_model_stub.sinusoidal_embedding_1d = lambda dim, timesteps: torch.zeros(timesteps.numel(), dim)
    sys.modules.setdefault("wan", wan_stub)
    sys.modules.setdefault("wan.modules", wan_modules_stub)
    sys.modules.setdefault("wan.modules.model", wan_model_stub)
    return originals


_ORIGINAL_MODULES = _install_lightweight_training_import_stubs()

from train_bidirectional_draft_head import (
    RCMStyleDraftHeadDMD,
    rcm_dmd_gradient_target,
    rcm_dmd_surrogate_loss,
    rcm_fake_score_loss,
    shifted_uniform_timesteps,
)

for _name, _module in _ORIGINAL_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module


class BidirectionalRCMDMDTests(unittest.TestCase):
    def test_shifted_uniform_timesteps_are_batched_and_bounded(self):
        torch.manual_seed(0)

        timesteps = shifted_uniform_timesteps(
            batch_size=4,
            frames=3,
            shift=5.0,
            min_timestep=20.0,
            max_timestep=980.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(timesteps.shape, (4, 3))
        self.assertGreaterEqual(float(timesteps.min()), 20.0)
        self.assertLessEqual(float(timesteps.max()), 980.0)
        self.assertTrue(torch.equal(timesteps[:, :1].expand_as(timesteps), timesteps))

    def test_rcm_dmd_target_uses_fake_score_minus_teacher(self):
        generator = torch.zeros(1, 2, 1, 1, 1)
        teacher = torch.ones_like(generator)
        fake_score = torch.ones_like(generator) * 3.0

        target, grad = rcm_dmd_gradient_target(generator, fake_score, teacher)

        self.assertTrue(torch.allclose(grad, torch.ones_like(generator) * 2.0))
        self.assertTrue(torch.allclose(target, torch.ones_like(generator) * -2.0))

    def test_rcm_dmd_surrogate_loss_sums_per_sample_like_rcm(self):
        generator = torch.zeros(1, 2, 1, 1, 1)
        teacher = torch.ones_like(generator)
        fake_score = torch.ones_like(generator) * 3.0

        loss, metrics = rcm_dmd_surrogate_loss(generator, fake_score, teacher)

        self.assertTrue(torch.allclose(loss, torch.tensor(8.0)))
        self.assertAlmostEqual(metrics["dmd_gradient_norm"], 2.0)

    def test_fake_score_loss_uses_sigma_squared_weighting(self):
        fake_score = torch.zeros(1, 2, 1, 1, 1)
        generator = torch.ones_like(fake_score)
        sigma = torch.full((1, 2, 1, 1, 1), 0.5)

        loss = rcm_fake_score_loss(fake_score, generator, sigma)

        self.assertTrue(torch.allclose(loss, torch.tensor(8.0)))

    def test_student_phase_matches_rcm_alternation(self):
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(1, warmup_steps=2, student_update_freq=5))
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(2, warmup_steps=2, student_update_freq=5))
        self.assertFalse(RCMStyleDraftHeadDMD.is_student_phase(3, warmup_steps=2, student_update_freq=5))
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(7, warmup_steps=2, student_update_freq=5))


if __name__ == "__main__":
    unittest.main()
