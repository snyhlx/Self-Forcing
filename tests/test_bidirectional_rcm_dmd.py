import unittest
import sys
import types

import torch
from torch import nn


def _install_lightweight_training_import_stubs():
    """The DMD math helpers are lightweight; the trainer's Wan classes are not."""
    module_names = [
        "sdvg_draft_head",
        "wan",
        "wan.modules",
        "wan.modules.model",
        "wan.utils",
        "wan.utils.fm_solvers_unipc",
        "wan.utils.rcm_rf",
    ]
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
    wan_utils_stub = types.ModuleType("wan.utils")
    wan_unipc_stub = types.ModuleType("wan.utils.fm_solvers_unipc")
    wan_rcm_rf_stub = types.ModuleType("wan.utils.rcm_rf")

    class _FlowUniPCMultistepScheduler:
        def __init__(self, *args, **kwargs):
            self.timesteps = torch.tensor([])

        def set_timesteps(self, num_inference_steps, device=None, shift=None):
            self.timesteps = torch.linspace(1000, 0, steps=num_inference_steps + 1, device=device)[:-1].long()

        def step(self, model_output, timestep, sample, return_dict=True):
            output = sample - model_output * 0.0
            return (output,) if not return_dict else types.SimpleNamespace(prev_sample=output)

    wan_unipc_stub.FlowUniPCMultistepScheduler = _FlowUniPCMultistepScheduler
    wan_rcm_rf_stub.rf_to_sigma = lambda rf_t: rf_t.clamp(max=1.0 - torch.finfo(rf_t.dtype).eps) / (
        1.0 - rf_t.clamp(max=1.0 - torch.finfo(rf_t.dtype).eps)
    )
    wan_rcm_rf_stub.rf_to_trig_time = lambda rf_t: torch.atan(wan_rcm_rf_stub.rf_to_sigma(rf_t))
    def _sample_shifted_uniform_rf_times(*, shape, shift, device, dtype=torch.float32):
        unit = torch.rand(shape, device=device, dtype=dtype)
        return (shift * unit / (1 + (shift - 1) * unit)).clamp(0.0, 1.0)

    def _sample_lognormal_rf_times(*, shape, mean=0.0, std=1.6, device, dtype=torch.float32):
        sigma = torch.exp(torch.randn(shape, device=device, dtype=dtype) * std + mean)
        return (sigma / (sigma + 1.0)).clamp(0.0, 1.0)

    wan_rcm_rf_stub.sample_shifted_uniform_rf_times = _sample_shifted_uniform_rf_times
    wan_rcm_rf_stub.sample_lognormal_rf_times = _sample_lognormal_rf_times
    wan_model_stub.Head = nn.Identity
    wan_model_stub.WanAttentionBlock = nn.Identity
    wan_model_stub.rope_params = lambda *args, **kwargs: torch.empty(0)
    wan_model_stub.sinusoidal_embedding_1d = lambda dim, timesteps: torch.zeros(timesteps.numel(), dim)
    sys.modules.setdefault("wan", wan_stub)
    sys.modules.setdefault("wan.modules", wan_modules_stub)
    sys.modules.setdefault("wan.modules.model", wan_model_stub)
    sys.modules.setdefault("wan.utils", wan_utils_stub)
    sys.modules.setdefault("wan.utils.fm_solvers_unipc", wan_unipc_stub)
    sys.modules.setdefault("wan.utils.rcm_rf", wan_rcm_rf_stub)
    return originals


_ORIGINAL_MODULES = _install_lightweight_training_import_stubs()

from train_bidirectional_draft_head import (
    RCMStyleDraftHeadDMD,
    default_rcm_rollout_schedule,
    parse_rollout_timestep_schedule,
    rcm_dmd_gradient_target,
    rcm_dmd_surrogate_loss,
    rcm_fake_score_loss,
    rcm_rf_training_timesteps,
    resolve_rollout_steps,
    resolve_rollout_timestep_list,
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

    def test_rcm_lognormal_training_timesteps_return_rf_trig_sigma(self):
        torch.manual_seed(0)

        timesteps, rf_time, trig_time, sigma = rcm_rf_training_timesteps(
            batch_size=4,
            frames=3,
            distribution="rcm_lognormal",
            shift=5.0,
            lognormal_mean=0.0,
            lognormal_std=1.6,
            min_timestep=20.0,
            max_timestep=980.0,
            device=torch.device("cpu"),
        )

        self.assertEqual(timesteps.shape, (4, 3))
        self.assertEqual(rf_time.shape, (4, 3))
        self.assertEqual(trig_time.shape, (4, 3))
        self.assertEqual(sigma.shape, (4, 3))
        self.assertGreaterEqual(float(timesteps.min()), 20.0)
        self.assertLessEqual(float(timesteps.max()), 980.0)
        self.assertTrue(torch.all(rf_time > 0))
        self.assertTrue(torch.all(trig_time >= 0))
        self.assertTrue(torch.all(sigma > 0))

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

        self.assertTrue(torch.allclose(loss, torch.tensor(4.0)))

    def test_student_phase_matches_rcm_alternation(self):
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(1, warmup_steps=2, student_update_freq=5))
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(2, warmup_steps=2, student_update_freq=5))
        self.assertFalse(RCMStyleDraftHeadDMD.is_student_phase(3, warmup_steps=2, student_update_freq=5))
        self.assertTrue(RCMStyleDraftHeadDMD.is_student_phase(7, warmup_steps=2, student_update_freq=5))

    def test_rcm_trig_x0_preconditioning_formula(self):
        class ZeroNet(nn.Module):
            def forward(self, x, t, context, seq_len):
                return torch.zeros_like(x)

        class DummyWrapper(nn.Module):
            uniform_timestep = True

            def __init__(self):
                super().__init__()
                self.model = ZeroNet()

            def _seq_len_for_latents(self, noisy):
                return 1

        dmd = RCMStyleDraftHeadDMD.__new__(RCMStyleDraftHeadDMD)
        dmd.dtype = torch.float32
        noisy = torch.ones(1, 2, 1, 1, 1)
        prompt_embeds = torch.zeros(1, 1, 1)
        trig_time = torch.full((1, 2), torch.pi / 4)

        pred = dmd._predict_rcm_trig_x0(DummyWrapper(), noisy, prompt_embeds, trig_time)

        expected = torch.ones_like(noisy) / (2.0**0.5)
        self.assertTrue(torch.allclose(pred, expected, atol=1e-6))

    def test_rcm_fd_consistency_loss_runs_with_student_flow(self):
        class ToyStudent(nn.Module):
            def forward(self, *, anchor_latents, future_noise, prompt_embeds, timestep):
                del anchor_latents, prompt_embeds, timestep
                return future_noise * 0.25

        dmd = RCMStyleDraftHeadDMD.__new__(RCMStyleDraftHeadDMD)
        dmd.consistency_objective = "rcm_fd"
        dmd.consistency_loss_weight = 1.0
        dmd.consistency_fd_epsilon = 1e-3
        dmd.score_scope = "future"
        dmd.dtype = torch.float32
        dmd._score_clean_and_slice = lambda generator_x0, anchor: (generator_x0, slice(None))
        dmd._sample_noisy_score_input = lambda clean_x0: (
            clean_x0 + 0.1,
            torch.full(clean_x0.shape[:2], 500.0),
            torch.ones(*clean_x0.shape[:2], 1, 1, 1),
            torch.full(clean_x0.shape[:2], 0.5),
            torch.full(clean_x0.shape[:2], 0.5),
        )
        dmd._teacher_cfg_flow = lambda noisy, prompt_embeds, timestep: torch.ones_like(noisy) * 0.5

        loss, metrics = dmd.consistency_loss(
            ToyStudent(),
            torch.zeros(1, 2, 1, 1, 1),
            anchor=None,
            prompt_embeds=torch.zeros(1, 1, 1),
        )

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss), 0.0)
        self.assertIn("dmd_consistency_loss_raw", metrics)

    def test_resolve_rollout_steps_prefers_explicit_steps(self):
        self.assertEqual(resolve_rollout_steps(rollout_steps=5, denoising_step_list=[1000, 0]), 5)
        self.assertEqual(resolve_rollout_steps(rollout_steps=0, denoising_step_list=[1000, 750, 0]), 3)
        with self.assertRaisesRegex(ValueError, "rollout_steps"):
            resolve_rollout_steps(rollout_steps=1, denoising_step_list=[1000, 0])

    def test_parse_rcm_rollout_schedule_accepts_fractions_and_sigma_max(self):
        self.assertEqual(
            parse_rollout_timestep_schedule("sigma_max 15/16 5/6 5/8 0", sigma_max=1600.0),
            [999, 938, 833, 625, 0],
        )
        self.assertEqual(default_rcm_rollout_schedule(rollout_steps=3, sigma_max=1600.0), [999, 625, 0])
        with self.assertRaisesRegex(ValueError, "descending"):
            parse_rollout_timestep_schedule("0 1/2 1", sigma_max=1600.0)

    def test_resolve_rollout_timestep_list_defaults_to_rcm_schedule(self):
        self.assertEqual(
            resolve_rollout_timestep_list(
                rollout_solver="rcm",
                rollout_steps=5,
                rollout_schedule="",
                rollout_sigma_max=1600.0,
                denoising_step_list=[1000, 750, 500, 250, 0],
            ),
            [999, 938, 833, 625, 0],
        )
        self.assertEqual(
            resolve_rollout_timestep_list(
                rollout_solver="euler",
                rollout_steps=0,
                rollout_schedule="",
                rollout_sigma_max=1600.0,
                denoising_step_list=[1000, 500, 0],
            ),
            [1000, 500, 0],
        )

    def test_fake_score_optimizer_is_created_after_optional_wrap(self):
        dmd = RCMStyleDraftHeadDMD.__new__(RCMStyleDraftHeadDMD)
        dmd.fake_score = nn.Linear(2, 2)
        dmd.raw_fake_score = dmd.fake_score
        dmd.fake_score_lr = 1e-4
        dmd.fake_score_weight_decay = 0.01
        dmd.fake_score_is_wrapped = False
        dmd.wrap_fake_score(
            strategy="none",
            is_distributed=False,
            local_rank=0,
            fsdp_min_num_params=1,
            fsdp_mixed_precision="none",
        )
        dmd.create_optimizer()

        self.assertIsNotNone(dmd.optimizer)
        self.assertIs(dmd.fake_score, dmd.raw_fake_score)


if __name__ == "__main__":
    unittest.main()
