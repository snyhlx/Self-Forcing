import unittest
from types import SimpleNamespace

import torch
from torch import nn

from sdvg_inference import commit_clean_block_with_feature_capture, mode_needs_draft_pipeline


class TinyGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 2),
            nn.SiLU(),
            nn.Linear(2, 2),
        )
        self.calls = 0

    def forward(
        self,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
    ):
        del conditional_dict, timestep, kv_cache, crossattn_cache, current_start
        self.calls += 1
        flat = noisy_image_or_video.reshape(-1, noisy_image_or_video.shape[-1])
        return None, self.model(flat)


class TinyPipeline:
    def __init__(self):
        self.generator = TinyGenerator()
        self.args = SimpleNamespace(context_noise=0)
        self.kv_cache1 = []
        self.crossattn_cache = []
        self.frame_seq_length = 1


class SDVGDraftHeadIntegrationTests(unittest.TestCase):
    def test_commit_clean_block_with_feature_capture_uses_commit_path(self):
        pipeline = TinyPipeline()
        block_latents = torch.randn(1, 1, 1, 1, 2)

        features = commit_clean_block_with_feature_capture(
            pipeline,
            block_latents,
            conditional_dict={"prompt_embeds": torch.zeros(1, 1, 2)},
            current_start_frame=3,
            layer_names=("0", "2"),
        )

        self.assertEqual(pipeline.generator.calls, 1)
        self.assertEqual(set(features), {"0", "2"})
        self.assertEqual(features["0"].shape, (1, 2))
        self.assertEqual(features["2"].shape, (1, 2))
        self.assertFalse(features["0"].requires_grad)

    def test_target_regen_loads_draft_pipeline(self):
        self.assertTrue(mode_needs_draft_pipeline("target_regen"))
        self.assertTrue(mode_needs_draft_pipeline("compare", compare_mode="target_regen"))
        self.assertTrue(mode_needs_draft_pipeline("compare", compare_mode="sdvg"))
        self.assertFalse(mode_needs_draft_pipeline("draft_head"))
        self.assertFalse(mode_needs_draft_pipeline("compare", compare_mode="draft_head"))
        self.assertFalse(mode_needs_draft_pipeline("target_only"))


if __name__ == "__main__":
    unittest.main()
