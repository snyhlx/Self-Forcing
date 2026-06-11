import tempfile
import unittest

import torch

from sdvg_draft_head import (
    KVCacheInjectedLatentDraftHead,
    KVInjectedLatentDraftHead,
    LatentDraftHead,
    TargetFeatureFuser,
    WanDFlashLatentDraftHead,
    load_draft_head_checkpoint,
    pool_target_features,
    save_draft_head_checkpoint,
)


class LatentDraftHeadTests(unittest.TestCase):
    def test_pool_target_features_concatenates_in_requested_order(self):
        features = {
            "late": torch.full((2, 5, 3), 2.0),
            "early": torch.full((2, 7, 4), 1.0),
        }

        pooled = pool_target_features(features, layer_names=("early", "late"))

        self.assertEqual(pooled.shape, (2, 7))
        self.assertTrue(torch.equal(pooled[:, :4], torch.ones(2, 4)))
        self.assertTrue(torch.equal(pooled[:, 4:], torch.full((2, 3), 2.0)))

    def test_pool_target_features_handles_already_pooled_features(self):
        features = {"pooled": torch.randn(2, 6)}

        pooled = pool_target_features(features)

        self.assertEqual(pooled.shape, (2, 6))
        self.assertTrue(torch.equal(pooled, features["pooled"]))

    def test_pool_target_features_rejects_missing_layer(self):
        with self.assertRaisesRegex(ValueError, "Missing target feature layer"):
            pool_target_features({"a": torch.zeros(1, 2)}, layer_names=("b",))

    def test_latent_draft_head_preserves_shape_and_initial_identity(self):
        torch.manual_seed(0)
        head = LatentDraftHead(latent_channels=4, feature_dim=6, hidden_channels=8, num_res_blocks=1)
        block_noise = torch.randn(2, 3, 4, 5, 6)
        feature_vector = torch.randn(2, 6)

        output = head(block_noise, feature_vector, block_index=1, num_blocks=9)

        self.assertEqual(output.shape, block_noise.shape)
        self.assertTrue(torch.allclose(output, block_noise))

    def test_latent_draft_head_backpropagates_to_parameters(self):
        torch.manual_seed(0)
        head = LatentDraftHead(latent_channels=4, feature_dim=6, hidden_channels=8, num_res_blocks=1)
        block_noise = torch.randn(2, 3, 4, 5, 6)
        feature_vector = torch.randn(2, 6)

        output = head(block_noise, feature_vector, block_index=torch.tensor([1, 2]), num_blocks=9)
        loss = output.square().mean()
        loss.backward()

        self.assertIsNotNone(head.out_proj.weight.grad)
        self.assertGreater(float(head.out_proj.weight.grad.abs().sum()), 0.0)

    def test_latent_draft_head_validates_input_shapes(self):
        head = LatentDraftHead(latent_channels=4, feature_dim=6, hidden_channels=8)
        with self.assertRaisesRegex(ValueError, "latent channels"):
            head(torch.randn(2, 3, 5, 5, 6), torch.randn(2, 6), block_index=0, num_blocks=9)
        with self.assertRaisesRegex(ValueError, "feature_dim"):
            head(torch.randn(2, 3, 4, 5, 6), torch.randn(2, 7), block_index=0, num_blocks=9)

    def test_target_feature_fuser_limits_tokens(self):
        fuser = TargetFeatureFuser({"a": 5, "b": 7}, hidden_channels=8, max_context_tokens=4)
        features = {
            "a": torch.randn(2, 10, 5),
            "b": torch.randn(2, 10, 7),
        }

        output = fuser(features)

        self.assertEqual(output.shape, (2, 4, 8))

    def test_kv_injected_head_preserves_shape_and_backpropagates(self):
        torch.manual_seed(0)
        head = KVInjectedLatentDraftHead(
            latent_channels=4,
            layer_feature_dims={"blocks.0": 6},
            hidden_channels=8,
            num_layers=1,
            num_heads=2,
            max_context_tokens=4,
            latent_pool=(1, 2, 2),
        )
        block_latents = torch.randn(2, 3, 4, 4, 4)
        features = {"blocks.0": torch.randn(2, 9, 6)}

        output = head(block_latents, features, block_index=torch.tensor([1, 2]), num_blocks=9)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(output.shape, block_latents.shape)
        self.assertIsNotNone(head.latent_out.weight.grad)

    def test_kv_cache_head_uses_target_cache_and_round_trips_checkpoint(self):
        torch.manual_seed(0)
        head = KVCacheInjectedLatentDraftHead(
            latent_channels=4,
            kv_layer_names=("blocks.0",),
            hidden_channels=8,
            num_layers=1,
            num_heads=2,
            max_context_tokens=4,
            latent_pool=(1, 2, 2),
        )
        block_latents = torch.randn(2, 3, 4, 4, 4)
        target_kv_cache = {
            "blocks.0": {
                "k": torch.randn(2, 6, 2, 4),
                "v": torch.randn(2, 6, 2, 4),
            }
        }

        output = head(block_latents, target_kv_cache, block_index=torch.tensor([1, 2]), num_blocks=9)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(output.shape, block_latents.shape)
        self.assertIsNotNone(head.latent_out.weight.grad)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_draft_head_checkpoint(head, f"{tmpdir}/head.pt", layer_names=("blocks.0",), num_blocks=9)
            loaded, metadata = load_draft_head_checkpoint(path)

        self.assertIsInstance(loaded, KVCacheInjectedLatentDraftHead)
        self.assertEqual(metadata["head_type"], "kv_cache_attention")

    def test_wan_dflash_head_preserves_shape_and_round_trips_checkpoint(self):
        torch.manual_seed(0)
        head = WanDFlashLatentDraftHead(
            latent_channels=4,
            layer_feature_dims={"blocks.0": 12},
            hidden_channels=12,
            num_layers=1,
            num_heads=3,
            patch_size=(1, 2, 2),
            ffn_dim=24,
            freq_dim=8,
            max_context_tokens=12,
        )
        block_latents = torch.randn(1, 3, 4, 4, 4)
        features = {"blocks.0": torch.randn(1, 12, 12)}

        output = head(block_latents, features, block_index=1, num_blocks=9, timestep=torch.ones(1, 3) * 500)
        loss = output.square().mean()
        loss.backward()

        self.assertEqual(output.shape, block_latents.shape)
        self.assertIsNotNone(head.patch_embedding.weight.grad)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_draft_head_checkpoint(head, f"{tmpdir}/head.pt", layer_names=("blocks.0",), num_blocks=9)
            loaded, metadata = load_draft_head_checkpoint(path)

        self.assertIsInstance(loaded, WanDFlashLatentDraftHead)
        self.assertEqual(metadata["head_type"], "wan_dflash_attention")
        self.assertTrue(loaded.per_block_context)
        self.assertTrue(loaded.mean_init_context_fuser)

    def test_wan_dflash_context_fuser_is_mean_initialized(self):
        head = WanDFlashLatentDraftHead(
            latent_channels=4,
            layer_feature_dims={"blocks.0": 12, "blocks.1": 12, "blocks.2": 12},
            hidden_channels=12,
            num_layers=3,
            num_heads=3,
            patch_size=(1, 2, 2),
            ffn_dim=24,
            freq_dim=8,
        )

        weight = head.context_fc.weight.detach()
        eye = torch.eye(12) / 3.0

        self.assertTrue(torch.allclose(weight[:, :12], eye))
        self.assertTrue(torch.allclose(weight[:, 12:24], eye))
        self.assertTrue(torch.allclose(weight[:, 24:], eye))

    def test_wan_dflash_per_block_context_routes_matching_layers(self):
        head = WanDFlashLatentDraftHead(
            latent_channels=4,
            layer_feature_dims={"blocks.0": 12, "blocks.1": 12},
            hidden_channels=12,
            num_layers=2,
            num_heads=3,
            patch_size=(1, 2, 2),
            ffn_dim=24,
            freq_dim=8,
            max_context_tokens=4,
        )
        features = {
            "blocks.0": torch.full((1, 6, 12), 1.0),
            "blocks.1": torch.full((1, 6, 12), 2.0),
        }

        contexts = head._per_block_contexts(features, torch.device("cpu"), torch.float32)

        self.assertIsNotNone(contexts)
        assert contexts is not None
        self.assertFalse(head.context_fc.weight.requires_grad)
        self.assertEqual(len(contexts), 2)
        self.assertEqual(contexts[0].shape, (1, 4, 12))
        self.assertEqual(contexts[1].shape, (1, 4, 12))
        self.assertTrue(torch.allclose(contexts[0], head.context_norm(torch.ones(1, 4, 12))))
        self.assertTrue(torch.allclose(contexts[1], head.context_norm(torch.full((1, 4, 12), 2.0))))

    def test_wan_dflash_legacy_checkpoint_keeps_fused_context(self):
        head = WanDFlashLatentDraftHead(
            latent_channels=4,
            layer_feature_dims={"blocks.0": 12},
            hidden_channels=12,
            num_layers=1,
            num_heads=3,
            patch_size=(1, 2, 2),
            ffn_dim=24,
            freq_dim=8,
            per_block_context=False,
            mean_init_context_fuser=False,
        )
        payload = {
            "format": "sdvg_latent_draft_head_v1",
            "head_type": "wan_dflash_attention",
            "model_state_dict": head.state_dict(),
            "model_config": {
                "latent_channels": 4,
                "layer_feature_dims": {"blocks.0": 12},
                "hidden_channels": 12,
                "num_layers": 1,
                "num_heads": 3,
                "patch_size": (1, 2, 2),
                "ffn_dim": 24,
                "freq_dim": 8,
                "max_context_tokens": 4680,
                "dropout": 0.0,
                "eps": 1e-6,
            },
            "layer_names": ["blocks.0"],
            "num_blocks": 9,
            "metadata": {},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/legacy.pt"
            torch.save(payload, path)
            loaded, _ = load_draft_head_checkpoint(path)

        self.assertIsInstance(loaded, WanDFlashLatentDraftHead)
        self.assertFalse(loaded.per_block_context)
        self.assertFalse(loaded.mean_init_context_fuser)


if __name__ == "__main__":
    unittest.main()
