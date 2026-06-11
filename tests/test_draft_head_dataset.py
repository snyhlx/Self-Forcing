import tempfile
import unittest
from pathlib import Path

import torch

from sdvg_draft_head import (
    DraftHeadDatasetWriter,
    load_draft_head_records,
    make_draft_head_record,
    prepare_target_features_for_storage,
)


class DraftHeadDatasetTests(unittest.TestCase):
    def test_prepare_target_features_pooled_storage_reduces_token_dimension(self):
        features = {
            "blocks.8": torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3),
        }

        pooled = prepare_target_features_for_storage(features, storage="pooled")

        self.assertEqual(pooled["blocks.8"].shape, (2, 3))
        self.assertTrue(torch.allclose(pooled["blocks.8"], features["blocks.8"].mean(dim=1)))

    def test_prepare_target_features_full_storage_preserves_shape(self):
        features = {"blocks.8": torch.randn(2, 4, 3)}

        full = prepare_target_features_for_storage(features, storage="full")

        self.assertEqual(full["blocks.8"].shape, (2, 4, 3))

    def test_make_record_moves_tensors_to_cpu_and_detaches(self):
        block_noise = torch.randn(1, 3, 2, 4, 4, requires_grad=True)
        target_latents = torch.randn(1, 3, 2, 4, 4, requires_grad=True)
        features = {"blocks.1": torch.randn(1, 12, 8, requires_grad=True)}

        record = make_draft_head_record(
            prompt="test prompt",
            prompt_index=7,
            block_index=2,
            block_noise=block_noise,
            target_latents=target_latents,
            target_features=features,
            delta=0.25,
        )

        self.assertEqual(record["prompt"], "test prompt")
        self.assertEqual(record["prompt_index"], 7)
        self.assertEqual(record["block_index"], 2)
        self.assertEqual(record["delta"], 0.25)
        self.assertEqual(record["block_noise"].device.type, "cpu")
        self.assertFalse(record["block_noise"].requires_grad)
        self.assertFalse(record["target_latents"].requires_grad)
        self.assertFalse(record["target_features"]["blocks.1"].requires_grad)
        self.assertEqual(record["target_features"]["blocks.1"].shape, (1, 8))

    def test_make_record_rejects_mismatched_latent_shapes(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            make_draft_head_record(
                prompt="bad",
                prompt_index=None,
                block_index=0,
                block_noise=torch.zeros(1, 3, 2, 4, 4),
                target_latents=torch.zeros(1, 2, 2, 4, 4),
                target_features={"blocks.0": torch.zeros(1, 4, 8)},
            )

    def test_make_record_requires_target_features(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            make_draft_head_record(
                prompt="bad",
                prompt_index=None,
                block_index=0,
                block_noise=torch.zeros(1, 3, 2, 4, 4),
                target_latents=torch.zeros(1, 3, 2, 4, 4),
                target_features={},
            )

    def test_writer_flushes_shards_and_loader_round_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DraftHeadDatasetWriter(tmpdir, shard_size=2)
            for index in range(3):
                writer.add(
                    make_draft_head_record(
                        prompt=f"prompt {index}",
                        prompt_index=index,
                        block_index=index % 2,
                        block_noise=torch.full((1, 3, 2, 4, 4), float(index)),
                        target_latents=torch.full((1, 3, 2, 4, 4), float(index + 1)),
                        target_features={"blocks.0": torch.full((1, 5, 8), float(index))},
                    )
                )
            manifest_path = writer.close()

            self.assertTrue(Path(manifest_path).is_file())
            self.assertTrue((Path(tmpdir) / "shard_000000.pt").is_file())
            self.assertTrue((Path(tmpdir) / "shard_000001.pt").is_file())

            records = load_draft_head_records(manifest_path)
            self.assertEqual(len(records), 3)
            self.assertEqual(records[0]["prompt"], "prompt 0")
            self.assertTrue(torch.equal(records[2]["block_noise"], torch.full((1, 3, 2, 4, 4), 2.0)))

    def test_writer_rejects_non_positive_shard_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "positive"):
                DraftHeadDatasetWriter(tmpdir, shard_size=0)


if __name__ == "__main__":
    unittest.main()
