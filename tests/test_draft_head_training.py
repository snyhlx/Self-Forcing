import tempfile
import unittest

import torch

from sdvg_draft_head import (
    DraftHeadDatasetWriter,
    DraftHeadRecordDataset,
    LatentDraftHead,
    collate_draft_head_records,
    load_draft_head_checkpoint,
    make_draft_head_record,
    save_draft_head_checkpoint,
    train_draft_head_step,
)


def make_record(index: int):
    block_noise = torch.zeros(1, 2, 2, 3, 3)
    target_latents = torch.ones(1, 2, 2, 3, 3) * (1.0 + 0.1 * index)
    features = {"blocks.0": torch.ones(1, 4, 3) * index}
    return make_draft_head_record(
        prompt=f"prompt {index}",
        prompt_index=index,
        block_index=index % 3,
        block_noise=block_noise,
        target_latents=target_latents,
        target_features=features,
    )


class DraftHeadTrainingTests(unittest.TestCase):
    def test_dataset_loads_manifest_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DraftHeadDatasetWriter(tmpdir, shard_size=1)
            writer.add(make_record(0))
            writer.add(make_record(1))
            manifest = writer.close()

            dataset = DraftHeadRecordDataset(manifest)

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[1]["prompt"], "prompt 1")

    def test_collate_records_builds_training_batch(self):
        batch = collate_draft_head_records([make_record(0), make_record(1)], layer_names=("blocks.0",))

        self.assertEqual(batch["block_noise"].shape, (2, 2, 2, 3, 3))
        self.assertEqual(batch["target_latents"].shape, (2, 2, 2, 3, 3))
        self.assertEqual(batch["target_feature_vector"].shape, (2, 3))
        self.assertEqual(batch["prompts"], ["prompt 0", "prompt 1"])
        self.assertTrue(torch.equal(batch["block_index"], torch.tensor([0, 1])))

    def test_train_step_reduces_loss_on_tiny_batch(self):
        torch.manual_seed(0)
        records = [make_record(i) for i in range(4)]
        batch = collate_draft_head_records(records, layer_names=("blocks.0",))
        model = LatentDraftHead(latent_channels=2, feature_dim=3, hidden_channels=8, num_res_blocks=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)

        initial_loss = train_draft_head_step(model, batch, optimizer, num_blocks=3)
        final_loss = initial_loss
        for _ in range(20):
            final_loss = train_draft_head_step(model, batch, optimizer, num_blocks=3)

        self.assertLess(final_loss, initial_loss)

    def test_collate_rejects_empty_records(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            collate_draft_head_records([])

    def test_checkpoint_round_trip_preserves_metadata(self):
        model = LatentDraftHead(latent_channels=2, feature_dim=3, hidden_channels=8, num_res_blocks=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_draft_head_checkpoint(
                model,
                f"{tmpdir}/draft_head.pt",
                layer_names=("blocks.0",),
                num_blocks=3,
                metadata={"source": "unit-test"},
            )

            loaded, metadata = load_draft_head_checkpoint(path)

        self.assertIsInstance(loaded, LatentDraftHead)
        self.assertEqual(metadata["layer_names"], ("blocks.0",))
        self.assertEqual(metadata["num_blocks"], 3)
        self.assertEqual(metadata["metadata"]["source"], "unit-test")


if __name__ == "__main__":
    unittest.main()
