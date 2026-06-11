import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.convert_draft_head_dataset import convert_dataset


class ConvertDraftHeadDatasetTests(unittest.TestCase):
    def test_convert_dataset_pools_full_target_features(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            output = root / "output"
            source.mkdir()

            records = [
                {
                    "prompt": "p0",
                    "prompt_index": 0,
                    "block_index": 1,
                    "block_noise": torch.zeros(1, 3, 2, 2, 2),
                    "target_latents": torch.ones(1, 3, 2, 2, 2),
                    "target_features": {
                        "blocks.8": torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
                    },
                    "context_latents": None,
                    "draft_latents": None,
                    "delta": 0.5,
                }
            ]
            torch.save({"records": records}, source / "shard_000000.pt")
            (source / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "sdvg_draft_head_v1",
                        "num_records": 1,
                        "shard_size": 128,
                        "shards": [{"path": "shard_000000.pt", "num_records": 1}],
                    }
                ),
                encoding="utf-8",
            )

            manifest_path = convert_dataset(
                source_manifest=source / "manifest.json",
                output_dir=output,
                feature_storage="pooled",
                overwrite=False,
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["feature_storage"], "pooled")
            payload = torch.load(output / "shard_000000.pt", map_location="cpu", weights_only=False)
            converted = payload["records"][0]
            self.assertEqual(converted["target_features"]["blocks.8"].shape, (1, 3))
            self.assertTrue(
                torch.allclose(
                    converted["target_features"]["blocks.8"],
                    records[0]["target_features"]["blocks.8"].mean(dim=1),
                )
            )


if __name__ == "__main__":
    unittest.main()
