import unittest

import torch
from torch import nn

from sdvg_draft_head import FeatureCaptureConfig, TargetFeatureCapture, first_tensor


class TupleBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, width)

    def forward(self, x):
        y = self.proj(x)
        return {"hidden": (y, y.mean(dim=-1))}


class TinyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(4, 4), TupleBlock(4)])

    def forward(self, x):
        x = self.blocks[0](x)
        output = self.blocks[1](x)
        return output["hidden"][0]


class TargetFeatureCaptureTests(unittest.TestCase):
    def test_first_tensor_finds_nested_tensor(self):
        expected = torch.ones(2, 3)
        output = {"ignored": None, "nested": [(), {"value": expected}]}

        self.assertIs(first_tensor(output), expected)

    def test_capture_selected_layers_and_detaches_outputs(self):
        model = TinyTarget()
        x = torch.randn(2, 4, requires_grad=True)
        config = FeatureCaptureConfig(layer_names=("blocks.0", "blocks.1"), detach=True, clone=True)

        with TargetFeatureCapture(model, config) as capture:
            y = model(x)
            latest = capture.latest()

        self.assertEqual(set(latest), {"blocks.0", "blocks.1"})
        self.assertEqual(latest["blocks.0"].shape, (2, 4))
        self.assertEqual(latest["blocks.1"].shape, (2, 4))
        self.assertFalse(latest["blocks.0"].requires_grad)
        self.assertFalse(latest["blocks.1"].requires_grad)
        self.assertEqual(y.shape, (2, 4))

    def test_capture_can_preserve_grad_when_requested(self):
        model = TinyTarget()
        x = torch.randn(2, 4, requires_grad=True)
        config = FeatureCaptureConfig(layer_names=("blocks.0",), detach=False)

        with TargetFeatureCapture(model, config) as capture:
            model(x)
            latest = capture.latest()

        self.assertTrue(latest["blocks.0"].requires_grad)

    def test_hooks_are_removed_after_context(self):
        model = TinyTarget()
        x = torch.randn(2, 4)
        config = FeatureCaptureConfig(layer_names=("blocks.0",))

        with TargetFeatureCapture(model, config) as capture:
            model(x)
            self.assertEqual(len(capture.features["blocks.0"]), 1)

        model(x)
        self.assertEqual(len(capture.features["blocks.0"]), 1)

    def test_unknown_layer_raises_helpful_error(self):
        model = TinyTarget()
        config = FeatureCaptureConfig(layer_names=("blocks.99",))

        with self.assertRaisesRegex(ValueError, "Unknown feature capture layer"):
            with TargetFeatureCapture(model, config):
                pass

    def test_latest_requires_at_least_one_capture(self):
        model = TinyTarget()
        config = FeatureCaptureConfig(layer_names=("blocks.0",))

        with TargetFeatureCapture(model, config) as capture:
            with self.assertRaisesRegex(RuntimeError, "No captured feature"):
                capture.latest()


if __name__ == "__main__":
    unittest.main()
