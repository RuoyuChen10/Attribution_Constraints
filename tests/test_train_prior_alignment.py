from __future__ import annotations

import argparse
import unittest

import torch
import torch.nn as nn

from prior_alignment_core import (
    AlignmentLossConfig,
    DeviationExample,
    RedundancyExample,
)
from train_prior_alignment import (
    DistributedContext,
    UnifiedClassifier,
    checkpoint_payload,
    make_grad_scaler,
    run_deviation_backward,
    run_redundancy_backward,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(12, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images.flatten(1))


class TrainPriorAlignmentTest(unittest.TestCase):
    def test_single_process_alignment_backward(self) -> None:
        torch.manual_seed(3)
        device = torch.device("cpu")
        context = DistributedContext(0, 1, 0, device)
        model = TinyClassifier()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = make_grad_scaler(enabled=False)
        image = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2) / 12
        initial = model.classifier.weight.detach().clone()

        optimizer.zero_grad(set_to_none=True)
        deviation = DeviationExample(
            insertion=image,
            deletion=1.0 - image,
            label=torch.tensor(0),
            best_human_gain=torch.tensor(0.1),
            full_confidence=torch.tensor(0.9),
        )
        deviation_diagnostics = run_deviation_backward(
            model=model,
            examples=[deviation],
            dummy_image=image,
            loss_config=AlignmentLossConfig(
                variant="adaptive_log", adaptive_beta=2.0
            ),
            loss_weight=0.5,
            chunk_size=1,
            scaler=scaler,
            amp=False,
            context=context,
        )

        redundancy = RedundancyExample(
            insertion_after=image,
            deletion_after=1.0 - image,
            insertion_before=image * 0.5,
            deletion_before=1.0 - image * 0.5,
            label=torch.tensor(0),
            best_human_gain=torch.tensor(0.05),
            full_confidence=torch.tensor(0.85),
        )
        redundancy_diagnostics = run_redundancy_backward(
            model=model,
            examples=[redundancy],
            dummy_image=image,
            loss_config=AlignmentLossConfig(variant="paper"),
            loss_weight=0.5,
            chunk_size=1,
            scaler=scaler,
            amp=False,
            context=context,
        )
        scaler.step(optimizer)
        scaler.update()

        self.assertEqual(deviation_diagnostics.count, 1.0)
        self.assertEqual(redundancy_diagnostics.count, 1.0)
        self.assertFalse(torch.equal(initial, model.classifier.weight))

    def test_checkpoint_contains_unwrapped_raw_state_dict(self) -> None:
        raw_model = TinyClassifier()
        wrapped = UnifiedClassifier(raw_model, "resnet")
        optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.1)
        scaler = make_grad_scaler(enabled=False)
        payload = checkpoint_payload(
            base_model=wrapped,
            optimizer=optimizer,
            scaler=scaler,
            args=argparse.Namespace(model="resnet"),
            label_to_idx={"class_a": 0, "class_b": 1},
            epoch=2,
            global_step=10,
            best_top1=0.5,
        )

        self.assertEqual(set(payload["model"]), set(raw_model.state_dict()))
        self.assertTrue(all(not key.startswith("raw_model.") for key in payload["model"]))
        restored = TinyClassifier()
        restored.load_state_dict(payload["model"], strict=True)


if __name__ == "__main__":
    unittest.main()
