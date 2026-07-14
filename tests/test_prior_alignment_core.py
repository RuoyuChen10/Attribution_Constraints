from __future__ import annotations

import math
import unittest

import torch

from prior_alignment_core import (
    AlignmentLossConfig,
    adaptive_gradient_weight,
    adaptive_log_penalty,
    best_human_gain,
    build_alignment_examples,
    excess_over_human_gain,
    normalized_ranking_gains,
    redundancy_loss_from_logits,
    reference_scale,
)


def binary_logits(probability: float, *, requires_grad: bool = False) -> torch.Tensor:
    values = torch.tensor(
        [[math.log(probability), math.log(1.0 - probability)]],
        dtype=torch.float64,
    )
    return values.requires_grad_(requires_grad)


class PriorAlignmentCoreTest(unittest.TestCase):
    def test_paper_redundancy_is_relu_of_single_delta(self) -> None:
        config = AlignmentLossConfig(variant="paper")
        result = redundancy_loss_from_logits(
            binary_logits(0.7),
            binary_logits(0.2),
            binary_logits(0.5),
            binary_logits(0.4),
            torch.tensor([0]),
            torch.tensor([0.0]),
            torch.tensor([0.9]),
            config,
        )
        # F_after=2*0.7+(1-0.2)=2.2; F_before=2*0.5+(1-0.4)=1.6.
        self.assertAlmostEqual(result.loss.item(), 0.6, places=7)

    def test_excess_matches_bad_minus_best_human_gain(self) -> None:
        human = torch.tensor(0.3, requires_grad=True)
        for bad, expected in ((0.5, 0.2), (0.3, 0.0), (0.2, 0.0)):
            bad_tensor = torch.tensor(bad, requires_grad=True)
            excess = excess_over_human_gain(bad_tensor, human)
            self.assertAlmostEqual(excess.item(), expected, places=6)
            excess.backward()
            self.assertIsNone(human.grad)
            expected_grad = 1.0 if bad > 0.3 else 0.0
            self.assertAlmostEqual(bad_tensor.grad.item(), expected_grad)

    def test_lower_full_confidence_reduces_adaptive_weight(self) -> None:
        human = torch.tensor([0.3, 0.3])
        confidence = torch.tensor([0.6, 0.9])
        reference = reference_scale(human, confidence, 1e-6)
        self.assertTrue(torch.allclose(reference, torch.tensor([0.4, 0.3])))
        excess = torch.tensor([0.1, 0.1])
        weight = adaptive_gradient_weight(excess, reference, beta=2.0)
        self.assertLess(weight[0].item(), weight[1].item())

    def test_adaptive_weight_decreases_with_gap(self) -> None:
        gaps = torch.tensor([0.3, 0.2, 0.1, 0.0])
        reference = torch.full_like(gaps, 0.3)
        weights = adaptive_gradient_weight(gaps, reference, beta=2.0)
        self.assertTrue(torch.all(weights[:-1] > weights[1:]))
        self.assertEqual(weights[-1].item(), 0.0)

    def test_zero_excess_has_exact_zero_loss_and_gradient(self) -> None:
        for beta in (1.0, 2.0, 4.0):
            for gap in (0.0, -0.1):
                raw = torch.tensor(gap, requires_grad=True)
                excess = torch.relu(raw)
                loss = adaptive_log_penalty(
                    excess, torch.tensor(0.3), beta=beta
                )
                self.assertEqual(loss.item(), 0.0)
                loss.backward()
                self.assertEqual(raw.grad.item(), 0.0)

    def test_references_and_before_prefix_are_detached(self) -> None:
        config = AlignmentLossConfig(variant="adaptive_log", adaptive_beta=2.0)
        after_insertion = binary_logits(0.8, requires_grad=True)
        after_deletion = binary_logits(0.1, requires_grad=True)
        before_insertion = binary_logits(0.3, requires_grad=True)
        before_deletion = binary_logits(0.7, requires_grad=True)
        human = torch.tensor([0.1], requires_grad=True)
        confidence = torch.tensor([0.9], requires_grad=True)
        result = redundancy_loss_from_logits(
            after_insertion,
            after_deletion,
            before_insertion,
            before_deletion,
            torch.tensor([0]),
            human,
            confidence,
            config,
        )
        result.loss.sum().backward()
        self.assertIsNotNone(after_insertion.grad)
        self.assertIsNotNone(after_deletion.grad)
        self.assertIsNone(before_insertion.grad)
        self.assertIsNone(before_deletion.grad)
        self.assertIsNone(human.grad)
        self.assertIsNone(confidence.grad)

    def test_ranking_gains_and_best_human_gain(self) -> None:
        gains = normalized_ranking_gains(torch.tensor([1.5, 2.4, 2.7]), 3.0)
        self.assertTrue(torch.allclose(gains, torch.tensor([0.5, 0.3, 0.1])))
        best = best_human_gain(gains, torch.tensor([False, True, False]))
        self.assertAlmostEqual(best.item(), 0.3, places=6)
        none = best_human_gain(gains, torch.zeros(3, dtype=torch.bool))
        self.assertEqual(none.item(), 0.0)

    def test_example_builder_keeps_deviation_and_higher_order_redundancy(self) -> None:
        image = torch.ones(3, 2, 2)
        regions = torch.tensor(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=torch.bool,
        )
        prior = torch.tensor([[0, 1], [0, 0]], dtype=torch.bool)
        examples = build_alignment_examples(
            image=image,
            prior_mask=prior,
            ranked_regions=regions,
            set_scores=[1.5, 2.4, 2.7],
            label=torch.tensor(0),
            full_confidence=torch.tensor(0.9),
            overlap_threshold=0.15,
            normalizer=3.0,
        )
        self.assertEqual(len(examples.deviations), 1)
        self.assertEqual(len(examples.redundancies), 1)
        self.assertTrue(
            torch.equal(
                examples.prior_consistent,
                torch.tensor([False, True, False]),
            )
        )
        self.assertAlmostEqual(examples.best_human_gain.item(), 0.3, places=6)

    def test_example_builder_handles_all_and_no_prior_regions(self) -> None:
        image = torch.ones(3, 2, 2)
        regions = torch.tensor(
            [
                [[1, 0], [0, 0]],
                [[0, 1], [0, 0]],
                [[0, 0], [1, 0]],
            ],
            dtype=torch.bool,
        )
        common = dict(
            image=image,
            ranked_regions=regions,
            set_scores=[1.5, 2.4, 2.7],
            label=torch.tensor(0),
            full_confidence=torch.tensor(0.9),
            overlap_threshold=0.15,
            normalizer=3.0,
        )

        all_prior = build_alignment_examples(
            prior_mask=torch.ones(2, 2, dtype=torch.bool), **common
        )
        self.assertEqual(len(all_prior.deviations), 0)
        self.assertEqual(len(all_prior.redundancies), 0)
        self.assertAlmostEqual(all_prior.best_human_gain.item(), 0.5)

        no_prior = build_alignment_examples(
            prior_mask=torch.zeros(2, 2, dtype=torch.bool), **common
        )
        self.assertEqual(len(no_prior.deviations), 1)
        self.assertEqual(len(no_prior.redundancies), 2)
        self.assertEqual(no_prior.best_human_gain.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
