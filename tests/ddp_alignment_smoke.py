"""Two-process CPU smoke test for uneven local alignment example counts."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from prior_alignment_core import (
    AlignmentLossConfig,
    DeviationExample,
    RedundancyExample,
)
from train_prior_alignment import (
    cleanup_distributed,
    make_grad_scaler,
    run_deviation_backward,
    run_redundancy_backward,
    setup_distributed,
)


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(12, 2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images.flatten(1))


def main() -> None:
    context = setup_distributed()
    if context.world_size != 2:
        raise RuntimeError("Run with --nproc-per-node=2")
    try:
        torch.manual_seed(7)
        base = TinyClassifier().to(context.device)
        ddp_kwargs = {"broadcast_buffers": False}
        if context.device.type == "cuda":
            ddp_kwargs["device_ids"] = [context.local_rank]
        model = DistributedDataParallel(base, **ddp_kwargs)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        scaler = make_grad_scaler(enabled=False)
        dummy = (
            torch.arange(12, dtype=torch.float32, device=context.device)
            .reshape(3, 2, 2)
            / 12
        )

        optimizer.zero_grad(set_to_none=True)
        labels = torch.tensor([0], device=context.device)
        main_loss = nn.functional.cross_entropy(model(dummy[None]), labels)
        scaler.scale(main_loss).backward()

        examples: list[DeviationExample] = []
        if context.rank == 0:
            examples.append(
                DeviationExample(
                    insertion=dummy,
                    deletion=1.0 - dummy,
                    label=torch.tensor(0),
                    best_human_gain=torch.tensor(0.1),
                    full_confidence=torch.tensor(0.9),
                )
            )
        diagnostics = run_deviation_backward(
            model=model,
            examples=examples,
            dummy_image=dummy,
            loss_config=AlignmentLossConfig(
                variant="adaptive_log", adaptive_beta=2.0
            ),
            loss_weight=0.5,
            chunk_size=1,
            scaler=scaler,
            amp=False,
            context=context,
        )
        if diagnostics.count != 1.0:
            raise AssertionError(f"Expected global count 1, got {diagnostics.count}")

        redundancies: list[RedundancyExample] = []
        if context.rank == 1:
            redundancies.append(
                RedundancyExample(
                    insertion_after=dummy,
                    deletion_after=1.0 - dummy,
                    insertion_before=dummy * 0.5,
                    deletion_before=1.0 - dummy * 0.5,
                    label=torch.tensor(0),
                    best_human_gain=torch.tensor(0.05),
                    full_confidence=torch.tensor(0.85),
                )
            )
        diagnostics = run_redundancy_backward(
            model=model,
            examples=redundancies,
            dummy_image=dummy,
            loss_config=AlignmentLossConfig(
                variant="adaptive_log", adaptive_beta=2.0
            ),
            loss_weight=0.5,
            chunk_size=1,
            scaler=scaler,
            amp=False,
            context=context,
        )
        if diagnostics.count != 1.0:
            raise AssertionError(f"Expected global count 1, got {diagnostics.count}")
        scaler.step(optimizer)
        scaler.update()

        parameters = torch.cat([parameter.detach().flatten() for parameter in model.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(context.world_size)]
        dist.all_gather(gathered, parameters)
        if not torch.allclose(gathered[0], gathered[1], atol=1e-7, rtol=1e-7):
            raise AssertionError("DDP parameters diverged across ranks")
        if context.rank == 0:
            print("DDP alignment smoke test passed")
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
