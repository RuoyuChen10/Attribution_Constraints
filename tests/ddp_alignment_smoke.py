"""Two-process CPU smoke test for uneven local alignment example counts."""

from __future__ import annotations

from types import SimpleNamespace

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
    train_one_epoch,
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

        # Rank 0 requests an intra-epoch stop at the first evaluation boundary;
        # train_one_epoch must broadcast it so both ranks exit after one step.
        batch = (
            dummy.unsqueeze(0),
            torch.zeros(1, 2, 2, device=context.device),
            torch.tensor([0], device=context.device),
            ["unused.jpg"],
            ["unused.npy"],
        )
        args = SimpleNamespace(
            amp=False,
            epochs=1,
            alignment_interval=100,
            evals_per_epoch=4,
            eval_without_alignment=True,
            lambda_deviation=0.5,
            lambda_redundancy=0.5,
            alignment_batch_size=1,
        )
        callback_calls = 0

        def request_stop(*_) -> bool:
            nonlocal callback_calls
            callback_calls += 1
            return True

        global_step, _, stopped = train_one_epoch(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            loader=[batch, batch, batch, batch],
            human_lima=None,
            loss_config=AlignmentLossConfig(variant="paper"),
            args=args,
            context=context,
            epoch=1,
            global_step=0,
            evaluation_callback=request_stop,
        )
        if not stopped or global_step != 1:
            raise AssertionError(
                f"Expected synchronized stop at step 1, got stopped={stopped}, "
                f"global_step={global_step}"
            )
        expected_calls = 1 if context.rank == 0 else 0
        if callback_calls != expected_calls:
            raise AssertionError(
                f"Rank {context.rank}: expected {expected_calls} callback calls, "
                f"got {callback_calls}"
            )

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
