"""Unified human-prior alignment training for image classifiers.

Examples:
    python train_prior_alignment.py --dataset saliency-bench --model vit \
        --loss-variant paper

    torchrun --standalone --nproc-per-node=2 train_prior_alignment.py \
        --dataset imagenet-s919 --model resnet \
        --loss-variant adaptive_log --adaptive-beta 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.transforms import InterpolationMode
try:
    from tqdm import tqdm
except ImportError:  # Keep --help/--print-config usable in minimal test envs.
    def tqdm(iterable, **_: Any):
        return iterable

from dataloader import (
    HF_VIT_MEAN,
    HF_VIT_STD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ImageNetSDataset,
    PascalSaliencyDataset,
    build_label_map,
)
from prior_alignment_core import (
    AlignmentBatchResult,
    AlignmentLossConfig,
    DeviationExample,
    RedundancyExample,
    build_alignment_examples,
    deviation_loss_from_logits,
    flatten_examples,
    redundancy_loss_from_logits,
)
from utils import SubRegionDivision


ROOT = Path(__file__).resolve().parent

DATASETS = {
    "saliency-bench": {
        "train_txt": "data_list/saliency-bench/train.txt",
        "test_txt": "data_list/saliency-bench/test.txt",
        "epochs": 20,
    },
    "imagenet-s919": {
        "train_txt": "data_list/imagenet-s919/train.txt",
        "test_txt": "data_list/imagenet-s919/test.txt",
        "epochs": 10,
    },
}

MODEL_DEFAULTS = {
    "clip": {
        "batch_size": 64,
        "lr": 1e-6,
        "weight_decay": 0.1,
        "train_scope": "vision",
        "lima_backend": "efficient",
        "mean": OPENAI_CLIP_MEAN,
        "std": OPENAI_CLIP_STD,
    },
    "vit": {
        "batch_size": 32,
        "lr": 1e-4,
        "weight_decay": 0.05,
        "train_scope": "full",
        "lima_backend": "exact",
        "mean": HF_VIT_MEAN,
        "std": HF_VIT_STD,
    },
    "resnet": {
        "batch_size": 32,
        "lr": 1e-5,
        "weight_decay": 0.05,
        "train_scope": "full",
        "lima_backend": "efficient",
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
}


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


@dataclass
class AlignmentDiagnostics:
    count: float = 0.0
    loss_sum: float = 0.0
    bad_gain_sum: float = 0.0
    best_human_gain_sum: float = 0.0
    excess_sum: float = 0.0
    reference_sum: float = 0.0
    adaptive_weight_sum: float = 0.0
    satisfied_sum: float = 0.0

    def add(self, other: "AlignmentDiagnostics") -> None:
        for name in asdict(self):
            setattr(self, name, getattr(self, name) + getattr(other, name))

    def averages(self, prefix: str) -> dict[str, float]:
        if self.count <= 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_loss": 0.0,
                f"{prefix}_mean_bad_gain": 0.0,
                f"{prefix}_mean_best_human_gain": 0.0,
                f"{prefix}_mean_excess_q": 0.0,
                f"{prefix}_mean_reference_r": 0.0,
                f"{prefix}_mean_adaptive_weight": 0.0,
                f"{prefix}_satisfied_fraction": 0.0,
            }
        return {
            f"{prefix}_count": self.count,
            f"{prefix}_loss": self.loss_sum / self.count,
            f"{prefix}_mean_bad_gain": self.bad_gain_sum / self.count,
            f"{prefix}_mean_best_human_gain": self.best_human_gain_sum
            / self.count,
            f"{prefix}_mean_excess_q": self.excess_sum / self.count,
            f"{prefix}_mean_reference_r": self.reference_sum / self.count,
            f"{prefix}_mean_adaptive_weight": self.adaptive_weight_sum
            / self.count,
            f"{prefix}_satisfied_fraction": self.satisfied_sum / self.count,
        }


class UnifiedClassifier(nn.Module):
    """Expose tensor logits for torchvision, Hugging Face ViT, and CLIP."""

    def __init__(
        self,
        raw_model: nn.Module,
        model_kind: str,
        text_features: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.raw_model = raw_model
        self.model_kind = model_kind
        if text_features is None:
            self.register_buffer("text_features", None)
        else:
            self.register_buffer("text_features", text_features.detach())

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.model_kind == "vit":
            return self.raw_model(pixel_values=images).logits
        if self.model_kind == "resnet":
            return self.raw_model(images)
        if self.model_kind == "clip":
            image_features = self.raw_model.get_image_features(pixel_values=images)
            image_features = F.normalize(image_features, dim=-1)
            if self.text_features is None:
                raise RuntimeError("CLIP text features have not been initialized")
            return (
                image_features @ self.text_features
            ) * self.raw_model.logit_scale.exp()
        raise RuntimeError(f"Unsupported model kind: {self.model_kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified subset-attribution human-prior alignment training"
    )
    parser.add_argument("--dataset", choices=DATASETS, default="saliency-bench")
    parser.add_argument("--model", choices=MODEL_DEFAULTS, default="vit")
    parser.add_argument(
        "--loss-variant", choices=["paper", "adaptive_log"], default="paper"
    )
    parser.add_argument("--adaptive-beta", type=float, default=2.0)
    parser.add_argument("--reference-epsilon", type=float, default=1e-6)
    parser.add_argument("--insertion-weight", type=float, default=2.0)
    parser.add_argument("--collaboration-weight", type=float, default=1.0)
    parser.add_argument("--lambda-deviation", type=float, default=0.5)
    parser.add_argument("--lambda-redundancy", type=float, default=0.5)
    parser.add_argument("--alignment-interval", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--prior-overlap-threshold", type=float, default=0.15)
    parser.add_argument("--attribution-stop-confidence", type=float, default=0.8)
    parser.add_argument(
        "--attribution-max-regions",
        "--lima-length",
        dest="attribution_max_regions",
        type=int,
        default=10,
    )
    parser.add_argument("--division-number", type=int, default=50)
    parser.add_argument(
        "--lima-backend", choices=["auto", "exact", "efficient"], default="auto"
    )
    parser.add_argument(
        "--alignment-batch-size",
        type=int,
        default=8,
        help="Number of deviation/redundancy groups per differentiable chunk",
    )
    parser.add_argument("--train-txt", type=str, default=None)
    parser.add_argument("--test-txt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size", "--batch_size", dest="batch_size", type=int, default=None
    )
    parser.add_argument(
        "--num-workers", "--num_workers", dest="num_workers", type=int, default=8
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--weight-decay",
        "--weight_decay",
        dest="weight_decay",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--train-scope",
        choices=["full", "head", "vision", "proj"],
        default=None,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", dest="amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--output-dir", "--output_dir", dest="output_dir", type=str, default=None
    )
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    return resolve_args(args)


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    dataset_defaults = DATASETS[args.dataset]
    model_defaults = MODEL_DEFAULTS[args.model]
    args.train_txt = args.train_txt or dataset_defaults["train_txt"]
    args.test_txt = args.test_txt or dataset_defaults["test_txt"]
    args.epochs = (
        dataset_defaults["epochs"] if args.epochs is None else args.epochs
    )
    args.batch_size = (
        model_defaults["batch_size"]
        if args.batch_size is None
        else args.batch_size
    )
    args.lr = model_defaults["lr"] if args.lr is None else args.lr
    args.weight_decay = (
        model_defaults["weight_decay"]
        if args.weight_decay is None
        else args.weight_decay
    )
    args.train_scope = args.train_scope or model_defaults["train_scope"]
    if args.lima_backend == "auto":
        args.lima_backend = model_defaults["lima_backend"]
    if args.output_dir is None:
        suffix = args.loss_variant
        if args.loss_variant == "adaptive_log":
            suffix += f"_beta{args.adaptive_beta:g}"
        args.output_dir = str(
            ROOT
            / "runs"
            / "prior_alignment"
            / args.dataset
            / args.model
            / suffix
            / f"seed_{args.seed}"
        )

    if args.alignment_interval <= 0:
        raise ValueError("alignment_interval must be positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.lr < 0 or args.weight_decay < 0:
        raise ValueError("lr and weight_decay must be non-negative")
    if args.adaptive_beta <= 0:
        raise ValueError("adaptive_beta must be positive")
    if args.reference_epsilon <= 0:
        raise ValueError("reference_epsilon must be positive")
    if args.insertion_weight < 0 or args.collaboration_weight < 0:
        raise ValueError("set-function weights must be non-negative")
    if args.insertion_weight + args.collaboration_weight <= 0:
        raise ValueError("at least one set-function weight must be positive")
    if args.attribution_max_regions <= 0:
        raise ValueError("attribution_max_regions must be positive")
    if args.division_number <= 0 or args.alignment_batch_size <= 0:
        raise ValueError("division_number and alignment_batch_size must be positive")
    if not 0 <= args.confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be in [0, 1]")
    if not 0 <= args.prior_overlap_threshold <= 1:
        raise ValueError("prior_overlap_threshold must be in [0, 1]")
    if not 0 <= args.attribution_stop_confidence <= 1:
        raise ValueError("attribution_stop_confidence must be in [0, 1]")
    if args.lambda_deviation < 0 or args.lambda_redundancy < 0:
        raise ValueError("alignment loss weights must be non-negative")
    if args.model in ("vit", "resnet") and args.train_scope not in ("full", "head"):
        raise ValueError(f"{args.model} supports train_scope=full|head")
    if args.model == "clip" and args.train_scope not in ("full", "vision", "proj"):
        raise ValueError("clip supports train_scope=full|vision|proj")
    return args


def setup_distributed() -> DistributedContext:
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=1800),
        )
    return DistributedContext(rank, world_size, local_rank, device)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def set_seed(seed: int, rank: int, deterministic: bool) -> None:
    local_seed = seed + rank
    random.seed(local_seed)
    np.random.seed(local_seed)
    torch.manual_seed(local_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(local_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_datasets(
    args: argparse.Namespace,
) -> tuple[Any, Any, dict[str, int]]:
    train_txt = (
        str((ROOT / args.train_txt).resolve())
        if not Path(args.train_txt).is_absolute()
        else args.train_txt
    )
    test_txt = (
        str((ROOT / args.test_txt).resolve())
        if not Path(args.test_txt).is_absolute()
        else args.test_txt
    )
    args.train_txt = train_txt
    args.test_txt = test_txt
    label_to_idx = build_label_map(train_txt, test_txt)
    model_defaults = MODEL_DEFAULTS[args.model]
    common = dict(
        label_to_idx=label_to_idx,
        target_size=(224, 224),
        image_mean=model_defaults["mean"],
        image_std=model_defaults["std"],
        normalize_image=True,
        image_interpolation=InterpolationMode.BICUBIC,
    )
    if args.dataset == "saliency-bench":
        train_dataset = PascalSaliencyDataset(
            list_file=train_txt, mask_interp="nearest", **common
        )
        test_dataset = PascalSaliencyDataset(
            list_file=test_txt, mask_interp="nearest", **common
        )
    else:
        train_dataset = ImageNetSDataset(list_file=train_txt, **common)
        test_dataset = ImageNetSDataset(list_file=test_txt, **common)
    return train_dataset, test_dataset, label_to_idx


def build_loaders(
    train_dataset: Any,
    test_dataset: Any,
    args: argparse.Namespace,
    context: DistributedContext,
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(args.seed + context.rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=context.device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        generator=generator,
    )
    test_loader = None
    if context.is_main:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=context.device.type == "cuda",
            drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
    return train_loader, test_loader, sampler


def build_clip_text_features(
    model: nn.Module,
    class_names: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    from transformers import AutoTokenizer

    model_id = "openai/clip-vit-large-patch14"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    features = []
    model.eval()
    with torch.inference_mode():
        for class_name in class_names:
            text = f"a photo of a {class_name.replace('_', ' ')}."
            tokens = tokenizer(
                [text], padding=True, truncation=True, return_tensors="pt"
            ).to(device)
            feature = model.get_text_features(**tokens)
            features.append(F.normalize(feature, dim=-1).squeeze(0))
    return torch.stack(features, dim=1)


def configure_clip_scope(model: nn.Module, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if scope in ("full", "vision"):
        for parameter in model.vision_model.parameters():
            parameter.requires_grad = True
        for parameter in model.visual_projection.parameters():
            parameter.requires_grad = True
    elif scope == "proj":
        for parameter in model.visual_projection.parameters():
            parameter.requires_grad = True
    model.logit_scale.requires_grad = False


def build_model(
    args: argparse.Namespace,
    label_to_idx: dict[str, int],
    context: DistributedContext,
) -> UnifiedClassifier:
    num_classes = len(label_to_idx)
    freeze_backbone = args.train_scope == "head"
    if args.model == "vit":
        from vit_model import build_vit_classifier

        raw_model = build_vit_classifier(num_classes, freeze_backbone)
        return UnifiedClassifier(raw_model.to(context.device), "vit")
    if args.model == "resnet":
        raw_model = resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
        in_features = raw_model.fc.in_features
        raw_model.fc = nn.Linear(in_features, num_classes)
        if freeze_backbone:
            for name, parameter in raw_model.named_parameters():
                parameter.requires_grad = name.startswith("fc.")
        return UnifiedClassifier(raw_model.to(context.device), "resnet")
    if args.model == "clip":
        from transformers import CLIPModel

        raw_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(context.device)
        class_names = [
            name for name, _ in sorted(label_to_idx.items(), key=lambda x: x[1])
        ]
        text_features = build_clip_text_features(
            raw_model, class_names, context.device
        )
        configure_clip_scope(raw_model, args.train_scope)
        return UnifiedClassifier(raw_model, "clip", text_features)
    raise RuntimeError(f"Unsupported model: {args.model}")


def build_human_lima(
    model: UnifiedClassifier,
    args: argparse.Namespace,
) -> Any:
    if args.lima_backend == "exact":
        from interpretation.HUMAN_LIMA import HumanLIMA
    else:
        from interpretation.HUMAN_LIMA_Efficient import HumanLIMA

    explainer = HumanLIMA(
        model,
        lambda1=args.insertion_weight,
        lambda2=args.collaboration_weight,
        threshold=args.attribution_stop_confidence,
        softmax=True,
    )
    # Both bundled implementations stop after appending when i == k.
    explainer.k = args.attribution_max_regions - 1
    return explainer


def unwrap_model(model: nn.Module) -> UnifiedClassifier:
    return model.module if isinstance(model, DistributedDataParallel) else model


def autocast_context(device: torch.device, enabled: bool):
    if device.type != "cuda" or not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)


def make_grad_scaler(enabled: bool):
    """Construct a GradScaler across the repository's supported torch versions."""

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def create_alignment_examples(
    base_model: UnifiedClassifier,
    human_lima: Any,
    images: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    image_paths: Sequence[str],
    logits: torch.Tensor,
    args: argparse.Namespace,
    loss_config: AlignmentLossConfig,
) -> tuple[list[DeviationExample], list[RedundancyExample], int]:
    probabilities = logits.softmax(dim=-1)
    predictions = probabilities.argmax(dim=-1)
    full_confidences = probabilities.gather(1, labels[:, None]).squeeze(1)
    selected = (predictions == labels) & (
        full_confidences > args.confidence_threshold
    )
    selected_indices = selected.nonzero(as_tuple=True)[0].tolist()
    if not selected_indices:
        return [], [], 0

    was_training = base_model.training
    base_model.eval()
    human_lima.model = base_model
    sample_examples = []
    try:
        for index in selected_indices:
            image = images[index]
            height, width = image.shape[-2:]
            raw_image = cv2.imread(image_paths[index])
            if raw_image is None:
                raise FileNotFoundError(f"Could not read image: {image_paths[index]}")
            if raw_image.shape[:2] != (height, width):
                raw_image = cv2.resize(
                    raw_image, (width, height), interpolation=cv2.INTER_AREA
                )
            region_size = max(
                1,
                int(
                    math.sqrt(
                        raw_image.shape[0]
                        * raw_image.shape[1]
                        / args.division_number
                    )
                ),
            )
            candidate_regions = SubRegionDivision(
                raw_image, mode="slico", region_size=region_size
            )
            selected_regions, saved = human_lima(
                image, candidate_regions, labels[index]
            )
            if not selected_regions:
                continue
            ranked_regions = torch.from_numpy(
                np.stack([region.squeeze(-1) for region in selected_regions])
            ).to(device=image.device, dtype=torch.bool)
            set_scores = saved.get("smdl_score", [])
            sample_examples.append(
                build_alignment_examples(
                    image=image,
                    prior_mask=masks[index],
                    ranked_regions=ranked_regions,
                    set_scores=set_scores,
                    label=labels[index],
                    full_confidence=full_confidences[index],
                    overlap_threshold=args.prior_overlap_threshold,
                    normalizer=loss_config.normalizer,
                )
            )
    finally:
        base_model.train(was_training)
    deviations, redundancies = flatten_examples(sample_examples)
    return deviations, redundancies, len(selected_indices)


def global_scalar(
    value: float,
    device: torch.device,
    context: DistributedContext,
    operation: dist.ReduceOp = dist.ReduceOp.SUM,
) -> float:
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    if context.distributed:
        dist.all_reduce(tensor, op=operation)
    return float(tensor.item())


def diagnostics_from_result(result: AlignmentBatchResult) -> AlignmentDiagnostics:
    return AlignmentDiagnostics(
        count=float(result.loss.numel()),
        loss_sum=float(result.loss.detach().sum().item()),
        bad_gain_sum=float(result.bad_gain.sum().item()),
        best_human_gain_sum=float(result.best_human_gain.sum().item()),
        excess_sum=float(result.excess.sum().item()),
        reference_sum=float(result.reference.sum().item()),
        adaptive_weight_sum=float(result.adaptive_weight.sum().item()),
        satisfied_sum=float(result.satisfied.float().sum().item()),
    )


def reduce_diagnostics(
    diagnostics: AlignmentDiagnostics,
    device: torch.device,
    context: DistributedContext,
) -> AlignmentDiagnostics:
    values = torch.tensor(
        list(asdict(diagnostics).values()), device=device, dtype=torch.float64
    )
    if context.distributed:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return AlignmentDiagnostics(**dict(zip(asdict(diagnostics), values.tolist())))


def stack_deviation_chunk(
    examples: Sequence[DeviationExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.cat(
        [
            torch.stack([item.insertion for item in examples]),
            torch.stack([item.deletion for item in examples]),
        ],
        dim=0,
    ).to(device=device, non_blocking=True)
    labels = torch.stack([item.label for item in examples]).to(device=device)
    human_gain = torch.stack([item.best_human_gain for item in examples]).to(
        device=device
    )
    confidence = torch.stack([item.full_confidence for item in examples]).to(
        device=device
    )
    return inputs, labels, human_gain, confidence


def stack_redundancy_chunk(
    examples: Sequence[RedundancyExample],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = torch.cat(
        [
            torch.stack([item.insertion_after for item in examples]),
            torch.stack([item.deletion_after for item in examples]),
            torch.stack([item.insertion_before for item in examples]),
            torch.stack([item.deletion_before for item in examples]),
        ],
        dim=0,
    ).to(device=device, non_blocking=True)
    labels = torch.stack([item.label for item in examples]).to(device=device)
    human_gain = torch.stack([item.best_human_gain for item in examples]).to(
        device=device
    )
    confidence = torch.stack([item.full_confidence for item in examples]).to(
        device=device
    )
    return inputs, labels, human_gain, confidence


def run_deviation_backward(
    model: nn.Module,
    examples: Sequence[DeviationExample],
    dummy_image: torch.Tensor,
    loss_config: AlignmentLossConfig,
    loss_weight: float,
    chunk_size: int,
    scaler: torch.amp.GradScaler,
    amp: bool,
    context: DistributedContext,
) -> AlignmentDiagnostics:
    local_count = len(examples)
    global_count = int(global_scalar(local_count, context.device, context))
    if global_count == 0:
        return AlignmentDiagnostics()
    local_chunks = math.ceil(local_count / chunk_size)
    max_chunks = int(
        global_scalar(
            local_chunks, context.device, context, operation=dist.ReduceOp.MAX
        )
    )
    diagnostics = AlignmentDiagnostics()
    for chunk_index in range(max_chunks):
        start = chunk_index * chunk_size
        chunk = examples[start : start + chunk_size]
        if chunk:
            inputs, labels, human_gain, confidence = stack_deviation_chunk(
                chunk, context.device
            )
            with autocast_context(context.device, amp):
                logits = model(inputs)
                batch = len(chunk)
                result = deviation_loss_from_logits(
                    logits[:batch],
                    logits[batch:],
                    labels,
                    human_gain,
                    confidence,
                    loss_config,
                )
                # DDP averages gradients, so multiply by world_size for a true
                # global sum/global_count reduction.
                loss = (
                    loss_weight
                    * result.loss.sum()
                    * context.world_size
                    / global_count
                )
            diagnostics.add(diagnostics_from_result(result))
        else:
            dummy = dummy_image.unsqueeze(0).expand(2, -1, -1, -1)
            with autocast_context(context.device, amp):
                loss = model(dummy).sum() * 0.0
        scaler.scale(loss).backward()
    return reduce_diagnostics(diagnostics, context.device, context)


def run_redundancy_backward(
    model: nn.Module,
    examples: Sequence[RedundancyExample],
    dummy_image: torch.Tensor,
    loss_config: AlignmentLossConfig,
    loss_weight: float,
    chunk_size: int,
    scaler: torch.amp.GradScaler,
    amp: bool,
    context: DistributedContext,
) -> AlignmentDiagnostics:
    local_count = len(examples)
    global_count = int(global_scalar(local_count, context.device, context))
    if global_count == 0:
        return AlignmentDiagnostics()
    local_chunks = math.ceil(local_count / chunk_size)
    max_chunks = int(
        global_scalar(
            local_chunks, context.device, context, operation=dist.ReduceOp.MAX
        )
    )
    diagnostics = AlignmentDiagnostics()
    for chunk_index in range(max_chunks):
        start = chunk_index * chunk_size
        chunk = examples[start : start + chunk_size]
        if chunk:
            inputs, labels, human_gain, confidence = stack_redundancy_chunk(
                chunk, context.device
            )
            with autocast_context(context.device, amp):
                logits = model(inputs)
                batch = len(chunk)
                result = redundancy_loss_from_logits(
                    logits[:batch],
                    logits[batch : 2 * batch],
                    logits[2 * batch : 3 * batch],
                    logits[3 * batch :],
                    labels,
                    human_gain,
                    confidence,
                    loss_config,
                )
                loss = (
                    loss_weight
                    * result.loss.sum()
                    * context.world_size
                    / global_count
                )
            diagnostics.add(diagnostics_from_result(result))
        else:
            dummy = dummy_image.unsqueeze(0).expand(4, -1, -1, -1)
            with autocast_context(context.device, amp):
                loss = model(dummy).sum() * 0.0
        scaler.scale(loss).backward()
    return reduce_diagnostics(diagnostics, context.device, context)


def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    loader: DataLoader,
    human_lima: Any,
    loss_config: AlignmentLossConfig,
    args: argparse.Namespace,
    context: DistributedContext,
    epoch: int,
    global_step: int,
) -> tuple[int, dict[str, float]]:
    model.train()
    deviation_epoch = AlignmentDiagnostics()
    redundancy_epoch = AlignmentDiagnostics()
    selected_local = 0.0
    ce_sum = 0.0
    sample_count = 0
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{args.epochs}",
        disable=not context.is_main,
    )
    for images, masks, labels, image_paths, _ in progress:
        global_step += 1
        images = images.to(context.device, non_blocking=True)
        labels = labels.to(context.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(context.device, args.amp):
            logits = model(images)
            ce_loss = F.cross_entropy(logits, labels)
        scaler.scale(ce_loss).backward()
        ce_sum += float(ce_loss.detach().item()) * labels.size(0)
        sample_count += labels.size(0)

        if global_step % args.alignment_interval == 0:
            base_model = unwrap_model(model)
            deviations, redundancies, selected = create_alignment_examples(
                base_model=base_model,
                human_lima=human_lima,
                images=images,
                masks=masks,
                labels=labels,
                image_paths=image_paths,
                logits=logits.detach(),
                args=args,
                loss_config=loss_config,
            )
            selected_local += selected
            model.eval()
            dev_diagnostics = run_deviation_backward(
                model=model,
                examples=deviations,
                dummy_image=images[0].detach(),
                loss_config=loss_config,
                loss_weight=args.lambda_deviation,
                chunk_size=args.alignment_batch_size,
                scaler=scaler,
                amp=args.amp,
                context=context,
            )
            red_diagnostics = run_redundancy_backward(
                model=model,
                examples=redundancies,
                dummy_image=images[0].detach(),
                loss_config=loss_config,
                loss_weight=args.lambda_redundancy,
                chunk_size=args.alignment_batch_size,
                scaler=scaler,
                amp=args.amp,
                context=context,
            )
            model.train()
            if context.is_main:
                deviation_epoch.add(dev_diagnostics)
                redundancy_epoch.add(red_diagnostics)

        scaler.step(optimizer)
        scaler.update()
        if context.is_main and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                ce=f"{ce_loss.item():.4f}",
                dev=int(deviation_epoch.count),
                red=int(redundancy_epoch.count),
            )

    ce_values = torch.tensor(
        [ce_sum, sample_count], device=context.device, dtype=torch.float64
    )
    if context.distributed:
        dist.all_reduce(ce_values, op=dist.ReduceOp.SUM)
    selected_global = global_scalar(selected_local, context.device, context)
    alignment_epoch = AlignmentDiagnostics()
    alignment_epoch.add(deviation_epoch)
    alignment_epoch.add(redundancy_epoch)
    metrics = {
        "train_ce": float(ce_values[0].item() / max(ce_values[1].item(), 1.0)),
        "alignment_selected_samples": selected_global,
        **alignment_epoch.averages("alignment"),
        **deviation_epoch.averages("deviation"),
        **redundancy_epoch.averages("redundancy"),
    }
    return global_step, metrics


@torch.no_grad()
def evaluate(
    model: UnifiedClassifier,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    model.eval()
    top1 = 0
    top2 = 0
    count = 0
    for images, _, labels, _, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast_context(device, amp):
            logits = model(images)
        top1 += int((logits.argmax(dim=1) == labels).sum().item())
        k = min(2, logits.size(1))
        top2 += int(
            (logits.topk(k, dim=1).indices == labels[:, None])
            .any(dim=1)
            .sum()
            .item()
        )
        count += labels.size(0)
    return top1 / max(count, 1), top2 / max(count, 1)


def checkpoint_payload(
    base_model: UnifiedClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    label_to_idx: dict[str, int],
    epoch: int,
    global_step: int,
    best_top1: float,
) -> dict[str, Any]:
    return {
        "model": base_model.raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_top1": best_top1,
        "config": vars(args),
        "label_to_idx": label_to_idx,
    }


def load_resume(
    path: str,
    base_model: UnifiedClassifier,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[int, int, float]:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    base_model.raw_model.load_state_dict(state, strict=True)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return (
        int(checkpoint.get("epoch", 0)) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_top1", 0.0)),
    )


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), indent=2, sort_keys=True))
        return
    context = setup_distributed()
    try:
        set_seed(args.seed, context.rank, args.deterministic)
        train_dataset, test_dataset, label_to_idx = build_datasets(args)
        train_loader, test_loader, train_sampler = build_loaders(
            train_dataset, test_dataset, args, context
        )
        base_model = build_model(args, label_to_idx, context)
        human_lima = build_human_lima(base_model, args)
        model: nn.Module = base_model
        if context.distributed:
            ddp_kwargs: dict[str, Any] = {"broadcast_buffers": False}
            if context.device.type == "cuda":
                ddp_kwargs["device_ids"] = [context.local_rank]
            model = DistributedDataParallel(base_model, **ddp_kwargs)

        parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("No trainable parameters for the selected train scope")
        optimizer = torch.optim.AdamW(
            parameters, lr=args.lr, weight_decay=args.weight_decay
        )
        amp_enabled = args.amp and context.device.type == "cuda"
        scaler = make_grad_scaler(amp_enabled)
        loss_config = AlignmentLossConfig(
            variant=args.loss_variant,
            insertion_weight=args.insertion_weight,
            collaboration_weight=args.collaboration_weight,
            adaptive_beta=args.adaptive_beta,
            epsilon=args.reference_epsilon,
        )

        output_dir = Path(args.output_dir)
        start_epoch = 1
        global_step = 0
        best_top1 = 0.0
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "config.json").write_text(
                json.dumps(vars(args), indent=2, sort_keys=True) + "\n"
            )
        barrier(context)
        if args.resume:
            start_epoch, global_step, best_top1 = load_resume(
                args.resume, base_model, optimizer, scaler, context.device
            )
        barrier(context)

        if context.is_main:
            print(
                f"[Setup] dataset={args.dataset} model={args.model} "
                f"classes={len(label_to_idx)} loss={args.loss_variant} "
                f"beta={args.adaptive_beta:g} world_size={context.world_size}"
            )

        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            global_step, train_metrics = train_one_epoch(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                loader=train_loader,
                human_lima=human_lima,
                loss_config=loss_config,
                args=args,
                context=context,
                epoch=epoch,
                global_step=global_step,
            )

            barrier(context)
            if context.is_main:
                assert test_loader is not None
                top1, top2 = evaluate(
                    base_model, test_loader, context.device, args.amp
                )
                improved = top1 > best_top1
                if improved:
                    best_top1 = top1
                metrics = {
                    "epoch": epoch,
                    "global_step": global_step,
                    "top1": top1,
                    "top2": top2,
                    "best_top1": best_top1,
                    **train_metrics,
                }
                print(
                    f"[Eval] Epoch {epoch}: top1={top1:.4f}, top2={top2:.4f}"
                )
                print("[Alignment] " + json.dumps(train_metrics, sort_keys=True))
                append_jsonl(output_dir / "metrics.jsonl", metrics)
                payload = checkpoint_payload(
                    base_model,
                    optimizer,
                    scaler,
                    args,
                    label_to_idx,
                    epoch,
                    global_step,
                    best_top1,
                )
                torch.save(payload, output_dir / "last.pt")
                if improved:
                    torch.save(payload, output_dir / "best.pt")
                if args.save_every_epoch:
                    torch.save(payload, output_dir / f"epoch_{epoch}.pt")
            barrier(context)
    finally:
        cleanup_distributed(context)


if __name__ == "__main__":
    main()
