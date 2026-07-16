"""Generate LIMA explanations and evaluate Point Game for best checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet101
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from transformers import AutoTokenizer, CLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from interpretation.LIMA import BlackBoxSingleModalCounterfactualSubModularExplanation
from utils import SubRegionDivision
from vit_model import LogitsOnlyViT, VIT_IMAGE_MEAN, VIT_IMAGE_STD, build_vit_classifier


DATASETS = {
    "saliency-bench": ROOT / "data_list/saliency-bench/test.txt",
    "imagenet-s919": ROOT / "data_list/imagenet-s919/test_one_per_class.txt",
}
MODELS = ("clip", "vit", "resnet")
METHODS = ("finetuning", "rrr", "xil", "megl")
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class CLIPAdaptor:
    def __init__(self, model: CLIPModel, text_features: torch.Tensor):
        self.model = model
        self.text_features = text_features

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self.model.get_image_features(pixel_values=pixel_values)
        features = features / features.norm(dim=-1, keepdim=True)
        return (features @ self.text_features) * self.model.logit_scale.exp()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate predicted-class LIMA explanations and Point Game metrics."
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--division-number", type=int, default=50)
    parser.add_argument("--point-game-threshold", type=float, default=0.2)
    parser.add_argument("--output-root", type=Path, default=ROOT / "interpretation_results")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_items(path: Path) -> list[tuple[str, str, str]]:
    return [tuple(line.split()) for line in path.read_text().splitlines() if line.strip()]


def build_label_map(items: list[tuple[str, str, str]]) -> dict[str, int]:
    return {label: index for index, label in enumerate(sorted({item[2] for item in items}))}


def load_state_dict(checkpoint: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict):
        for key in ("model", "state_dict", "net", "ema"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint is not a state dict: {checkpoint}")
    return {
        (key[len("module."):] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def build_model(
    model_name: str,
    label_to_idx: dict[str, int],
    checkpoint: Path,
    device: torch.device,
) -> tuple[Callable[[torch.Tensor], torch.Tensor], object]:
    state = load_state_dict(checkpoint)
    num_classes = len(label_to_idx)

    if model_name == "clip":
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        idx_to_label = {index: label for label, index in label_to_idx.items()}
        features = []
        with torch.no_grad():
            for index in range(num_classes):
                text = f"a photo of a {idx_to_label[index]}."
                tokens = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
                feature = model.get_text_features(**tokens)
                features.append(feature[0] / feature[0].norm())
        adaptor = CLIPAdaptor(model, torch.stack(features, dim=1))
        return adaptor, model

    if model_name == "vit":
        model = build_vit_classifier(num_classes)
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        adaptor = LogitsOnlyViT(model).to(device).eval()
        return adaptor, model

    model = resnet101(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, model


def spatial_transforms(
    dataset: str, model: str, method: str
) -> tuple[Callable[[Image.Image], Image.Image], Callable[[Image.Image], Image.Image], tuple, tuple, str]:
    if model == "clip":
        image_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
        mask_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        return image_spatial, mask_spatial, CLIP_MEAN, CLIP_STD, "resize_224_bicubic"
    if model == "vit":
        image_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC)
        mask_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        return image_spatial, mask_spatial, VIT_IMAGE_MEAN, VIT_IMAGE_STD, "resize_224_bicubic"

    # Saliency-Bench fine-tuning used torchvision's standard 256-resize and
    # 224-center-crop at test time. The other ResNet scripts resize to 224x224.
    if dataset == "saliency-bench" and method == "finetuning":
        image_spatial = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
        ])
        mask_spatial = transforms.Compose([
            transforms.Resize(256, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(224),
        ])
        geometry = "resize_256_center_crop_224_bilinear"
    else:
        image_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)
        mask_spatial = transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST)
        geometry = "resize_224_bilinear"
    return image_spatial, mask_spatial, IMAGENET_MEAN, IMAGENET_STD, geometry


def load_binary_mask(mask_path: Path) -> Image.Image:
    if mask_path.suffix.lower() == ".npy":
        mask = np.load(mask_path)
        if mask.ndim == 3:
            mask = mask.max(axis=-1)
    else:
        raw = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(mask_path)
        mask = raw.max(axis=-1) if raw.ndim == 3 else raw
    binary = (mask > 0).astype(np.uint8) * 255
    return Image.fromarray(binary, mode="L")


def output_stem(index: int, image_path: str) -> str:
    # Basenames are not unique in Saliency-Bench, even among the first 100.
    return f"{index:04d}_{Path(image_path).stem}"


def evaluate_saved(
    items: list[tuple[str, str, str]],
    label_to_idx: dict[str, int],
    output_dir: Path,
    mask_spatial: Callable[[Image.Image], Image.Image],
    threshold: float,
) -> dict:
    rows = []
    for index, (image_path, mask_path, label_name) in enumerate(items):
        stem = output_stem(index, image_path)
        explanation_path = output_dir / "npy" / f"{stem}.npy"
        metadata_path = output_dir / "json" / f"{stem}.json"
        if not explanation_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"Missing explanation pair for {image_path}")
        explanation = np.load(explanation_path)
        metadata = json.loads(metadata_path.read_text())
        selected = np.asarray(explanation[0]).squeeze() > 0
        gt_mask = np.asarray(mask_spatial(load_binary_mask(ROOT / mask_path))) > 0
        selected_pixels = int(selected.sum())
        overlap_ratio = (
            float(np.logical_and(selected, gt_mask).sum() / selected_pixels)
            if selected_pixels else 0.0
        )
        pg_success = overlap_ratio > threshold
        correct = int(metadata["predict_label"]) == label_to_idx[label_name]
        rows.append({"pg_success": pg_success, "correct": correct, "overlap_ratio": overlap_ratio})

    n = len(rows)
    pg_rows = [row for row in rows if row["pg_success"]]
    correct_rows = [row for row in rows if row["correct"]]
    wrong_rows = [row for row in rows if not row["correct"]]
    return {
        "n": n,
        "point_game": statistics.mean(row["pg_success"] for row in rows),
        "sample_top1_accuracy": statistics.mean(row["correct"] for row in rows),
        "top1_accuracy_given_pg1": (
            statistics.mean(row["correct"] for row in pg_rows) if pg_rows else None
        ),
        "point_game_given_correct": (
            statistics.mean(row["pg_success"] for row in correct_rows) if correct_rows else None
        ),
        "point_game_given_wrong": (
            statistics.mean(row["pg_success"] for row in wrong_rows) if wrong_rows else None
        ),
        "pg1_count": len(pg_rows),
        "correct_count": len(correct_rows),
        "mean_overlap_ratio": statistics.mean(row["overlap_ratio"] for row in rows),
    }


def run_one(
    dataset: str,
    model_name: str,
    method: str,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict | None:
    run_dir = ROOT / "seed_results" / dataset / model_name / method / f"seed_{seed}"
    training_metrics_path = run_dir / "metrics.json"
    if not training_metrics_path.is_file():
        print(f"[missing] {dataset}/{model_name}/{method}/seed_{seed}: no training metrics")
        return None
    training_metrics = json.loads(training_metrics_path.read_text())
    epoch = int(training_metrics["epoch"])
    checkpoint = run_dir / "checkpoints" / f"best_epoch{epoch}.pt"
    if not checkpoint.is_file():
        print(f"[missing] {dataset}/{model_name}/{method}/seed_{seed}: {checkpoint}")
        return None

    output_dir = (
        args.output_root / dataset / model_name / method / f"seed_{seed}" / f"best_epoch_{epoch}"
    )
    metrics_path = output_dir / "point_game_metrics.json"
    if metrics_path.is_file() and not args.force:
        metrics = json.loads(metrics_path.read_text())
        if metrics.get("max_samples") == args.max_samples:
            print(f"[skip] {dataset}/{model_name}/{method}/seed_{seed}")
            return metrics

    print(
        f"[run] {dataset}/{model_name}/{method}/seed_{seed} "
        f"checkpoint=best_epoch{epoch}.pt samples={args.max_samples}"
    )
    if args.dry_run:
        return None

    all_items = read_items(DATASETS[dataset])
    label_to_idx = build_label_map(all_items)
    items = all_items[:args.max_samples]
    image_spatial, mask_spatial, mean, std, geometry = spatial_transforms(
        dataset, model_name, method
    )
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    predictor, owning_model = build_model(model_name, label_to_idx, checkpoint, device)
    explainer = BlackBoxSingleModalCounterfactualSubModularExplanation(
        predictor, lambda1=1, lambda2=1, softmax=True, device=str(device)
    )

    npy_dir = output_dir / "npy"
    json_dir = output_dir / "json"
    npy_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    for index, (image_path, mask_path, label_name) in enumerate(
        tqdm(items, desc=f"{dataset}/{model_name}/{method}/s{seed}")
    ):
        stem = output_stem(index, image_path)
        explanation_path = npy_dir / f"{stem}.npy"
        metadata_path = json_dir / f"{stem}.json"
        if explanation_path.is_file() and metadata_path.is_file() and not args.force:
            continue
        image = Image.open(ROOT / image_path).convert("RGB")
        processed = image_spatial(image)
        image_tensor = normalize(processed).to(device)
        with torch.no_grad():
            predicted_label = int(predictor(image_tensor.unsqueeze(0)).argmax(dim=1).item())
        cv_image = cv2.cvtColor(np.asarray(processed), cv2.COLOR_RGB2BGR)
        region_size = max(1, int((224 * 224 / args.division_number) ** 0.5))
        regions = SubRegionDivision(cv_image, mode="slico", region_size=region_size)
        selected, metadata = explainer(image_tensor, regions, predicted_label)
        metadata.update({
            "predict_label": predicted_label,
            "true_label": label_to_idx[label_name],
            "correct": predicted_label == label_to_idx[label_name],
            "image_path": image_path,
            "mask_path": mask_path,
            "checkpoint": str(checkpoint),
            "preprocessing": {"geometry": geometry, "mean": mean, "std": std},
        })
        np.save(explanation_path, np.asarray(selected, dtype=np.uint8))
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    result = {
        "dataset": dataset,
        "model": model_name,
        "method": method,
        "seed": seed,
        "best_epoch": epoch,
        "checkpoint": str(checkpoint),
        "max_samples": args.max_samples,
        "division_number": args.division_number,
        "point_game_threshold": args.point_game_threshold,
        "preprocessing": {"geometry": geometry, "mean": mean, "std": std},
        **evaluate_saved(items, label_to_idx, output_dir, mask_spatial, args.point_game_threshold),
    }
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[result] PG={result['point_game']:.4f} "
        f"Top1(PG=1)={result['top1_accuracy_given_pg1']} "
        f"sample-Top1={result['sample_top1_accuracy']:.4f}"
    )
    del explainer, predictor, owning_model
    torch.cuda.empty_cache()
    return result


def aggregate(records: list[dict], output_root: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        groups.setdefault((record["dataset"], record["model"], record["method"]), []).append(record)
    rows = []
    metric_names = ("point_game", "top1_accuracy_given_pg1", "sample_top1_accuracy")
    for (dataset, model, method), values in sorted(groups.items()):
        row = {
            "dataset": dataset,
            "model": model,
            "method": method,
            "n_seeds": len(values),
            "seeds": ",".join(str(value["seed"]) for value in sorted(values, key=lambda x: x["seed"])),
        }
        for metric in metric_names:
            samples = [float(value[metric]) for value in values if value[metric] is not None]
            row[f"{metric}_mean"] = statistics.mean(samples) if samples else ""
            row[f"{metric}_std"] = statistics.stdev(samples) if len(samples) > 1 else ""
        rows.append(row)

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    md = [
        "| Dataset | Model | Method | Seeds | Point Game | Top-1 Acc. (PG=1) | Sample Top-1 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        def display(metric: str) -> str:
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            if mean == "":
                return "N/A"
            return f"{mean:.4f}" if std == "" else f"{mean:.4f} ± {std:.4f}"
        md.append(
            f"| {row['dataset']} | {row['model']} | {row['method']} | {row['seeds']} | "
            f"{display('point_game')} | {display('top1_accuracy_given_pg1')} | "
            f"{display('sample_top1_accuracy')} |"
        )
    (output_root / "summary.md").write_text("\n".join(md) + "\n")
    print(f"[done] {csv_path} and {output_root / 'summary.md'}")


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0 or args.division_number <= 0:
        raise SystemExit("--max-samples and --division-number must be positive")
    if not args.dry_run and not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for explanation generation")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    records = []
    for dataset in args.datasets:
        for model in args.models:
            for method in args.methods:
                for seed in args.seeds:
                    result = run_one(dataset, model, method, seed, args, device)
                    if result is not None:
                        records.append(result)
    if not args.dry_run:
        aggregate(records, args.output_root)


if __name__ == "__main__":
    main()
