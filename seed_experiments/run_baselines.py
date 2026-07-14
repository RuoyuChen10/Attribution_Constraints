"""Run baseline training with multiple seeds and aggregate final-epoch accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = Path(__file__).with_name("seed_entrypoint.py")
EVAL_RE = re.compile(
    r"\[Eval\]\s+Epoch\s+(?P<epoch>\d+):\s+"
    r"top1=(?P<top1>[0-9.]+),\s+top2=(?P<top2>[0-9.]+)"
)

DATASETS = {
    "saliency-bench": {
        "directory": "vision_task_saliency_bench",
        "train_txt": "data_list/saliency-bench/train.txt",
        "test_txt": "data_list/saliency-bench/test.txt",
    },
    "imagenet-s919": {
        "directory": "vision_task_imagenet-s",
        "train_txt": "data_list/imagenet-s919/train.txt",
        "test_txt": "data_list/imagenet-s919/test.txt",
    },
}
METHOD_FILES = {
    "finetuning": "baseline_{model}.py",
    "rrr": "RRR_{model}.py",
    "xil": "XIL_{model}.py",
    "megl": "MEGL_{model}.py",
}


@dataclass(frozen=True)
class Run:
    dataset: str
    model: str
    method: str
    seed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline methods for several seeds. Metrics are taken from the "
            "last epoch, then summarized with the sample standard deviation."
        )
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=["saliency-bench"])
    parser.add_argument("--models", nargs="+", choices=["clip", "vit", "resnet"], default=["clip", "vit", "resnet"])
    parser.add_argument("--methods", nargs="+", choices=METHOD_FILES, default=list(METHOD_FILES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path("seed_results"))
    parser.add_argument("--python", default=sys.executable, help="Python executable in the training environment")
    parser.add_argument("--force", action="store_true", help="rerun completed seed directories")
    parser.add_argument("--dry-run", action="store_true", help="print commands without training")
    return parser.parse_args()


def iter_runs(args: argparse.Namespace) -> Iterable[Run]:
    for dataset in args.datasets:
        for model in args.models:
            for method in args.methods:
                for seed in args.seeds:
                    yield Run(dataset, model, method, seed)


def read_rows(list_file: Path) -> list[list[str]]:
    rows = [line.split() for line in list_file.read_text().splitlines() if line.strip()]
    malformed = [row for row in rows if len(row) != 3]
    if malformed:
        raise ValueError(f"Malformed rows in {list_file}: expected image mask label")
    return rows


def preflight(datasets: list[str]) -> None:
    errors: list[str] = []
    for dataset in datasets:
        config = DATASETS[dataset]
        for split in ("train_txt", "test_txt"):
            list_file = ROOT / config[split]
            if not list_file.is_file():
                errors.append(f"missing list file: {list_file}")
                continue
            rows = read_rows(list_file)
            missing_images = sum(not (ROOT / row[0]).is_file() for row in rows)
            missing_masks = sum(not (ROOT / row[1]).is_file() for row in rows)
            if missing_images or missing_masks:
                errors.append(
                    f"{dataset}/{split}: {missing_images}/{len(rows)} images and "
                    f"{missing_masks}/{len(rows)} masks are missing"
                )
    if errors:
        raise SystemExit("Dataset preflight failed:\n  - " + "\n  - ".join(errors))


def parse_metrics(log_text: str) -> dict[str, float | int]:
    matches = list(EVAL_RE.finditer(log_text))
    if not matches:
        raise RuntimeError("No '[Eval] Epoch ... top1=..., top2=...' line found")
    last = matches[-1]
    return {
        "epoch": int(last.group("epoch")),
        "top1": float(last.group("top1")),
        "top2": float(last.group("top2")),
    }


def run_one(run: Run, args: argparse.Namespace, output_root: Path) -> dict:
    config = DATASETS[run.dataset]
    script = ROOT / config["directory"] / METHOD_FILES[run.method].format(model=run.model)
    run_dir = output_root / run.dataset / run.model / run.method / f"seed_{run.seed}"
    metrics_file = run_dir / "metrics.json"
    log_file = run_dir / "train.log"
    if metrics_file.is_file() and not args.force:
        print(f"[skip] {run.dataset}/{run.model}/{run.method}/seed={run.seed}")
        return json.loads(metrics_file.read_text())

    command = [
        args.python,
        str(ENTRYPOINT),
        "--seed", str(run.seed),
        "--script", str(script),
        "--",
        "--train_txt", str(ROOT / config["train_txt"]),
        "--test_txt", str(ROOT / config["test_txt"]),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--output_dir", str(run_dir / "checkpoints"),
    ]
    print("[run] " + " ".join(command), flush=True)
    if args.dry_run:
        return {}

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(run.seed)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    lines: list[str] = []
    with log_file.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            lines.append(line)
        return_code = process.wait()
    if return_code:
        raise SystemExit(f"Training failed with exit code {return_code}; see {log_file}")

    metrics = {
        "dataset": run.dataset,
        "model": run.model,
        "method": run.method,
        "seed": run.seed,
        **parse_metrics("".join(lines)),
    }
    metrics_file.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def aggregate(records: list[dict], output_root: Path) -> None:
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for record in records:
        key = (record["dataset"], record["model"], record["method"])
        groups.setdefault(key, []).append(record)

    rows = []
    for (dataset, model, method), values in sorted(groups.items()):
        values.sort(key=lambda item: item["seed"])
        top1 = [item["top1"] for item in values]
        top2 = [item["top2"] for item in values]
        rows.append({
            "dataset": dataset,
            "model": model,
            "method": method,
            "n": len(values),
            "seeds": ",".join(str(item["seed"]) for item in values),
            "top1_mean": statistics.mean(top1),
            "top1_std": statistics.stdev(top1) if len(top1) > 1 else 0.0,
            "top2_mean": statistics.mean(top2),
            "top2_std": statistics.stdev(top2) if len(top2) > 1 else 0.0,
        })

    output_root.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (output_root / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "| Dataset | Model | Method | Seeds | Top-1 (mean ± std) | Top-2 (mean ± std) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            f"| {row['dataset']} | {row['model']} | {row['method']} | {row['seeds']} | "
            f"{row['top1_mean']:.4f} ± {row['top1_std']:.4f} | "
            f"{row['top2_mean']:.4f} ± {row['top2_std']:.4f} |"
        )
    (output_root / "summary.md").write_text("\n".join(table) + "\n")
    print(f"\n[done] summaries: {output_root / 'summary.csv'} and {output_root / 'summary.md'}")


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("--seeds contains duplicates")
    if not args.dry_run:
        preflight(args.datasets)
    output_root = (ROOT / args.output_root).resolve() if not args.output_root.is_absolute() else args.output_root
    records = [run_one(run, args, output_root) for run in iter_runs(args)]
    if not args.dry_run:
        aggregate(records, output_root)


if __name__ == "__main__":
    main()
