"""Run one existing training script after setting all relevant random seeds."""

from __future__ import annotations

import argparse
import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except (AttributeError, TypeError):
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    set_seed(args.seed)
    script = args.script.resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    forwarded = args.script_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    print(f"[SeedRunner] seed={args.seed} script={script}", flush=True)
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
