#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/prior_alignment/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT_DIR}"

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi
if [[ "${FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi

"${PYTHON_BIN}" seed_experiments/run_point_game.py \
  --datasets saliency-bench imagenet-s919 \
  --models clip vit resnet \
  --methods finetuning rrr xil megl \
  --seeds 0 1 2 \
  --max-samples "${MAX_SAMPLES:-100}" \
  --division-number "${DIVISION_NUMBER:-50}" \
  "${extra_args[@]}" \
  "$@"

echo "Point Game results are in interpretation_results/summary.csv and summary.md"
