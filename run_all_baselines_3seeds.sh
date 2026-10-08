#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/prior_alignment/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

completed_runs=0
if [[ -d seed_results ]]; then
  completed_runs="$(find seed_results -type f -name metrics.json | wc -l)"
fi
echo "[info] Resume mode: skipping ${completed_runs} completed runs with metrics.json."
echo "[info] Interrupted runs without metrics.json will restart from epoch 1."

datasets=(saliency-bench)
if "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

complete = True
for list_name in (
    "data_list/imagenet-s919/train.txt",
    "data_list/imagenet-s919/test.txt",
):
    list_file = Path(list_name)
    if not list_file.is_file():
        complete = False
        break
    for line in list_file.read_text().splitlines():
        if not line.strip():
            continue
        image_path, mask_path, _ = line.split()
        if not Path(image_path).is_file() or not Path(mask_path).is_file():
            complete = False
            break
    if not complete:
        break
raise SystemExit(0 if complete else 1)
PY
then
  datasets+=(imagenet-s919)
else
  echo "[info] ImageNet-S919 images are incomplete; running Saliency-Bench only."
  echo "[info] Re-run this script after populating ImageNetS919 images; completed runs will be reused."
fi

extra_args=()
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON_BIN}" seed_experiments/run_baselines.py \
  --python "${PYTHON_BIN}" \
  --datasets "${datasets[@]}" \
  --models clip vit resnet \
  --methods finetuning rrr xil megl \
  --seeds 0 1 2 \
  --epochs "${EPOCHS:-10}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  "${extra_args[@]}" \
  "$@"

echo "Results are available in seed_results/summary.csv and seed_results/summary.md"
