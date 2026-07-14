#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/egr/research-optml/ruoyu.chen/anaconda3/envs/prior_alignment/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-seed_results}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

cd "${ROOT_DIR}"

# Space-separated environment overrides are supported, for example:
#   MODELS="clip" SEEDS="0" EPOCHS=2 DRY_RUN=1 ./run_prior_alignment_experiments.sh
read -r -a models <<< "${MODELS:-clip vit resnet}"
read -r -a seeds <<< "${SEEDS:-0 1 2}"
read -r -a betas <<< "${BETAS:-1 2 4}"

if [[ -n "${DATASETS:-}" ]]; then
  read -r -a datasets <<< "${DATASETS}"
else
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
    echo "[info] ImageNet-S919 is incomplete; running Saliency-Bench only."
  fi
fi

variants=(paper)
for beta in "${betas[@]}"; do
  variants+=("adaptive_log_beta${beta}")
done

optional_args=()
[[ -n "${EPOCHS:-}" ]] && optional_args+=(--epochs "${EPOCHS}")
[[ -n "${BATCH_SIZE:-}" ]] && optional_args+=(--batch-size "${BATCH_SIZE}")
[[ -n "${NUM_WORKERS:-}" ]] && optional_args+=(--num-workers "${NUM_WORKERS}")
[[ -n "${ALIGNMENT_INTERVAL:-}" ]] && optional_args+=(--alignment-interval "${ALIGNMENT_INTERVAL}")
[[ -n "${ALIGNMENT_BATCH_SIZE:-}" ]] && optional_args+=(--alignment-batch-size "${ALIGNMENT_BATCH_SIZE}")
[[ -n "${EVALS_PER_EPOCH:-}" ]] && optional_args+=(--evals-per-epoch "${EVALS_PER_EPOCH}")
[[ -n "${EARLY_STOP_ACC_DROP:-}" ]] && optional_args+=(--early-stop-acc-drop "${EARLY_STOP_ACC_DROP}")
[[ -n "${LAMBDA_DEVIATION:-}" ]] && optional_args+=(--lambda-deviation "${LAMBDA_DEVIATION}")
[[ -n "${LAMBDA_REDUNDANCY:-}" ]] && optional_args+=(--lambda-redundancy "${LAMBDA_REDUNDANCY}")
[[ -n "${LIMA_LENGTH:-}" ]] && optional_args+=(--lima-length "${LIMA_LENGTH}")
[[ -n "${LIMA_BACKEND:-}" ]] && optional_args+=(--lima-backend "${LIMA_BACKEND}")
[[ -n "${CONFIDENCE_THRESHOLD:-}" ]] && optional_args+=(--confidence-threshold "${CONFIDENCE_THRESHOLD}")
[[ -n "${PRIOR_OVERLAP_THRESHOLD:-}" ]] && optional_args+=(--prior-overlap-threshold "${PRIOR_OVERLAP_THRESHOLD}")
[[ -n "${ATTRIBUTION_STOP_CONFIDENCE:-}" ]] && optional_args+=(--attribution-stop-confidence "${ATTRIBUTION_STOP_CONFIDENCE}")
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${EXTRA_ARGS}"
  optional_args+=("${extra_args[@]}")
fi

nproc="${NPROC_PER_NODE:-1}"
if (( nproc > 1 )); then
  launcher=(
    "${PYTHON_BIN}" -m torch.distributed.run
    --standalone
    --nproc_per_node="${nproc}"
    "${ROOT_DIR}/train_prior_alignment.py"
  )
else
  launcher=("${PYTHON_BIN}" "${ROOT_DIR}/train_prior_alignment.py")
fi

total_runs=$(( ${#datasets[@]} * ${#models[@]} * ${#variants[@]} * ${#seeds[@]} ))
echo "[setup] datasets=${datasets[*]}"
echo "[setup] models=${models[*]} variants=${variants[*]} seeds=${seeds[*]}"
echo "[setup] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} nproc=${nproc} total_runs=${total_runs}"

for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for variant in "${variants[@]}"; do
      if [[ "${variant}" == "paper" ]]; then
        method="prior_alignment_paper_segmented"
        loss_args=(--loss-variant paper)
      else
        beta="${variant##*beta}"
        method="prior_alignment_adaptive_beta${beta}_segmented"
        loss_args=(--loss-variant adaptive_log --adaptive-beta "${beta}")
      fi

      for seed in "${seeds[@]}"; do
        run_dir="${OUTPUT_ROOT}/${dataset}/${model}/${method}/seed_${seed}"
        metrics_file="${run_dir}/metrics.json"
        log_file="${run_dir}/train.log"

        if [[ -f "${metrics_file}" ]]; then
          echo "[skip] ${dataset}/${model}/${method}/seed_${seed}"
          continue
        fi

        mkdir -p "${run_dir}"
        command=(
          "${launcher[@]}"
          --dataset "${dataset}"
          --model "${model}"
          "${loss_args[@]}"
          --seed "${seed}"
          --output-dir "${run_dir}"
          "${optional_args[@]}"
        )

        resume_mode=0
        if [[ -f "${run_dir}/last.pt" ]]; then
          command+=(--resume "${run_dir}/last.pt")
          resume_mode=1
        fi

        echo "[run] ${command[*]}"
        if [[ "${DRY_RUN:-0}" == "1" ]]; then
          continue
        fi

        export PYTHONHASHSEED="${seed}"
        if (( resume_mode )); then
          "${command[@]}" 2>&1 | tee -a "${log_file}"
        else
          "${command[@]}" 2>&1 | tee "${log_file}"
        fi

        "${PYTHON_BIN}" - "${run_dir}" "${dataset}" "${model}" "${method}" "${seed}" <<'PY'
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
metrics_path = run_dir / "metrics.jsonl"
records = []
if metrics_path.is_file():
    records = [
        json.loads(line)
        for line in metrics_path.read_text().splitlines()
        if line.strip()
    ]
if records:
    best = max(records, key=lambda item: item["top1"])
    summary = {
        "dataset": sys.argv[2],
        "model": sys.argv[3],
        "method": sys.argv[4],
        "seed": int(sys.argv[5]),
        "epoch": best["epoch"],
        "epoch_fraction": best.get("epoch_fraction", 1.0),
        "top1": best["top1"],
        "top2": best["top2"],
        "selection": "best_top1",
        "status": "complete",
    }
else:
    summary = {
        "dataset": sys.argv[2],
        "model": sys.argv[3],
        "method": sys.argv[4],
        "seed": int(sys.argv[5]),
        "epoch": None,
        "epoch_fraction": None,
        "top1": None,
        "top2": None,
        "selection": "skipped_no_alignment_loss",
        "status": "no_alignment_evaluation",
    }
(run_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
if summary["top1"] is None:
    print("[best] skipped: training produced no segment with alignment loss")
else:
    print(
        f"[best] epoch={summary['epoch']} fraction={summary['epoch_fraction']:.2f} "
        f"top1={summary['top1']:.4f} top2={summary['top2']:.4f}"
    )
PY
      done
    done
  done
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  "${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for path in root.glob("*/*/prior_alignment_*/seed_*/metrics.json"):
    record = json.loads(path.read_text())
    if record.get("top1") is not None:
        records.append(record)

groups = {}
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

csv_path = root / "prior_alignment_summary.csv"
md_path = root / "prior_alignment_summary.md"
if rows:
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

table = [
    "| Dataset | Model | Method | Seeds | Top-1 (mean +/- std) | Top-2 (mean +/- std) |",
    "|---|---|---|---:|---:|---:|",
]
for row in rows:
    table.append(
        f"| {row['dataset']} | {row['model']} | {row['method']} | {row['seeds']} | "
        f"{row['top1_mean']:.4f} +/- {row['top1_std']:.4f} | "
        f"{row['top2_mean']:.4f} +/- {row['top2_std']:.4f} |"
    )
md_path.write_text("\n".join(table) + "\n")
print(f"[done] summaries: {csv_path} and {md_path}")
PY
fi
