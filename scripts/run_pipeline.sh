#!/usr/bin/env bash
# Select an architecture and run its datasets with shared runtime settings.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"

usage() {
    cat <<'USAGE'
Usage: run_pipeline.sh [--method METHOD] --dataset DATASET [options]

Methods:  aegis (default), rlfe_ids, scala_nids, ma_ids, sera_ids
Datasets: kdd, cicids2018, ton_iot, unsw_nb15, all

Common options: --all-flows | --max-flows N, --dry-run, --output-dir PATH
Baseline options: --construction-flows N, --epochs N, --task binary|family
Single-method option: --config PATH

Run each architecture separately:
  run_pipeline.sh --method aegis --dataset all --all-flows
  run_pipeline.sh --method rlfe_ids --dataset all --all-flows
  run_pipeline.sh --method scala_nids --dataset all --all-flows
  run_pipeline.sh --method ma_ids --dataset all --all-flows
  run_pipeline.sh --method sera_ids --dataset all --all-flows

Datasets run sequentially and stop on the first failure. Options pass to Python.
Optional method groups: 'baselines' runs four baselines; 'all' includes AEGIS.
USAGE
}

method="aegis"
dataset=""
options=()
baseline_only=false
custom_config=false
while (($#)); do
    case "$1" in
        --method|--dataset)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            if [[ "$1" == --method ]]; then method="$2"; else dataset="$2"; fi
            shift 2
            ;;
        --method=*) method="${1#*=}"; shift ;;
        --dataset=*) dataset="${1#*=}"; shift ;;
        --help|-h) usage; exit 0 ;;
        --construction-flows|--epochs|--task)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            baseline_only=true
            options+=("$1" "$2")
            shift 2
            ;;
        --construction-flows=*|--epochs=*|--task=*)
            baseline_only=true; options+=("$1"); shift ;;
        --config|--config=*) custom_config=true; options+=("$1"); shift ;;
        *) options+=("$1"); shift ;;
    esac
done

case "$method" in
    aegis|rlfe_ids|scala_nids|ma_ids|sera_ids) methods=("$method") ;;
    baselines) methods=(rlfe_ids scala_nids ma_ids sera_ids) ;;
    all) methods=(aegis rlfe_ids scala_nids ma_ids sera_ids) ;;
    *) echo "Unknown method: $method" >&2; usage >&2; exit 2 ;;
esac
case "$dataset" in
    kdd|cicids2018|ton_iot|unsw_nb15) datasets=("$dataset") ;;
    all) datasets=(kdd cicids2018 ton_iot unsw_nb15) ;;
    *) echo "Specify a supported --dataset." >&2; usage >&2; exit 2 ;;
esac
if [[ "$baseline_only" == true && ( "$method" == aegis || "$method" == all ) ]]; then
    echo "Baseline-only options require a baseline --method or --method baselines." >&2
    exit 2
fi
if [[ "$custom_config" == true && "$method" == all ]]; then
    echo "AEGIS and baselines use different configuration schemas; select a method for --config." >&2
    exit 2
fi

export OMP_NUM_THREADS="${ZETA_NATIVE_THREADS:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

cd "$project_root"
if [[ ! -x .venv/bin/python ]]; then
    echo "Missing Python environment: $project_root/.venv/bin/python" >&2
    exit 1
fi
for selected_method in "${methods[@]}"; do
    for selected_dataset in "${datasets[@]}"; do
        printf 'Starting method: %s | dataset: %s\n' "$selected_method" "$selected_dataset"
        if [[ "$selected_method" == aegis ]]; then
            module=(src)
        else
            module=(src.baselines --method "$selected_method")
        fi
        taskset --cpu-list "${ZETA_CPU_AFFINITY:-0-5,8-31}" \
            .venv/bin/python -m "${module[@]}" --dataset "$selected_dataset" "${options[@]}"
    done
done
