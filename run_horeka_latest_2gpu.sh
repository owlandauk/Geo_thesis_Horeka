#!/bin/bash
# Submit with:
#   sbatch run_horeka_latest_2gpu.sh
#
# Defaults to a 300-image HoreKa validation run on 2 A100 GPUs. Override at
# submit time when needed, for example:
#   sbatch --export=ALL,LIMIT=50,BATCH_SIZE=4,RUN_TAG=smoke run_horeka_latest_2gpu.sh
#   sbatch --export=ALL,LIMIT=full,RUN_TAG=full run_horeka_latest_2gpu.sh

#SBATCH --job-name=geo-horeka-latest
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025551
#SBATCH --constraint=LSDF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=geo_pipeline/results/horeka_latest_2gpu_%j.out
#SBATCH --error=geo_pipeline/results/horeka_latest_2gpu_%j.err

set -euo pipefail

START_TS=$(date +%s)
START_HUMAN=$(date)

finish() {
  local rc=$?
  local end_ts elapsed
  end_ts=$(date +%s)
  elapsed=$((end_ts - START_TS))

  echo
  echo "========== Runtime =========="
  echo "Started: ${START_HUMAN}"
  echo "Ended:   $(date)"
  echo "Exit code: ${rc}"
  echo "Elapsed seconds: ${elapsed}"
  echo "Elapsed minutes: $((elapsed / 60))"
  awk -v elapsed="${elapsed}" 'BEGIN {printf "Elapsed hours: %.2f\n", elapsed / 3600}'
  echo "============================="
}
trap finish EXIT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
ENV_DIR="${ENV_DIR:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/envs/geo-vllm}"

cd "${REPO_DIR}"
mkdir -p geo_pipeline/results

LIMIT="${LIMIT:-300}"
BATCH_SIZE="${BATCH_SIZE:-8}"
RUN_TAG="${RUN_TAG:-latest_limit${LIMIT}_2gpu}"
OUT_JSON="${OUT_JSON:-geo_pipeline/results/horeka_${RUN_TAG}.json}"
STRICT_CHILD_GEOCODE="${STRICT_CHILD_GEOCODE:-0}"
ALLOW_BARE_CITY_GEOCODE="${ALLOW_BARE_CITY_GEOCODE:-1}"

JOB_TMP_DIR="/scratch/slurm_tmpdir/job_${SLURM_JOB_ID:-manual}"
if [ ! -d "${JOB_TMP_DIR}" ] || [ ! -w "${JOB_TMP_DIR}" ]; then
  JOB_TMP_DIR="/tmp/${USER}_${SLURM_JOB_ID:-manual}"
  mkdir -p "${JOB_TMP_DIR}"
fi
export TMPDIR="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"
export TORCHINDUCTOR_CACHE_DIR="${JOB_TMP_DIR}/torchinductor"
export TRITON_CACHE_DIR="${JOB_TMP_DIR}/triton"
export CUDA_CACHE_PATH="${JOB_TMP_DIR}/cuda"
export XDG_CACHE_HOME="${JOB_TMP_DIR}/xdg"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate "${ENV_DIR}"

export MLLM_BACKEND=vllm
export VLLM_TP=2
export MODEL_PATH="${MODEL_PATH:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/models/qwen2.5-vl-7b}"
export YFCC4K_IMG_DIR="${YFCC4K_IMG_DIR:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k}"
export YFCC4K_GPS_CSV="${YFCC4K_GPS_CSV:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k_gps.csv}"
export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.80}"

EVAL_ARGS=(
  --batch_size "${BATCH_SIZE}"
  --out "${OUT_JSON}"
)

if [ "${LIMIT}" != "full" ] && [ "${LIMIT}" != "none" ] && [ -n "${LIMIT}" ]; then
  EVAL_ARGS+=(--limit "${LIMIT}")
fi
if [ "${STRICT_CHILD_GEOCODE}" = "1" ]; then
  EVAL_ARGS+=(--strict_child_geocode)
fi
if [ "${ALLOW_BARE_CITY_GEOCODE}" = "0" ]; then
  EVAL_ARGS+=(--disable_bare_city_geocode)
fi

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: ${START_HUMAN}"
echo "Repo: ${REPO_DIR}"
echo "Git branch: $(git branch --show-current 2>/dev/null || true)"
echo "Git commit: $(git rev-parse --short HEAD 2>/dev/null || true)"
echo "Git status:"
git status --short 2>/dev/null || true
echo "Env: ${ENV_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "YFCC4K_IMG_DIR: ${YFCC4K_IMG_DIR}"
echo "YFCC4K_GPS_CSV: ${YFCC4K_GPS_CSV}"
echo "VLLM_TP: ${VLLM_TP}"
echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "LIMIT: ${LIMIT}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "RUN_TAG: ${RUN_TAG}"
echo "OUT_JSON: ${OUT_JSON}"
echo "STRICT_CHILD_GEOCODE: ${STRICT_CHILD_GEOCODE}"
echo "ALLOW_BARE_CITY_GEOCODE: ${ALLOW_BARE_CITY_GEOCODE}"

nvidia-smi

python geo_pipeline/evaluate.py "${EVAL_ARGS[@]}"

python geo_pipeline/analyze_results.py --pred "${OUT_JSON}"
