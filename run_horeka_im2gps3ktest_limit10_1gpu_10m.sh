#!/bin/bash
# Submit with:
#   sbatch run_horeka_im2gps3ktest_limit10_1gpu_10m.sh
#
# This HoreKa job runs the 10-image IM2GPS3KTEST smoke evaluation on 1 A100 GPU for up to
# 10 minutes and prints the wall-clock runtime at the end of the Slurm log.

#SBATCH --job-name=geo-im2gps3ktest-10-1gpu
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025551
#SBATCH --constraint=LSDF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH --time=00:10:00
#SBATCH --chdir=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation
#SBATCH --output=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation/geo_pipeline/results/horeka_im2gps3ktest_limit10_1gpu_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation/geo_pipeline/results/horeka_im2gps3ktest_limit10_1gpu_%j.err

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

REPO_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation
ENV_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/envs/geo-vllm

cd "${REPO_DIR}"
mkdir -p geo_pipeline/results

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
export VLLM_TP=1
export MODEL_PATH=/hkfs/work/workspace/scratch/tj3409-SichengZuo/models/qwen2.5-vl-7b
export IM2GPS3KTEST_IMG_DIR="${IM2GPS3KTEST_IMG_DIR:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/im2gps3ktest/im2gps3ktest}"
export IM2GPS3KTEST_GPS_CSV="${IM2GPS3KTEST_GPS_CSV:-/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/im2gps3ktest/im2gps3ktest_gps.csv}"
export VLLM_GPU_MEMORY_UTILIZATION=0.80

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: ${START_HUMAN}"
echo "Repo: ${REPO_DIR}"
echo "Env: ${ENV_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "TORCHINDUCTOR_CACHE_DIR: ${TORCHINDUCTOR_CACHE_DIR}"
echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "IM2GPS3KTEST_IMG_DIR: ${IM2GPS3KTEST_IMG_DIR}"
echo "IM2GPS3KTEST_GPS_CSV: ${IM2GPS3KTEST_GPS_CSV}"
echo "VLLM_TP: ${VLLM_TP}"
echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION}"
echo "WEB_SEARCH_ENABLED: ${WEB_SEARCH_ENABLED:-0}"

if [ ! -d "${IM2GPS3KTEST_IMG_DIR}" ]; then
  echo "[ERROR] IM2GPS3KTEST_IMG_DIR not found: ${IM2GPS3KTEST_IMG_DIR}"
  exit 2
fi
if [ ! -f "${IM2GPS3KTEST_GPS_CSV}" ]; then
  echo "[ERROR] IM2GPS3KTEST_GPS_CSV not found: ${IM2GPS3KTEST_GPS_CSV}"
  exit 2
fi

nvidia-smi

python -u geo_pipeline/evaluate.py \
  --batch_size 4 \
  --start 0 \
  --limit 10 \
  --img_dir "${IM2GPS3KTEST_IMG_DIR}" \
  --gps_csv "${IM2GPS3KTEST_GPS_CSV}" \
  --out geo_pipeline/results/horeka_v5_im2gps3ktest_limit10.json

python geo_pipeline/analyze_results.py \
  --pred geo_pipeline/results/horeka_v5_im2gps3ktest_limit10.json
