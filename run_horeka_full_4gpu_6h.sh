#!/bin/bash
# Submit with:
#   sbatch run_horeka_full_4gpu_6h.sh
#
# This HoreKa job runs the full YFCC4K evaluation on 4 A100 GPUs for up to
# 6 hours and prints the wall-clock runtime at the end of the Slurm log.

#SBATCH --job-name=geo-full-4gpu
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025551
#SBATCH --constraint=LSDF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=geo_pipeline/results/horeka_full_4gpu_%j.out
#SBATCH --error=geo_pipeline/results/horeka_full_4gpu_%j.err

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

JOB_TMP_DIR="${REPO_DIR}/.tmp/slurm_${SLURM_JOB_ID:-manual}"
mkdir -p "${JOB_TMP_DIR}"
export TMPDIR="${JOB_TMP_DIR}"
export TMP="${JOB_TMP_DIR}"
export TEMP="${JOB_TMP_DIR}"

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate "${ENV_DIR}"

export MLLM_BACKEND=vllm
export VLLM_TP=4
export MODEL_PATH=/hkfs/work/workspace/scratch/tj3409-SichengZuo/models/qwen2.5-vl-7b
export YFCC4K_IMG_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k
export YFCC4K_GPS_CSV=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k_gps.csv
export VLLM_GPU_MEMORY_UTILIZATION=0.80

echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "Node: $(hostname)"
echo "Start: ${START_HUMAN}"
echo "Repo: ${REPO_DIR}"
echo "Env: ${ENV_DIR}"
echo "TMPDIR: ${TMPDIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "YFCC4K_IMG_DIR: ${YFCC4K_IMG_DIR}"
echo "YFCC4K_GPS_CSV: ${YFCC4K_GPS_CSV}"
echo "VLLM_TP: ${VLLM_TP}"
echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION}"

nvidia-smi

python geo_pipeline/evaluate.py \
  --batch_size 16 \
  --out geo_pipeline/results/horeka_v10_delta_full_4gpu.json

python geo_pipeline/analyze_results.py \
  --pred geo_pipeline/results/horeka_v10_delta_full_4gpu.json
