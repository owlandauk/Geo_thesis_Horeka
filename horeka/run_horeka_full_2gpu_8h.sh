#!/bin/bash
# Submit with:
#   sbatch run_horeka_full_2gpu_8h.sh
#
# This HoreKa job runs the full YFCC4K evaluation on 2 A100 GPUs for up to
# 8 hours and prints the wall-clock runtime at the end of the Slurm log.

#SBATCH --job-name=geo-full-2gpu
#SBATCH --partition=accelerated
#SBATCH --account=hk-project-p0025551
#SBATCH --constraint=LSDF
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation/geo_pipeline/results/horeka_full_2gpu_%j.out
#SBATCH --error=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation/geo_pipeline/results/horeka_full_2gpu_%j.err

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
export VLLM_TP=2
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
echo "TORCHINDUCTOR_CACHE_DIR: ${TORCHINDUCTOR_CACHE_DIR}"
echo "TRITON_CACHE_DIR: ${TRITON_CACHE_DIR}"
echo "MODEL_PATH: ${MODEL_PATH}"
echo "YFCC4K_IMG_DIR: ${YFCC4K_IMG_DIR}"
echo "YFCC4K_GPS_CSV: ${YFCC4K_GPS_CSV}"
echo "VLLM_TP: ${VLLM_TP}"
echo "VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION}"

nvidia-smi

python geo_pipeline/evaluate.py \
  --batch_size 8 \
  --out geo_pipeline/results/horeka_v10_delta_full_2gpu.json

python geo_pipeline/analyze_results.py \
  --pred geo_pipeline/results/horeka_v10_delta_full_2gpu.json
