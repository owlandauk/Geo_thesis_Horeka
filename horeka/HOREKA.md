# HoreKa Notes

Use these paths on HoreKa (`hkn1990` / `tj3409`) for this project.

## Repository

```bash
cd /hkfs/work/workspace/scratch/tj3409-SichengZuo/Multi-agent-MLLM-geolocation
```

If already inside `geo_pipeline/`, the repository root is one level up:

```bash
cd ..
```

## Environment And Data

```bash
ENV_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/envs/geo-vllm
MODEL_PATH=/hkfs/work/workspace/scratch/tj3409-SichengZuo/models/qwen2.5-vl-7b
YFCC4K_IMG_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k
YFCC4K_GPS_CSV=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k_gps.csv
```

Activate the environment from a Slurm script with:

```bash
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate /hkfs/work/workspace/scratch/tj3409-SichengZuo/envs/geo-vllm
```

## Current Baseline

Use `geo_pipeline/results/full_v5.json` as the baseline result.

```text
Street <1km:        5.34%
City <25km:        16.16%
Region <200km:     26.30%
Country <750km:    43.85%
Continent <2500km: 62.63%
Unknown country:    0.88%
```

## 200-Image Test

Run from the repository root:

```bash
MLLM_BACKEND=vllm \
MODEL_PATH=/hkfs/work/workspace/scratch/tj3409-SichengZuo/models/qwen2.5-vl-7b \
YFCC4K_IMG_DIR=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k \
YFCC4K_GPS_CSV=/hkfs/work/workspace/scratch/tj3409-SichengZuo/Dataset/yfcc4k/yfcc4k_gps.csv \
python geo_pipeline/evaluate.py \
  --start 0 \
  --limit 200 \
  --batch_size 20 \
  --out geo_pipeline/results/v7_limit200_default.json
```

Analyze:

```bash
python geo_pipeline/analyze_results.py --pred geo_pipeline/results/v7_limit200_default.json
```

If already inside `geo_pipeline/`, use:

```bash
python analyze_results.py --pred results/v7_limit200_default.json
```

Do not use `--strict_child_geocode` for baseline-comparable runs; it is only for ablation.

## Torch/Triton Cache Permission Fix

If a Slurm log shows a failure like:

```text
PermissionError: [Errno 13] Permission denied: '/scratch/slurm_tmpdir/job_<old_job_id>'
```

then Torch Inductor or Triton is trying to reuse a cache path from an old job.
Set cache directories under the current writable job tmpdir before starting vLLM:

```bash
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
```

The repository Slurm scripts already include these exports.

## Slurm Scripts

Existing scripts in the repository root:

```bash
sbatch run_horeka_limit50_1gpu_30m.sh
sbatch run_horeka_limit1000_2gpu_3h.sh
sbatch run_horeka_full_4gpu_6h.sh
```
