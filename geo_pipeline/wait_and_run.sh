#!/bin/bash
# Wait until ALL required GPUs are free, then launch the full 4K eval.
#
# Old bug: previous version took max free across GPUs, so when 1/4 cards
# were busy the script still fired and vLLM TP=4 OOM'd on the busy one.
# New logic: every required GPU must clear BOTH thresholds (free VRAM and
# utilization), and must stay clear for N consecutive checks (stability
# window) before launch — protects against a neighbour briefly releasing
# memory mid-job and grabbing it back.

set -u

# ── Config ────────────────────────────────────────────────────────────────────
GPUS="${GPUS:-0,1,2,3}"          # which GPUs we need (comma-separated indices)
REQUIRED_FREE_MB=10000           # min free VRAM per GPU. 11GB cards: 10GB free = idle
MAX_UTIL=5                       # max SM utilization % to consider idle
CHECK_INTERVAL=30                # seconds between probes
STABLE_CHECKS=3                  # consecutive idle probes before launching
                                 # (3 × 30s = 90s of confirmed idle)

OUT="${OUT:-results/full_v3.json}"
BATCH_SIZE="${BATCH_SIZE:-20}"
EXTRA_ARGS="${EXTRA_ARGS:-}"     # e.g. EXTRA_ARGS="--limit 100 --start 0"

# ── Probe helpers ─────────────────────────────────────────────────────────────
IFS=',' read -ra GPU_LIST <<< "$GPUS"

probe_all_idle() {
    # Returns 0 if every GPU in GPU_LIST is idle (free VRAM >= REQUIRED_FREE_MB
    # AND utilization <= MAX_UTIL). Prints a per-GPU status line either way.
    local all_ok=1
    local report=""
    for g in "${GPU_LIST[@]}"; do
        # nvidia-smi -i <id> returns one line: "free_mb, util_pct"
        local line
        line=$(nvidia-smi -i "$g" \
               --query-gpu=memory.free,utilization.gpu \
               --format=csv,noheader,nounits 2>/dev/null)
        if [ -z "$line" ]; then
            report+="  GPU $g: query failed\n"
            all_ok=0
            continue
        fi
        # strip spaces, split on comma
        local free util
        free=$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')
        util=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')

        local ok="ok"
        if [ "$free" -lt "$REQUIRED_FREE_MB" ] || [ "$util" -gt "$MAX_UTIL" ]; then
            ok="BUSY"
            all_ok=0
        fi
        report+="  GPU $g: free=${free}MB util=${util}%  [$ok]\n"
    done
    printf "$report"
    return $((1 - all_ok))   # 0 = success when all_ok=1
}

# ── Wait loop ─────────────────────────────────────────────────────────────────
echo "[$(date)] Waiting for GPUs [$GPUS] to be idle"
echo "  thresholds: free >= ${REQUIRED_FREE_MB} MB  AND  util <= ${MAX_UTIL}%"
echo "  must stay idle for $STABLE_CHECKS consecutive checks (${CHECK_INTERVAL}s each)"

streak=0
while true; do
    echo
    echo "[$(date)] probe (streak=$streak/$STABLE_CHECKS)"
    if probe_all_idle; then
        streak=$((streak + 1))
        echo "  → all idle (streak=$streak/$STABLE_CHECKS)"
        if [ "$streak" -ge "$STABLE_CHECKS" ]; then
            break
        fi
    else
        if [ "$streak" -gt 0 ]; then
            echo "  → a GPU is busy, streak reset"
        fi
        streak=0
    fi
    sleep "$CHECK_INTERVAL"
done

# ── Launch ────────────────────────────────────────────────────────────────────
echo
echo "[$(date)] All GPUs stable. Launching full eval."
echo "  out=$OUT  batch_size=$BATCH_SIZE  extra=$EXTRA_ARGS"

source /home/szuo/.local/opt/miniconda3/etc/profile.d/conda.sh
conda activate /cvhci/temp/szuo/vllm-env

cd "$(dirname "$0")"   # cd into geo_pipeline/

MLLM_BACKEND=vllm \
CUDA_VISIBLE_DEVICES="$GPUS" \
MODEL_PATH=/cvhci/temp/szuo/models/qwen2.5-vl-7b \
python evaluate.py --batch_size "$BATCH_SIZE" --out "$OUT" $EXTRA_ARGS

echo "[$(date)] eval done."
