import os

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH = os.environ.get("MODEL_PATH", "/cvhci/temp/szuo/models/qwen2.5-vl-7b")
DEVICE = "cuda"

# ── Dataset ────────────────────────────────────────────────────────────────────
YFCC4K_IMG_DIR  = os.environ.get("YFCC4K_IMG_DIR",  "/cvhci/temp/szuo/yfcc4k/yfcc4k")
YFCC4K_GPS_CSV  = os.environ.get("YFCC4K_GPS_CSV",  "/cvhci/temp/szuo/yfcc4k/yfcc4k_gps.csv")
RESULTS_DIR     = os.environ.get("RESULTS_DIR",     "/cvhci/temp/szuo/geo_results")

# ── GeoBayes hyperparams (kept identical to paper) ────────────────────────────
PRIOR_TEMP      = 1.5    # T  in Eq.5
PRIOR_CUTOFF    = 0.6    # τp in Eq.5
TRANSITION_THR  = 0.7    # τ_transition
ENHANCE_THR     = 0.05   # τ_enhance  (ΔP threshold)
BETA            = 0.693  # ln2

# ── SL (single-source uncertainty) ────────────────────────────────────────────
SL_N_SAMPLES    = 5      # samples per hypothesis for uncertainty estimation in SLModule
SL_TEMPERATURE  = 0.8    # sampling temperature

# ── DST (Dempster-Shafer fusion) ───────────────────────────────────────────────
DST_CONFLICT_THR = 0.5   # K > this → treat as high-conflict, apply cautious rule

# ── POMDP ─────────────────────────────────────────────────────────────────────
POMDP_MAX_STEPS = 8      # full experiments
POMDP_GAMMA     = 0.95   # discount factor (used if computing cumulative reward)

# ── Evaluation thresholds (km) ─────────────────────────────────────────────────
EVAL_THRESHOLDS = [1, 25, 200, 750, 2500]

# ── Generation ────────────────────────────────────────────────────────────────
MAX_NEW_TOKENS  = 512

# ── Batch inference ────────────────────────────────────────────────────────────
MAX_SL_BATCH_SIZE = 20   # max hypotheses per batch in SLModule; reduce if OOM
