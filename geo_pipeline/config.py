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
TRANSITION_THR  = 0.55   # τ_transition (lowered from 0.7: coarse levels rarely hit 0.7 with many hypotheses)
ENHANCE_THR     = 0.05   # τ_enhance  (ΔP threshold)
BETA            = 0.693  # ln2

# ── Hierarchical control (v9) ─────────────────────────────────────────────────
# A level is considered stable for descent when either the top posterior is
# strong, or it clears the transition threshold with enough top1-top2 margin / low
# normalized entropy. This prevents flat country posteriors from blindly driving
# city/street prompts.
STRONG_POSTERIOR_THR = 0.60
STABLE_MARGIN_THR    = 0.04
STABLE_ENTROPY_THR   = 0.98
GUARDED_DESCENT_THR  = 0.40
COUNTRY_REPLACE_TOP_THR    = 0.45
COUNTRY_REPLACE_MARGIN_THR = 0.02
COUNTRY_REPLACE_ATTEMPTS = 1

# Optional GeoBayes-style web evidence enhancement. Disabled by default because
# HPC compute nodes may not have outbound network access and web search can be slow.
WEB_SEARCH_TOP_THR = 0.60
WEB_SEARCH_MARGIN_THR = ENHANCE_THR
WEB_SEARCH_MAX_RESULTS = 3
WEB_SEARCH_TIMEOUT = 8
WEB_SEARCH_REQUIRE_ENTITY = True

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
MAX_NEW_TOKENS         = 384    # default cap (used by hypothesize)
SL_MAX_NEW_TOKENS      = 48     # SL responses are tiny: "Rating: X / Confidence: Y / one sentence"
VERIFY_MAX_NEW_TOKENS  = 160    # verify observation: a paragraph is plenty
POMDP_MAX_NEW_TOKENS   = 64     # policy returns {"task_index": N, "reason": "..."}

# ── Batch inference ────────────────────────────────────────────────────────────
MAX_SL_BATCH_SIZE = 8    # original prompts per SL batch; multiplied by SL_N_SAMPLES → actual GPU batch. Reduce if OOM.
