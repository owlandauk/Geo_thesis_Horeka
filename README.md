# Multi-Agent MLLM Geolocation

Master's thesis project extending [GeoBayes](https://arxiv.org/abs/2501.04304) (AAAI-26) with three modules that replace its single-point inference with uncertainty-aware, multi-source reasoning.

## Overview

GeoBayes frames image geo-localization as MAP estimation over geographic hypotheses, updated sequentially via an MLLM. This project extends that framework with:

| Module | Replaces | What it does |
|--------|----------|--------------|
| **SL** (Single-source uncertainty) | Point estimate (c_t, α_t) | Samples the MLLM N times per evidence item; shrinks likelihood toward neutral when variance is high |
| **DST** (Dempster-Shafer fusion) | Naive product ∏W | Combines evidence BBAs with Dempster's rule; falls back to Yager's cautious rule under high conflict |
| **POMDP** (Evidence selection) | Sequential task traversal | LLM-as-policy selects the next verification task by expected information gain |

The pipeline runs coarse-to-fine: Country → City → Street, transitioning when posterior confidence exceeds a threshold.

## Repository Structure

```
geo_pipeline/
├── config.py              # All hyperparameters and paths
├── pipeline.py            # GeoPipeline: predict() / predict_batch()
├── evaluate.py            # YFCC4K evaluation script
├── models/
│   └── mllm_client.py     # MLLM wrapper (DashScope API or local transformers)
├── modules/
│   ├── sl.py              # SL: uncertainty-aware likelihood scoring
│   ├── dst.py             # DST: Dempster-Shafer evidence fusion
│   └── pomdp.py           # POMDP: LLM-based verification task selection
└── data/
    └── yfcc4k_loader.py   # YFCC4K dataset loader
```

## Setup

```bash
pip install -r geo_pipeline/requirements.txt
```

**Model:** Qwen2.5-VL-7B-Instruct (local or via DashScope API)

## Running

### API backend (testing)

```bash
export MLLM_BACKEND=dashscope
export DASHSCOPE_API_KEY=your_key
python geo_pipeline/evaluate.py --limit 100 --out results/run1.json
```

### Local backend (server)

```bash
export MLLM_BACKEND=local
export MODEL_PATH=/path/to/Qwen2.5-VL-7B-Instruct
CUDA_VISIBLE_DEVICES=0 python geo_pipeline/evaluate.py --batch_size 4 --out results/run1.json
```

Resume from a checkpoint:

```bash
python geo_pipeline/evaluate.py --start 1000 --out results/run2.json
```

## Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SL_N_SAMPLES` | 5 | MLLM samples per evidence item for uncertainty estimation |
| `POMDP_MAX_STEPS` | 8 | Max verification steps per hierarchy level |
| `TRANSITION_THR` | 0.7 | Posterior threshold to advance to next level |
| `DST_CONFLICT_THR` | 0.5 | Conflict mass K above which Yager's rule is applied |
| `MAX_SL_BATCH_SIZE` | 6 | Max hypotheses per GPU batch in SL (reduce if OOM) |

## Evaluation

Standard distance-threshold accuracy on YFCC4K (4,000 Flickr images):

| Threshold | Metric |
|-----------|--------|
| 1 km | Street |
| 25 km | City |
| 200 km | Region |
| 750 km | Country |
| 2500 km | Continent |

## Baseline

GeoBayes (Qwen2.5-VL-7B, YFCC4K):

| Threshold | Accuracy |
|-----------|----------|
| 2500 km | 75.4% |
| 750 km | 55.8% |
| 200 km | 30.9% |
| 25 km | 16.1% |
| 1 km | 4.9% |
