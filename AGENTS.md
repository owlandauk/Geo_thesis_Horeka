# Repository Guidelines

## Project Structure & Module Organization

This repository contains a thesis-oriented geolocation pipeline plus literature assets. Source code lives in `geo_pipeline/`:

- `pipeline.py`: main `GeoPipeline` orchestration for country -> city -> street inference.
- `modules/`: core reasoning modules: `sl.py`, `dst.py`, and `pomdp.py`.
- `models/mllm_client.py`: backend wrapper for DashScope, local transformers, and vLLM.
- `data/yfcc4k_loader.py`: YFCC4K dataset loader.
- `evaluate.py`: batch evaluation, geocoding, and distance-threshold metrics.
- `config.py`: paths, model settings, thresholds, and generation limits.
- `results/`: stored evaluation outputs; avoid committing large transient runs unless they are intentional experiment artifacts.

No formal test directory currently exists.

## Build, Test, and Development Commands

Install dependencies:

```bash
pip install -r geo_pipeline/requirements.txt
```

Run a small API-backed smoke test:

```bash
export MLLM_BACKEND=dashscope
export DASHSCOPE_API_KEY=your_key
python3 geo_pipeline/evaluate.py --limit 10 --out geo_pipeline/results/test_run.json
```

Run local or server evaluation with vLLM:

```bash
MLLM_BACKEND=vllm MODEL_PATH=/path/to/qwen2.5-vl-7b \
python3 geo_pipeline/evaluate.py --batch_size 20 --out geo_pipeline/results/run.json
```

Check the HoreKa Slurm script syntax before server use:

```bash
bash -n run_horeka_latest_2gpu.sh
```

## Coding Style & Naming Conventions

Use Python 3 with 4-space indentation, type hints where helpful, and concise comments for non-obvious logic. Follow the existing naming style: `snake_case` for functions and variables, `CamelCase` for classes, and uppercase constants in `config.py`. Keep prompts and hyperparameters centralized where possible.

## Testing Guidelines

There is no dedicated test suite yet. For changes to inference logic, run at least a small `--limit` evaluation and inspect `records`, `country_posterior`, and raw response fields. For pure utility changes, prefer adding small deterministic checks or scripts before running expensive GPU jobs.

## Commit & Pull Request Guidelines

Git history uses short imperative commit messages, for example `Collapse country aliases to canonical form` and `Add geocode fallbacks to evaluate.py`. Keep commits focused and mention affected modules. Pull requests should include the motivation, changed files, commands run, and before/after metrics when evaluation behavior changes.

## Security & Configuration Tips

Do not commit API keys, model credentials, or private server paths beyond documented defaults. Prefer environment variables such as `DASHSCOPE_API_KEY`, `MLLM_BACKEND`, `MODEL_PATH`, `YFCC4K_IMG_DIR`, and `YFCC4K_GPS_CSV`. Keep model weights, datasets, and large caches outside the repository.
