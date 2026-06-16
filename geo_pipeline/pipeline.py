"""
Main pipeline: SL + DST + POMDP on YFCC4K.

Flow (one image):
  1. Hypothesize  — MLLM global analysis → hypothesis set H_0 + verification plan V_0
  2. Per level (country → city → street):
       a. SL: score each pending evidence against each hypothesis (uncertainty-aware)
       b. DST: fuse all evidence BBAs into updated posterior
       c. POMDP: select next verification task (LLM policy)
       d. Repeat until POMDP stopping condition
       e. Hierarchical transition if max_posterior > TRANSITION_THR
  3. Output MAP location → geocode → (lat, lon)
"""

import json
import re
import math
from PIL import Image

from models.mllm_client import MLLMClient
from modules.sl import SLModule
from modules.dst import DSTModule
from modules.pomdp import POMDPModule
from config import PRIOR_TEMP, PRIOR_CUTOFF, TRANSITION_THR

LEVELS = ["country", "city", "street"]

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _try_parse_json(text: str):
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _softmax_prior(scores: dict[str, float]) -> dict[str, float]:
    """Eq.5 from GeoBayes: temperature-scaled softmax with score cutoff."""
    import math
    clipped = {h: min(s, PRIOR_CUTOFF) for h, s in scores.items()}
    exps = {h: math.exp(s / PRIOR_TEMP) for h, s in clipped.items()}
    total = sum(exps.values())
    return {h: v / total for h, v in exps.items()}


# ── Prompt builders ────────────────────────────────────────────────────────────

def _hypothesize_prompt(image: Image.Image, level: str, context: str = "") -> list:
    level_hint = {
        "country": "Identify the most likely countries and generate a plan to verify.",
        "city":    "Identify the most likely cities and generate a plan to verify.",
        "street":  "Identify the most likely streets/districts and generate a plan to verify.",
    }[level]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"You are a geolocation expert. {level_hint}\n"
                    + (f"Prior context: {context}\n" if context else "")
                    + "\nAnalyze this image and respond with JSON:\n"
                    '{\n'
                    '  "hypotheses": [{"location": "<name>", "confidence": <0-1>}, ...],\n'
                    '  "verification_plan": [{"desc": "<what to check>", "bbox": [x,y,w,h] or null}, ...]\n'
                    '}'
                )},
            ],
        }
    ]


def _verify_prompt(image: Image.Image, task: dict, hypotheses: list[str], level: str) -> list:
    hyp_str = ", ".join(hypotheses[:5])
    bbox = task.get("bbox")
    region_note = f" Focus on region [x,y,w,h]={bbox}." if bbox else ""
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Task: {task['desc']}.{region_note}\n"
                    f"Current hypotheses: {hyp_str}\n"
                    f"Reasoning level: {level}\n\n"
                    "Describe what you observe and how it relates to the hypotheses.\n"
                    "Respond with: <observation text>"
                )},
            ],
        }
    ]


# ── Main pipeline class ────────────────────────────────────────────────────────

BATCH_SIZE = 20  # number of images to process in parallel; reduce if OOM


class GeoPipeline:
    def __init__(self, mllm: MLLMClient):
        self.mllm  = mllm
        self.sl    = SLModule(mllm)
        self.dst   = DSTModule()
        self.pomdp = POMDPModule(mllm)

    def _hypothesize(self, image: Image.Image, level: str, context: str = "") -> tuple[dict, list]:
        """Returns (prior_dict, verification_plan_list)."""
        messages = _hypothesize_prompt(image, level, context)
        response = self.mllm.generate(messages)
        parsed = _try_parse_json(response)
        if parsed is None or "hypotheses" not in parsed:
            # fallback: single hypothesis with uniform prior
            return {"Unknown": 1.0}, []

        raw_scores = {h["location"]: h.get("confidence", 0.5) for h in parsed["hypotheses"]}
        prior = _softmax_prior(raw_scores)
        plan  = parsed.get("verification_plan", [])
        return prior, plan

    def _run_level(
        self,
        image: Image.Image,
        level: str,
        initial_posterior: dict[str, float],
        initial_plan: list[dict],
        key_evidence: list[str],
    ) -> tuple[dict, list[str]]:
        """
        Run one hierarchy level. Returns (final_posterior, updated_key_evidence).
        """
        posterior = dict(initial_posterior)
        pending   = list(initial_plan)
        step      = 0
        evidence_scores_all: list[dict[str, float]] = []

        while True:
            exhausted = len(pending) == 0
            if self.pomdp.should_stop(posterior, step, level, exhausted):
                break

            # POMDP: select best action (skip if only one task)
            if len(pending) == 1:
                task_idx = 0
            else:
                task_idx = self.pomdp.select_action(posterior, pending, level, step)
            task = pending.pop(task_idx)

            # Verify: get evidence description from MLLM
            hyps = list(posterior.keys())
            v_messages = _verify_prompt(image, task, hyps, level)
            evidence_desc = self.mllm.generate(v_messages, max_new_tokens=256)

            # SL: uncertainty-aware per-hypothesis scores
            w_scores = self.sl.score(evidence_desc, hyps, level)
            evidence_scores_all.append(w_scores)

            # DST: fuse all evidence so far into new posterior
            posterior = self.dst.fuse(initial_posterior, evidence_scores_all)

            # track key evidence (high-information clues)
            max_w = max(w_scores.values(), default=1.0)
            if max_w > 1.5:
                key_evidence.append(evidence_desc[:120])

            step += 1

        return posterior, key_evidence

    def predict(self, image: Image.Image) -> dict:
        """
        Full coarse-to-fine inference for one image.
        Returns {level: best_location_name, "posterior": final_posterior_dict}.
        """
        result       = {}
        key_evidence = []
        context      = ""

        for level in LEVELS:
            prior, plan = self._hypothesize(image, level, context)

            # at city/street level, seed hypotheses from prior level result
            if level != "country" and result:
                parent = result.get(LEVELS[LEVELS.index(level) - 1], "")
                context = f"Located in {parent}. Key clues: {'; '.join(key_evidence[-3:])}"

            posterior, key_evidence = self._run_level(
                image, level, prior, plan, key_evidence
            )

            best = max(posterior, key=posterior.get)
            result[level] = best
            result[f"{level}_posterior"] = posterior

            # stop early if confidence is very low (model has no signal)
            if posterior.get(best, 0) < 0.3 and level == "country":
                break

        result["posterior"] = posterior
        return result

    def predict_batch(self, images: list) -> list[dict]:
        """
        Process a batch of images together, grouping MLLM calls across images
        at each pipeline step to maximise GPU utilisation.
        Returns a list of result dicts in the same order as images.
        """
        n = len(images)
        # per-image state
        results      = [{} for _ in range(n)]
        key_evidence = [[] for _ in range(n)]
        contexts     = [""] * n

        for level in LEVELS:
            # ── Hypothesize: one batch call for all images ──────────────────────
            hyp_messages = [
                _hypothesize_prompt(images[i], level, contexts[i]) for i in range(n)
            ]
            hyp_responses = self.mllm.batch_generate(hyp_messages)

            priors = []
            plans  = []
            for resp in hyp_responses:
                parsed = _try_parse_json(resp)
                if parsed is None or "hypotheses" not in parsed:
                    priors.append({"Unknown": 1.0})
                    plans.append([])
                else:
                    raw_scores = {h["location"]: h.get("confidence", 0.5)
                                  for h in parsed["hypotheses"]}
                    priors.append(_softmax_prior(raw_scores))
                    plans.append(parsed.get("verification_plan", []))

            # seed context from parent level
            if level != "country":
                parent_level = LEVELS[LEVELS.index(level) - 1]
                for i in range(n):
                    parent = results[i].get(parent_level, "")
                    clues  = "; ".join(key_evidence[i][-3:])
                    contexts[i] = f"Located in {parent}. Key clues: {clues}" if parent else ""

            # ── POMDP loop across all images simultaneously ─────────────────────
            posteriors    = [dict(p) for p in priors]
            pending       = [list(pl) for pl in plans]
            steps         = [0] * n
            ev_scores_all = [[] for _ in range(n)]

            while True:
                # find images still running
                active = [
                    i for i in range(n)
                    if not self.pomdp.should_stop(
                        posteriors[i], steps[i], level, len(pending[i]) == 0
                    )
                ]
                if not active:
                    break

                # ── Select actions for all active images (batch) ────────────────
                policy_msgs = []
                policy_idx  = []  # which active images need a policy call
                task_choices = {}
                for i in active:
                    if len(pending[i]) == 1:
                        task_choices[i] = 0
                    else:
                        policy_msgs.append(
                            self.pomdp._make_policy_prompt(
                                posteriors[i], pending[i], level, steps[i]
                            )
                        )
                        policy_idx.append(i)

                if policy_msgs:
                    policy_resps = self.mllm.batch_generate(policy_msgs, max_new_tokens=128)
                    for i, resp in zip(policy_idx, policy_resps):
                        match = __import__("re").search(r'"?task_index"?\s*:\s*(\d+)', resp)
                        idx = int(match.group(1)) if match else 0
                        task_choices[i] = min(idx, len(pending[i]) - 1)

                tasks = {i: pending[i].pop(task_choices[i]) for i in active}

                # ── Verify: batch call for all active images ────────────────────
                verify_msgs = [
                    _verify_prompt(images[i], tasks[i], list(posteriors[i].keys()), level)
                    for i in active
                ]
                verify_resps = self.mllm.batch_generate(verify_msgs, max_new_tokens=256)
                evidence_descs = {i: resp for i, resp in zip(active, verify_resps)}

                # ── SL scoring + DST fusion (CPU, per image) ───────────────────
                for i in active:
                    hyps     = list(posteriors[i].keys())
                    w_scores = self.sl.score(evidence_descs[i], hyps, level)
                    ev_scores_all[i].append(w_scores)
                    posteriors[i] = self.dst.fuse(priors[i], ev_scores_all[i])

                    max_w = max(w_scores.values(), default=1.0)
                    if max_w > 1.5:
                        key_evidence[i].append(evidence_descs[i][:120])

                    steps[i] += 1

            # ── Collect level results ───────────────────────────────────────────
            for i in range(n):
                best = max(posteriors[i], key=posteriors[i].get)
                results[i][level] = best
                results[i][f"{level}_posterior"] = posteriors[i]

                if posteriors[i].get(best, 0) < 0.3 and level == "country":
                    # no signal at country level — skip finer levels for this image
                    for remaining in LEVELS[LEVELS.index(level) + 1:]:
                        results[i][remaining] = "Unknown"
                        results[i][f"{remaining}_posterior"] = {}

            results_i_posterior = posteriors  # noqa: F841 — kept for debuggability

        for i in range(n):
            final_level = next(
                (lv for lv in reversed(LEVELS) if results[i].get(lv, "Unknown") != "Unknown"),
                LEVELS[-1]
            )
            results[i]["posterior"] = results[i].get(f"{final_level}_posterior", {})

        return results
