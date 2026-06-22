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
from config import (
    PRIOR_TEMP, PRIOR_CUTOFF, TRANSITION_THR,
    VERIFY_MAX_NEW_TOKENS, POMDP_MAX_NEW_TOKENS,
)

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
    """
    Country-level prompt fuses two ideas from the literature:
      - GLOBE (NeurIPS-25, Fig. 2): force structured reasoning across 4 visual cue
        categories BEFORE naming any country — prevents commitment to nearby-but-
        wrong countries because the model has to enumerate cues first.
      - GeoBayes (AAAI-26, Tab. 3): 5 country candidates is the sweet spot —
        Top-5 country recall on YFCC4K = 74.3% (Top-1 = 50.7%). Listing more
        than 5 dilutes probability mass; listing fewer drops recall.

    City/street levels follow GeoBayes (5 candidates, conditioned on parent context).
    """
    if level == "country":
        instruction = (
            "Step 1 — Analyze the image across these FOUR visual cue categories. "
            "For each, briefly describe what you observe (one phrase each):\n"
            "  (a) Architecture & building style\n"
            "  (b) Signage & written language/script\n"
            "  (c) Street layout, vegetation & terrain\n"
            "  (d) License plates, road signs & vehicle types\n\n"
            "Step 2 — Based on those cues, list the TOP 5 most likely countries "
            "(across different regions if cues are ambiguous). For each, assign a "
            "confidence in [0,1] that reflects how strongly the cues support it.\n\n"
            "Step 3 — Build a verification plan: 4-6 short, specific tasks that "
            "would DISTINGUISH between the candidates (not just confirm the top one). "
            "Each task should target a specific bbox or visual feature."
        )
    elif level == "city":
        instruction = (
            "List the TOP 5 most likely cities given the parent country context. "
            "Then build a 3-5 task verification plan that distinguishes between them."
        )
    else:  # street
        instruction = (
            "List the most likely streets, districts, or neighborhoods within the "
            "parent city. Build a short verification plan."
        )

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"You are an expert geolocation analyst. {instruction}\n"
                    + (f"\nPrior context: {context}\n" if context else "")
                    + "\nRespond with JSON only:\n"
                    '{\n'
                    '  "cues": {"architecture": "<phrase>", "signage": "<phrase>", '
                    '"layout": "<phrase>", "plates_signs": "<phrase>"},\n'
                    '  "hypotheses": [{"location": "<name>", "confidence": <0-1>}, ...],\n'
                    '  "verification_plan": [{"desc": "<what to check>", "bbox": [x,y,w,h] or null}, ...]\n'
                    '}\n'
                    "Note: the 'cues' field is optional for city/street levels."
                )},
            ],
        }
    ]


def _verify_prompt(image: Image.Image, task: dict, hypotheses: list[str], level: str) -> list:
    """
    Implements GeoBayes 'Probability Thought' (AAAI-26, Fig. 1d): instead of a
    freeform evidence description, the model is explicitly asked to rate the
    evidence against EACH candidate hypothesis. This is the load-bearing trick
    in the original paper — every clue is scored against every candidate, not
    just the leading one, so contradictory evidence can cancel a wrong prior.
    """
    # cap at 5 hypotheses to match GeoBayes Top-5 setting
    hyps = hypotheses[:5]
    hyp_lines = "\n".join(f"  - {h}" for h in hyps)
    bbox = task.get("bbox")
    region_note = f" Focus on region [x,y,w,h]={bbox}." if bbox else ""

    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    f"Verification task: {task['desc']}.{region_note}\n"
                    f"Reasoning level: {level}\n\n"
                    f"Candidate hypotheses:\n{hyp_lines}\n\n"
                    "Step 1 — Describe what you observe in 1-2 sentences "
                    "(the visual evidence, only what is actually visible).\n\n"
                    "Step 2 — For EACH candidate hypothesis above, state whether "
                    "this evidence supports it (S), contradicts it (C), or is "
                    "neutral (N). Be honest — most evidence will be neutral for "
                    "most candidates.\n\n"
                    "Respond in this exact format:\n"
                    "Observation: <what you see>\n"
                    "Support: <hypothesis_1>=S/C/N; <hypothesis_2>=S/C/N; ..."
                )},
            ],
        }
    ]


def _geo_reasoner_prompt(image: Image.Image) -> list:
    """
    GeoReasoner (ICML-24) freeform prompt — used as a complementary signal to
    the structured 4-cue prompt. Empirically the two prompts fail on different
    images, so ensembling them at country-level boosts Top-1 country recall.

    Output format follows GeoReasoner Fig. 3 verbatim: {'country', 'city', 'reasons'}.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "According to the content of the image, please think step by "
                    "step and deduce in which country and city the image is most "
                    "likely located and give the most important reason. Output "
                    "in JSON format, e.g. "
                    '{"country":"", "city":"", "reasons":""}.'
                )},
            ],
        }
    ]


def _merge_geo_reasoner_seed(prior: dict[str, float],
                             reasoner_country: str | None,
                             boost: float = 0.35) -> dict[str, float]:
    """
    Inject the GeoReasoner top-1 country guess into the structured prior.
    If the guess is already in the prior: boost it. If not: add with `boost`
    probability mass, renormalize. The boost is calibrated so that a strong
    consensus (both prompts agree) reliably crosses TRANSITION_THR.
    """
    if not reasoner_country:
        return prior
    name = reasoner_country.strip()
    if not name or name.lower() in ("unknown", ""):
        return prior

    merged = dict(prior)
    if name in merged:
        merged[name] = merged[name] + boost
    else:
        merged[name] = boost

    total = sum(merged.values())
    if total <= 0:
        return prior
    return {k: v / total for k, v in merged.items()}


# ── Main pipeline class ────────────────────────────────────────────────────────

BATCH_SIZE = 20  # number of images to process in parallel; reduce if OOM


class GeoPipeline:
    def __init__(self, mllm: MLLMClient):
        self.mllm  = mllm
        self.sl    = SLModule(mllm)
        self.dst   = DSTModule()
        self.pomdp = POMDPModule(mllm)

    def _hypothesize(self, image: Image.Image, level: str, context: str = "") -> tuple[dict, list]:
        """Returns (prior_dict, verification_plan_list).

        Country level: ensembles the GLOBE 4-cue structured prompt with a
        GeoReasoner freeform prompt — the two have largely disjoint failure
        modes, so combining them boosts Top-1 country recall.
        """
        messages = _hypothesize_prompt(image, level, context)
        response = self.mllm.generate(messages)
        parsed = _try_parse_json(response)
        if parsed is None or "hypotheses" not in parsed:
            return {"Unknown": 1.0}, []

        raw_scores = {h["location"]: h.get("confidence", 0.5) for h in parsed["hypotheses"]}
        prior = _softmax_prior(raw_scores)
        plan  = parsed.get("verification_plan", [])

        if level == "country":
            r_msg = _geo_reasoner_prompt(image)
            r_resp = self.mllm.generate(r_msg)
            r_parsed = _try_parse_json(r_resp)
            rc = r_parsed.get("country") if r_parsed else None
            if rc:
                prior = _merge_geo_reasoner_seed(prior, rc)

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
            evidence_desc = self.mllm.generate(v_messages, max_new_tokens=VERIFY_MAX_NEW_TOKENS)

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

            # ── Country-level: add GeoReasoner freeform second prompt as a ──────
            # complementary signal (ensemble across two prompt structures). Only
            # at country level — at city/street the context-conditioned single
            # prompt is enough.
            if level == "country":
                reasoner_msgs = [_geo_reasoner_prompt(images[i]) for i in range(n)]
                reasoner_resps = self.mllm.batch_generate(reasoner_msgs)
                for i, resp in enumerate(reasoner_resps):
                    parsed = _try_parse_json(resp)
                    rc = parsed.get("country") if parsed else None
                    if rc and "Unknown" not in priors[i]:
                        priors[i] = _merge_geo_reasoner_seed(priors[i], rc)

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
                    policy_resps = self.mllm.batch_generate(policy_msgs, max_new_tokens=POMDP_MAX_NEW_TOKENS)
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
                verify_resps = self.mllm.batch_generate(verify_msgs, max_new_tokens=VERIFY_MAX_NEW_TOKENS)
                evidence_descs = {i: resp for i, resp in zip(active, verify_resps)}

                # ── SL scoring: ONE big batch across all active images ──────────
                sl_items = [
                    (evidence_descs[i], list(posteriors[i].keys()))
                    for i in active
                ]
                sl_results = self.sl.score_many(sl_items, level)

                # ── DST fusion (CPU, per image) ────────────────────────────────
                for k, i in enumerate(active):
                    w_scores = sl_results[k]
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

            results_i_posterior = posteriors  # noqa: F841 — kept for debuggability

        for i in range(n):
            final_level = next(
                (lv for lv in reversed(LEVELS) if results[i].get(lv, "Unknown") != "Unknown"),
                LEVELS[-1]
            )
            results[i]["posterior"] = results[i].get(f"{final_level}_posterior", {})

        return results
