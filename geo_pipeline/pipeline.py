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

from __future__ import annotations

import json
import re
import math
from PIL import Image

from models.mllm_client import MLLMClient
from modules.sl import SLModule
from modules.dst import DSTModule
from modules.pomdp import POMDPModule
from country_aliases import canonicalize_country
from config import (
    PRIOR_TEMP, PRIOR_CUTOFF, TRANSITION_THR,
    VERIFY_MAX_NEW_TOKENS, POMDP_MAX_NEW_TOKENS,
    STRONG_POSTERIOR_THR, STABLE_MARGIN_THR, STABLE_ENTROPY_THR,
    COUNTRY_REPLACE_ATTEMPTS,
)

LEVELS = ["country", "city", "street"]

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _try_parse_json(text: str):
    """Parse the first JSON object/array from raw model text."""
    if isinstance(text, (dict, list)):
        return text
    if not isinstance(text, str):
        return None

    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    fenced = _FENCED_JSON_RE.search(stripped)
    if fenced:
        parsed = _try_parse_json(fenced.group(1))
        if parsed is not None:
            return parsed

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(stripped):
        if ch not in "[{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[idx:])
            return value
        except json.JSONDecodeError:
            continue
    return None


def _parse_hypothesis_payload(text: str) -> dict | None:
    """Normalize model/wrapper outputs to {hypotheses, verification_plan}."""
    parsed = _try_parse_json(text)
    if parsed is None:
        return None
    if isinstance(parsed, list):
        return {"hypotheses": parsed, "verification_plan": []}
    if not isinstance(parsed, dict):
        return None
    if "hypotheses" in parsed:
        return parsed

    hypotheses: list[dict] = []
    verification_plan: list[dict] = []

    general = _parse_hypothesis_payload(parsed.get("general", ""))
    if general:
        hypotheses.extend(general.get("hypotheses", []))
        verification_plan = general.get("verification_plan", []) or []

    for cue_text in parsed.get("cue_responses", []) or []:
        cue = _parse_hypothesis_payload(cue_text)
        if cue:
            hypotheses.extend(cue.get("hypotheses", []))

    if hypotheses:
        return {"hypotheses": hypotheses, "verification_plan": verification_plan}
    return None


def _softmax_prior(scores: dict[str, float]) -> dict[str, float]:
    """Eq.5 from GeoBayes: temperature-scaled softmax with score cutoff."""
    import math
    clipped = {h: min(s, PRIOR_CUTOFF) for h, s in scores.items()}
    exps = {h: math.exp(s / PRIOR_TEMP) for h, s in clipped.items()}
    total = sum(exps.values())
    return {h: v / total for h, v in exps.items()}


def _collect_scores(hypotheses: list, level: str) -> dict[str, float]:
    """Collect {location: confidence} from parsed hypotheses.

    At the country level we canonicalize location names via the alias map
    before scoring, so "USA"/"California, USA"/"Southeast Asia" don't leak
    through as distinct entries (see full_v4 diagnosis: raw MLLM strings
    that didn't match a country killed 60% of records to Unknown). When
    multiple candidates map to the same canonical country, keep the max
    confidence — we don't want to double-count "USA" and "United States".
    Non-country levels keep the raw string.
    """
    scores: dict[str, float] = {}
    for h in hypotheses:
        loc = h.get("location")
        if not loc:
            continue
        conf = h.get("confidence", 0.5)
        if level == "country":
            canon = canonicalize_country(loc)
            if canon is None:
                continue  # drop non-country strings like "Southeast Asia"
            loc = canon
        scores[loc] = max(scores.get(loc, 0.0), conf)
    return scores


def _format_top_candidates(posterior: dict[str, float], k: int = 3) -> str:
    items = sorted((posterior or {}).items(), key=lambda x: -x[1])[:k]
    return ", ".join(f"{loc} ({prob:.2f})" for loc, prob in items)


def _posterior_stats(posterior: dict[str, float]) -> dict[str, float]:
    """Return top mass, top1-top2 margin, and normalized entropy."""
    vals = sorted((float(v) for v in (posterior or {}).values()), reverse=True)
    if not vals:
        return {"top": 0.0, "margin": 0.0, "entropy": 1.0}
    top = vals[0]
    margin = top - (vals[1] if len(vals) > 1 else 0.0)
    if len(vals) <= 1:
        entropy = 0.0
    else:
        entropy = -sum(v * math.log(max(v, 1e-12)) for v in vals) / math.log(len(vals))
    return {"top": top, "margin": margin, "entropy": entropy}


def _stable_for_descent(posterior: dict[str, float]) -> bool:
    """Multi-signal gate for hierarchical descent.

    Top probability alone is not enough: v8 showed that flat country posteriors
    around 0.51 can still drive noisy city/street guesses. We descend only when
    the top candidate is strong, or when it clears the transition threshold with
    a meaningful margin or low normalized entropy.
    """
    stats = _posterior_stats(posterior)
    if stats["top"] >= STRONG_POSTERIOR_THR:
        return True
    if stats["top"] < TRANSITION_THR:
        return False
    return stats["margin"] >= STABLE_MARGIN_THR or stats["entropy"] <= STABLE_ENTROPY_THR


def _country_candidate_set(country_posterior: dict[str, float], k: int = 3) -> set[str]:
    return {
        country
        for country, _ in sorted((country_posterior or {}).items(), key=lambda x: -x[1])[:k]
    }


def _child_country_conflict(location: str, country_posterior: dict[str, float]) -> bool:
    child_country = canonicalize_country(location or "")
    if not child_country:
        return False
    return child_country not in _country_candidate_set(country_posterior)


def _filter_child_posterior(
    posterior: dict[str, float],
    country_posterior: dict[str, float],
) -> tuple[dict[str, float], list[str]]:
    """Drop child hypotheses that name countries outside top country candidates."""
    if not posterior:
        return posterior, []
    kept = {}
    conflicts = []
    for loc, prob in posterior.items():
        if _child_country_conflict(loc, country_posterior):
            conflicts.append(loc)
        else:
            kept[loc] = prob
    if not conflicts:
        return posterior, []
    if not kept:
        return {"Unknown": 1.0}, conflicts
    total = sum(kept.values())
    return ({k: v / total for k, v in kept.items()} if total > 0 else kept), conflicts


def _replace_context(level: str, posterior: dict[str, float], key_evidence: list[str]) -> str:
    stats = _posterior_stats(posterior)
    clues = "; ".join(key_evidence[-3:])
    return (
        "Previous country candidates remained unstable after verification. "
        f"Top candidates were {_format_top_candidates(posterior, 5)}; "
        f"top={stats['top']:.2f}, margin={stats['margin']:.2f}, entropy={stats['entropy']:.2f}. "
        "Re-analyze from scratch and return a diverse country candidate set. "
        "Avoid defaulting to any country from weak generic cues, but do not over-correct away from North America: "
        "United States, Canada, and Mexico remain valid when road signs, traffic infrastructure, license plates, "
        "landmarks, vegetation, or architecture clearly support them. "
        f"Previous useful clues: {clues}"
    )


def _context_for_level(level: str, result: dict, key_evidence: list[str]) -> str:
    clues = "; ".join(key_evidence[-3:])
    if level == "city":
        countries = _format_top_candidates(result.get("country_posterior", {}))
        return (
            f"Country candidates: {countries}. "
            "Hypothesize cities within these candidate countries. Do not introduce a city "
            "from a different country unless visible text or a landmark directly names it. "
            "If the country remains ambiguous, keep alternatives within the listed candidates. "
            f"Key clues: {clues}"
        )
    if level == "street":
        countries = _format_top_candidates(result.get("country_posterior", {}))
        cities = _format_top_candidates(result.get("city_posterior", {}))
        return (
            f"Country candidates: {countries}. City candidates: {cities}. "
            "Refine within these city/country candidates. Return a street, district, or landmark; "
            "do not append a different country unless directly visible in the image. "
            f"Key clues: {clues}"
        )
    return ""


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
                    "Return 3-5 plausible hypotheses when the image is ambiguous. "
                    "For country-level reasoning, return country names only, not continents or regions. "
                    "Do not default to any country from weak generic cues. English text, generic roads, "
                    "vegetation, architecture, online media, or product branding alone are not enough for "
                    "United States or Canada; however North America is valid when concrete road signs, "
                    "traffic infrastructure, license plates, landmarks, vegetation, or architecture support it. "
                    "Assign high confidence only when there are explicit local clues. "
                    + (f"Prior context: {context}\n" if context else "")
                    + "\nAnalyze this image and respond with valid JSON only, no markdown fences:\n"
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

    def _hypothesize(self, image: Image.Image, level: str, context: str = "") -> tuple[dict, list, str]:
        """Returns (prior_dict, verification_plan_list, raw_response)."""
        messages = _hypothesize_prompt(image, level, context)
        response = self.mllm.generate(messages)
        parsed = _parse_hypothesis_payload(response)
        if parsed is None or "hypotheses" not in parsed:
            # fallback: single hypothesis with uniform prior
            return {"Unknown": 1.0}, [], response

        raw_scores = _collect_scores(parsed["hypotheses"], level)
        prior = _softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0}
        plan  = parsed.get("verification_plan", [])
        return prior, plan, response

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
            # at city/street level, seed hypotheses from prior level result
            if level != "country" and result:
                context = _context_for_level(level, result, key_evidence)

            prior, plan, raw_resp = self._hypothesize(image, level, context)
            result[f"{level}_raw_response"] = raw_resp

            posterior, key_evidence = self._run_level(
                image, level, prior, plan, key_evidence
            )

            if level == "country" and not _stable_for_descent(posterior):
                for _ in range(COUNTRY_REPLACE_ATTEMPTS):
                    replace_context = _replace_context(level, posterior, key_evidence)
                    prior, plan, raw_resp = self._hypothesize(image, level, replace_context)
                    result[f"{level}_raw_response"] = raw_resp
                    posterior, key_evidence = self._run_level(
                        image, level, prior, plan, key_evidence
                    )
                    result["country_replaced"] = True
                    if _stable_for_descent(posterior):
                        break

            if level in ("city", "street"):
                filtered, conflicts = _filter_child_posterior(
                    posterior, result.get("country_posterior", {})
                )
                if conflicts:
                    result[f"{level}_backtrack_conflicts"] = conflicts[:5]
                    posterior = filtered

            best = max(posterior, key=posterior.get)
            result[level] = best
            result[f"{level}_posterior"] = posterior
            result[f"{level}_stable"] = _stable_for_descent(posterior)

            # stop early if confidence is very low (model has no signal)
            if posterior.get(best, 0) < 0.3 and level == "country":
                break
            if level == "country" and not _stable_for_descent(posterior):
                # Replace could not produce a reliable country distribution.
                # Avoid propagating a flat parent posterior into noisy child prompts.
                for remaining in LEVELS[LEVELS.index(level) + 1:]:
                    result[remaining] = "Unknown"
                    result[f"{remaining}_posterior"] = {}
                break
            if level == "city" and best == "Unknown":
                result["street"] = "Unknown"
                result["street_posterior"] = {}
                break

        result["posterior"] = posterior
        return result

    def _run_level_batch(
        self,
        images: list,
        level: str,
        contexts: list[str],
        key_evidence: list[list[str]],
    ) -> tuple[list[str], list[dict[str, float]]]:
        """Run one hierarchy level for a batch and update key_evidence in place."""
        n = len(images)
        hyp_messages = [_hypothesize_prompt(images[i], level, contexts[i]) for i in range(n)]
        hyp_responses = self.mllm.batch_generate(hyp_messages)

        priors = []
        plans = []
        for resp in hyp_responses:
            parsed = _parse_hypothesis_payload(resp)
            if parsed is None or "hypotheses" not in parsed:
                priors.append({"Unknown": 1.0})
                plans.append([])
            else:
                raw_scores = _collect_scores(parsed["hypotheses"], level)
                priors.append(_softmax_prior(raw_scores) if raw_scores else {"Unknown": 1.0})
                plans.append(parsed.get("verification_plan", []))

        posteriors = [dict(p) for p in priors]
        pending = [list(pl) for pl in plans]
        steps = [0] * n
        ev_scores_all = [[] for _ in range(n)]

        while True:
            active = [
                i for i in range(n)
                if not self.pomdp.should_stop(
                    posteriors[i], steps[i], level, len(pending[i]) == 0
                )
            ]
            if not active:
                break

            policy_msgs = []
            policy_idx = []
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
                policy_resps = self.mllm.batch_generate(
                    policy_msgs, max_new_tokens=POMDP_MAX_NEW_TOKENS
                )
                for i, resp in zip(policy_idx, policy_resps):
                    match = re.search(r'"?task_index"?\s*:\s*(\d+)', resp)
                    idx = int(match.group(1)) if match else 0
                    task_choices[i] = min(idx, len(pending[i]) - 1)

            tasks = {i: pending[i].pop(task_choices[i]) for i in active}

            verify_msgs = [
                _verify_prompt(images[i], tasks[i], list(posteriors[i].keys()), level)
                for i in active
            ]
            verify_resps = self.mllm.batch_generate(
                verify_msgs, max_new_tokens=VERIFY_MAX_NEW_TOKENS
            )
            evidence_descs = {i: resp for i, resp in zip(active, verify_resps)}

            sl_items = [
                (evidence_descs[i], list(posteriors[i].keys()))
                for i in active
            ]
            sl_results = self.sl.score_many(sl_items, level)

            for k, i in enumerate(active):
                w_scores = sl_results[k]
                ev_scores_all[i].append(w_scores)
                posteriors[i] = self.dst.fuse(priors[i], ev_scores_all[i])

                max_w = max(w_scores.values(), default=1.0)
                if max_w > 1.5:
                    key_evidence[i].append(evidence_descs[i][:120])

                steps[i] += 1

        return hyp_responses, posteriors

    def predict_batch(self, images: list) -> list[dict]:
        """
        Process a batch of images together, grouping MLLM calls across images
        at each pipeline step to maximise GPU utilisation.
        Returns a list of result dicts in the same order as images.
        """
        n = len(images)
        # per-image state
        results = [{} for _ in range(n)]
        key_evidence = [[] for _ in range(n)]
        contexts = [""] * n
        skip_finer = [False] * n

        for level in LEVELS:
            level_indices = [i for i in range(n) if not skip_finer[i]]
            if not level_indices:
                break

            # seed context from parent level before hypothesizing the next level
            if level != "country":
                for i in level_indices:
                    contexts[i] = _context_for_level(level, results[i], key_evidence[i])

            subset_images = [images[i] for i in level_indices]
            subset_contexts = [contexts[i] for i in level_indices]
            subset_key_evidence = [key_evidence[i] for i in level_indices]
            raw_responses, posteriors_subset = self._run_level_batch(
                subset_images, level, subset_contexts, subset_key_evidence
            )
            posteriors_by_idx = {
                idx: post for idx, post in zip(level_indices, posteriors_subset)
            }
            raw_by_idx = {
                idx: raw for idx, raw in zip(level_indices, raw_responses)
            }

            # Replace: if country remains flat after verification, regenerate the
            # country candidate set once with a prompt that asks for diverse but
            # evidence-grounded candidates.
            if level == "country" and COUNTRY_REPLACE_ATTEMPTS > 0:
                unstable = [idx for idx in level_indices if not _stable_for_descent(posteriors_by_idx[idx])]
                for _ in range(COUNTRY_REPLACE_ATTEMPTS):
                    if not unstable:
                        break
                    replace_images = [images[i] for i in unstable]
                    replace_contexts = [
                        _replace_context(level, posteriors_by_idx[i], key_evidence[i])
                        for i in unstable
                    ]
                    replace_key_evidence = [key_evidence[i] for i in unstable]
                    repl_raw, repl_posts = self._run_level_batch(
                        replace_images, level, replace_contexts, replace_key_evidence
                    )
                    for idx, raw, post in zip(unstable, repl_raw, repl_posts):
                        raw_by_idx[idx] = raw
                        posteriors_by_idx[idx] = post
                        results[idx]["country_replaced"] = True
                    unstable = [idx for idx in unstable if not _stable_for_descent(posteriors_by_idx[idx])]

            # ── Collect level results ───────────────────────────────────────────
            for i in level_indices:
                posterior = posteriors_by_idx[i]
                results[i][f"{level}_raw_response"] = raw_by_idx[i]

                if level in ("city", "street"):
                    filtered, conflicts = _filter_child_posterior(
                        posterior, results[i].get("country_posterior", {})
                    )
                    if conflicts:
                        results[i][f"{level}_backtrack_conflicts"] = conflicts[:5]
                        posterior = filtered

                best = max(posterior, key=posterior.get)
                results[i][level] = best
                results[i][f"{level}_posterior"] = posterior
                results[i][f"{level}_stable"] = _stable_for_descent(posterior)

                if level == "country" and (
                    posterior.get(best, 0) < 0.3 or not _stable_for_descent(posterior)
                ):
                    # No reliable country distribution — avoid propagating a flat
                    # parent into noisy child prompts.
                    for remaining in LEVELS[LEVELS.index(level) + 1:]:
                        results[i][remaining] = "Unknown"
                        results[i][f"{remaining}_posterior"] = {}
                    skip_finer[i] = True
                elif level == "city" and best == "Unknown":
                    results[i]["street"] = "Unknown"
                    results[i]["street_posterior"] = {}
                    skip_finer[i] = True

        for i in range(n):
            final_level = next(
                (lv for lv in reversed(LEVELS) if results[i].get(lv, "Unknown") != "Unknown"),
                LEVELS[-1]
            )
            results[i]["posterior"] = results[i].get(f"{final_level}_posterior", {})

        return results
