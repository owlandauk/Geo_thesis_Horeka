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

def _pre_analysis_prompt(image: Image.Image) -> list:
    """
    Pre-analysis stage (training-free) executed BEFORE country hypothesize.
    Combines three ideas from the literature:

    - GeoChain (EMNLP-25, Q1): hemisphere is a near-free win. Sun azimuth
      + vegetation biome → N/S hemisphere with ~95% reliability. This halves
      the country search space and prevents catastrophic cross-hemisphere
      errors (a major continent-level miss source).
    - GeoChain locatability + IMAGEO-Bench failure regression: rural / indoor
      / no-landmark images degrade geolocation by ~30-50%. Detect these
      upfront and route to continent fallback instead of forcing a guess.
    - GEO-R1 step 1 (Visual Cue Identification): solar elevation + shadows
      as a cue for latitude band.

    Returns a JSON dict the caller uses to (a) bias the country prior,
    (b) optionally skip street-level inference.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "Quick geographic pre-analysis. Look at the image and answer:\n"
                    "1. Hemisphere — N or S? Use sun position, shadows, vegetation, "
                    "season cues. Answer 'N', 'S', or 'unknown'.\n"
                    "2. Climate band — tropical / arid / temperate / boreal / polar / unknown.\n"
                    "3. Setting — urban / suburban / rural / natural / indoor.\n"
                    "4. Locatability — high / medium / low. HIGH = clear landmarks, "
                    "signage, distinctive architecture. LOW = generic interior, sky, "
                    "ocean, featureless landscape, no readable text or unique features.\n"
                    "5. Continent — your single best guess (Asia/Europe/Africa/"
                    "North America/South America/Oceania), or 'unknown'.\n\n"
                    "Respond with JSON only:\n"
                    '{"hemisphere":"N|S|unknown", "climate":"<band>", '
                    '"setting":"<type>", "locatability":"high|medium|low", '
                    '"continent":"<name>"}'
                )},
            ],
        }
    ]


def _hypothesize_prompt(image: Image.Image, level: str, context: str = "",
                        pre: dict | None = None) -> list:
    """
    Country-level prompt now uses the GRE Suite 5-step cue framework (Section A.2,
    p.21) which is broader than GLOBE's 4 cues — adds textual/script decoding and
    transport mode. Imperative "must include hypothesis even with partial evidence"
    clause from GRE + LLMGeo "Must" framing kills the refusal/Unknown failure mode.

    City/street levels follow GeoBayes (5 candidates, conditioned on parent context).
    Pre-analysis (hemisphere/continent) is injected as soft prior context if available.
    """
    if level == "country":
        pre_hint = ""
        if pre:
            hemi = pre.get("hemisphere", "unknown")
            climate = pre.get("climate", "unknown")
            cont = pre.get("continent", "unknown")
            if hemi != "unknown" or cont != "unknown":
                pre_hint = (
                    f"\nPre-analysis hints (use as soft prior, not absolute truth): "
                    f"hemisphere={hemi}, climate={climate}, continent guess={cont}.\n"
                )

        instruction = (
            "You MUST provide a country hypothesis list — do NOT respond with "
            "'unknown' or refuse. Prioritize these FIVE analysis steps:\n"
            "  (1) Architecture, infrastructure, street furniture.\n"
            "  (2) Textual clues — decode the SCRIPT (Latin/Cyrillic/Arabic/"
            "Han/Devanagari/Thai/Hangul/Hebrew/Greek) and any readable language "
            "on signage, license plates, billboards. The script alone narrows "
            "the country set dramatically.\n"
            "  (3) Vegetation, biome and climate cues — match to regional climate bands.\n"
            "  (4) Terrain, topography, coastline patterns.\n"
            "  (5) Transportation modes — vehicle types, road markings, drive-side "
            "(left in UK/Japan/Australia/India, right elsewhere).\n\n"
            f"{pre_hint}"
            "Then list the TOP 5 most likely countries with confidence in [0,1]. "
            "Even with partial evidence, you must commit to 5 candidates ranked by "
            "plausibility. Drive-side and script are particularly high-information "
            "cues — weight them strongly.\n\n"
            "Finally, build a 4-6 task verification plan that DISTINGUISHES between "
            "the candidates (not just confirms the top one)."
        )
    elif level == "city":
        instruction = (
            "List the TOP 5 most likely cities given the parent country context. "
            "Then build a 3-5 task verification plan that distinguishes between them. "
            "You MUST commit to a city list — do not say 'unknown'."
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
                    '  "cues": {"architecture": "<phrase>", "script_language": "<phrase>", '
                    '"vegetation_biome": "<phrase>", "terrain": "<phrase>", '
                    '"transport": "<phrase>"},\n'
                    '  "hypotheses": [{"location": "<name>", "confidence": <0-1>}, ...],\n'
                    '  "verification_plan": [{"desc": "<what to check>", "bbox": [x,y,w,h] or null}, ...]\n'
                    '}\n'
                    "The 'cues' field is optional for city/street levels."
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


# Lightweight country → continent map. Used to align country candidates with
# the pre-analysis continent guess. Covers the most common geolocation
# benchmark countries; falls back to no boost on unknown countries.
_COUNTRY_TO_CONTINENT = {
    # Asia
    "china": "Asia", "japan": "Asia", "south korea": "Asia", "korea": "Asia",
    "north korea": "Asia", "india": "Asia", "pakistan": "Asia", "bangladesh": "Asia",
    "thailand": "Asia", "vietnam": "Asia", "indonesia": "Asia", "malaysia": "Asia",
    "singapore": "Asia", "philippines": "Asia", "taiwan": "Asia", "myanmar": "Asia",
    "cambodia": "Asia", "laos": "Asia", "nepal": "Asia", "sri lanka": "Asia",
    "mongolia": "Asia", "kazakhstan": "Asia", "uzbekistan": "Asia",
    "iran": "Asia", "iraq": "Asia", "saudi arabia": "Asia", "uae": "Asia",
    "israel": "Asia", "turkey": "Asia", "jordan": "Asia", "lebanon": "Asia",
    "syria": "Asia", "qatar": "Asia", "kuwait": "Asia", "oman": "Asia",
    # Europe
    "germany": "Europe", "france": "Europe", "italy": "Europe", "spain": "Europe",
    "portugal": "Europe", "united kingdom": "Europe", "uk": "Europe",
    "england": "Europe", "scotland": "Europe", "wales": "Europe", "ireland": "Europe",
    "netherlands": "Europe", "belgium": "Europe", "switzerland": "Europe",
    "austria": "Europe", "poland": "Europe", "czech republic": "Europe",
    "czechia": "Europe", "hungary": "Europe", "greece": "Europe",
    "sweden": "Europe", "norway": "Europe", "finland": "Europe", "denmark": "Europe",
    "iceland": "Europe", "russia": "Europe", "ukraine": "Europe",
    "romania": "Europe", "bulgaria": "Europe", "serbia": "Europe",
    "croatia": "Europe", "slovenia": "Europe", "slovakia": "Europe",
    "estonia": "Europe", "latvia": "Europe", "lithuania": "Europe",
    "luxembourg": "Europe", "malta": "Europe", "cyprus": "Europe",
    # Africa
    "egypt": "Africa", "morocco": "Africa", "south africa": "Africa",
    "kenya": "Africa", "nigeria": "Africa", "ethiopia": "Africa", "ghana": "Africa",
    "algeria": "Africa", "tunisia": "Africa", "uganda": "Africa",
    "tanzania": "Africa", "senegal": "Africa", "zimbabwe": "Africa",
    "namibia": "Africa", "botswana": "Africa", "madagascar": "Africa",
    # North America
    "united states": "North America", "usa": "North America", "us": "North America",
    "canada": "North America", "mexico": "North America", "cuba": "North America",
    "jamaica": "North America", "guatemala": "North America",
    "panama": "North America", "costa rica": "North America",
    "honduras": "North America", "nicaragua": "North America",
    "el salvador": "North America", "dominican republic": "North America",
    # South America
    "brazil": "South America", "argentina": "South America", "chile": "South America",
    "peru": "South America", "colombia": "South America",
    "venezuela": "South America", "ecuador": "South America",
    "bolivia": "South America", "uruguay": "South America",
    "paraguay": "South America", "guyana": "South America",
    # Oceania
    "australia": "Oceania", "new zealand": "Oceania", "fiji": "Oceania",
    "papua new guinea": "Oceania",
}

# Rough N/S hemisphere assignment for top-prevalent geolocation countries.
_COUNTRY_HEMISPHERE = {
    # Southern hemisphere (everything south of equator, plus mostly-southern)
    "australia": "S", "new zealand": "S", "argentina": "S", "chile": "S",
    "uruguay": "S", "paraguay": "S", "bolivia": "S", "peru": "S",
    "brazil": "S",  # mostly south
    "south africa": "S", "namibia": "S", "botswana": "S", "zimbabwe": "S",
    "madagascar": "S", "tanzania": "S",  # mostly south
    "indonesia": "S",  # mostly south
    "papua new guinea": "S", "fiji": "S",
}


def _continent_of(country: str) -> str | None:
    if not country:
        return None
    return _COUNTRY_TO_CONTINENT.get(country.strip().lower())


def _hemisphere_of(country: str) -> str:
    """Default 'N' for any country not explicitly listed as 'S'."""
    if not country:
        return "N"
    return _COUNTRY_HEMISPHERE.get(country.strip().lower(), "N")


def _apply_pre_analysis_bias(prior: dict[str, float], pre: dict | None,
                             boost: float = 1.5,
                             penalty: float = 0.5) -> dict[str, float]:
    """
    Multiplicatively bias the country prior by the pre-analysis hemisphere and
    continent guesses. Countries consistent with hints get multiplied by `boost`;
    inconsistent ones by `penalty`. Then renormalize.

    Conservative: only applied when the pre-analysis is explicit (not 'unknown')
    and a candidate has a known continent/hemisphere mapping.
    """
    if not pre:
        return prior
    hemi_hint = pre.get("hemisphere", "unknown")
    cont_hint = pre.get("continent", "unknown")
    if hemi_hint == "unknown" and cont_hint == "unknown":
        return prior

    adjusted = {}
    for hyp, p in prior.items():
        factor = 1.0
        cont = _continent_of(hyp)
        hemi = _hemisphere_of(hyp)
        if cont_hint != "unknown" and cont is not None:
            factor *= boost if cont == cont_hint else penalty
        if hemi_hint != "unknown" and hemi_hint in ("N", "S"):
            factor *= boost if hemi == hemi_hint else penalty
        adjusted[hyp] = p * factor

    total = sum(adjusted.values())
    if total <= 0:
        return prior
    return {k: v / total for k, v in adjusted.items()}


def _fact_check_prompt(country: str, city: str) -> list:
    """
    GEO-R1 Fact-Check Engine (lightweight, text-only): after city prediction,
    verify city actually belongs to the predicted country. A common failure mode
    is the city/country fields being independently produced and incompatible
    (e.g., 'Paris' as city of 'United States'). One small LLM call rejects these.
    """
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    f"Is '{city}' a real city located in '{country}'?\n"
                    "Answer with JSON only: "
                    '{"consistent": true|false, "true_country": "<country if you '
                    "know where this city is, else null>\"}"
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

    def _pre_analyze(self, image: Image.Image) -> dict | None:
        """Run the pre-analysis prompt; returns None on parse failure."""
        resp = self.mllm.generate(_pre_analysis_prompt(image))
        parsed = _try_parse_json(resp)
        return parsed if isinstance(parsed, dict) else None

    def _hypothesize(self, image: Image.Image, level: str, context: str = "",
                     pre: dict | None = None) -> tuple[dict, list]:
        """Returns (prior_dict, verification_plan_list).

        Country level: ensembles the GRE 5-cue structured prompt with a
        GeoReasoner freeform prompt and biases by the pre-analysis hemisphere/
        continent hints. Three independent signals (structured, freeform,
        pre-analysis) with disjoint failure modes.
        """
        messages = _hypothesize_prompt(image, level, context, pre=pre)
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
            # Bias by the pre-analysis hemisphere/continent hints (very cheap
            # and helps the most catastrophic cross-hemisphere errors).
            prior = _apply_pre_analysis_bias(prior, pre)

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
        Returns {level: best_location_name, "posterior": final_posterior_dict,
                 "pre_analysis": <dict from pre-analysis prompt>}.
        """
        result       = {}
        key_evidence = []
        context      = ""

        # Pre-analysis: hemisphere/continent/locatability/setting hints.
        pre = self._pre_analyze(image)
        result["pre_analysis"] = pre

        for level in LEVELS:
            # Locatability gate: skip street-level inference when the image is
            # judged un-localizable (rural, indoor, no-landmark). Saves compute
            # and prevents the pipeline from forcing a wrong street guess.
            if pre and level == "street" and pre.get("locatability") == "low":
                result[level] = "Unknown"
                result[f"{level}_posterior"] = {}
                continue

            prior, plan = self._hypothesize(image, level, context, pre=pre)

            # at city/street level, seed hypotheses from prior level result
            if level != "country" and result:
                parent = result.get(LEVELS[LEVELS.index(level) - 1], "")
                context = f"Located in {parent}. Key clues: {'; '.join(key_evidence[-3:])}"

            posterior, key_evidence = self._run_level(
                image, level, prior, plan, key_evidence
            )

            best = max(posterior, key=posterior.get) if posterior else "Unknown"
            result[level] = best
            result[f"{level}_posterior"] = posterior

        # Fact-check: ensure predicted city is actually in predicted country.
        # If inconsistent, blank the city so evaluate.py geocodes the country.
        if (result.get("city") not in (None, "Unknown")
                and result.get("country") not in (None, "Unknown")):
            fc_resp = self.mllm.generate(_fact_check_prompt(result["country"], result["city"]))
            fc = _try_parse_json(fc_resp)
            if fc and fc.get("consistent") is False:
                result["city"] = "Unknown"
                result["city_posterior"] = {}

        # Set top-level posterior to the deepest non-Unknown level (mirrors
        # predict_batch). Falls back to country if everything below is Unknown.
        final_level = next(
            (lv for lv in reversed(LEVELS) if result.get(lv, "Unknown") != "Unknown"),
            LEVELS[0],
        )
        result["posterior"] = result.get(f"{final_level}_posterior", {})
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

        # ── Pre-analysis: one batched LLM call for all images ──────────────────
        # Produces hemisphere / climate / setting / locatability / continent
        # hints, used to (a) bias country prior, (b) gate street-level inference.
        pre_msgs = [_pre_analysis_prompt(img) for img in images]
        pre_resps = self.mllm.batch_generate(pre_msgs)
        pre_list: list[dict | None] = []
        for resp in pre_resps:
            parsed = _try_parse_json(resp)
            pre_list.append(parsed if isinstance(parsed, dict) else None)
        for i in range(n):
            results[i]["pre_analysis"] = pre_list[i]

        for level in LEVELS:
            # Locatability gate: skip street-level for low-locatability images.
            # `active_imgs_idx` is the set of images we ACTUALLY process this level.
            # Skipped images get level="Unknown" and a copy of the previous level's
            # posterior (so geocode falls back to country-level location).
            if level == "street":
                active_imgs_idx = [
                    i for i in range(n)
                    if not (pre_list[i] and pre_list[i].get("locatability") == "low")
                ]
                skipped = [i for i in range(n) if i not in active_imgs_idx]
                for i in skipped:
                    results[i][level] = "Unknown"
                    results[i][f"{level}_posterior"] = {}
            else:
                active_imgs_idx = list(range(n))
                skipped = []

            # ── Hypothesize: one batch call for ACTIVE images only ──────────────
            hyp_messages = [
                _hypothesize_prompt(images[i], level, contexts[i], pre=pre_list[i])
                for i in active_imgs_idx
            ]
            hyp_responses = self.mllm.batch_generate(hyp_messages) if hyp_messages else []

            # priors/plans keyed by image index (not by active position)
            priors: dict[int, dict] = {}
            plans:  dict[int, list] = {}
            for i, resp in zip(active_imgs_idx, hyp_responses):
                parsed = _try_parse_json(resp)
                if parsed is None or "hypotheses" not in parsed:
                    priors[i] = {"Unknown": 1.0}
                    plans[i]  = []
                else:
                    raw_scores = {h["location"]: h.get("confidence", 0.5)
                                  for h in parsed["hypotheses"]}
                    priors[i] = _softmax_prior(raw_scores)
                    plans[i]  = parsed.get("verification_plan", [])

            # ── Country-level: add GeoReasoner freeform second prompt as a ──────
            # complementary signal, then apply pre-analysis hemisphere/continent
            # bias. Three signals (structured 5-cue + freeform + pre-analysis)
            # produce the country prior.
            if level == "country" and active_imgs_idx:
                reasoner_msgs = [_geo_reasoner_prompt(images[i]) for i in active_imgs_idx]
                reasoner_resps = self.mllm.batch_generate(reasoner_msgs)
                for i, resp in zip(active_imgs_idx, reasoner_resps):
                    parsed = _try_parse_json(resp)
                    rc = parsed.get("country") if parsed else None
                    if rc and "Unknown" not in priors[i]:
                        priors[i] = _merge_geo_reasoner_seed(priors[i], rc)
                    # Apply pre-analysis bias only to non-degenerate priors
                    # ({"Unknown":1.0} has nothing to bias).
                    if "Unknown" not in priors[i]:
                        priors[i] = _apply_pre_analysis_bias(priors[i], pre_list[i])

            # seed context from parent level (only for images we'll process)
            if level != "country":
                parent_level = LEVELS[LEVELS.index(level) - 1]
                for i in active_imgs_idx:
                    parent = results[i].get(parent_level, "")
                    clues  = "; ".join(key_evidence[i][-3:])
                    contexts[i] = f"Located in {parent}. Key clues: {clues}" if parent else ""

            # ── POMDP loop across all active images simultaneously ──────────────
            posteriors: dict[int, dict]    = {i: dict(priors[i]) for i in active_imgs_idx}
            pending:    dict[int, list]    = {i: list(plans[i])  for i in active_imgs_idx}
            steps:      dict[int, int]     = {i: 0               for i in active_imgs_idx}
            ev_scores_all: dict[int, list] = {i: []              for i in active_imgs_idx}

            while True:
                # find images still running among ACTIVE ones.
                active = [
                    i for i in active_imgs_idx
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

            # ── Collect level results (only for active images) ──────────────────
            for i in active_imgs_idx:
                best = max(posteriors[i], key=posteriors[i].get) if posteriors[i] else "Unknown"
                results[i][level] = best
                results[i][f"{level}_posterior"] = posteriors[i]

            # ── Fact-check (after city level): batch text-only LLM call to ─────
            # verify each (country, city) pair is geographically consistent.
            # GEO-R1 idea: city + country are produced from separate hierarchical
            # steps and occasionally disagree (e.g. country=Italy, city=Paris).
            # Blanking inconsistent cities lets evaluate.py geocode the country
            # instead — recovers country-level accuracy at no city-level cost.
            if level == "city":
                fc_idx = [
                    i for i in range(n)
                    if results[i].get("city") not in (None, "Unknown")
                    and results[i].get("country") not in (None, "Unknown")
                ]
                if fc_idx:
                    fc_msgs = [
                        _fact_check_prompt(results[i]["country"], results[i]["city"])
                        for i in fc_idx
                    ]
                    fc_resps = self.mllm.batch_generate(fc_msgs)
                    for i, resp in zip(fc_idx, fc_resps):
                        fc = _try_parse_json(resp)
                        if fc and fc.get("consistent") is False:
                            results[i]["city"] = "Unknown"
                            results[i]["city_posterior"] = {}

        for i in range(n):
            final_level = next(
                (lv for lv in reversed(LEVELS) if results[i].get(lv, "Unknown") != "Unknown"),
                LEVELS[-1]
            )
            results[i]["posterior"] = results[i].get(f"{final_level}_posterior", {})

        return results
