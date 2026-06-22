"""
SL — Single-source Uncertainty Modeling

For each visual clue, instead of a single (c_t, α_t) point estimate (GeoBayes),
we estimate a full uncertainty-aware likelihood by:
  1. Sampling N responses from the MLLM for the same evidence prompt.
  2. Parsing each response into a (c, α) tuple.
  3. Computing mean and variance across samples.
  4. Returning an uncertainty-weighted likelihood W_sl that shrinks toward 1.0
     (neutral) when variance is high — i.e. the model is unsure.

W_sl(e|l) = exp[ α_mean · β · (c_mean − 3) · (1 − λ · σ_c) ]
  where λ is an uncertainty penalty (default 1.0) and σ_c is the std of c across samples.
"""

import re
import math
import numpy as np
from models.mllm_client import MLLMClient
from config import SL_N_SAMPLES, BETA, SL_MAX_NEW_TOKENS


_SCORE_RE = re.compile(
    r"support[_\s]*rating[:\s]+([1-5])|rating[:\s]+([1-5])|score[:\s]+([1-5])",
    re.IGNORECASE,
)
_CONF_RE = re.compile(r"confidence[:\s]+(0\.\d+|1\.0|1)", re.IGNORECASE)

# Fast path: parse the Support line emitted by the new Probability-Thought
# verify prompt (pipeline.py:_verify_prompt). Format: "Hypothesis_A=S; Hyp_B=N; ..."
_SUPPORT_LINE_RE = re.compile(r"support\s*[:：]\s*(.+)", re.IGNORECASE)
# matches "<hyp_name> = S|C|N" tolerating whitespace/punctuation
_SUPPORT_PAIR_RE = re.compile(r"([^;,=]+?)\s*=\s*([SCN])", re.IGNORECASE)


def _parse_support_line(text: str, hypotheses: list[str]) -> dict[str, float] | None:
    """
    If the evidence text contains a structured Support: A=S; B=C; C=N line,
    convert it directly to W scores without a second LLM call.

    Mapping (centered around 3 per GeoBayes Eq.6 with α=0.7, c ∈ {1,3,5}):
        S → c=5 → W = exp[0.7 * ln2 * 2]  = 2.639
        N → c=3 → W = exp[0]              = 1.000  (neutral)
        C → c=1 → W = exp[0.7 * ln2 * -2] = 0.379

    Hypothesis names in the response are matched against `hypotheses` by
    case-insensitive substring (the model often abbreviates "United States" → "US").
    Returns None if parsing fails — caller falls back to sampled SL scoring.
    """
    m = _SUPPORT_LINE_RE.search(text)
    if not m:
        return None
    line = m.group(1)
    pairs = _SUPPORT_PAIR_RE.findall(line)
    if not pairs:
        return None

    mapping = {"S": 2.639, "N": 1.000, "C": 0.379}
    scores: dict[str, float] = {h: 1.0 for h in hypotheses}  # default neutral
    matched_any = False

    hyp_lower = [(h, h.lower()) for h in hypotheses]
    for raw_name, support in pairs:
        name_norm = raw_name.strip().lower()
        if not name_norm:
            continue
        # match by substring either direction (handles US/USA/United States)
        for orig, lh in hyp_lower:
            if name_norm in lh or lh in name_norm:
                scores[orig] = mapping[support.upper()]
                matched_any = True
                break
    return scores if matched_any else None


def _parse_ct_alpha(text: str) -> tuple[float, float]:
    """Extract (c_t, α_t) from an MLLM response string. Returns (3, 0.5) if parsing fails."""
    c_match = _SCORE_RE.search(text)
    a_match = _CONF_RE.search(text)
    c = float(next(g for g in c_match.groups() if g) ) if c_match else 3.0
    a = float(a_match.group(1)) if a_match else 0.5
    return c, a


def _w_single(c: float, alpha: float) -> float:
    return math.exp(alpha * BETA * (c - 3))


class SLModule:
    """
    Produces uncertainty-aware likelihood scores for a single evidence item
    across all current location hypotheses.
    """

    def __init__(self, mllm: MLLMClient, n_samples: int = SL_N_SAMPLES, uncertainty_penalty: float = 1.0):
        self.mllm = mllm
        self.n_samples = n_samples
        self.lam = uncertainty_penalty

    def _make_prompt(self, evidence_desc: str, hypothesis: str, level: str) -> list:
        """Build the MLLM message asking for support rating + confidence."""
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"You are evaluating geographic evidence for the location hypothesis: '{hypothesis}'.\n"
                        f"Current reasoning level: {level}.\n\n"
                        f"Evidence: {evidence_desc}\n\n"
                        "Rate how strongly this evidence supports the hypothesis.\n"
                        "Respond with:\n"
                        "  Support Rating: <1-5>  (1=strong contradiction, 3=neutral, 5=strong support)\n"
                        "  Confidence: <0.0-1.0>  (your confidence in this rating)\n"
                        "  Reasoning: <one sentence>"
                    )},
                ],
            }
        ]

    def score(
        self,
        evidence_desc: str,
        hypotheses: list[str],
        level: str = "country",
    ) -> dict[str, float]:
        """
        Returns W_sl(e | l) for each hypothesis l in hypotheses.

        Fast path: if the evidence text already contains a structured
        "Support: A=S; B=C; ..." line (emitted by the Probability-Thought
        verify prompt), parse it directly — no extra LLM calls needed.
        This both saves N×n_samples GPU forwards per evidence and uses the
        verify-time judgement which had full image context.

        Fallback: re-prompt the MLLM per hypothesis with n_samples sampling
        for uncertainty estimation.
        """
        fast = _parse_support_line(evidence_desc, hypotheses)
        if fast is not None:
            return fast

        from config import MAX_SL_BATCH_SIZE
        messages_list = [self._make_prompt(evidence_desc, hyp, level) for hyp in hypotheses]

        all_responses: list[list[str]] = []
        for i in range(0, len(messages_list), MAX_SL_BATCH_SIZE):
            batch = messages_list[i:i + MAX_SL_BATCH_SIZE]
            all_responses.extend(
                self.mllm.batch_sample_n(batch, n=self.n_samples, max_new_tokens=SL_MAX_NEW_TOKENS)
            )

        scores = {}
        for hyp, responses in zip(hypotheses, all_responses):
            parsed = [_parse_ct_alpha(r) for r in responses]
            cs     = np.array([p[0] for p in parsed])
            alphas = np.array([p[1] for p in parsed])

            c_mean  = cs.mean()
            c_std   = cs.std()
            a_mean  = alphas.mean()

            uncertainty_factor = max(0.3, 1.0 - self.lam * c_std)
            w = math.exp(a_mean * BETA * (c_mean - 3) * uncertainty_factor)
            scores[hyp] = w

        return scores

    def score_many(
        self,
        items: list[tuple[str, list[str]]],
        level: str = "country",
    ) -> list[dict[str, float]]:
        """
        Score multiple (evidence_desc, hypotheses) pairs in ONE big GPU batch.

        Builds a flat list of all (evidence, hypothesis) prompts across every
        item, sends them as one batch_sample_n call, then routes responses
        back to per-item score dicts.

        Fast path: items whose evidence already contains a structured Support
        line are scored directly without LLM calls.
        """
        from config import MAX_SL_BATCH_SIZE

        # Pre-split: items with structured Support line use fast path; others
        # fall into the LLM-sampling path below.
        results: list[dict[str, float]] = [dict() for _ in items]
        slow_items: list[tuple[int, tuple[str, list[str]]]] = []
        for item_idx, (evidence, hyps) in enumerate(items):
            fast = _parse_support_line(evidence, hyps)
            if fast is not None:
                results[item_idx] = fast
            else:
                slow_items.append((item_idx, (evidence, hyps)))

        if not slow_items:
            return results

        flat_msgs: list = []
        owners: list[tuple[int, str]] = []  # (item_idx, hyp_name)
        for item_idx, (evidence, hyps) in slow_items:
            for hyp in hyps:
                flat_msgs.append(self._make_prompt(evidence, hyp, level))
                owners.append((item_idx, hyp))

        if not flat_msgs:
            return results

        flat_responses: list[list[str]] = []
        for i in range(0, len(flat_msgs), MAX_SL_BATCH_SIZE):
            batch = flat_msgs[i:i + MAX_SL_BATCH_SIZE]
            flat_responses.extend(
                self.mllm.batch_sample_n(batch, n=self.n_samples, max_new_tokens=SL_MAX_NEW_TOKENS)
            )

        # do NOT re-init results — fast-path entries are already populated.
        for (item_idx, hyp), responses in zip(owners, flat_responses):
            parsed = [_parse_ct_alpha(r) for r in responses]
            cs     = np.array([p[0] for p in parsed])
            alphas = np.array([p[1] for p in parsed])

            c_mean = cs.mean()
            c_std  = cs.std()
            a_mean = alphas.mean()

            uncertainty_factor = max(0.3, 1.0 - self.lam * c_std)
            w = math.exp(a_mean * BETA * (c_mean - 3) * uncertainty_factor)
            results[item_idx][hyp] = w

        return results
