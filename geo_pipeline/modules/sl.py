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
        Uses batch inference to process all hypotheses in parallel.
        """
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

            w = math.exp(a_mean * BETA * (c_mean - 3) * max(0.0, 1.0 - self.lam * c_std))
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

        This is the path that lets GPU utilization stay high: instead of
        N images × M hypotheses × n_samples scattered into N small forwards,
        we fire one giant forward with sum(M_i) × n_samples inputs.
        """
        from config import MAX_SL_BATCH_SIZE

        flat_msgs: list = []
        owners: list[tuple[int, str]] = []  # (item_idx, hyp_name)
        for item_idx, (evidence, hyps) in enumerate(items):
            for hyp in hyps:
                flat_msgs.append(self._make_prompt(evidence, hyp, level))
                owners.append((item_idx, hyp))

        if not flat_msgs:
            return [dict() for _ in items]

        flat_responses: list[list[str]] = []
        for i in range(0, len(flat_msgs), MAX_SL_BATCH_SIZE):
            batch = flat_msgs[i:i + MAX_SL_BATCH_SIZE]
            flat_responses.extend(
                self.mllm.batch_sample_n(batch, n=self.n_samples, max_new_tokens=SL_MAX_NEW_TOKENS)
            )

        results: list[dict[str, float]] = [dict() for _ in items]
        for (item_idx, hyp), responses in zip(owners, flat_responses):
            parsed = [_parse_ct_alpha(r) for r in responses]
            cs     = np.array([p[0] for p in parsed])
            alphas = np.array([p[1] for p in parsed])

            c_mean = cs.mean()
            c_std  = cs.std()
            a_mean = alphas.mean()

            w = math.exp(a_mean * BETA * (c_mean - 3) * max(0.0, 1.0 - self.lam * c_std))
            results[item_idx][hyp] = w

        return results
