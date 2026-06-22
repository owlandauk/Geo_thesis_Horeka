"""
Posterior update module.

GeoBayes (AAAI-26, Eq.7) updates the posterior via simple multiplicative
Bayesian update:

    P(L=l | E_{1:t}) ∝ P_0(l) · ∏_t W(e_t | L=l)

where W(e_t | L=l) = exp[α_t · β · (c_t − 3)] is a centered support weight
(β = ln 2). The centering at 3 is critical — it lets contradictory evidence
(c_t < 3) cancel positive priors instead of always reinforcing.

This module previously implemented Dempster-Shafer fusion. The DST formulation
loses information because it L1-normalizes the W scores into BBA mass before
combination, throwing away the magnitude that distinguishes (W=1.87 vs W=0.54)
from (W=1.05 vs W=0.95). We restore GeoBayes's faithful multiplicative update
as the primary path, and keep DST Yager-cautious combination as an optional
fallback for high-conflict regimes (multiple very strong but contradictory
clues), gated by the conflict mass K.
"""

from __future__ import annotations
import math
import numpy as np
from config import DST_CONFLICT_THR


def _renormalize(d: dict[str, float]) -> dict[str, float]:
    total = sum(d.values())
    if total <= 0:
        n = max(len(d), 1)
        return {k: 1.0 / n for k in d}
    return {k: v / total for k, v in d.items()}


def _bayesian_update(prior: dict[str, float],
                     evidence_scores: list[dict[str, float]]) -> dict[str, float]:
    """GeoBayes Eq.7: P_t(l) ∝ P_0(l) · ∏ W(e_t | l).

    W is already exponential ( = exp[α·β·(c−3)] ), so log-sum then exp keeps
    numerics stable when many evidence items multiply.
    """
    if not evidence_scores:
        return dict(prior)

    hyps = list(prior.keys())
    log_posterior = {h: math.log(max(prior[h], 1e-12)) for h in hyps}

    for w_scores in evidence_scores:
        for h in hyps:
            w = w_scores.get(h, 1.0)
            log_posterior[h] += math.log(max(w, 1e-12))

    # softmax over log-posterior for numerical stability
    m = max(log_posterior.values())
    exps = {h: math.exp(lp - m) for h, lp in log_posterior.items()}
    return _renormalize(exps)


def _evidence_conflict(evidence_scores: list[dict[str, float]]) -> float:
    """Measure pairwise conflict across evidence items.

    Returns a value in [0, 1]: 0 = all evidence points to the same hypothesis,
    1 = every pair of evidence items points to a completely different hypothesis.

    Used as a gating signal: if conflict is very high, the multiplicative
    update can collapse to a wrong winner; we then fall back to Yager-cautious
    DST so conflicting mass goes to ignorance instead.
    """
    if len(evidence_scores) < 2:
        return 0.0

    # for each evidence, the hypothesis it most supports
    top_hyps = []
    for ws in evidence_scores:
        if not ws:
            continue
        top = max(ws.items(), key=lambda kv: kv[1])
        if top[1] > 1.0:  # only count items that genuinely support something
            top_hyps.append(top[0])

    if len(top_hyps) < 2:
        return 0.0

    from collections import Counter
    counts = Counter(top_hyps)
    most_common_count = counts.most_common(1)[0][1]
    # fraction of evidence items NOT pointing to the modal hypothesis
    return 1.0 - most_common_count / len(top_hyps)


# ── Yager DST fallback (only invoked under heavy conflict) ─────────────────────

def _likelihood_to_bba(w_scores: dict[str, float],
                       base_ignorance: float = 0.15) -> dict[str, float]:
    """Map W scores to a BBA. Unlike the previous version, we preserve W's
    magnitude by using softmax-on-log(W) so a strong likelihood ratio stays
    informative."""
    theta = "__ignorance__"
    hyps = list(w_scores.keys())
    if not hyps:
        return {theta: 1.0}

    log_ws = np.array([math.log(max(w_scores[h], 1e-12)) for h in hyps])
    log_ws -= log_ws.max()
    exps = np.exp(log_ws)
    probs = exps / exps.sum()

    bba = {h: float(p) * (1.0 - base_ignorance) for h, p in zip(hyps, probs)}
    bba[theta] = base_ignorance
    return bba


def _yager_combine(bba1: dict, bba2: dict) -> dict:
    """Yager's cautious rule: conflict mass goes to ignorance (no renormalization).
    Used as fallback when multiplicative update would collapse under high conflict."""
    theta = "__ignorance__"
    hyps = [k for k in bba1 if k != theta]
    combined: dict[str, float] = {h: 0.0 for h in hyps}
    combined[theta] = 0.0
    K = 0.0

    for k1, m1 in bba1.items():
        for k2, m2 in bba2.items():
            product = m1 * m2
            if k1 == k2:
                combined[k1] = combined.get(k1, 0.0) + product
            elif k1 == theta:
                combined[k2] = combined.get(k2, 0.0) + product
            elif k2 == theta:
                combined[k1] = combined.get(k1, 0.0) + product
            else:
                K += product

    combined[theta] = combined.get(theta, 0.0) + K
    return _renormalize(combined)


def _dst_fallback(prior: dict[str, float],
                  evidence_scores: list[dict[str, float]]) -> dict[str, float]:
    """Yager-cautious DST combination — robust to high-conflict evidence streams."""
    theta = "__ignorance__"
    hyps = list(prior.keys())
    bba = {h: prior[h] * 0.85 for h in hyps}
    bba[theta] = 0.15
    for ws in evidence_scores:
        bba = _yager_combine(bba, _likelihood_to_bba(ws))
    posterior = {h: bba.get(h, 0.0) for h in hyps}
    return _renormalize(posterior)


# ── Public module ──────────────────────────────────────────────────────────────

class DSTModule:
    """
    Posterior update. Primary path: GeoBayes Eq.7 multiplicative Bayesian update.
    Fallback (when evidence is highly inconsistent): Yager DST combination so
    conflict mass routes to ignorance instead of producing a confident wrong answer.
    """

    def fuse(self,
             prior: dict[str, float],
             evidence_scores: list[dict[str, float]]) -> dict[str, float]:
        if not prior:
            return {}
        # always run the Bayesian update; it is the GeoBayes-faithful path
        bayes_post = _bayesian_update(prior, evidence_scores)

        # fall back to Yager only when conflict is extreme — protects against
        # the case where two strong contradictory clues would collapse posterior
        conflict = _evidence_conflict(evidence_scores)
        if conflict > DST_CONFLICT_THR:
            return _dst_fallback(prior, evidence_scores)
        return bayes_post
