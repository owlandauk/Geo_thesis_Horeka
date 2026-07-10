"""
POMDP — Sequential Decision-Making for Evidence Selection

GeoBayes selects the next verification task by iterating through V_t in order
(essentially a fixed policy). This module replaces that with a POMDP-style
policy that selects the action (verification task) expected to yield the
highest information gain given the current belief state.

State  s_t  = current posterior distribution over hypotheses (belief)
Action a_t  = which verification task v_i to execute next
Reward r_t  = ΔP_t = change in posterior entropy after executing the action
Policy π    = LLM-based: the model is shown the current belief and pending tasks
              and asked to pick the most informative one.

For offline / no-search setting (CVHCI servers, no internet):
  - The model itself acts as the oracle for expected information gain.
  - No RL training is needed — we use the LLM as a zero-shot policy.
"""

import json
import re
from models.mllm_client import MLLMClient
from config import POMDP_MAX_STEPS, POMDP_MAX_NEW_TOKENS


_CHOICE_RE = re.compile(r'"?task_index"?\s*:\s*(\d+)', re.IGNORECASE)


class POMDPModule:
    """
    LLM-as-policy: given the current belief state and a list of pending
    verification tasks, ask the MLLM which task to execute next.
    """

    def __init__(self, mllm: MLLMClient, max_steps: int = POMDP_MAX_STEPS):
        self.mllm = mllm
        self.max_steps = max_steps

    def _belief_summary(self, posterior: dict[str, float]) -> str:
        items = sorted(posterior.items(), key=lambda x: -x[1])
        return ", ".join(f"{h}: {p:.3f}" for h, p in items[:5])

    def _make_policy_prompt(
        self,
        posterior: dict[str, float],
        pending_tasks: list[dict],
        level: str,
        step: int,
    ) -> list:
        task_list = "\n".join(
            f"  [{i}] desc: {t['desc']}, region: {t.get('bbox', 'full image')}"
            for i, t in enumerate(pending_tasks)
        )
        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"You are a geolocation agent at reasoning level: {level} (step {step}).\n\n"
                        f"Current belief over locations:\n  {self._belief_summary(posterior)}\n\n"
                        f"Pending verification tasks:\n{task_list}\n\n"
                        "Select the single task that would most reduce uncertainty — i.e., "
                        "the task most likely to produce evidence that strongly supports or "
                        "contradicts the current top hypotheses.\n\n"
                        'Respond with JSON only: {"task_index": <int>, "reason": "<short reason>"}'
                    )},
                ],
            }
        ]

    def select_action(
        self,
        posterior: dict[str, float],
        pending_tasks: list[dict],
        level: str,
        step: int,
    ) -> int:
        """
        Returns the index of the selected task in pending_tasks.
        Falls back to index 0 if parsing fails.
        """
        if len(pending_tasks) == 1:
            return 0

        messages = self._make_policy_prompt(posterior, pending_tasks, level, step)
        response = self.mllm.generate(messages, max_new_tokens=POMDP_MAX_NEW_TOKENS)

        match = _CHOICE_RE.search(response)
        if match:
            idx = int(match.group(1))
            return min(idx, len(pending_tasks) - 1)
        return 0  # fallback: first task

    def should_stop(
        self,
        posterior: dict[str, float],
        step: int,
        level: str,
        all_tasks_exhausted: bool,
    ) -> bool:
        """
        POMDP stopping condition — mirrors GeoBayes Eq.11 but adds a step cap.
        At street level: stop when tasks exhausted.
        At other levels: stop when max_prob > TRANSITION_THR after at least
        one verification step, or when tasks are exhausted.
        """
        from config import TRANSITION_THR
        max_prob = max(posterior.values(), default=0.0)

        if step >= self.max_steps:
            return True
        if all_tasks_exhausted:
            return True
        if level != "street" and step > 0 and max_prob >= TRANSITION_THR:
            return True
        return False
