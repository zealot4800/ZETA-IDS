"""Binary LLM adjudication for data-plane misses."""
from __future__ import annotations

import json
import math

from .llm import answer, structured_request


def request_body(flow: dict, source: dict, cfg: dict) -> dict:
    prompts = cfg["prompts"]["binary_llm"]
    # Opt-in ablation: a "blind" prompt omits the source examples and the
    # source-support block (decision, ensemble score, neighbour agreement).
    blind = cfg["binary_llm"].get("blind_prompt", False)
    sections = ("task", "features", "question") if blind else ("task", "features", "source_examples", "question")
    instructions = "\n\n".join(prompts[key] for key in sections)
    context = flow.get("controller_context") or {}
    # Key order is kept identical to the original prompt in the default mode.
    state = {"feature_names": source["feature_names"],
             "flow_under_test": flow["features"]}
    if not blind:
        state["source_examples"] = context.get("source_examples", [])
        state["source_support"] = {key: value for key, value in context.items()
                                   if key != "source_examples"}
    state["limitations"] = source["limitations"]
    state["missing_feature_handling"] = source["missing_feature_handling"]
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"p_attack": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["p_attack"]}
    return structured_request(cfg["binary_llm"], [
        {"role": "system", "content": instructions},
        {"role": "user", "content": json.dumps(state, allow_nan=False)}],
        "binary_attack_probability", schema)


def probability(body: dict) -> float:
    value = answer(body)["p_attack"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("invalid_attack_probability")
    return float(value)


def binary_decision(p_attack: float, thresholds: dict) -> str:
    if p_attack < thresholds["benign_max"]:
        return "BENIGN"
    if p_attack > thresholds["attack_min"]:
        return "ATTACK"
    return "UNCERTAIN"


def combine_decision(decision: str, context: dict) -> tuple[str, str]:
    """Keep the raw LLM answer separate from the supported final decision."""
    if not context.get("supported"):
        return "UNCERTAIN", "insufficient_training_support"
    support = context["decision"]
    if decision == "UNCERTAIN":
        return support, "training_support_resolved_abstention"
    if decision != support:
        return "UNCERTAIN", "model_llm_disagreement"
    return decision, "model_neighbours_llm_agree"
