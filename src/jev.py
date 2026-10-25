"""Categorical OpenRouter verification and source-only safety certification."""
from __future__ import annotations

import json
import numpy as np
from jsonschema import validate
from scipy.stats import beta

from .evidence import public_flow


class APIFailure(Exception):
    """Sanitized provider failure without provider text or credentials."""
    def __init__(self, code: str, status: int | None = None):
        super().__init__(code)
        self.status = status


def decision_schema(cfg: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {"decision": {"enum": ["BENIGN", "ATTACK", "UNCERTAIN"]},
                           "reason": {"type": "string", "minLength": 1,
                                      "maxLength": cfg["verifier"]["reason_max_chars"]}},
            "required": ["decision", "reason"]}


def request_body(flow: dict, source: dict, cfg: dict) -> dict:
    options = cfg["verifier"]
    rates = cfg["budgets"]["pricing"]["verifier"]
    evidence = {**public_flow(flow, source), "source_benign_statistics": source["baseline"],
                "source_examples": source["examples"]}
    return {"model": options["model"], "temperature": options["temperature"],
            "seed": cfg["general"]["seed"], "max_tokens": options["max_tokens"],
            "reasoning": options["reasoning"],
            "provider": {**options["provider_options"],
                         "max_price": {"prompt": rates["input_per_million"],
                                       "completion": rates["output_per_million"]}},
            "messages": [{"role": "system", "content": cfg["prompts"]["verifier"]["system"]},
                         {"role": "user", "content": json.dumps(evidence, allow_nan=False)}],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": options["schema_name"], "strict": True, "schema": decision_schema(cfg)}}}


def parse_decision(body: dict, cfg: dict) -> dict:
    if body.get("model") != cfg["verifier"]["model"]:
        raise ValueError("Unexpected verifier model")
    choice = body["choices"][0]
    if choice.get("finish_reason") != "stop" or choice["message"].get("tool_calls"):
        raise ValueError("Incomplete or tool-bearing verifier response")
    result = json.loads(choice["message"]["content"])
    validate(result, decision_schema(cfg))
    if not result["reason"].strip():
        raise ValueError("Empty reason")
    return result


def error_upper(errors: int, count: int, confidence: float) -> float:
    """One-sided Clopper–Pearson bound for a frozen categorical policy."""
    return 1.0 if count == 0 or errors == count else float(beta.ppf(confidence, errors + 1, count - errors))


def policy_evidence(decisions: list[str], labels: list[int], cfg: dict) -> dict:
    d, y = np.asarray(decisions), np.asarray(labels)
    result = {"correctly_resolved": int(np.sum(((y == 0) & (d == "BENIGN")) | ((y == 1) & (d == "ATTACK")))),
              "uncertain": int(np.sum(d == "UNCERTAIN")), "total": len(labels)}
    for name, mask, errors in (("false_benign", y == 1, d == "BENIGN"),
                               ("false_attack", y == 0, d == "ATTACK")):
        count, failed = int(mask.sum()), int(np.sum(mask & errors))
        # Two error constraints: union-bound adjustment for joint confidence.
        upper = error_upper(failed, count, 1 - (1 - cfg["confidence"]) / 2)
        result[name] = {"count": count, "errors": failed, "upper": upper}
    result["passes"] = all(result[n]["count"] >= cfg["min_samples_per_class"] and
                           result[n]["upper"] <= cfg[n + "_budget"]
                           for n in ("false_benign", "false_attack"))
    return result


def validated_policy(artifact: dict, context_hash: str, cfg: dict) -> dict:
    if not isinstance(artifact, dict):
        raise ValueError("Invalid calibration artifact")
    if (artifact.get("kind") != "categorical_verifier_v1" or artifact.get("context_hash") != context_hash
            or artifact.get("model") != cfg["verifier"]["model"] or artifact.get("status") != "certified"):
        raise ValueError("Missing, stale or uncertified verifier policy")
    samples = artifact["samples"]
    rows = [{s["row_position"] for s in samples[stage]} for stage in ("tuning", "certification")]
    if rows[0] & rows[1] or any(len(rows[i]) != len(samples[stage]) for i, stage in enumerate(("tuning", "certification"))):
        raise ValueError("Calibration partitions overlap or contain duplicate rows")
    for stage in ("tuning", "certification"):
        records = samples[stage]
        if len(records) != artifact["planned"][stage] or any(
                s.get("error") or s["decision"] not in ("BENIGN", "ATTACK", "UNCERTAIN") or
                type(s["binary_label"]) is not int or s["binary_label"] not in (0, 1) for s in records):
            raise ValueError("Incomplete calibration")
        evidence = policy_evidence([s["decision"] for s in records], [s["binary_label"] for s in records], cfg["verifier"]["calibration"])
        if stage == "tuning" and any(evidence[n]["count"] < cfg["verifier"]["calibration"]["min_samples_per_class"] or
                evidence[n]["errors"] / evidence[n]["count"] > cfg["verifier"]["calibration"][n + "_budget"]
                for n in ("false_benign", "false_attack")):
            raise ValueError("Source tuning failed")
        if stage == "certification" and not evidence["passes"]:
            raise ValueError("Source safety constraints failed")
    return {"certified": True}
