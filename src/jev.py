"""Official TypeSafe SDK binary inference and source-only threshold calibration."""

from __future__ import annotations

import logging
import math
import os

import numpy as np
from scipy.stats import beta


class APIFailure(Exception):
    """Sanitized provider failure: never retains provider text or credentials."""

    def __init__(self, code: str, status: int | None = None):
        super().__init__(code)
        self.status = status


def request_body(flow: dict, source: dict, cfg: dict) -> dict:
    prompts, options = cfg["prompts"]["jev"], cfg["jev"]
    if set(flow["features"]) != set(source["feature_names"]):
        raise ValueError("Flow schema differs from Jev source schema")
    return {"model": options["model"],
            "state": {"task": prompts["task"], "feature_instructions": prompts["features"],
                      "source_example_instructions": prompts["source_examples"],
                      "feature_names": source["feature_names"],
                      "flow_under_test": flow["features"], "source_examples": source["examples"],
                      "missing_feature_handling": source["missing_feature_handling"]},
            "questions": {options["request"]["question_id"]: {
                "type": options["request"]["question_type"], "instructions": prompts["question"]}}}


class JevDetector:
    def __init__(self, cfg: dict):
        self.options = cfg["jev"]
        self.log_level = cfg["general"]["logging"]["sdk_level"]
        self.client = None

    def send(self, body: dict) -> dict:
        # Lazy import keeps first-layer and dry-run use independent of API credentials.
        from typesafe_sdk import (RetryPolicy, TypeSafeAPIConnectionError, TypeSafeAPIError,
                                  TypeSafeAPIResponseValidationError, TypeSafeClient, TypeSafeError)
        if self.client is None:
            logging.getLogger("typesafe_sdk").setLevel(self.log_level)
            key = os.environ.get(self.options["api_key_env"])
            if not key:
                raise APIFailure("missing_jev_api_key")
            try:
                self.client = TypeSafeClient(api_key=key, base_url=self.options["base_url"],
                                            timeout=self.options["timeout_seconds"],
                                            retry=RetryPolicy(max_retries=0, timeout=None),
                                            compression=self.options["request"]["compression"])
            except TypeSafeError as exc:
                raise APIFailure("invalid_jev_client_configuration") from exc
        try:
            response = self.client.system_one(body["state"], body["questions"], model=body["model"])
        except TypeSafeAPIConnectionError as exc:
            raise APIFailure("connection_or_timeout") from exc
        except TypeSafeAPIResponseValidationError as exc:
            raise APIFailure("malformed_response") from exc
        except TypeSafeAPIError as exc:
            raise APIFailure("provider_http_error", exc.status) from exc
        except TypeSafeError as exc:
            raise APIFailure("invalid_jev_request") from exc
        return response.raw_http_response.json()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


def probability(body: dict, cfg: dict) -> float:
    if body.get("model") != cfg["jev"]["model"]:
        raise ValueError("Jev response model does not match the pinned version")
    answer = body["answers"][cfg["jev"]["request"]["question_id"]]
    value = answer["noul"]
    if (answer.get("type") != "noul" or isinstance(value, bool) or
            not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("Invalid Jev attack probability")
    return float(value)


def binary_decision(p_attack: float, thresholds: dict | None) -> str:
    if thresholds is None:
        return "UNCERTAIN"
    if p_attack <= thresholds["benign_max"]:
        return "BENIGN"
    if p_attack >= thresholds["attack_min"]:
        return "ATTACK"
    return "UNCERTAIN"


def error_upper(errors: int, count: int, confidence: float) -> float:
    """One-sided exact binomial bound for a fixed threshold on source certification."""
    if count == 0 or errors == count:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, count - errors))


def threshold_evidence(probabilities: list[float], labels: list[int], thresholds: dict,
                       cfg: dict) -> dict:
    p, y = np.asarray(probabilities), np.asarray(labels)
    result = {}
    for name, mask, errors in (
            ("false_benign", y == 1, p <= thresholds["benign_max"]),
            ("false_attack", y == 0, p >= thresholds["attack_min"])):
        count, failed = int(mask.sum()), int(np.sum(mask & errors))
        upper = error_upper(failed, count, cfg["confidence"])
        result[name] = {"count": count, "errors": failed, "upper": upper}
    result["passes"] = all(result[name]["count"] >= cfg["min_samples_per_class"] and
                           result[name]["upper"] <= cfg[name + "_budget"]
                           for name in ("false_benign", "false_attack"))
    return result


def choose_thresholds(probabilities: list[float], labels: list[int], cfg: dict) -> dict | None:
    """Select on source tuning only; certification must be scored afterwards."""
    p, y = np.asarray(probabilities), np.asarray(labels)
    feasible = []
    for lower in cfg["benign_candidates"]:
        for upper in cfg["attack_candidates"]:
            if lower >= upper:
                continue
            thresholds = {"benign_max": lower, "attack_min": upper}
            evidence = threshold_evidence(probabilities, labels, thresholds, cfg)
            # Tune on empirical error; only a separate frozen-threshold certification
            # may establish the configured upper error bounds.
            if all(evidence[name]["count"] >= cfg["min_samples_per_class"] and
                   evidence[name]["errors"] / evidence[name]["count"] <= cfg[name + "_budget"]
                   for name in ("false_benign", "false_attack")):
                correct = int(np.sum((y == 0) & (p <= lower)) + np.sum((y == 1) & (p >= upper)))
                feasible.append((correct, -lower, upper, thresholds))
    return max(feasible, key=lambda item: item[:3])[-1] if feasible else None


def validated_thresholds(artifact: dict, context_hash: str, cfg: dict) -> dict:
    options = cfg["jev"]["calibration"]
    if (artifact.get("context_hash") != context_hash or artifact.get("model") != cfg["jev"]["model"]
            or artifact.get("status") != "certified"):
        raise ValueError("Calibration is missing, uncertified, or belongs to a different source context")
    thresholds = artifact["thresholds"]
    lower, upper = thresholds["benign_max"], thresholds["attack_min"]
    if (lower not in options["benign_candidates"] or upper not in options["attack_candidates"]
            or not 0 <= lower < upper <= 1):
        raise ValueError("Calibration thresholds are outside configured candidates")
    for name in ("false_benign", "false_attack"):
        item = artifact["certification"][name]
        count, errors = item["count"], item["errors"]
        if (type(count) is not int or type(errors) is not int or not 0 <= errors <= count
                or count < options["min_samples_per_class"]
                or error_upper(errors, count, options["confidence"]) > options[name + "_budget"]):
            raise ValueError("Calibration does not meet the configured source error budget")
    return thresholds
