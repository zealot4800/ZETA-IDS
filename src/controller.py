"""Source-assessed benign filtering, categorical verification and bounded investigation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import pickle
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from copy import deepcopy
import yaml
from jsonschema import validate, ValidationError

from .agent import openrouter_request, investigate, initial_messages, request_body as investigation_request
from .evidence import build_source, controller_partition, dataplane_partition, digest, flow_record, public_flow, unresolved_indices
from .jev import (APIFailure, parse_decision, policy_evidence, error_upper,
                  request_body as verifier_request, validated_policy)
from .loao import fit_fold
from .loao_data import _read_dataset


def record_schema(cfg: dict) -> dict:
    properties = {
        "evidence_state": {"enum": ["CONFLICTING", "ATTACK_SUPPORTED", "BENIGN_SUPPORTED", "INSUFFICIENT"]},
        "security_decision": {"enum": ["BENIGN", "ATTACK", "UNRESOLVED"]},
        "final_disposition": {"enum": ["ACCEPT", "ALERT", "RETAIN_UNRESOLVED"]},
        "benign_release_policy": {"type": ["string", "null"]},
        "escalation_reason": {"type": "string"},
        "flow_id": {"type": "string"}, "dataset": {"type": "string"},
        "first_layer_decision": {"const": "UNRESOLVED"},
        "verifier_decision": {"enum": ["BENIGN", "ATTACK", "UNCERTAIN"]},
        "raw_decision": {"enum": [None, "BENIGN", "ATTACK", "UNCERTAIN"]},
        "reason": {"type": ["string", "null"]}, "verifier_model": {"type": "string"},
        "controller_status": {"enum": ["BENIGN", "UNRESOLVED", "ATTACK_PENDING_INVESTIGATION"]},
        "handoff": {"type": ["object", "null"]},
        "latency_ms": {"type": "number", "minimum": 0},
        "token_usage": {"type": "object"},
        "estimated_cost": {"type": ["number", "null"], "minimum": 0},
        "error": {"type": ["string", "null"]},
        "calibration_status": {"enum": ["certified", "unavailable"]},
        "cache_hits": {"type": "integer", "minimum": 0}}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for section in ("general", "verifier", "advanced_llm", "prompts", "evidence", "budgets", "output"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing controller configuration section: {section}")
    v = cfg["verifier"]
    if v["provider"] != "openrouter" or v["model"] not in v["open_weight_models"]:
        raise ValueError("Verifier requires an explicitly configured open-weight OpenRouter model")
    inv = cfg["advanced_llm"]
    if type(inv["enabled"]) is not bool or inv["selected"] not in inv["models"]:
        raise ValueError("Invalid investigation selection")
    for name in inv["comparison_models"]:
        if name not in inv["models"]:
            raise ValueError("Unknown comparison model")
    for options in inv["models"].values():
        if options["model"] not in inv["open_weight_models"]:
            raise ValueError("Investigation requires an explicitly allowed open-weight model")
        if options.get("reasoning") is not None and options["reasoning"].get("effort") not in ("low", "medium", "high"):
            raise ValueError("Invalid reasoning setting")
        if set(options["pricing"]) != {"input_per_million", "output_per_million"} or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in options["pricing"].values()):
            raise ValueError("Invalid investigation prices")
    if inv["provider"] != "openrouter" or inv["base_url"] != "https://openrouter.ai/api/v1":
        raise ValueError("Investigation must use OpenRouter")
    if not set(inv["allowed_tools"]) <= {"inspect_flow", "compare_source_baseline", "retrieve_attack_knowledge"}:
        raise ValueError("Unknown investigation tool")
    if inv["endpoint"] != "/chat/completions" or inv["provider_options"].get("require_parameters") is not True:
        raise ValueError("Investigation requires supported structured-output parameters")
    if not 0 <= inv["temperature"] <= 2 or not math.isfinite(inv["timeout_seconds"]) or inv["timeout_seconds"] <= 0:
        raise ValueError("Invalid investigation generation settings")
    if inv["response_format"]["type"] not in ("json_schema", "json_object") or inv["tool_choice"] != "auto":
        raise ValueError("Invalid investigation response/tool settings")
    if type(inv["investigate_uncertain"]) is not bool or not inv["comparison_models"] or len(inv["comparison_models"]) != len(set(inv["comparison_models"])):
        raise ValueError("Invalid investigation routing/comparison configuration")
    for k in ("max_tokens", "max_steps", "max_tool_calls", "sample_per_stratum", "repeats"):
        if type(inv[k]) is not int or inv[k] < 1:
            raise ValueError("Investigation limits must be positive integers")
    if not 0 <= cfg["routing"]["benign_acceptance_max_fraction"] <= 1:
        raise ValueError("Invalid controller benign-acceptance cap")
    quantiles = cfg["routing"]["coverage_region_quantiles"]
    if not quantiles or quantiles != sorted(set(quantiles)) or not all(0 < q <= 1 for q in quantiles):
        raise ValueError("Invalid coverage region quantiles")
    candidates = cfg["routing"]["coverage_candidates"]
    if not candidates or len({c["name"] for c in candidates}) != len(candidates):
        raise ValueError("Coverage candidates must have unique names")
    for c in candidates:
        if not {"name", "max_depth", "max_leaf_nodes", "top_features", "min_benign_retention"} <= set(c) or not set(c) <= {"name", "max_depth", "max_leaf_nodes", "top_features", "min_benign_retention", "region_quantiles"} or any(type(c[k]) is not int or c[k] < 1 for k in ("max_depth", "max_leaf_nodes", "top_features")):
            raise ValueError("Invalid source coverage candidate")
        q = c.get("region_quantiles", quantiles)
        if not 0 < c["min_benign_retention"] <= 1 or not q or q != sorted(set(q)) or not all(0 < v <= 1 for v in q):
            raise ValueError("Invalid coverage retention/radii")
    fractions = cfg["routing"]["source_fractions"]
    if set(fractions) != {"fit", "gate", "tuning", "certification"} or any(not 0 < v < 1 for v in fractions.values()) or not math.isclose(sum(fractions.values()), 1):
        raise ValueError("Source fractions must be positive and sum to one")
    url = urlparse(v["base_url"])
    if url.scheme != "https" or url.hostname != "openrouter.ai" or url.username or url.password or url.query:
        raise ValueError("Verifier must use credential-free HTTPS OpenRouter URL")
    if not v["api_key_env"] or v["endpoint"] != "/chat/completions":
        raise ValueError("Invalid verifier endpoint or credential environment name")
    if v["provider_options"].get("require_parameters") is not True:
        raise ValueError("Structured outputs require provider parameter support")
    cal = v["calibration"]
    for key in ("false_benign_budget", "false_attack_budget", "confidence"):
        if not 0 < cal[key] < 1:
            raise ValueError(f"Invalid calibration {key}")
    for value in (cfg["general"]["max_flows"], v["max_tokens"], v["reason_max_chars"],
                  cal["min_samples_per_class"], cal["tuning_per_class"], cal["certification_per_class"], cal["progress_every"]):
        if type(value) is not int or value < 1:
            raise ValueError("Controller limits must be positive integers")
    for value in (v["few_shot_per_class"], cfg["budgets"]["token_reserve_overhead"]):
        if type(value) is not int or value < 0:
            raise ValueError("Example and token reserve counts must be nonnegative integers")
    if not 0 <= v["temperature"] <= 2 or v["reasoning"].get("effort") not in ("low", "medium", "high"):
        raise ValueError("Invalid verifier generation settings")
    if not math.isfinite(v["timeout_seconds"]) or v["timeout_seconds"] <= 0:
        raise ValueError("Invalid timeout")
    if not isinstance(cfg["prompts"]["verifier"]["system"], str) or not cfg["prompts"]["verifier"]["system"].strip():
        raise ValueError("Verifier system prompt is required")
    if type(cfg["general"]["shuffle_flows"]) is not bool:
        raise ValueError("shuffle_flows must be boolean")
    if cfg["evidence"]["source_reference"] != "benign_training":
        raise ValueError("Only source-training references are permitted")
    if set(cfg["output"]["record_fields"]) != set(record_schema(cfg)["properties"]):
        raise ValueError("Configured record fields must match the controller interface")
    budget = cfg["budgets"]["max_cost_usd"]
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Per-run budget must be finite and positive")
    for stage in ("verifier", "advanced_llm"):
        rates = cfg["budgets"]["pricing"][stage]
        if any(type(rates.get(k)) not in (int, float) or not math.isfinite(rates[k]) or rates[k] < 0
               for k in ("input_per_million", "output_per_million")):
            raise ValueError("Invalid configured token prices")
    return cfg


class Requests:
    """One attempt per request, one dollar budget per run and duplicate-request caching."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.count = 0
        self.tokens = 0
        self.cost = 0.0
        self.usage_complete = True
        self.cost_complete = True
        self.events = defaultdict(list)
        self.cache = {}

    def _budget_error(self, stage: str, body: dict) -> str | None:
        if not self.cost_complete:
            return "cost_usage_unavailable"
        limits = self.cfg["budgets"]
        # Reserve one input token per serialized byte plus protocol overhead.
        reserve_in = len(json.dumps(body, allow_nan=False).encode()) + limits["token_reserve_overhead"]
        reserve_out = self.cfg[stage]["max_tokens"]
        rates = limits["pricing"][stage]
        reserve_cost = (reserve_in * rates["input_per_million"] +
                        reserve_out * rates["output_per_million"]) / 1_000_000
        if self.cost + reserve_cost > limits["max_cost_usd"]:
            return "cost_budget_exhausted"
        return None

    def _usage(self, stage: str, body: dict | None) -> tuple[dict | None, float | None]:
        raw = (body or {}).get("usage")
        if not isinstance(raw, dict):
            raw = {}
        usage = {"input_tokens": raw.get("input_tokens", raw.get("prompt_tokens")),
                 "output_tokens": raw.get("output_tokens", raw.get("completion_tokens"))}
        if any(type(value) is not int or value < 0 for value in usage.values()):
            self.usage_complete = self.cost_complete = False
            return None, None
        self.tokens += sum(usage.values())
        rates = self.cfg["budgets"]["pricing"][stage]
        if self.cfg[stage]["provider"] == "openrouter" and "cost" in raw:
            cost = raw["cost"]
        else:
            cost = sum(usage[key + "_tokens"] * rates[key + "_per_million"]
                       for key in ("input", "output")) / 1_000_000
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            self.cost_complete = False
            return usage, None
        self.cost += cost
        return usage, cost

    def call(self, stage: str, flow_id: str, body: dict, send) -> dict:
        key = digest([stage, body])
        cache_hit = stage != "advanced_llm" and key in self.cache
        usage, cost = None, 0.0
        provider_usage = None
        attempted = False
        if cache_hit:
            result = self.cache[key]
        elif error := self._budget_error(stage, body):
            result = {"body": None, "error": error}
        else:
            self.count += 1
            attempted = True
            try:
                response = send(body)
                if not isinstance(response, dict):
                    raise APIFailure("malformed_response")
                if "error" in response:
                    detail = response["error"]
                    status = detail.get("code") if isinstance(detail, dict) else None
                    raise APIFailure("provider_error_response", status if type(status) is int else None)
                result = {"body": response, "error": None}
            except APIFailure as exc:
                error = str(exc) + (f"_{exc.status}" if exc.status else "")
                result = {"body": None, "error": error}
            except (ValueError, TypeError, KeyError):
                result = {"body": None, "error": "malformed_response"}
            except BaseException:
                self.cost_complete = False  # An interrupted request may still be billed.
                raise
            usage, cost = self._usage(stage, result["body"])
            provider_usage = (result["body"] or {}).get("usage")
            if stage != "advanced_llm":
                self.cache[key] = result
        self.events[flow_id].append({"stage": stage, "usage": usage, "cost": cost,
                                     "cache_hit": cache_hit, "provider_usage": provider_usage, "attempted": attempted})
        return result

    def measurements(self, flow_id: str) -> dict:
        events = self.events[flow_id]
        usage = {}
        for stage in ("verifier", "advanced_llm"):
            charged = [event for event in events if event["stage"] == stage and not event["cache_hit"]]
            usage[stage] = ({key: sum(event["usage"][key] for event in charged)
                             for key in ("input_tokens", "output_tokens")}
                            if charged and all(event["usage"] is not None for event in charged) else None)
        return {"token_usage": usage,
                "estimated_cost": (sum(event["cost"] for event in events)
                                   if events and all(event["cost"] is not None for event in events) else None),
                "cache_hits": sum(event["cache_hit"] for event in events)}


def resolve(flow: dict, source: dict, cfg: dict, policy: dict | None,
            calibration_status: str, requests: Requests, verifier_send) -> dict | None:
    if flow["first_layer_decision"] != "UNRESOLVED":
        return None
    started = time.perf_counter()
    record = dict.fromkeys(cfg["output"]["record_fields"])
    record.update(flow_id=flow["flow_id"], dataset=flow["dataset"],
                  first_layer_decision="UNRESOLVED", verifier_model=cfg["verifier"]["model"],
                  verifier_decision="UNCERTAIN", controller_status="UNRESOLVED",
                  calibration_status=calibration_status)
    if policy is None or policy.get("certified") is not True or calibration_status != "certified":
        record["error"] = "missing_or_unsafe_calibration"
    else:
        reply = requests.call("verifier", flow["flow_id"], verifier_request(flow, source, cfg), verifier_send)
        if reply["error"]:
            record["error"] = reply["error"]
        else:
            try:
                answer = parse_decision(reply["body"], cfg)
                decision = answer["decision"]
                record.update(raw_decision=decision, verifier_decision=decision, reason=answer["reason"])
                if decision == "BENIGN":
                    record["controller_status"] = "BENIGN"
                elif decision == "ATTACK":
                    record.update(controller_status="ATTACK_PENDING_INVESTIGATION",
                                  handoff=public_flow(flow, source))
            except (KeyError, IndexError, TypeError, ValueError, ValidationError):
                record["error"] = "invalid_verifier_response"
    decision = record["verifier_decision"]
    record.update(evidence_state=flow["first_layer_evidence"].get("evidence_state", "INSUFFICIENT"),
                  security_decision="UNRESOLVED" if decision == "UNCERTAIN" else decision,
                  final_disposition={"BENIGN": "ACCEPT", "ATTACK": "ALERT", "UNCERTAIN": "RETAIN_UNRESOLVED"}[decision],
                  benign_release_policy="source_certified_categorical_verifier" if decision == "BENIGN" else None,
                  escalation_reason=record["error"] or ("verified_attack" if decision == "ATTACK" else
                                    "verifier_abstained" if decision == "UNCERTAIN" else "certified_benign_recommendation"))
    record.update(requests.measurements(flow["flow_id"]))
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    validate(record, record_schema(cfg))
    return record


def source_partitions(data: dict, cfg: dict, attack: str) -> dict:
    """Reserve feature-group-disjoint source pools before any detector fitting."""
    groups = pd.util.hash_pandas_object(data["x"], index=False).to_numpy()
    eligible = data["validation"] & (data["family"] != attack)
    unique = np.unique(groups[eligible])
    shuffled = np.random.default_rng(cfg["routing"]["split_seed"]).permutation(unique)
    pools, start = {}, 0
    fractions = cfg["routing"]["source_fractions"]
    cumulative = 0.0
    for name, fraction in fractions.items():
        cumulative += fraction
        end = round(len(shuffled) * cumulative)
        pools[name] = eligible & np.isin(groups, shuffled[start:end])
        start = end
    return pools


def fit_controller_fold(data: dict, loao_cfg: dict, cfg: dict, attack: str):
    pools = source_partitions(data, cfg, attack)
    source_data = {**data, "validation": pools["fit"]}
    controller_loao = deepcopy(loao_cfg)
    controller_loao["benign_acceptance"]["max_fraction"] = cfg["routing"]["benign_acceptance_max_fraction"]
    controller_loao["safe_regions"]["quantiles"] = cfg["routing"]["coverage_region_quantiles"]
    chosen = None
    coverage_audit = []
    for spec in cfg["routing"]["coverage_candidates"]:
        candidate_cfg = deepcopy(controller_loao)
        candidate_cfg["tree"].update(max_depth=spec["max_depth"], max_leaf_nodes=spec["max_leaf_nodes"])
        candidate_cfg["safe_regions"]["top_features"] = spec["top_features"]
        candidate_cfg["safe_regions"]["quantiles"] = spec.get("region_quantiles", cfg["routing"]["coverage_region_quantiles"])
        candidate_cfg["membership"]["min_benign_retention"] = spec["min_benign_retention"]
        candidate_cfg["conditional"]["min_benign_retention"] = spec["min_benign_retention"]
        candidate, candidate_fold = fit_fold(source_data, candidate_cfg, attack)
        decisions = dataplane_partition(data, candidate, candidate_fold["tuning"])
        accepted = decisions["BENIGN"]
        benign = int(np.sum(data["y"][accepted] == 0))
        errors = int(np.sum(data["y"][accepted] == 1))
        supported = candidate["policy_status"] == "certified"
        coverage_audit.append({**spec, "source_tuning_benign_accepted": benign,
                               "source_tuning_attack_accepted": errors,
                               "inner_policy_status": candidate["policy_status"]})
        # Existing inner fitting/certification is development data. Only the
        # selected frozen candidate is subsequently assessed on the outer gate.
        score = (int(supported), benign, -errors)
        if chosen is None or score > chosen[0]:
            chosen = (score, candidate, candidate_fold, spec)
        if cfg["general"]["logging"]["print_summary"]:
            print(f"Source coverage {spec['name']}: benign={benign}, attacks={errors}, status={candidate['policy_status']}", flush=True)
    _, model, fold, selected = chosen
    internal = dataplane_partition(data, model, pools["gate"])
    groups = pd.util.hash_pandas_object(data["x"], index=False).to_numpy()
    attack_groups = np.unique(groups[pools["gate"] & (data["y"] == 1)])
    accepted = internal["BENIGN"]
    leaked = np.unique(groups[accepted[data["y"][accepted] == 1]])
    confidence = cfg["verifier"]["calibration"]["confidence"]
    upper = error_upper(len(leaked), len(attack_groups), confidence)
    budget = loao_cfg["safe_regions"]["known_attack_leakage_budget"]
    useful = model["policy_status"] == "certified" and bool(np.any(data["y"][accepted] == 0))
    audit = {"status": "certified" if upper <= budget and useful else "uncertified",
             "useful_benign_coverage": useful,
             "attack_groups": len(attack_groups), "leaked_attack_groups": len(leaked),
             "upper": upper, "budget": budget, "confidence": confidence,
             "inner_policy_status": model["policy_status"],
             "coverage_selected": selected, "coverage_candidates": coverage_audit,
             "coverage_region_quantiles": cfg["routing"]["coverage_region_quantiles"],
             "benign_acceptance_max_fraction": cfg["routing"]["benign_acceptance_max_fraction"],
             "benign_rows": int(np.sum(pools["gate"] & (data["y"] == 0))),
             "benign_rows_accepted": int(np.sum(data["y"][accepted] == 0)),
             "source_pools": {k: int(v.sum()) for k, v in pools.items()},
             "unit": "Distinct feature groups; any accepted attack row makes its group a leakage error."}
    model["controller_benign_authorized"] = audit["status"] == "certified"
    return model, {**fold, "tuning": pools["tuning"], "certification": pools["certification"], "gate_audit": audit}


def calibration_run(data: dict, model: dict, fold: dict, source: dict, cfg: dict,
                    requests: Requests, verifier_send, context_hash: str, checkpoint=None) -> dict:
    options = cfg["verifier"]["calibration"]
    pools = {}
    rng = np.random.default_rng(cfg["general"]["seed"])
    for stage in ("tuning", "certification"):
        mask = fold[stage]
        if np.any(mask & data["test"]) or np.any(mask & data["train"]):
            raise ValueError("Verifier calibration must use source validation only")
        unresolved = unresolved_indices(data, model, mask)
        # One randomly chosen representative per input pattern before class sampling.
        shuffled = rng.permutation(unresolved)
        groups = pd.util.hash_pandas_object(data["x"].iloc[shuffled], index=False).to_numpy()
        unresolved = shuffled[np.unique(groups, return_index=True)[1]]
        pools[stage] = rng.permutation(np.concatenate([
            rng.permutation(unresolved[data["y"][unresolved] == label])[:options[stage + "_per_class"]]
            for label in (0, 1)]))
        if any(np.sum(data["y"][pools[stage]] == label) < options["min_samples_per_class"] for label in (0, 1)):
            return {"kind": "categorical_verifier_v1", "context_hash": context_hash,
                    "model": cfg["verifier"]["model"], "status": "uncertified",
                    "error": "insufficient_source_" + stage + "_support",
                    "source_support": {"benign": int(np.sum(data["y"][pools[stage]] == 0)),
                                       "attack": int(np.sum(data["y"][pools[stage]] == 1))}}
    if np.intersect1d(pools["tuning"], pools["certification"]).size:
        raise ValueError("Source tuning and certification overlap")
    artifact = {"kind": "categorical_verifier_v1", "context_hash": context_hash,
                "model": cfg["verifier"]["model"], "status": "uncertified", "error": None,
                "sampling_unit": "One representative per distinct feature group, before binary stratification.",
                "policy": "Use categorical decisions directly; reject the whole policy if certification fails.",
                "planned": {stage: len(pool) for stage, pool in pools.items()},
                "samples": {stage: [] for stage in pools}}
    if checkpoint:
        checkpoint(artifact)
    for stage in ("tuning", "certification"):
        for position in pools[stage]:
            flow = flow_record(data, model, int(position))
            reply = requests.call("verifier", flow["flow_id"], verifier_request(flow, source, cfg), verifier_send)
            error, decision = reply["error"], "UNCERTAIN"
            if not error:
                try:
                    decision = parse_decision(reply["body"], cfg)["decision"]
                except (KeyError, IndexError, TypeError, ValueError, ValidationError) as exc:
                    error = "invalid_verifier_response"
                    # Keep structural diagnostics, never provider text or flow evidence.
                    body = reply["body"] if isinstance(reply["body"], dict) else {}
                    choices = body.get("choices")
                    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
                    message = choice.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    artifact["response_diagnostics"] = {
                        "validation_error_type": type(exc).__name__,
                        "model_matches": body.get("model") == cfg["verifier"]["model"],
                        "finished_normally": choice.get("finish_reason") == "stop",
                        "output_truncated": choice.get("finish_reason") == "length",
                        "content_is_string": isinstance(content, str),
                        "content_length": len(content) if isinstance(content, str) else None,
                    }
            artifact["samples"][stage].append({"row_position": int(data["row_positions"][position]),
                                               "binary_label": int(data["y"][position]),
                                               "decision": decision, "error": error})
            rows = artifact["samples"][stage]
            artifact[stage] = policy_evidence([r["decision"] for r in rows], [r["binary_label"] for r in rows], options)
            artifact["error"] = error
            if checkpoint:
                checkpoint(artifact)
            if error:
                return artifact  # Failed rows are retained; incomplete runs cannot certify.
            if cfg["general"]["logging"]["print_summary"] and len(rows) % options["progress_every"] == 0:
                print(f"Verifier source {stage}: {len(rows)}/{len(pools[stage])}; cost=${requests.cost:.4f}", flush=True)
        if stage == "tuning" and any(artifact[stage][n]["errors"] / artifact[stage][n]["count"] > options[n + "_budget"]
                                     for n in ("false_benign", "false_attack")):
            artifact["error"] = "source_tuning_failed"
            return artifact
    if artifact["certification"]["passes"]:
        artifact["status"] = "certified"
    else:
        artifact["error"] = "source_certification_failed"
    return artifact


def context_fingerprint(data_path: Path, loao_cfg: dict, source: dict, cfg: dict, attack: str) -> str:
    with data_path.open("rb") as stream:
        data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    code = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")}
    return digest({"data": data_hash, "first_layer": loao_cfg, "source": source, "attack": attack,
                   "verifier": cfg["verifier"], "routing": cfg["routing"], "seed": cfg["general"]["seed"],
                   "prompts": cfg["prompts"]["verifier"], "pricing": cfg["budgets"]["pricing"]["verifier"], "code": code})


def _path(base: Path, template: str, **values) -> Path:
    target = (base / template.format(**values)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError("Controller output path escapes its configured directory")
    return target


def csv_record(record: dict) -> dict:
    """Keep nested evidence/usage as JSON in properly quoted CSV cells."""
    return {key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
            for key, value in record.items()}


def evaluation_summary(states: list[dict]) -> dict:
    """Offline truth only; failures/abstentions stay in every evaluated denominator."""
    def rate(n, d):
        return {"numerator": n, "denominator": d, "percent": 100 * n / d if d else None}
    result = {}
    for group in ("benign", "known_attack", "heldout_attack", "all_attacks"):
        plane, observed = Counter(), Counter()
        groups = ("known_attack", "heldout_attack") if group == "all_attacks" else (group,)
        for state in states:
            for name in groups:
                plane.update(state["data_plane_by_truth"][name])
                observed.update(state["observed"][name])
        total, eligible, logged = sum(plane.values()), plane["UNRESOLVED"], observed["logged"]
        evaluated = plane["BENIGN"] + logged
        accepted = plane["BENIGN"] + observed["BENIGN"]
        attacks = observed["ATTACK"]
        result[group] = {
            "test_flows": total, "data_plane": dict(plane), "controller": dict(observed),
            "controller_forwarding": rate(eligible, total),
            "conditional_verifier": {
                "resolution": rate(observed["BENIGN" if group == "benign" else "ATTACK"], logged),
                "benign_to_attack_error" if group == "benign" else "attack_to_benign_error":
                    rate(observed["ATTACK" if group == "benign" else "BENIGN"], logged),
                "uncertain": rate(observed["UNCERTAIN"], logged),
                "invalid_response": rate(observed["invalid"], logged),
                "failed": rate(observed["errors"], logged)},
            "evaluation_coverage": rate(evaluated, total),
            "benign_acceptance_on_evaluated": rate(accepted, evaluated),
            "attack_decisions_on_evaluated": rate(attacks, evaluated),
            "full_population_evaluated": logged == eligible,
            "full_fold_error_rate": rate(attacks if group == "benign" else accepted, total) if logged == eligible else None}
    return result


def performance(states: list[dict], requests: Requests) -> dict:
    latencies = [value for state in states for value in state["latencies"]]
    n = len(latencies)
    observed = Counter()
    for state in states:
        for group in state["observed"].values():
            observed.update(group)
    events = [e for values in requests.events.values() for e in values if e["stage"] == "verifier" and not e["cache_hit"]]
    tokens = sum(sum(e["usage"].values()) for e in events if e["usage"] is not None)
    complete = all(e["usage"] is not None for e in events)
    costs = sum(e["cost"] for e in events if e["cost"] is not None)
    return {"flows": n,
            "uncertain_rate": {"numerator": observed["UNCERTAIN"], "denominator": n,
                               "percent": 100 * observed["UNCERTAIN"] / n if n else None},
            "invalid_response_rate": {"numerator": observed["invalid"], "denominator": n,
                                      "percent": 100 * observed["invalid"] / n if n else None}, "mean_latency_ms": float(np.mean(latencies)) if n else None,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if n else None,
            "billed_tokens_per_flow": tokens / n if n and complete else None,
            "cost_usd": costs if all(e["cost"] is not None for e in events) else None,
            "cost_basis": "Provider usage.cost when supplied; configured token-price estimate otherwise. Cache hits cost zero."}


def investigation_config(cfg: dict, profile: str, mode: str, repeat: int) -> dict:
    selected = deepcopy(cfg)
    options = cfg["advanced_llm"]["models"][profile]
    selected["advanced_llm"].update({k: v for k, v in options.items() if k != "pricing"})
    selected["advanced_llm"]["mode"] = mode
    selected["budgets"]["pricing"]["advanced_llm"] = options["pricing"]
    selected["general"]["seed"] += repeat
    return selected


def investigation_metrics(rows: list[dict]) -> dict:
    """Offline scoring; abstentions/failures remain in the fixed population."""
    from sklearn.metrics import f1_score
    def rate(n, d):
        return {"numerator": n, "denominator": d, "percent": 100 * n / d if d else None}
    result = {}
    for profile, mode in sorted({(r["profile"], r["mode"]) for r in rows}):
        group = [r for r in rows if r["profile"] == profile and r["mode"] == mode]
        supported = [r for r in group if r["true_catalogue_family"] is not None and r["truth_group"] != "benign"]
        actual = [r["true_catalogue_family"] for r in supported]
        predicted = [(r["report"] or {}).get("attack_family") or "NO_REPORT" for r in supported]
        valid = [r for r in group if r["report"] is not None]
        latencies = [r["latency_ms"] for r in group]
        cost_complete = all(r["estimated_cost"] is not None for r in group)
        usage_complete = all(r["usage_complete"] for r in group)
        result[profile + "/" + mode] = {
            "scheduled": len(group), "attempted": sum(r["api_requests"] > 0 for r in group),
            "api_requests": sum(r["api_requests"] for r in group),
            "support_by_true_family": dict(Counter(r["true_family"] for r in group)),
            "family_accuracy": rate(sum(a == b for a, b in zip(actual, predicted)), len(supported)),
            "macro_f1": float(f1_score(actual, predicted, labels=sorted(set(actual)), average="macro", zero_division=0)) if actual else None,
            "family_metric_scope": "Mapped malicious rows; errors and abstentions are incorrect family predictions. Unmapped and benign rows reported separately.",
            "unmapped_or_benign_rows": len(group) - len(supported),
            "abstentions": sum((r["report"] or {}).get("attack_family") in ("UNKNOWN_ATTACK", "INSUFFICIENT_EVIDENCE") or (r["report"] or {}).get("classification") == "INSUFFICIENT_EVIDENCE" for r in group),
            "appropriate_abstention": None, "unsupported_claim_rate": None,
            "semantic_review_status": "Requires independent human review using configured rubric; no automatic semantic truth labels.",
            "evidence_fidelity_scope": "Feature/value pairs in schema-valid reports, including reports rejected by evidence checks; not semantic correctness.",
            "evidence_fidelity": rate(sum(r["matching_observations"] for r in group), sum(r["observations"] for r in group)),
            "json_validity": rate(sum(r["json_valid"] for r in group), len(group)),
            "schema_validity": rate(sum(r["schema_valid"] for r in group), len(group)),
            "tool_success_rate": rate(sum(r["tool_successes"] for r in group), sum(r["tool_calls"] for r in group)),
            "mean_tool_calls": sum(r["tool_calls"] for r in group) / len(group),
            "mean_latency_ms": float(np.mean(latencies)), "p95_latency_ms": float(np.percentile(latencies, 95)),
            "input_tokens": sum(r["input_tokens"] for r in group), "output_tokens": sum(r["output_tokens"] for r in group),
            "usage_complete": usage_complete,
            "cost_per_scheduled_investigation": sum(r["estimated_cost"] for r in group) / len(group) if cost_complete else None,
            "cost_complete": cost_complete, "failure_rate": rate(len(group) - len(valid), len(group)),
            "statuses": dict(Counter(r["investigation_status"] for r in group))}
    return result


def pipeline_comparison(states: list[dict], models: list[str]) -> dict:
    """Same evaluated rows for every architecture; investigation never changes security actions."""
    baseline, verified = {}, {}
    for group, summary in evaluation_summary(states).items():
        plane, observed = summary["data_plane"], Counter(summary["controller"])
        n = plane.get("BENIGN", 0) + observed["logged"]
        def stage(accepted, attacks, unresolved):
            return {"evaluated": n, "accepted": accepted, "attack_alerts": attacks, "unresolved": unresolved,
                    "benign_acceptance_or_attack_bypass_pct": 100 * accepted / n if n else None,
                    "attack_alert_pct": 100 * attacks / n if n else None,
                    "full_test_population_evaluated": summary["full_population_evaluated"]}
        baseline[group] = stage(plane.get("BENIGN", 0), 0, observed["logged"])
        verified[group] = stage(plane.get("BENIGN", 0) + observed["BENIGN"], observed["ATTACK"], observed["UNCERTAIN"])
    return {"first_layer_only": baseline, "first_layer_plus_verifier": verified,
            **{"complete_" + name: verified for name in models},
            "interpretation": "Investigation changes report quality only; all complete variants share binary decisions. Alert recall is not unknown-family identification."}


def run_investigations(queue: list, data: dict, cfg: dict, requests: Requests,
                       models: list[str], run_dir: Path | None, states: list[dict]) -> dict:
    """Freeze upstream once, select offline, then run the same rows for each model."""
    strata = defaultdict(list)
    for item in queue:
        upstream = item[2]
        mode = "attack" if upstream["verifier_decision"] == "ATTACK" else "uncertainty"
        strata[(upstream["heldout_attack"], mode, upstream["true_family"])].append(item)
    selected = []
    support = []
    for key, items in sorted(strata.items()):
        ranked = sorted(items, key=lambda item: digest([cfg["general"]["seed"], item[2]["flow_id"]]))
        chosen = ranked[:cfg["advanced_llm"]["sample_per_stratum"]]
        selected.extend(chosen)
        support.append({"fold": key[0], "mode": key[1], "true_family": key[2], "available": len(items), "selected": len(chosen)})
    rows = []
    summary = {"status": "running", "profiles": models, "sampling_support": support,
               "selected_flow_ids": [item[2]["flow_id"] for item in selected],
               "upstream_digest": digest([item[2] for item in selected]),
               "source_evidence_digests": {s["attack"]: digest(s["source"]) for s in states},
               "sampling": "Seeded hash ordering within fold/mode/original-label strata. Labels used offline only; no population-weighted quality claim.",
               "paired_scope": "All selected identities scheduled for every model/repeat; failures and budget-denied cases retained.",
               "prior_knowledge": "Full general catalogue supplied in every fold. Family recognition is prior-assisted, not proof of previously unseen-family discovery.",
               "review_rubric": cfg["advanced_llm"]["review_rubric"],
               "binary_pipeline_comparison": pipeline_comparison(states, models),
               "original_vs_revised": "Attack-only investigation vs optional uncertainty tools; same upstream decisions. Uncertainty recommendations never authorize acceptance or alerts.",
               "first_layer_inference_ms": sum(s.get("first_layer_inference_ms", 0) for s in states),
               "effective_configurations": {name: investigation_config(cfg, name, "attack", 0)["advanced_llm"] for name in models}}
    def save():
        summary["metrics"] = investigation_metrics(rows)
        summary["completed_records"] = len(rows)
        summary["api_requests"] = {"verification": sum(e["attempted"] for events in requests.events.values() for e in events if e["stage"] == "verifier"),
                                   "investigation": sum(r["api_requests"] for r in rows)}
        summary["total_accounted_cost_usd"] = requests.cost
        summary["cost_complete"] = requests.cost_complete
        summary["routing"] = {"accepted_layer1": sum(s["data_plane"]["BENIGN"] for s in states),
                              "eligible_layer2": sum(s["available"] for s in states),
                              "evaluated_layer2": sum(sum(o["logged"] for o in s["observed"].values()) for s in states),
                              "accepted_layer2": sum(sum(o["BENIGN"] for o in s["observed"].values()) for s in states),
                              "uncertain_layer2": sum(sum(o["UNCERTAIN"] for o in s["observed"].values()) for s in states),
                              "attack_alerts_layer2": sum(sum(o["ATTACK"] for o in s["observed"].values()) for s in states),
                              "unique_selected_layer3": len(selected),
                              "successful_reports": sum(r["report"] is not None for r in rows),
                              "failed_investigations": sum(r["report"] is None for r in rows)}
        if run_dir:
            target = _path(run_dir, cfg["output"]["investigation_summary"])
            temp = target.with_suffix(".tmp")
            temp.write_text(json.dumps(summary, allow_nan=False, indent=2))
            temp.replace(target)
    with ExitStack() as files:
        stream = csv_stream = writer = None
        if run_dir:
            stream = files.enter_context(_path(run_dir, cfg["output"]["investigation_jsonl"]).open("x"))
            csv_stream = files.enter_context(_path(run_dir, cfg["output"]["investigation_csv"]).open("x", newline=""))
        save()
        try:
            for i, (state, position, upstream) in enumerate(selected):
                mode = "attack" if upstream["verifier_decision"] == "ATTACK" else "uncertainty"
                original = flow_record(data, state["model"], position)
                # Rotate order to avoid consistently giving one model budget/latency priority.
                order = models[i % len(models):] + models[:i % len(models)]
                for repeat in range(cfg["advanced_llm"]["repeats"]):
                    for profile in order:
                        icfg = investigation_config(cfg, profile, mode, repeat)
                        flow = {**original, "flow_id": digest([upstream["flow_id"], profile, repeat, mode])}
                        requests.cfg = icfg
                        started, before = time.perf_counter(), requests.count
                        try:
                            result = investigate(flow, state["source"], icfg, requests)
                        finally:
                            requests.cfg = cfg
                        elapsed = (time.perf_counter() - started) * 1000
                        report = result["report"]
                        status = ("ATTACK_INVESTIGATION_ERROR" if mode == "attack" else "UNCERTAINTY_INVESTIGATION_ERROR")
                        if report:
                            family = report["attack_family"]
                            status = ({"UNKNOWN_ATTACK": "ATTACK_UNKNOWN", "INSUFFICIENT_EVIDENCE": "ATTACK_INSUFFICIENT_EVIDENCE"}.get(family, "ATTACK_IDENTIFIED")
                                      if mode == "attack" else "RECOMMENDATION_ONLY")
                        events = requests.events[flow["flow_id"]]
                        usage = [e["usage"] for e in events]
                        cost = [e["cost"] for e in events]
                        record = {"flow_id": upstream["flow_id"], "row_position": upstream["row_position"],
                                  "heldout_attack": state["attack"], "true_family": upstream["true_family"],
                                  "truth_group": upstream["truth_group"],
                                  "true_catalogue_family": cfg["investigation_evaluation"]["label_mapping"][data["name"]].get(upstream["true_family"]),
                                  "profile": profile, "mode": mode, "repeat": repeat, "seed": icfg["general"]["seed"],
                                  "verifier_decision": upstream["verifier_decision"],
                                  "evidence_state": upstream["evidence_state"],
                                  "security_decision": upstream["security_decision"], "final_disposition": upstream["final_disposition"],
                                  "investigation_status": status, "api_requests": requests.count - before,
                                  "latency_ms": elapsed, "end_to_end_controller_latency_ms": upstream["latency_ms"] + elapsed,
                                  "amortized_pipeline_latency_ms": upstream["latency_ms"] + elapsed +
                                      state.get("first_layer_inference_ms", 0) / max(1, sum(state["data_plane"].values())),
                                  "input_tokens": sum(u["input_tokens"] for u in usage if u is not None),
                                  "output_tokens": sum(u["output_tokens"] for u in usage if u is not None),
                                  "usage_complete": all(e["usage"] is not None for e in events if e["attempted"]),
                                  "estimated_cost": sum(cost) if all(v is not None for v in cost) else None,
                                  "provider_usage": [e["provider_usage"] for e in events],
                                  "permitted_flow": public_flow(original, state["source"]), **result}
                        rows.append(record)
                        if stream:
                            if writer is None:
                                writer = csv.DictWriter(csv_stream, fieldnames=list(record))
                                writer.writeheader()
                            writer.writerow(csv_record(record))
                            csv_stream.flush()
                            stream.write(json.dumps(record, allow_nan=False) + "\n")
                            stream.flush()
                        save()
            summary["status"] = "complete" if all(r["error"] not in ("cost_budget_exhausted", "cost_usage_unavailable") for r in rows) else "budget_limited"
        except BaseException:
            summary["status"] = "interrupted"
            raise
        finally:
            save()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/controller.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attack", help="Withheld family; omitted runs every configured family")
    population = parser.add_mutually_exclusive_group()
    population.add_argument("--max-flows", type=int, help="Maximum controller flows per fold")
    population.add_argument("--all-flows", action="store_true",
                            help="Process all unresolved test flows, subject to the shared dollar budget")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--source-audit", action="store_true", help="Report independent source support without fitting or API calls")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--verify-only", action="store_true", help="Stop at frozen verifier decisions")
    parser.add_argument("--investigation-model", help="Configured model profile, or both for paired evaluation")
    parser.add_argument("--investigate-uncertain", action="store_true", help="Also evaluate unresolved recommendations; never release automatically")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.verify_only:
        cfg["advanced_llm"]["enabled"] = False
    if args.investigate_uncertain:
        cfg["advanced_llm"]["investigate_uncertain"] = True
    selected = args.investigation_model or cfg["advanced_llm"]["selected"]
    investigation_models = cfg["advanced_llm"]["comparison_models"] if selected == "both" else [selected]
    if not set(investigation_models) <= set(cfg["advanced_llm"]["models"]):
        parser.error("Unknown investigation model profile")
    root = Path(__file__).resolve().parent.parent
    if not args.dry_run and not args.source_audit and not cfg["general"]["enabled"]:
        parser.error("Set general.enabled in the controller config before a live run")
    limit = None if args.all_flows else (args.max_flows if args.max_flows is not None
                                       else cfg["general"]["max_flows"])
    if limit is not None and limit < 1:
        parser.error("--max-flows must be positive")
    if args.calibrate and args.attack is None:
        parser.error("Calibration requires one explicit --attack")
    loao_cfg = yaml.safe_load((root / cfg["general"]["first_layer_config"]).read_text())
    if args.dataset not in loao_cfg["datasets"]:
        parser.error("Select a configured dataset")
    mapping = cfg["investigation_evaluation"]["label_mapping"][args.dataset]
    if set(mapping) != set(loao_cfg["expected_attacks"][args.dataset]) or not set(v for v in mapping.values() if v is not None) <= set(cfg["attack_knowledge"]["families"]):
        parser.error("Offline family mapping must explicitly cover every dataset attack label")
    attacks = [args.attack] if args.attack else loao_cfg["expected_attacks"][args.dataset]
    if not set(attacks) <= set(loao_cfg["expected_attacks"][args.dataset]):
        parser.error("Select a configured dataset and attack family")
    data = _read_dataset(args.dataset, loao_cfg["datasets"][args.dataset], loao_cfg, root, include_raw=True)
    if args.source_audit:
        groups = pd.util.hash_pandas_object(data["x"], index=False).to_numpy()
        budget = loao_cfg["safe_regions"]["known_attack_leakage_budget"]
        confidence = cfg["verifier"]["calibration"]["confidence"]
        diagnostics = {}
        for attack in attacks:
            pools = source_partitions(data, cfg, attack)
            counts = {k: {"rows": int(mask.sum()), "feature_groups": len(np.unique(groups[mask])),
                          "attack_groups": len(np.unique(groups[mask & (data["y"] == 1)]))}
                      for k, mask in pools.items()}
            support = counts["gate"]["attack_groups"]
            diagnostics[attack] = {"source_support": counts, "best_case_gate_upper": error_upper(0, support, confidence),
                                   "gate_budget": budget, "gate_certification_possible": error_upper(0, support, confidence) <= budget,
                                   "minimum_zero_error_attack_groups": math.ceil(math.log(1-confidence) / math.log(1-budget)),
                                   "benign_acceptance_cap": cfg["routing"]["benign_acceptance_max_fraction"]}
        print(json.dumps({"dataset": args.dataset, "source_audit": diagnostics, "api_requests": 0, "first_layer_fitted": False}))
        return
    out = args.output_dir or root / cfg["output"]["results_directory"]
    states, previews = [], []
    rng = np.random.default_rng(cfg["general"]["seed"])
    requests = Requests(cfg)
    verifier_send = lambda body: openrouter_request(body, cfg, "verifier")
    try:
        for attack in attacks:
            train = data["train"] & (data["family"] != attack)
            source = build_source(data, train, cfg)
            if args.dry_run:
                placeholder = {"features": dict.fromkeys(source["feature_names"]), "first_layer_evidence": {}}
                bodies = {"verifier": verifier_request(placeholder, source, cfg)}
                if cfg["advanced_llm"]["enabled"]:
                    for profile in investigation_models:
                        icfg = investigation_config(cfg, profile, "attack", 0)
                        bodies[profile] = investigation_request(initial_messages(placeholder, source, icfg), icfg, False)
                preview = {"attack": attack, "feature_count": len(source["feature_names"]),
                           "source_examples": len(source["examples"]),
                           "request_bytes": {key: len(json.dumps(body).encode()) for key, body in bodies.items()}}
                if cfg["general"]["dry_run_show_requests"]:
                    preview["prepared_requests"] = bodies
                previews.append(preview)
                continue
            context = context_fingerprint(root / loao_cfg["datasets"][args.dataset]["path"],
                                          loao_cfg, source, cfg, attack)
            fold_name = re.sub(r"[^A-Za-z0-9_-]+", "_", attack)
            artifact_path = _path(out, cfg["verifier"]["calibration"]["artifact"], dataset=args.dataset, fold=fold_name)
            policy, calibration_status = None, "unavailable"
            if args.calibrate and artifact_path.exists():
                parser.error("Calibration artifact already exists; choose another output directory")
            if not args.calibrate and artifact_path.exists():
                try:
                    policy = validated_policy(json.loads(artifact_path.read_text()), context, cfg)
                    calibration_status = "certified"
                except (ValueError, KeyError, TypeError, AttributeError, IndexError, ValidationError):
                    policy = None  # Stale/unsafe artifacts fail closed without API requests.
            model, fold = fit_controller_fold(data, loao_cfg, cfg, attack)
            if any(np.any(fold[k] & (data["family"] == attack)) for k in ("train", "tuning", "certification")):
                raise AssertionError("Withheld attack leaked into source partitions")
            state = {"attack": attack, "model": model, "source": source, "context": context,
                     "policy": policy, "calibration_status": calibration_status,
                     "frozen": hashlib.sha256(pickle.dumps(model)).hexdigest(), "counts": Counter(),
                     "latencies": [], "gate_audit": fold["gate_audit"],
                     "observed": {group: Counter() for group in ("benign", "known_attack", "heldout_attack")}}
            states.append(state)
            if args.calibrate:
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_written = False
                def checkpoint(artifact):
                    nonlocal checkpoint_written
                    artifact.update(source_dataset=args.dataset, withheld_attack=attack,
                                    gate_audit=fold["gate_audit"], source_partition_configuration=cfg["routing"],
                                    verifier_configuration=cfg["verifier"], prompts=cfg["prompts"]["verifier"],
                                    first_layer_configuration=loao_cfg,
                                    code_version={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                                  for p in Path(__file__).parent.glob("*.py")},
                                    api_requests=requests.count, accounted_cost_usd=requests.cost,
                                    cost_complete=requests.cost_complete,
                                    max_cost_usd=cfg["budgets"]["max_cost_usd"])
                    if not checkpoint_written:
                        with artifact_path.open("x") as stream:
                            json.dump(artifact, stream, allow_nan=False, indent=2)
                        checkpoint_written = True
                        return
                    temporary = artifact_path.with_suffix(".json.tmp")
                    temporary.write_text(json.dumps(artifact, allow_nan=False, indent=2))
                    temporary.replace(artifact_path)
                artifact = calibration_run(data, model, fold, source, cfg, requests, verifier_send, context, checkpoint)
                checkpoint(artifact)
                print(json.dumps({"calibration": artifact["status"], "artifact": str(artifact_path),
                                  "api_requests": requests.count, "error": artifact["error"],
                                  "max_cost_usd": cfg["budgets"]["max_cost_usd"],
                                  "estimated_cost_usd": requests.cost if requests.cost_complete else None}))
                return
            plane_started = time.perf_counter()
            internal = dataplane_partition(data, model, data["test"])
            state["first_layer_inference_ms"] = (time.perf_counter() - plane_started) * 1000
            state["internal_data_plane"] = {decision: len(rows) for decision, rows in internal.items()}
            partition = controller_partition(internal, model["controller_benign_authorized"])
            state["data_plane"] = {decision: len(rows) for decision, rows in partition.items()}
            groups = np.where(data["y"] == 0, "benign", np.where(
                data["family"] == attack, "heldout_attack", "known_attack"))
            state["data_plane_by_truth"] = {
                group: {decision: int(np.sum(groups[rows] == group)) for decision, rows in partition.items()}
                for group in state["observed"]}
            indices = partition["UNRESOLVED"]
            state["available"] = len(indices)
            state["positions"] = (rng.permutation(indices) if cfg["general"]["shuffle_flows"] else indices)[:limit]
            if cfg["general"]["logging"]["print_summary"]:
                handled = state["data_plane"]
                print(f"Prepared {args.dataset}/{attack}: data plane processed {sum(handled.values()):,} test flows; "
                      f"BENIGN={handled['BENIGN']:,}; "
                      f"internal KNOWN_ATTACK={state['internal_data_plane']['KNOWN_ATTACK']:,} (forwarded); "
                      f"controller eligible={len(indices):,}, selected={len(state['positions']):,}", flush=True)
        if args.dry_run:
            print(json.dumps({"mode": "dry_run", "dataset": args.dataset, "folds": previews,
                              "max_flows_per_fold": limit, "api_requests": 0, "first_layer_fitted": False,
                              "preview": "schema_only_no_flow_prediction"}, allow_nan=False))
            return
        counts, run_dir, stop_reason = Counter(), None, None
        investigation_queue, investigation_result = [], None
        if cfg["output"]["save_results"]:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = _path(out, cfg["output"]["run_directory"], dataset=args.dataset,
                            fold=fold_name if len(attacks) == 1 else "all_attacks", run_id=run_id)
            run_dir.mkdir(parents=True, exist_ok=False)
            metadata = {"configurations": cfg, "first_layer_configuration": loao_cfg,
                        "dataset": args.dataset, "withheld_attacks": attacks,
                        "max_flows_per_fold": limit, "feature_names": list(data["raw_x"].columns),
                        "selection": "Seeded permutations when general.shuffle_flows is true; round-robin across folds.",
                        "identity": "row_position identifies the original dataset row; flow_id includes the withheld fold.",
                        "labels": "true_family and truth_group are local evaluation fields, never sent to models.",
                        "classification_policy": "Source-certified categorical verifier; ATTACK awaits investigation. UNCERTAIN is not malicious detection.",
                        "metric_denominators": "Forwarding uses all test flows in each truth group. Partial evaluated metrics include data-plane benign decisions and ALL logged controller flows, including failures and abstentions. Full-fold error rates are unavailable until every eligible flow in that group completes. Source certification does not guarantee unseen-attack safety.",
                        "investigation_models": investigation_models,
                        "contexts": {state["attack"]: state["context"] for state in states}}
            _path(run_dir, cfg["output"]["metadata_filename"]).write_text(json.dumps(metadata, allow_nan=False, indent=2))
            print(f"Controller CSV: {_path(run_dir, cfg['output']['csv_filename'])}", flush=True)

        def snapshot(status: str) -> dict:
            summary = {"status": status, "dataset": args.dataset, "heldout_attack": args.attack,
                       "context_hash": states[0]["context"] if len(states) == 1 else None,
                       "calibration_status": states[0]["calibration_status"] if len(states) == 1 else None,
                       "unresolved_available": sum(state["available"] for state in states),
                       "data_plane": {decision: sum(state["data_plane"][decision] for state in states)
                                      for decision in states[0]["data_plane"]},
                       "internal_data_plane": {decision: sum(state["internal_data_plane"][decision] for state in states)
                                                for decision in states[0]["internal_data_plane"]},
                       "evaluation_by_truth": evaluation_summary(states),
                       "investigation": investigation_result,
                       "processed": sum(counts.values()), "statuses": dict(counts), "api_requests": requests.count,
                       "verification_performance": performance(states, requests),
                       "reported_tokens": requests.tokens, "usage_complete": requests.usage_complete,
                       "estimated_cost_usd": requests.cost if requests.cost_complete else None,
                       "accounted_cost_usd": requests.cost, "cost_complete": requests.cost_complete,
                       "max_cost_usd": cfg["budgets"]["max_cost_usd"], "stop_reason": stop_reason,
                       "results_directory": str(run_dir) if run_dir else None,
                       "folds": [{"heldout_attack": state["attack"], "context_hash": state["context"],
                                  "calibration_status": state["calibration_status"], "gate_audit": state["gate_audit"],
                                  "first_layer_inference_ms": state["first_layer_inference_ms"],
                                  "unresolved_available": state["available"], "selected": len(state["positions"]),
                                  "data_plane": state["data_plane"],
                                  "internal_data_plane": state["internal_data_plane"],
                                  "evaluation_by_truth": evaluation_summary([state]),
                                  "processed": sum(state["counts"].values()), "statuses": dict(state["counts"])}
                                 for state in states]}
            if run_dir:
                target = _path(run_dir, cfg["output"]["summary_filename"])
                temporary = target.with_suffix(target.suffix + ".tmp")
                temporary.write_text(json.dumps(summary, allow_nan=False, indent=2))
                temporary.replace(target)
            return summary

        with ExitStack() as files:
            stream = csv_stream = writer = None
            if run_dir:
                stream = files.enter_context(_path(run_dir, cfg["output"]["results_filename"]).open("x"))
                csv_stream = files.enter_context(_path(run_dir, cfg["output"]["csv_filename"]).open("x", newline=""))
                writer = csv.DictWriter(csv_stream, fieldnames=["heldout_attack", "row_position", "true_family",
                                                              "internal_first_layer_decision",
                                                              "truth_group", *cfg["output"]["record_fields"]])
                writer.writeheader()
                csv_stream.flush()
            snapshot("running")
            jobs = ((state, int(position)) for group in zip_longest(*(s["positions"] for s in states))
                    for state, position in zip(states, group) if position is not None)
            try:
                for state, position in jobs:
                    flow = flow_record(data, state["model"], position)
                    flow["flow_id"] = digest([state["attack"], flow["flow_id"]])
                    record = resolve(flow, state["source"], cfg, state["policy"],
                                     state["calibration_status"], requests, verifier_send)
                    if record is None:
                        raise AssertionError("A flow already handled by the data plane reached the controller queue")
                    counts[record["controller_status"]] += 1
                    state["counts"][record["controller_status"]] += 1
                    state["latencies"].append(record["latency_ms"])
                    truth = str(data["family"][position])
                    group = ("benign" if data["y"][position] == 0 else
                             "heldout_attack" if truth == state["attack"] else "known_attack")
                    observed = state["observed"][group]
                    observed["logged"] += 1
                    observed[record["verifier_decision"]] += 1
                    observed["errors"] += int(record["error"] is not None)
                    observed["invalid"] += int(record["error"] in ("invalid_verifier_response", "malformed_response"))
                    exported = {"heldout_attack": state["attack"], "row_position": int(data["row_positions"][position]),
                                "internal_first_layer_decision": flow["first_layer_evidence"]["internal_decision"],
                                "true_family": truth, "truth_group": group, **record}
                    if cfg["advanced_llm"]["enabled"] and record["error"] is None and (record["verifier_decision"] == "ATTACK" or
                            cfg["advanced_llm"]["investigate_uncertain"] and record["verifier_decision"] == "UNCERTAIN"):
                        investigation_queue.append((state, position, exported))
                    if writer:
                        writer.writerow(csv_record(exported))
                        csv_stream.flush()
                        stream.write(json.dumps(exported, allow_nan=False) + "\n")
                        stream.flush()
                    if record["error"] in ("cost_budget_exhausted", "cost_usage_unavailable"):
                        stop_reason = record["error"]
                    snapshot("stopped" if stop_reason else "running")
                    if stop_reason:
                        break
                if cfg["advanced_llm"]["enabled"]:
                    investigation_result = run_investigations(investigation_queue, data, cfg, requests,
                                                              investigation_models, run_dir, states)
                    if investigation_result["status"] == "budget_limited":
                        stop_reason = "investigation_budget_limited"
            except BaseException as exc:
                stop_reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else "execution_error"
                raise
            finally:
                summary = snapshot("stopped" if stop_reason else "complete")
                if cfg["general"]["logging"]["print_summary"]:
                    print(json.dumps(summary, allow_nan=False))
    finally:
        for state in states:
            if hashlib.sha256(pickle.dumps(state["model"])).hexdigest() != state["frozen"]:
                raise AssertionError("Frozen first-layer state changed during controller execution")


if __name__ == "__main__":
    main()
