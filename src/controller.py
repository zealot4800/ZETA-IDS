"""Run the independent UNRESOLVED -> Jev -> attack investigation controller."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
import re
import time
from urllib.parse import urlparse

import numpy as np
import yaml
from jsonschema import validate

from .agent import initial_messages, investigate, report_schema, request_body as agent_request
from .evidence import build_source, digest, flow_record, source_sample, unresolved_indices
from .jev import (APIFailure, JevDetector, binary_decision, choose_thresholds, probability,
                  request_body as jev_request, threshold_evidence, validated_thresholds)
from .loao import fit_fold
from .loao_data import _read_dataset


def record_schema(cfg: dict) -> dict:
    properties = {name: {"anyOf": [schema, {"type": "null"}]}
                  for name, schema in report_schema(cfg)["properties"].items()}
    properties.update({
        "flow_id": {"type": "string"}, "dataset": {"type": "string"},
        "first_layer_decision": {"const": "UNRESOLVED"},
        "jev_decision": {"enum": ["BENIGN", "ATTACK", "UNCERTAIN", "ERROR"]},
        "jev_p_attack": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
        "jev_model": {"type": "string"}, "advanced_model": {"type": ["string", "null"]},
        "controller_status": {"enum": ["BENIGN", "UNRESOLVED", "UNCALIBRATED",
                                         "ATTACK_IDENTIFIED", "ATTACK_UNKNOWN",
                                         "ATTACK_INSUFFICIENT_EVIDENCE", "ATTACK_INVESTIGATION_ERROR"]},
        "latency_ms": {"type": "number", "minimum": 0},
        "token_usage": {"type": "object", "additionalProperties": False,
                        "properties": {stage: {"anyOf": [{"type": "null"}, {
                            "type": "object", "additionalProperties": False,
                            "properties": {key: {"type": "integer", "minimum": 0}
                                           for key in ("input_tokens", "output_tokens")},
                            "required": ["input_tokens", "output_tokens"]}]}
                                       for stage in ("jev", "advanced_llm")},
                        "required": ["jev", "advanced_llm"]},
        "estimated_cost": {"type": ["number", "null"], "minimum": 0},
        "error": {"type": ["string", "null"]},
        "calibration_status": {"enum": ["certified", "experimental", "unavailable"]},
        "tool_calls": {"type": "integer", "minimum": 0},
        "cache_hits": {"type": "integer", "minimum": 0}})
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for section in ("general", "jev", "advanced_llm", "prompts", "attack_knowledge",
                    "evidence", "budgets", "output"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Missing controller configuration section: {section}")
    if cfg["jev"]["provider"] != "typesafe" or cfg["advanced_llm"]["provider"] != "openrouter":
        raise ValueError("Unsupported controller provider")
    if not re.fullmatch(r"jev-\d+\.\d+\.\d+", cfg["jev"]["model"]):
        raise ValueError("Jev requires an exact versioned model ID")
    for stage in ("jev", "advanced_llm"):
        options = cfg[stage]
        url = urlparse(options["base_url"])
        if url.scheme != "https" or not url.netloc or url.username or url.password or url.query:
            raise ValueError("API base URLs must be HTTPS and contain no credentials")
        if not math.isfinite(options["timeout_seconds"]) or options["timeout_seconds"] <= 0:
            raise ValueError("Invalid API timeout")
        if not options["model"] or not options["api_key_env"]:
            raise ValueError("Missing model or API-key environment-variable name")
    if cfg["jev"]["request"]["question_type"] != "noul":
        raise ValueError("Jev must ask only a binary noul question")
    lower, upper = (cfg["jev"]["thresholds"][key] for key in ("benign_max", "attack_min"))
    if not 0 <= lower < upper <= 1:
        raise ValueError("Invalid experimental binary thresholds")
    cal = cfg["jev"]["calibration"]
    if cal["mode"] not in ("required", "experimental"):
        raise ValueError("Calibration mode must be required or experimental")
    for key in ("false_benign_budget", "false_attack_budget", "confidence"):
        if not 0 < cal[key] < 1:
            raise ValueError(f"Invalid calibration {key}")
    for key in ("benign_candidates", "attack_candidates"):
        values = cal[key]
        if not values or values != sorted(set(values)) or not all(0 <= v <= 1 for v in values):
            raise ValueError("Invalid calibration threshold candidates")
    positive = [cfg["general"]["max_flows"], cfg["advanced_llm"]["max_steps"],
                cfg["advanced_llm"]["max_tokens"], cal["min_samples_per_class"],
                cal["tuning_per_class"], cal["certification_per_class"],
                cfg["evidence"]["max_features_per_tool"], cfg["evidence"]["max_families_per_tool"],
                cfg["evidence"]["categorical_top_k"]]
    if any(type(value) is not int or value < 1 for value in positive):
        raise ValueError("Controller limits must be positive integers")
    for value in (cfg["advanced_llm"]["max_tool_calls"], cfg["jev"]["few_shot_per_class"],
                  cfg["budgets"]["token_reserve_overhead"], cfg["budgets"]["jev_output_token_reserve"]):
        if type(value) is not int or value < 0:
            raise ValueError("Tool, example and token-reservation counts must be nonnegative integers")
    if not 0 <= cfg["advanced_llm"]["temperature"] <= 2:
        raise ValueError("Invalid model temperature")
    if cfg["advanced_llm"]["tool_choice"] != "auto":
        raise ValueError("Investigation uses automatic tool selection")
    for stage, names in (("jev", ("task", "features", "source_examples", "question")),
                         ("advanced_llm", ("system", "investigation", "hypotheses", "evidence",
                                           "classification", "uncertainty", "final_report"))):
        if any(not isinstance(cfg["prompts"][stage].get(name), str) or
               not cfg["prompts"][stage][name].strip() for name in names):
            raise ValueError("Missing controller prompt text")
    allowed = set(cfg["advanced_llm"]["allowed_tools"])
    if not allowed <= {"inspect_flow", "compare_source_baseline", "retrieve_attack_knowledge"}:
        raise ValueError("Unknown configured evidence tool")
    if not allowed <= set(cfg["prompts"]["advanced_llm"]["tool_descriptions"]):
        raise ValueError("Missing configured tool description")
    if cfg["evidence"]["source_reference"] != "benign_training":
        raise ValueError("Only a benign source-training baseline is supported")
    if not set(cfg["evidence"]["numeric_statistics"]) <= {"count", "missing", "min", "max", "mean", "std", "quantiles"}:
        raise ValueError("Unsupported comparison statistic")
    if not all(0 <= q <= 1 for q in cfg["evidence"]["quantiles"]):
        raise ValueError("Invalid baseline quantile")
    if cfg["advanced_llm"]["response_format"]["type"] not in ("json_schema", "json_object", "none"):
        raise ValueError("Unsupported structured response format")
    if set(cfg["output"]["record_fields"]) != set(record_schema(cfg)["properties"]):
        raise ValueError("Configured record fields must match the controller interface")
    budget = cfg["budgets"]["max_cost_usd"]
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("The per-run dollar budget must be finite and positive")
    for stage in ("jev", "advanced_llm"):
        rates = cfg["budgets"]["pricing"].get(stage)
        if not isinstance(rates, dict) or any(
                type(rates.get(key)) not in (int, float) or not math.isfinite(rates[key]) or rates[key] < 0
                for key in ("input_per_million", "output_per_million")):
            raise ValueError(f"Cost budgeting requires finite nonnegative {stage} prices")
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
        reserve_out = (limits["jev_output_token_reserve"] if stage == "jev"
                       else self.cfg[stage]["max_tokens"])
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
        if stage == "advanced_llm" and "cost" in raw:
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
        cache_hit = key in self.cache
        usage, cost = None, 0.0
        if cache_hit:
            result = self.cache[key]
        elif error := self._budget_error(stage, body):
            result = {"body": None, "error": error}
        else:
            self.count += 1
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
            usage, cost = self._usage(stage, result["body"])
            self.cache[key] = result
        self.events[flow_id].append({"stage": stage, "usage": usage, "cost": cost,
                                     "cache_hit": cache_hit})
        return result

    def measurements(self, flow_id: str) -> dict:
        events = self.events[flow_id]
        usage = {}
        for stage in ("jev", "advanced_llm"):
            charged = [event for event in events if event["stage"] == stage and not event["cache_hit"]]
            usage[stage] = ({key: sum(event["usage"][key] for event in charged)
                             for key in ("input_tokens", "output_tokens")}
                            if charged and all(event["usage"] is not None for event in charged) else None)
        return {"token_usage": usage,
                "estimated_cost": (sum(event["cost"] for event in events)
                                   if events and all(event["cost"] is not None for event in events) else None),
                "cache_hits": sum(event["cache_hit"] for event in events)}


def resolve(flow: dict, source: dict, cfg: dict, thresholds: dict | None,
            calibration_status: str, requests: Requests, jev_send, advanced_send=None) -> dict | None:
    if flow["first_layer_decision"] != "UNRESOLVED":
        return None
    started = time.perf_counter()
    record = dict.fromkeys(cfg["output"]["record_fields"])
    record.update(flow_id=flow["flow_id"], dataset=flow["dataset"],
                  first_layer_decision="UNRESOLVED", jev_model=cfg["jev"]["model"],
                  jev_decision="UNCERTAIN", controller_status="UNCALIBRATED",
                  calibration_status=calibration_status, tool_calls=0)
    if thresholds is not None:
        reply = requests.call("jev", flow["flow_id"], jev_request(flow, source, cfg), jev_send)
        if reply["error"]:
            record.update(jev_decision="ERROR", controller_status="UNRESOLVED", error=reply["error"])
        else:
            try:
                p_attack = probability(reply["body"], cfg)
                decision = binary_decision(p_attack, thresholds)
                record.update(jev_p_attack=p_attack, jev_decision=decision,
                              controller_status="BENIGN" if decision == "BENIGN" else "UNRESOLVED")
            except (KeyError, TypeError, ValueError):
                record.update(jev_decision="ERROR", controller_status="UNRESOLVED", error="invalid_jev_response")
        if record["jev_decision"] == "ATTACK":
            investigation = investigate(flow, source, cfg, requests, advanced_send)
            record.update(advanced_model=investigation["model"], tool_calls=investigation["tool_calls"],
                          error=investigation["error"])
            if investigation["report"] is None:
                record["controller_status"] = "ATTACK_INVESTIGATION_ERROR"
            else:
                record.update(investigation["report"])
                family = record["attack_family"]
                record["controller_status"] = ("ATTACK_UNKNOWN" if family == "UNKNOWN_ATTACK" else
                    "ATTACK_INSUFFICIENT_EVIDENCE" if family == "INSUFFICIENT_EVIDENCE" else "ATTACK_IDENTIFIED")
    record.update(requests.measurements(flow["flow_id"]))
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    validate(record, record_schema(cfg))
    json.dumps(record, allow_nan=False)
    return record


def calibration_run(data: dict, model: dict, fold: dict, source: dict, cfg: dict,
                    requests: Requests, jev_send, context_hash: str) -> dict:
    options = cfg["jev"]["calibration"]
    pools = {}
    for stage in ("tuning", "certification"):
        unresolved = unresolved_indices(data, model, fold[stage])
        pools[stage] = source_sample(data["x"], data["y"], unresolved,
                                     options[stage + "_per_class"], cfg["general"]["seed"])
        if any(np.sum(data["y"][pools[stage]] == label) < options["min_samples_per_class"]
               for label in (0, 1)):
            raise ValueError(f"Insufficient unresolved source {stage} support for Jev calibration")
    artifact = {"context_hash": context_hash, "model": cfg["jev"]["model"],
                "status": "uncertified", "thresholds": None, "certification": None, "error": None}
    for stage in ("tuning", "certification"):
        probabilities, labels = [], []
        for position in pools[stage]:
            flow = flow_record(data, model, int(position))
            reply = requests.call("jev", flow["flow_id"], jev_request(flow, source, cfg), jev_send)
            if reply["error"]:
                artifact["error"] = reply["error"]
                return artifact  # Never certify after dropping failed source predictions.
            try:
                probabilities.append(probability(reply["body"], cfg))
            except (KeyError, TypeError, ValueError):
                artifact["error"] = "invalid_jev_calibration_response"
                return artifact
            labels.append(int(data["y"][position]))
        if stage == "tuning":
            artifact["thresholds"] = choose_thresholds(probabilities, labels, options)
            if artifact["thresholds"] is None:
                artifact["error"] = "no_source_threshold_meets_budgets"
                return artifact
        else:
            artifact["certification"] = threshold_evidence(probabilities, labels, artifact["thresholds"], options)
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
                   "model": cfg["jev"]["model"], "request": cfg["jev"]["request"],
                   "prompts": cfg["prompts"]["jev"], "code": code})


def _path(base: Path, template: str, **values) -> Path:
    target = (base / template.format(**values)).resolve()
    if not target.is_relative_to(base.resolve()):
        raise ValueError("Controller output path escapes its configured directory")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/controller.yaml"))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--attack", required=True)
    parser.add_argument("--max-flows", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(__file__).resolve().parent.parent
    if not args.dry_run and not cfg["general"]["enabled"]:
        parser.error("Set general.enabled in the controller config before a live run")
    limit = args.max_flows if args.max_flows is not None else cfg["general"]["max_flows"]
    if limit < 1:
        parser.error("--max-flows must be positive")
    loao_cfg = yaml.safe_load((root / cfg["general"]["first_layer_config"]).read_text())
    if args.dataset not in loao_cfg["datasets"] or args.attack not in loao_cfg["expected_attacks"][args.dataset]:
        parser.error("Select a configured dataset and attack family")
    data = _read_dataset(args.dataset, loao_cfg["datasets"][args.dataset], loao_cfg, root, include_raw=True)
    train = data["train"] & (data["family"] != args.attack)
    source = build_source(data, train, cfg)
    if args.dry_run:
        placeholder = {"features": dict.fromkeys(source["feature_names"]), "first_layer_evidence": {}}
        bodies = {"jev": jev_request(placeholder, source, cfg),
                  "advanced_llm": agent_request(initial_messages(placeholder, source, cfg), cfg, False)}
        summary = {"mode": "dry_run", "dataset": args.dataset, "attack": args.attack,
                   "feature_count": len(source["feature_names"]), "source_examples": len(source["examples"]),
                   "max_flows": limit, "api_requests": 0, "first_layer_fitted": False,
                   "preview": "schema_only_no_flow_prediction",
                   "request_bytes": {key: len(json.dumps(body).encode()) for key, body in bodies.items()}}
        if cfg["general"]["dry_run_show_requests"]:
            summary["prepared_requests"] = bodies
        print(json.dumps(summary, allow_nan=False))
        return
    context = context_fingerprint(root / loao_cfg["datasets"][args.dataset]["path"],
                                  loao_cfg, source, cfg, args.attack)
    out = args.output_dir or root / cfg["output"]["results_directory"]
    fold_name = re.sub(r"[^A-Za-z0-9_-]+", "_", args.attack)
    artifact_path = _path(out, cfg["jev"]["calibration"]["artifact"], dataset=args.dataset, fold=fold_name)
    thresholds, calibration_status = None, "unavailable"
    if args.calibrate and artifact_path.exists():
        parser.error("Calibration artifact already exists; choose another output directory")
    if not args.calibrate:
        if cfg["jev"]["calibration"]["mode"] == "experimental":
            thresholds, calibration_status = cfg["jev"]["thresholds"], "experimental"
        elif artifact_path.exists():
            thresholds = validated_thresholds(json.loads(artifact_path.read_text()), context, cfg)
            calibration_status = "certified"
    model, fold = fit_fold(data, loao_cfg, args.attack)
    frozen = hashlib.sha256(pickle.dumps(model)).hexdigest()
    requests, detector = Requests(cfg), JevDetector(cfg)
    try:
        if args.calibrate:
            artifact = calibration_run(data, model, fold, source, cfg, requests, detector.send, context)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with artifact_path.open("x") as stream:
                json.dump(artifact, stream, allow_nan=False, indent=2)
            print(json.dumps({"calibration": artifact["status"], "artifact": str(artifact_path),
                              "api_requests": requests.count, "error": artifact["error"],
                              "max_cost_usd": cfg["budgets"]["max_cost_usd"],
                              "estimated_cost_usd": requests.cost if requests.cost_complete else None}))
            return
        indices = unresolved_indices(data, model, data["test"])
        counts, stream, run_dir = Counter(), None, None
        stop_reason = None
        if cfg["output"]["save_results"]:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            run_dir = _path(out, cfg["output"]["run_directory"], dataset=args.dataset, fold=fold_name, run_id=run_id)
            run_dir.mkdir(parents=True, exist_ok=False)
            stream = _path(run_dir, cfg["output"]["results_filename"]).open("x")
        try:
            for position in indices[:limit]:
                record = resolve(flow_record(data, model, int(position)), source, cfg, thresholds,
                                 calibration_status, requests, detector.send)
                counts[record["controller_status"]] += 1
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    stream.flush()
                if record["error"] in ("cost_budget_exhausted", "cost_usage_unavailable"):
                    stop_reason = record["error"]
                    break
        finally:
            if stream:
                stream.close()
        summary = {"dataset": args.dataset, "heldout_attack": args.attack, "context_hash": context,
                   "calibration_status": calibration_status, "unresolved_available": len(indices),
                   "processed": sum(counts.values()), "statuses": dict(counts), "api_requests": requests.count,
                   "reported_tokens": requests.tokens, "usage_complete": requests.usage_complete,
                   "estimated_cost_usd": requests.cost if requests.cost_complete else None,
                   "max_cost_usd": cfg["budgets"]["max_cost_usd"], "stop_reason": stop_reason,
                   "results_directory": str(run_dir) if run_dir else None}
        if run_dir:
            _path(run_dir, cfg["output"]["summary_filename"]).write_text(json.dumps(summary, allow_nan=False, indent=2))
        if cfg["general"]["logging"]["print_summary"]:
            print(json.dumps(summary, allow_nan=False))
    finally:
        detector.close()
        if hashlib.sha256(pickle.dumps(model)).hexdigest() != frozen:
            raise AssertionError("Frozen first-layer state changed during controller execution")


if __name__ == "__main__":
    main()
