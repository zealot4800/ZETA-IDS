"""Source-only read-only evidence tools and budgeted investigation agent."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from urllib.request import Request, urlopen

import joblib
import numpy as np

from . import detector


TOOL_NAMES = ("inspect_flow", "inspect_attack_evidence", "retrieve_source_reference",
              "analyze_benign_region", "check_feature_consistency")


def load_reference(path: str, model_path: str, policy_path: str, model: dict,
                   fingerprint: str, feature_names: list[str], version: str,
                   template_rows: int) -> dict:
    """Check a source-training-only reference against unchanged frozen artifacts."""
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != version:
        raise ValueError("Unsupported Phase 8 reference artifact")
    expected = {"source_dataset": model["source_dataset"],
                "source_fingerprint_sha256": model["source_artifact"]["data_fingerprint_sha256"],
                "training_partition_sha256": model["partition"]["partition_index_sha256"]["training"],
                "feature_schema_fingerprint_sha256": fingerprint,
                "model_sha256": detector.hash_file(model_path),
                "policy_sha256": detector.hash_file(policy_path),
                "family_names": list(model["family_names"])}
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("Reference provenance differs from frozen source artifacts")
    counts = artifact.get("cell_family_counts")
    regions = artifact.get("region_family_counts")
    nodes = model["binary_tree"].tree_.node_count
    region_count = len(model["family_model"]["benign_regions"]["codes"])
    shape = ((region_count + 1) * nodes, len(model["family_names"]))
    if (not isinstance(counts, np.ndarray) or counts.shape != shape or
            not isinstance(regions, np.ndarray) or
            regions.shape != (region_count + 1, len(model["family_names"])) or
            np.any(counts < 0) or not np.array_equal(
                counts.reshape(region_count + 1, nodes, -1).sum(axis=1), regions) or
            int(counts.sum()) != artifact.get("training_rows") or
            artifact.get("binary_node_count") != nodes or
            artifact.get("region_count") != region_count or
            artifact.get("benign_code") != model["benign_code"]):
        raise ValueError("Reference aggregate integrity failure")
    templates = artifact.get("family_templates")
    if (not isinstance(templates, dict) or
            set(templates) != set(model["family_names"]) or
            artifact.get("provenance") !=
            "source_training_only; aggregate labels; no calibration, validation, or target rows"):
        raise ValueError("Reference template provenance failure")
    for code, name in enumerate(model["family_names"]):
        template = templates[name]
        support = int(counts[:, code].sum())
        if (not isinstance(template, dict) or
                set(template) != {"training_rows", "sampled_rows", "median", "q25", "q75"} or
                template["training_rows"] != support or
                template["sampled_rows"] != min(template_rows, support)):
            raise ValueError("Reference family template support differs")
        for statistic in ("median", "q25", "q75"):
            values = template[statistic]
            if (not isinstance(values, list) or len(values) != len(feature_names) or
                    not all(isinstance(value, (int, float)) and np.isfinite(value)
                            for value in values)):
                raise ValueError("Reference family template features differ")
    return artifact


def tool_specs(descriptions: dict, enabled: list[str]) -> list[dict]:
    if set(descriptions) != set(TOOL_NAMES):
        raise ValueError("Tool descriptions must cover all five read-only tools")
    tools = []
    for name in enabled:
        properties = ({"family": {"type": "string", "description": "Exact source family name"}}
                      if name == "retrieve_source_reference" else {})
        tools.append({"type": "function", "function": {
            "name": name, "description": descriptions[name],
            "parameters": {"type": "object", "properties": properties,
                           "additionalProperties": False}}})
    return tools


class EvidenceTools:
    """No evaluation label, frame metadata, or target distribution enters this object."""

    def __init__(self, model: dict, reference: dict, features: np.ndarray,
                 options: dict, feature_names: list[str], chunk_rows: int,
                 observed: dict | None = None, position: int = 0):
        self.reference = reference
        self.options = options
        self.feature_names = feature_names
        self.features = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if self.features.shape[1] != len(feature_names):
            raise ValueError("Flow feature schema mismatch")
        self.evidence = observed if observed is not None else detector.evidence(model, self.features, chunk_rows)
        self.position = position if observed is not None else 0
        self.key = int(observed["decision_keys"][position] if observed is not None else
                       detector.decision_keys(model, self.features, self.evidence)[0])
        self.region = int(self.evidence["family"]["region"][self.position])
        self.leaf = int(self.key % reference["binary_node_count"])

    def flow_summary(self) -> dict:
        family = self.evidence["family"]
        prediction = int(self.evidence["predicted_family_code"][self.position])
        return {"reference_id": "flow:current",
                "features": {name: float(value) if np.isfinite(value) else None
                             for name, value in zip(self.feature_names, self.features[0])},
                "valid_features": bool(self.evidence["valid"][self.position]),
                "binary_attack_score": (float(self.evidence["attack_score"][self.position])
                                        if self.evidence["valid"][self.position] else None),
                "predicted_family_from_frozen_model": self.reference["family_names"][prediction],
                "source_prototype_attack_support": int(family["prototype_attack_count"][self.position]),
                "source_prototype_benign_support": int(family["prototype_benign_count"][self.position]),
                "benign_region": self.region, "binary_leaf": self.leaf}

    def run(self, name: str, arguments: dict | None = None) -> dict:
        if name not in self.options["enabled"] or not isinstance(arguments, dict):
            raise ValueError("Unknown tool or invalid arguments")
        if name != "retrieve_source_reference" and arguments:
            raise ValueError("This tool takes no arguments")
        if name == "retrieve_source_reference" and set(arguments) - {"family"}:
            raise ValueError("Unexpected source-reference argument")
        if name == "inspect_flow":
            return self.flow_summary()
        if name == "inspect_attack_evidence":
            row = self.reference["cell_family_counts"][self.key]
            names = self.reference["family_names"]
            benign = int(row[self.reference["benign_code"]])
            attack = sorted(((names[i], int(row[i])) for i in range(len(row))
                             if i != self.reference["benign_code"] and row[i] > 0),
                            key=lambda item: (-item[1], item[0]))[:self.options["max_attack_families"]]
            return {"reference_id": f"attack:cell:{self.key}",
                    "source_training_benign_rows": benign,
                    "source_training_attack_rows": int(row.sum()) - benign,
                    "top_source_training_attack_families": dict(attack),
                    "source_prototype_attack_support":
                    int(self.evidence["family"]["prototype_attack_count"][self.position])}
        if name == "retrieve_source_reference":
            family = arguments.get("family")
            if not family:
                predicted = int(self.evidence["predicted_family_code"][self.position])
                family = self.reference["family_names"][predicted]
            if not isinstance(family, str) or family not in self.reference["family_templates"]:
                raise ValueError("Unknown source family")
            return {"reference_id": f"source:family:{family}",
                    "family": family, "feature_names": self.feature_names,
                    **self.reference["family_templates"][family]}
        if name == "analyze_benign_region":
            if self.region < 0:
                return {"reference_id": "benign:region:none",
                        "source_training_rows": 0, "benign_fraction": None}
            row = self.reference["region_family_counts"][self.region + 1]
            benign = int(row[self.reference["benign_code"]])
            total = int(row.sum())
            return {"reference_id": f"benign:region:{self.region}",
                    "source_training_rows": total, "source_training_benign_rows": benign,
                    "source_training_attack_rows": total - benign,
                    "benign_fraction": benign / max(total, 1)}
        return self._consistency()

    def _consistency(self) -> dict:
        value = dict(zip(self.feature_names, (float(v) for v in self.features[0])))
        issues = []
        for name, number in value.items():
            if not np.isfinite(number):
                issues.append(f"{name}:nonfinite")
            elif number < 0:
                issues.append(f"{name}:negative")
        for prefix in ("fwd", "bwd"):
            low = value[f"{prefix}_packet_length_min_bytes"]
            mean = value[f"{prefix}_packet_length_mean_bytes"]
            high = value[f"{prefix}_packet_length_max_bytes"]
            valid_lengths = np.isfinite([low, mean, high]).all()
            if valid_lengths and low > mean + self.options["consistency_tolerance"]:
                issues.append(f"{prefix}_packet_length_min_exceeds_mean")
            if valid_lengths and mean > high + self.options["consistency_tolerance"]:
                issues.append(f"{prefix}_packet_length_mean_exceeds_max")
            if value[f"{prefix}_packet_count"] == 0 and value[f"{prefix}_bytes_total"] > 0:
                issues.append(f"{prefix}_bytes_without_packets")
        return {"reference_id": "consistency:current", "issues": issues,
                "consistent_under_basic_flow_rules": not issues}


def verdict_schema(config: dict) -> dict:
    return {"type": "json_schema", "json_schema": {
        "name": config["schema_name"], "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {"verdict": {"type": "string", "enum": config["allowed"]},
                                  "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                  "rationale": {"type": "string"}},
                   "required": config["required_fields"]}}}


def family_schema(families: list[str], benign_code: int) -> dict:
    candidates = [name for code, name in enumerate(families) if code != benign_code] + ["UNKNOWN"]
    return {"type": "json_schema", "json_schema": {
        "name": "source_attack_family", "strict": True,
        "schema": {"type": "object", "additionalProperties": False,
                   "properties": {"attack_family": {"type": "string", "enum": candidates},
                                  "evidence_refs": {"type": "array", "items": {"type": "string"}}},
                   "required": ["attack_family", "evidence_refs"]}}}


class BudgetError(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    max_requests: int
    max_tokens: int
    max_tool_calls: int
    max_cost_usd: float
    max_seconds: float
    max_requests_per_flow: int
    max_tools_per_flow: int
    max_output_tokens: int
    max_input_tokens_reserve: int
    input_framing_bytes: int
    max_rationale_chars: int
    max_cost_per_request_usd: float
    request_timeout_seconds: float

    def __post_init__(self):
        integer_fields = {"max_requests", "max_tokens", "max_tool_calls", "max_requests_per_flow",
                          "max_tools_per_flow", "max_output_tokens", "max_input_tokens_reserve",
                          "input_framing_bytes", "max_rationale_chars"}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if (name in integer_fields and type(value) is not int or
                    isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid agent limit: {name}")
        if min(self.max_output_tokens, self.max_rationale_chars,
               self.max_requests_per_flow, self.request_timeout_seconds,
               self.max_seconds, self.max_input_tokens_reserve) <= 0:
            raise ValueError("Output token and timeout limits must be positive")


class Ledger:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.started = time.monotonic()
        self.requests = 0
        self.tokens = 0
        self.tool_calls = 0
        self.cost_usd = 0.0
        self.usage_unknown = False
        self.exceeded = False

    def remaining_seconds(self) -> float:
        return self.limits.max_seconds - (time.monotonic() - self.started)

    def begin_request(self, prompt: list[dict], paid: bool,
                      max_output_tokens: int | None = None) -> tuple[int, float]:
        if self.exceeded or self.remaining_seconds() <= 0:
            raise BudgetError("time_or_prior_budget_exhausted")
        if self.requests >= self.limits.max_requests:
            raise BudgetError("request_budget_exhausted")
        # UTF-8 bytes plus message framing is deliberately conservative for a
        # local preflight estimate. Provider-reported usage is authoritative.
        message_bytes = len(json.dumps(prompt, ensure_ascii=False, allow_nan=False).encode())
        if paid and message_bytes + self.limits.input_framing_bytes > self.limits.max_input_tokens_reserve:
            raise BudgetError("input_reserve_too_small")
        reserve_tokens = (self.limits.max_input_tokens_reserve +
                          (self.limits.max_output_tokens if max_output_tokens is None else max_output_tokens)) if paid else 0
        reserve_cost = self.limits.max_cost_per_request_usd if paid else 0.0
        if self.tokens + reserve_tokens > self.limits.max_tokens:
            raise BudgetError("token_budget_exhausted")
        if self.cost_usd + reserve_cost > self.limits.max_cost_usd + 1e-12:
            raise BudgetError("cost_budget_exhausted")
        self.requests += 1
        self.tokens += reserve_tokens
        self.cost_usd += reserve_cost
        return reserve_tokens, reserve_cost

    def finish_request(self, reservation: tuple[int, float], usage: dict, paid: bool) -> None:
        reserved_tokens, reserved_cost = reservation
        if not isinstance(usage, dict):
            self.usage_unknown = paid
            self.exceeded = True
            raise BudgetError("missing_usage")
        tokens = usage.get("total_tokens")
        cost = usage.get("cost")
        if not (isinstance(tokens, int) and tokens >= 0 and
                isinstance(cost, (int, float)) and math.isfinite(float(cost)) and cost >= 0):
            self.usage_unknown = paid
            self.exceeded = True
            raise BudgetError("missing_or_invalid_usage")
        self.tokens += tokens - reserved_tokens
        self.cost_usd += float(cost) - reserved_cost
        if ((paid and (tokens > reserved_tokens or float(cost) > reserved_cost + 1e-12)) or
                self.tokens > self.limits.max_tokens or
                self.cost_usd > self.limits.max_cost_usd + 1e-12 or
                self.remaining_seconds() < 0):
            self.exceeded = True
            raise BudgetError("provider_usage_exceeded_budget")

    def check_tool(self) -> None:
        if self.tool_calls >= self.limits.max_tool_calls or self.remaining_seconds() <= 0:
            raise BudgetError("tool_or_time_budget_exhausted")

    def use_tool(self) -> None:
        self.check_tool()
        self.tool_calls += 1

    def snapshot(self) -> dict:
        return {"requests": self.requests, "tokens": self.tokens,
                "tool_calls": self.tool_calls, "cost_usd": self.cost_usd,
                "usage_unknown": self.usage_unknown}


class MockBackend:
    paid = False

    def __init__(self, first_tool: str):
        self.first_tool = first_tool

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int,
                 timeout: float) -> dict:
        if tools and not any(message.get("role") == "tool" for message in messages):
            message = {"role": "assistant", "content": None,
                       "tool_calls": [{"id": "mock-tool-1", "type": "function",
                                       "function": {"name": self.first_tool, "arguments": "{}"}}]}
        else:
            message = {"role": "assistant", "content": json.dumps({
                "verdict": "ABSTAIN", "evidence_refs": [],
                "rationale": "Mock backend has no investigator verdict."})}
        return {"choices": [{"message": message}], "usage": {"total_tokens": 0, "cost": 0.0}}


class OpenRouterBackend:
    paid = True

    def __init__(self, model: str, temperature: float, endpoint: str,
                 api_key: str, schema: dict):
        self.model = model
        self.temperature = temperature
        self.endpoint = endpoint
        self.schema = schema
        self.api_key = api_key
        if not model or not api_key:
            raise ValueError("Live mode requires a configured model and API key")

    def complete(self, messages: list[dict], tools: list[dict], max_tokens: int,
                 timeout: float) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": self.temperature,
                   "stream": False, "max_tokens": max_tokens}
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = False
        else:
            payload["response_format"] = self.schema
        body = json.dumps(payload, allow_nan=False).encode()
        request = Request(self.endpoint, data=body,
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


class JevBackend:
    """OpenRouter Decisions API; tool selection is orchestrated locally."""
    paid = True

    def __init__(self, model: str, endpoint: str, api_key: str):
        self.model, self.endpoint, self.api_key = model, endpoint, api_key
        self.family_backend = None

    def decide(self, request_body: dict, timeout: float) -> dict:
        request = Request(self.endpoint, data=json.dumps(request_body, allow_nan=False).encode(),
                          headers={"Authorization": f"Bearer {self.api_key}",
                                   "Content-Type": "application/json"}, method="POST")
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())


def _parse_verdict(message: dict, allowed_refs: set[str], max_rationale_chars: int,
                   config: dict) -> tuple[str, list[str], str]:
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("missing_verdict_content")
    value = json.loads(content)
    if (not isinstance(value, dict) or set(value) != set(config["required_fields"]) or
            value["verdict"] not in config["allowed"] or not isinstance(value["evidence_refs"], list) or
            not all(isinstance(ref, str) and ref in allowed_refs for ref in value["evidence_refs"]) or
            not isinstance(value["rationale"], str) or len(value["rationale"]) > max_rationale_chars):
        raise ValueError("invalid_structured_verdict")
    if config["require_evidence_for_decision"] and value["verdict"] != "ABSTAIN" and not value["evidence_refs"]:
        raise ValueError("decision_without_evidence")
    return value["verdict"], value["evidence_refs"], value["rationale"]


def failure_result(backend, reason: str) -> dict:
    return {"verdict": "ABSTAIN", "evidence_refs": [], "rationale": "",
            "failure_reason": reason, "tool_names": [],
            "tool_call_count": 0, "request_count": 0, "token_usage": 0,
            "investigation_cost_usd": 0.0, "usage_unknown": False,
            "backend": "openrouter" if backend.paid else "mock"}


def _classify_attack_family(tools: EvidenceTools, backend: JevBackend, ledger: Ledger,
                            config: dict, flow: dict, evidence: list[dict],
                            executed: list[str]) -> tuple[str | None, list[str]]:
    family_backend = backend.family_backend
    if family_backend is None:
        raise ValueError("family_model_unconfigured")
    if not any(item["reference_id"].startswith("attack:cell:") for item in evidence):
        if len(executed) < ledger.limits.max_tools_per_flow and ledger.tool_calls < ledger.limits.max_tool_calls:
            ledger.check_tool()
            evidence.append(tools.run("inspect_attack_evidence", {}))
            ledger.use_tool()
            executed.append("inspect_attack_evidence")
    attack = next((item for item in evidence if item["reference_id"].startswith("attack:cell:")), None)
    candidates = list(attack["top_source_training_attack_families"]) if attack else []
    for family in candidates[:config["attack_family"]["max_reference_families"]]:
        if len(executed) >= ledger.limits.max_tools_per_flow or ledger.tool_calls >= ledger.limits.max_tool_calls:
            break
        ledger.check_tool()
        evidence.append(tools.run("retrieve_source_reference", {"family": family}))
        ledger.use_tool()
        executed.append("retrieve_source_reference")
    families = [name for code, name in enumerate(tools.reference["family_names"])
                if code != tools.reference["benign_code"]]
    settings = config["attack_family"]
    messages = [{"role": "system", "content": settings["system_prompt"]},
                {"role": "user", "content": json.dumps(
                    {"flow": flow, "evidence": evidence, "source_attack_families": families,
                     "task": settings["task_prompt"]}, allow_nan=False)}]
    reservation = None
    completed_usage = False
    try:
        reservation = ledger.begin_request(messages + [{"response_format": family_backend.schema}],
                                           True, settings["max_output_tokens"])
        timeout = min(ledger.limits.request_timeout_seconds, ledger.remaining_seconds())
        if timeout <= 0:
            raise BudgetError("time_budget_exhausted")
        response = family_backend.complete(messages, [], settings["max_output_tokens"], timeout)
        completed_usage = True
        ledger.finish_request(reservation, response.get("usage"), True)
    except Exception:
        if reservation is not None and not completed_usage:
            ledger.usage_unknown = True
            ledger.exceeded = True
        raise
    content = response["choices"][0]["message"]["content"]
    value = json.loads(content)
    allowed_refs = {flow["reference_id"], *(item["reference_id"] for item in evidence)}
    if (not isinstance(value, dict) or set(value) != {"attack_family", "evidence_refs"} or
            value["attack_family"] not in families + ["UNKNOWN"] or
            not isinstance(value["evidence_refs"], list) or
            not all(isinstance(ref, str) and ref in allowed_refs for ref in value["evidence_refs"]) or
            (value["attack_family"] != "UNKNOWN" and not value["evidence_refs"])):
        raise ValueError("invalid_attack_family_response")
    return (None if value["attack_family"] == "UNKNOWN" else value["attack_family"],
            value["evidence_refs"])


def investigate_jev(tools: EvidenceTools, backend: JevBackend, ledger: Ledger, config: dict) -> dict:
    before = ledger.snapshot()
    try:
        flow = tools.flow_summary()
        valid_flow = flow["valid_features"] is True
    except Exception as exc:
        return failure_result(backend, f"flow_inspection_failed:{type(exc).__name__}")
    if not valid_flow:
        return failure_result(backend, "invalid_flow_features")
    evidence, executed = [], []
    reason, verdict, refs = None, "ABSTAIN", []
    finalize = False
    for step in range(ledger.limits.max_requests_per_flow):
        reservation = None
        completed_usage = False
        try:
            available = [name for name in config["tools"]["enabled"] if name not in executed]
            can_tool = (not finalize and available and step + 1 < ledger.limits.max_requests_per_flow and
                        len(executed) < ledger.limits.max_tools_per_flow and
                        ledger.tool_calls < ledger.limits.max_tool_calls)
            settings = config["decisions"]
            criteria = ({name: config["tools"]["descriptions"][name] for name in available}
                        | {"finalize": settings["finalize_description"]}) if can_tool else settings["verdict_criteria"]
            question = "next_tool" if can_tool else "verdict"
            request_body = {"model": backend.model, "state": {"flow": flow, "evidence": evidence},
                            "questions": {question: {"type": "choice", "criteria": criteria,
                                                    "instructions": settings["tool_instructions"] if can_tool
                                                    else settings["verdict_instructions"]}}}
            reservation = ledger.begin_request([request_body], True)
            timeout = min(ledger.limits.request_timeout_seconds, ledger.remaining_seconds())
            if timeout <= 0:
                raise BudgetError("time_budget_exhausted")
            response = backend.decide(request_body, timeout)
            usage = response.get("usage")
            if isinstance(usage, dict):
                incoming, outgoing = usage.get("input_tokens"), usage.get("output_tokens")
                if type(incoming) is not int or type(outgoing) is not int or min(incoming, outgoing) < 0:
                    raise ValueError("invalid_decision_usage")
                usage = {"total_tokens": incoming + outgoing, "cost": usage.get("cost")}
            completed_usage = True
            ledger.finish_request(reservation, usage, True)
            answer = response["answers"][question]
            choice = answer["choice"]
            if answer.get("type") != "choice" or choice not in criteria:
                raise ValueError("invalid_decision_response")
            if can_tool and choice != "finalize":
                ledger.check_tool()
                result = tools.run(choice, {})
                ledger.use_tool()
                executed.append(choice)
                evidence.append(result)
                continue
            if can_tool:
                finalize = True
                continue
            probability = answer.get("probabilities", {}).get(choice)
            if (isinstance(probability, bool) or not isinstance(probability, (int, float)) or
                    not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("invalid_decision_probability")
            if probability < settings["min_verdict_probability"]:
                reason = "low_decision_probability"
            elif choice != "ABSTAIN":
                verdict = choice
                refs = [flow["reference_id"], *(item["reference_id"] for item in evidence)]
            break
        except Exception as exc:
            if reservation is not None and not completed_usage:
                ledger.usage_unknown = True
                ledger.exceeded = True
            reason = type(exc).__name__ if not isinstance(exc, (BudgetError, ValueError)) else str(exc)
            break
    else:
        reason = "request_limit_before_final_verdict"
    attack_family, family_refs, family_failure = None, [], None
    if verdict == "MALICIOUS" and reason is None:
        try:
            attack_family, family_refs = _classify_attack_family(
                tools, backend, ledger, config, flow, evidence, executed)
        except Exception as exc:
            family_failure = type(exc).__name__ if not isinstance(exc, (BudgetError, ValueError)) else str(exc)
    after = ledger.snapshot()
    unknown = after["usage_unknown"] and not before["usage_unknown"]
    return {"verdict": verdict, "evidence_refs": refs, "rationale": "",
            "failure_reason": reason, "attack_family_candidate": attack_family,
            "attack_family_evidence_refs": family_refs, "family_failure_reason": family_failure,
            "tool_names": executed,
            "tool_call_count": after["tool_calls"] - before["tool_calls"],
            "request_count": after["requests"] - before["requests"],
            "token_usage": None if unknown else after["tokens"] - before["tokens"],
            "investigation_cost_usd": None if unknown else round(after["cost_usd"] - before["cost_usd"], 8),
            "usage_unknown": unknown, "backend": "openrouter"}


def investigate(tools: EvidenceTools, backend, ledger: Ledger, config: dict) -> dict:
    if isinstance(backend, JevBackend):
        return investigate_jev(tools, backend, ledger, config)
    before = ledger.snapshot()
    try:
        flow = tools.flow_summary()
    except Exception as exc:
        return failure_result(backend, f"flow_inspection_failed:{type(exc).__name__}")
    if flow.get("valid_features") is False:
        return failure_result(backend, "invalid_flow_features")
    allowed = {flow["reference_id"]}
    messages = [{"role": "system", "content": config["instructions"]["system_prompt"]},
                {"role": "user", "content": json.dumps(
                    {"flow": flow, "task": config["instructions"]["task_prompt"]},
                    allow_nan=False)}]
    executed = []
    try:
        available_tools = tool_specs(tools.options["descriptions"], tools.options["enabled"])
        schema = verdict_schema(config["verdict"])
    except (KeyError, ValueError) as exc:
        return failure_result(backend, f"invalid_agent_configuration:{type(exc).__name__}")
    reason = None
    verdict, refs, rationale = "ABSTAIN", [], ""
    for _ in range(ledger.limits.max_requests_per_flow):
        reservation = None
        completed_usage = False
        try:
            active_tools = (available_tools
                            if len(executed) < ledger.limits.max_tools_per_flow and
                            ledger.tool_calls < ledger.limits.max_tool_calls else [])
            overhead = {"tools": active_tools} if active_tools else {"response_format": schema}
            reservation = ledger.begin_request(messages + [overhead], backend.paid)
            timeout = min(ledger.limits.request_timeout_seconds, ledger.remaining_seconds())
            if timeout <= 0:
                raise BudgetError("time_budget_exhausted")
            response = backend.complete(messages, active_tools, ledger.limits.max_output_tokens, timeout)
            usage = response.get("usage")
            completed_usage = True
            ledger.finish_request(reservation, usage, backend.paid)
            message = response["choices"][0]["message"]
            calls = message.get("tool_calls") or []
            if calls:
                if (not active_tools or len(calls) > ledger.limits.max_tools_per_flow - len(executed) or
                        len({call.get("id") for call in calls}) != len(calls)):
                    raise ValueError("invalid_or_excess_tool_calls")
                parsed = []
                for call in calls:
                    arguments = json.loads(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise ValueError("invalid_tool_arguments")
                    parsed.append((call["id"], call["function"]["name"], arguments))
                messages.append({"role": "assistant", "content": message.get("content"),
                                 "tool_calls": calls})
                for call_id, name, arguments in parsed:
                    ledger.check_tool()
                    result = tools.run(name, arguments)
                    ledger.use_tool()
                    executed.append(name)
                    allowed.add(result["reference_id"])
                    messages.append({"role": "tool", "tool_call_id": call_id,
                                     "name": name, "content": json.dumps(result, allow_nan=False)})
                continue
            verdict, refs, rationale = _parse_verdict(message, allowed, ledger.limits.max_rationale_chars,
                                                      config["verdict"])
            break
        except Exception as exc:
            if reservation is not None and not completed_usage and backend.paid:
                ledger.usage_unknown = True
                ledger.exceeded = True
            reason = type(exc).__name__ if not isinstance(exc, (BudgetError, ValueError)) else str(exc)
            break
    else:
        reason = "request_limit_before_final_verdict"
    after = ledger.snapshot()
    return {"verdict": verdict if reason is None else "ABSTAIN",
            "evidence_refs": refs if reason is None else [],
            "rationale": rationale if reason is None else "",
            "failure_reason": reason, "tool_names": executed,
            "tool_call_count": after["tool_calls"] - before["tool_calls"],
            "request_count": after["requests"] - before["requests"],
            "token_usage": (None if after["usage_unknown"] and not before["usage_unknown"]
                            else after["tokens"] - before["tokens"]),
            "investigation_cost_usd": (None if after["usage_unknown"] and not before["usage_unknown"]
                                       else round(after["cost_usd"] - before["cost_usd"], 8)),
            "usage_unknown": after["usage_unknown"] and not before["usage_unknown"],
            "backend": "openrouter" if backend.paid else "mock"}
