"""Bounded OpenRouter investigation with three authorized read-only evidence tools."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from jsonschema import ValidationError, validate

from .evidence import digest, evidence_tool, public_flow
from .jev import APIFailure


def report_schema(cfg: dict) -> dict:
    attack_types = [*cfg["attack_knowledge"]["families"], "UNKNOWN_ATTACK", "INSUFFICIENT_EVIDENCE"]
    family = {"type": ["string", "null"], "enum": [*attack_types, None]}
    observation = {"type": "object", "additionalProperties": False,
                   "properties": {"feature": {"type": "string"},
                                  "value": {"type": ["number", "string", "boolean", "null"]},
                                  "interpretation": {"type": "string", "minLength": 1}},
                   "required": ["feature", "value", "interpretation"]}
    alternative = {"type": "object", "additionalProperties": False,
                   "properties": {"family": {"type": "string", "enum": [
                       *attack_types, "benign_explanation"]},
                                  "reason": {"type": "string", "minLength": 1},
                                  "why_not_selected": {"type": "string", "minLength": 1}},
                   "required": ["family", "reason", "why_not_selected"]}
    properties = {"classification": {"type": "string", "enum": ["BENIGN", "ATTACK", "INSUFFICIENT_EVIDENCE"]},
                  "attack_family": family, "attack_subtype": {"type": ["string", "null"]},
                  "classification_uncertain": {"type": "boolean"},
                  "alternative_hypotheses": {"type": "array", "items": alternative, "minItems": 1},
                  "evidence_strength": {"type": "string", "enum": ["none", "weak", "moderate", "strong"]},
                  "supporting_evidence": {"type": "array", "items": observation},
                  "contradicting_evidence": {"type": "array", "items": observation},
                  "missing_evidence": {"type": "array", "items": {"type": "string"}},
                  "investigation_summary": {"type": "string", "minLength": 1}}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def validate_report(report: dict, flow: dict, cfg: dict) -> None:
    validate(report, report_schema(cfg))
    json.dumps(report, allow_nan=False)
    mode = cfg["advanced_llm"].get("mode", "attack")
    if mode == "attack" and report["classification"] != "ATTACK":
        raise ValueError("Investigation must preserve the upstream attack decision")
    if report["classification"] == "BENIGN":
        if report["attack_family"] is not None or report["attack_subtype"] is not None:
            raise ValueError("A benign prediction cannot have an attack label")
    elif report["classification"] == "INSUFFICIENT_EVIDENCE":
        if report["attack_family"] != "INSUFFICIENT_EVIDENCE":
            raise ValueError("Insufficient recommendation must abstain from family assignment")
    elif report["attack_family"] is None:
        raise ValueError("An attack prediction requires a family or explicit abstention")
    context = flow.get("first_layer_evidence", {})
    observations = {**context, **flow["features"],
                    **{"first_layer_evidence." + name: value for name, value in context.items()}}
    for item in report["supporting_evidence"] + report["contradicting_evidence"]:
        name = item["feature"]
        observed, reported = observations.get(name), item["value"]
        # JSON numbers such as 0 and 0.0 denote the same measurement. Preserve
        # exact numeric equality; strings, booleans and null still match by type.
        numeric = type(observed) in (int, float) and type(reported) in (int, float)
        matches = observed == reported if numeric else digest(observed) == digest(reported)
        if name not in observations or not matches:
            raise ValueError("Report contains an unobserved feature or altered value")
    abstains = report["classification"] == "INSUFFICIENT_EVIDENCE" or report["attack_family"] == "INSUFFICIENT_EVIDENCE"
    if abstains and (not report["classification_uncertain"] or report["attack_subtype"] is not None):
        raise ValueError("Abstention must record uncertainty without a subtype")
    if report["attack_family"] in ("UNKNOWN_ATTACK", "INSUFFICIENT_EVIDENCE") and report["attack_subtype"] is not None:
        raise ValueError("Unknown/insufficient family cannot invent a subtype")
    if not abstains and not any(item["feature"] in flow["features"] for item in report["supporting_evidence"]):
        raise ValueError("Classification requires an observed traffic feature")
    if not report["alternative_hypotheses"]:
        raise ValueError("Classification requires an alternative explanation")
    if any(item["value"] is None or item["value"] in ("inf", "-inf")
           for item in report["supporting_evidence"]):
        raise ValueError("Missing values cannot support a classification")
    if (report["evidence_strength"] in cfg["advanced_llm"]["uncertain_evidence_strengths"]
            and not report["classification_uncertain"]):
        raise ValueError("Weak predictions must record uncertainty")


def tool_schemas(cfg: dict) -> list[dict]:
    tools = []
    for name in cfg["advanced_llm"]["allowed_tools"]:
        key = "families" if name == "retrieve_attack_knowledge" else "features"
        maximum = cfg["evidence"]["max_families_per_tool" if key == "families"
                                   else "max_features_per_tool"]
        parameters = {"type": "object", "additionalProperties": False,
                      "properties": {key: {"type": "array", "items": {"type": "string"},
                                           "uniqueItems": True, "maxItems": maximum}},
                      "required": [key] if name == "compare_source_baseline" else []}
        tools.append({"type": "function", "function": {
            "name": name, "description": cfg["prompts"]["advanced_llm"]["tool_descriptions"][name],
            "parameters": parameters}})
    return tools


def initial_messages(flow: dict, source: dict, cfg: dict) -> list[dict]:
    prompts = cfg["prompts"]["advanced_llm"]
    instructions = prompts["system"] + "\n\n" + prompts[cfg["advanced_llm"].get("mode", "attack")] + "\n\n" + prompts["report"]
    content = {"flow": public_flow(flow, source), "feature_groups": source["feature_groups"],
               "source_known_references": source["examples"],
               "attack_types": [label for label in report_schema(cfg)["properties"]["attack_family"]["enum"]
                                if label is not None],
               "uncertain_evidence_strengths": cfg["advanced_llm"]["uncertain_evidence_strengths"],
               "family_catalogue": list(cfg["attack_knowledge"]["families"]),
               "report_schema": report_schema(cfg)}
    return [{"role": "system", "content": instructions},
            {"role": "user", "content": json.dumps(content, allow_nan=False)}]


def request_body(messages: list[dict], cfg: dict, final_step: bool) -> dict:
    options = cfg["advanced_llm"]
    response_format = dict(options["response_format"])
    if response_format["type"] == "json_schema":
        response_format = {"type": "json_schema", "json_schema": {
            "name": response_format["name"], "strict": response_format["strict"],
            "schema": report_schema(cfg)}}
    else:
        response_format = {"type": response_format["type"]}
    rates = cfg["budgets"]["pricing"]["advanced_llm"]
    provider = {**options["provider_options"], "max_price": {
        "prompt": rates["input_per_million"], "completion": rates["output_per_million"]}}
    body = {"model": options["model"], "messages": messages,
            "temperature": options["temperature"], "max_tokens": options["max_tokens"],
            "provider": provider, "stream": False, "seed": cfg["general"]["seed"]}
    if options.get("reasoning") is not None:
        body["reasoning"] = options["reasoning"]
    if response_format["type"] != "none":
        body["response_format"] = response_format
    if options["allowed_tools"]:
        body.update(tools=tool_schemas(cfg),
                    tool_choice="none" if final_step else options["tool_choice"])
    return body


def openrouter_request(body: dict, cfg: dict, stage: str = "advanced_llm") -> dict:
    options = cfg[stage]
    key = os.environ.get(options["api_key_env"])
    if not key:
        raise APIFailure("missing_openrouter_api_key")
    request = Request(options["base_url"].rstrip("/") + options["endpoint"],
                      data=json.dumps(body, allow_nan=False).encode(),
                      headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                      method="POST")
    try:
        with urlopen(request, timeout=options["timeout_seconds"]) as response:
            result = json.load(response)
    except HTTPError as exc:
        raise APIFailure("provider_http_error", exc.code) from exc
    except OSError as exc:
        raise APIFailure("connection_or_timeout") from exc
    except (ValueError, UnicodeError) as exc:
        raise APIFailure("malformed_response") from exc
    if not isinstance(result, dict):
        raise APIFailure("malformed_response")
    return result


def investigate(flow: dict, source: dict, cfg: dict, requests, send=None) -> dict:
    """One bounded agent, specialized by attack/uncertainty mode; no security actions."""
    messages = initial_messages(flow, source, cfg)
    options = cfg["advanced_llm"]
    send = send or (lambda body: openrouter_request(body, cfg))
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tool_schemas(cfg)}
    cached_tools, calls = {}, 0
    last_model = options["model"]
    diagnostics = {"json_valid": False, "schema_valid": False, "observations": 0,
                   "matching_observations": 0, "tool_successes": 0, "tool_errors": [],
                   "tool_cache_hits": 0, "tool_trace": []}
    def result(report, error):
        return {"report": report, "model": last_model, "error": error, "tool_calls": calls, **diagnostics}
    for step in range(options["max_steps"]):
        final = step == options["max_steps"] - 1 or calls >= options["max_tool_calls"]
        reply = requests.call("advanced_llm", flow["flow_id"], request_body(messages, cfg, final), send)
        if reply["error"]:
            return result(None, reply["error"])
        body = reply["body"]
        try:
            reported_model = body.get("model")
            if reported_model != options["model"]:
                raise ValueError("Invalid investigation model identifier")
            last_model = reported_model
            choices = body["choices"]
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("Invalid investigation choices")
            choice = choices[0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError("Invalid investigation message")
            tool_calls = message.get("tool_calls")
            if tool_calls is None:
                tool_calls = []
            if not isinstance(tool_calls, list) or any(not isinstance(item, dict) for item in tool_calls):
                raise ValueError("Invalid investigation tool calls")
            if not tool_calls:
                if choice.get("finish_reason") != "stop":
                    return result(None, "incomplete_investigation_response")
                report = json.loads(message["content"])
                diagnostics["json_valid"] = True
                validate(report, report_schema(cfg))
                diagnostics["schema_valid"] = True
                observations = {**flow.get("first_layer_evidence", {}), **flow["features"],
                                **{"first_layer_evidence." + k: v for k, v in flow.get("first_layer_evidence", {}).items()}}
                for item in report["supporting_evidence"] + report["contradicting_evidence"]:
                    diagnostics["observations"] += 1
                    observed = observations.get(item["feature"])
                    numeric = type(observed) in (int, float) and type(item["value"]) in (int, float)
                    matches = observed == item["value"] if numeric else digest(observed) == digest(item["value"])
                    diagnostics["matching_observations"] += int(item["feature"] in observations and matches)
                validate_report(report, flow, cfg)
                if options.get("mode") == "uncertainty" and report["classification"] != "INSUFFICIENT_EVIDENCE" and not diagnostics["tool_successes"]:
                    return result(None, "no_additional_evidence_consulted")
                return result(report, None)
            if final or len(tool_calls) > options["max_tool_calls"] - calls:
                raise ValueError("Investigation tool budget exceeded")
            if len({item["id"] for item in tool_calls}) != len(tool_calls):
                raise ValueError("Duplicate tool call identifier")
            messages.append({"role": "assistant", **{key: message[key] for key in
                ("content", "tool_calls", "reasoning", "reasoning_details") if key in message}})
            for call in tool_calls:
                calls += 1
                name = call["function"]["name"]
                try:
                    if name not in schemas:
                        raise ValueError("Unauthorized tool")
                    args = json.loads(call["function"]["arguments"])
                    validate(args, schemas[name])
                    key = digest([name, args])
                    if key not in cached_tools:
                        cached_tools[key] = evidence_tool(name, args, flow, source, cfg)
                    else:
                        diagnostics["tool_cache_hits"] += 1
                    observation = cached_tools[key]
                    diagnostics["tool_successes"] += 1
                except (KeyError, ValueError, TypeError, ValidationError):
                    diagnostics["tool_errors"].append("invalid_or_unavailable_tool_request")
                    observation = {"error": "invalid_or_unavailable_tool_request"}
                diagnostics["tool_trace"].append({"tool": name, "observation": observation})
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(observation, allow_nan=False)})
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            return result(None, "invalid_investigation_response")
    return result(None, "investigation_step_limit")
