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
    family = {"type": "string", "enum": [*cfg["attack_knowledge"]["families"],
                                         "UNKNOWN_ATTACK", "INSUFFICIENT_EVIDENCE"]}
    observation = {"type": "object", "additionalProperties": False,
                   "properties": {"feature": {"type": "string"},
                                  "value": {"type": ["number", "string", "boolean", "null"]},
                                  "interpretation": {"type": "string", "minLength": 1}},
                   "required": ["feature", "value", "interpretation"]}
    alternative = {"type": "object", "additionalProperties": False,
                   "properties": {"family": {"type": "string", "enum": [
                       *family["enum"], "benign_explanation"]},
                                  "reason": {"type": "string", "minLength": 1},
                                  "why_not_selected": {"type": "string", "minLength": 1}},
                   "required": ["family", "reason", "why_not_selected"]}
    properties = {"attack_family": family, "attack_subtype": {"type": ["string", "null"]},
                  "alternative_hypotheses": {"type": "array", "items": alternative},
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
    for item in report["supporting_evidence"] + report["contradicting_evidence"]:
        name = item["feature"]
        if name not in flow["features"] or digest(item["value"]) != digest(flow["features"][name]):
            raise ValueError("Report contains an unobserved feature or altered value")
    if report["attack_family"] in cfg["attack_knowledge"]["families"]:
        if not report["supporting_evidence"] or report["evidence_strength"] == "none":
            raise ValueError("Specific attack family lacks supporting observations")
        if not report["alternative_hypotheses"]:
            raise ValueError("Specific attack family requires an alternative explanation")
        if any(item["value"] is None or item["value"] in ("inf", "-inf")
               for item in report["supporting_evidence"]):
            raise ValueError("Missing values cannot support a specific attack family")
    elif report["attack_subtype"] is not None:
        raise ValueError("Abstention cannot claim a subtype")


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
    instructions = "\n\n".join(value for key, value in prompts.items() if key != "tool_descriptions")
    content = {"flow": public_flow(flow, source), "feature_groups": source["feature_groups"],
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
            "provider": provider, "stream": False}
    if response_format["type"] != "none":
        body["response_format"] = response_format
    if options["allowed_tools"]:
        body.update(tools=tool_schemas(cfg),
                    tool_choice="none" if final_step else options["tool_choice"])
    return body


def openrouter_request(body: dict, cfg: dict) -> dict:
    options = cfg["advanced_llm"]
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
    """Called only after Jev ATTACK; no references to a model, labels or dataset frame."""
    messages = initial_messages(flow, source, cfg)
    options = cfg["advanced_llm"]
    send = send or (lambda body: openrouter_request(body, cfg))
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tool_schemas(cfg)}
    cached_tools, calls = {}, 0
    last_model = options["model"]
    for step in range(options["max_steps"]):
        final = step == options["max_steps"] - 1 or calls >= options["max_tool_calls"]
        reply = requests.call("advanced_llm", flow["flow_id"], request_body(messages, cfg, final), send)
        if reply["error"]:
            return {"report": None, "model": last_model, "error": reply["error"], "tool_calls": calls}
        body = reply["body"]
        try:
            reported_model = body.get("model", options["model"])
            if not isinstance(reported_model, str) or not reported_model:
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
                    raise ValueError("Incomplete investigation response")
                report = json.loads(message["content"])
                validate_report(report, flow, cfg)
                return {"report": report, "model": last_model, "error": None, "tool_calls": calls}
            if final or len(tool_calls) > options["max_tool_calls"] - calls:
                raise ValueError("Investigation tool budget exceeded")
            if len({item["id"] for item in tool_calls}) != len(tool_calls):
                raise ValueError("Duplicate tool call identifier")
            messages.append({"role": "assistant", "content": message.get("content"),
                             "tool_calls": tool_calls})
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
                    observation = cached_tools[key]
                except (ValueError, TypeError, ValidationError):
                    observation = {"error": "invalid_or_unavailable_tool_request"}
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": json.dumps(observation, allow_nan=False)})
        except (KeyError, IndexError, TypeError, ValueError, ValidationError):
            return {"report": None, "model": last_model,
                    "error": "invalid_investigation_response", "tool_calls": calls}
    return {"report": None, "model": last_model,
            "error": "investigation_step_limit", "tool_calls": calls}
