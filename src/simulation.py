"""Load the two YAML files and run a sequential source-only replay."""

from __future__ import annotations

import math
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml

from . import agent, controller, data, detector
from .memory import PatternMemory


def _keys(value: object, expected: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"Invalid {name} keys")
    return value


def _number(value: object, name: str, *, positive: bool = False, integer: bool = False) -> None:
    if (isinstance(value, bool) or
            (type(value) is not int if integer else not isinstance(value, (int, float))) or
            not math.isfinite(value) or value < 0 or (positive and value <= 0)):
        raise ValueError(f"Invalid {name}")


def _path(value: object, name: str, *, exists: bool = True) -> None:
    if not isinstance(value, str) or not value or (exists and not Path(value).is_file()):
        raise ValueError(f"Missing or invalid {name}")


def limits(cfg: dict) -> dict:
    return {"max_requests": cfg["budgets"]["max_requests"],
            "max_tokens": cfg["budgets"]["max_tokens"],
            "max_tool_calls": cfg["budgets"]["max_tool_calls"],
            "max_cost_usd": cfg["budgets"]["max_cost_usd"],
            "max_seconds": cfg["budgets"]["max_seconds"],
            "max_requests_per_flow": cfg["investigation"]["max_reasoning_steps"],
            "max_tools_per_flow": cfg["investigation"]["max_tools_per_flow"],
            "max_output_tokens": cfg["generation"]["max_output_tokens"],
            "max_input_tokens_reserve": cfg["budgets"]["max_input_tokens_reserve"],
            "input_framing_bytes": cfg["investigation"]["input_framing_bytes"],
            "max_rationale_chars": cfg["investigation"]["max_rationale_chars"],
            "max_cost_per_request_usd": cfg["budgets"]["max_cost_per_request_usd"],
            "request_timeout_seconds": cfg["generation"]["request_timeout_seconds"]}


def read_environment(path: str) -> dict[str, str]:
    """Read local .env values without evaluating shell syntax or logging secrets."""
    values = {}
    file = Path(path)
    if file.exists():
        for line in file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ValueError("Invalid environment file")
            name, value = line.removeprefix("export ").split("=", 1)
            name, value = name.strip(), value.strip()
            if not name.isidentifier():
                raise ValueError("Invalid environment variable name")
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]
            values[name] = value
    values.update(os.environ)
    return values


def load_configuration(system_path: str = "config/system.yaml",
                       agent_path: str = "config/agent.yaml") -> tuple[dict, dict]:
    """Validate settings, supported options, and configured input paths at startup."""
    with open(system_path, encoding="utf-8") as stream:
        system = yaml.safe_load(stream)
    with open(agent_path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    _keys(system, {"dataset", "datasets", "sampling", "source_partitions", "features",
                   "frozen", "detector", "patterns", "memory", "simulation", "output"}, "system")
    _keys(cfg, {"provider", "model", "environment", "instructions", "generation",
                "investigation", "decisions", "attack_family", "tools", "budgets", "verdict"}, "agent")
    if system["dataset"] not in ("cic2017", "cic2018"):
        raise ValueError("Unsupported dataset")
    _keys(system["datasets"], {"cic2017", "cic2018"}, "datasets")
    for dataset, paths in system["datasets"].items():
        _keys(paths, {"source_year", "flows", "review_source", "model", "policy",
                      "reference", "memory"}, dataset)
        if paths["source_year"] != dataset.removeprefix("cic"):
            raise ValueError("Dataset source year differs")
        for name in ("flows", "review_source", "model", "policy", "reference"):
            _path(paths[name], f"{dataset}.{name}")
        _path(paths["memory"], f"{dataset}.memory", exists=False)
        if Path(paths["memory"]).suffix not in (".yaml", ".yml"):
            raise ValueError("Memory output must be YAML")
    sampling = _keys(system["sampling"], {"ratio", "seed"}, "sampling")
    _number(sampling["ratio"], "sampling ratio", positive=True)
    _number(sampling["seed"], "sampling seed", integer=True)
    if sampling["ratio"] > 1:
        raise ValueError("Sampling ratio exceeds one")
    partition = _keys(system["source_partitions"], {"mode", "review_partitions", "partition_seed"}, "partitions")
    if partition["mode"] != "frozen_source_only" or partition["review_partitions"] != ["training", "calibration"]:
        raise ValueError("Only frozen source training and calibration reviews are supported")
    _number(partition["partition_seed"], "partition seed", integer=True)
    features = _keys(system["features"], {"compact_schema_version", "full_schema_version",
                                           "metadata_columns", "dtype", "columns", "preprocessing"}, "features")
    if (features["dtype"] != "float64" or
            not all(isinstance(features[k], str) and features[k] for k in ("compact_schema_version", "full_schema_version")) or
            not isinstance(features["metadata_columns"], list) or not features["metadata_columns"] or
            not all(isinstance(v, str) and v for v in features["metadata_columns"]) or
            not isinstance(features["columns"], list) or not features["columns"]):
        raise ValueError("Invalid feature schema")
    for column in features["columns"]:
        _keys(column, {"name", "unit", "cic2017", "cic2018"}, "feature")
        if not all(isinstance(v, str) and v for v in column.values()):
            raise ValueError("Invalid feature column")
    preprocessing = _keys(features["preprocessing"], {"quantizer_bins", "missing_policy",
                                                        "model_input_dtype"}, "preprocessing")
    _number(preprocessing["quantizer_bins"], "quantizer bins", positive=True, integer=True)
    if (not isinstance(preprocessing["missing_policy"], str) or not preprocessing["missing_policy"] or
            preprocessing["model_input_dtype"] != "float32"):
        raise ValueError("Invalid missing-value policy")
    names = data.feature_names(features)
    if len(names) != len(set(names)) or len(features["metadata_columns"]) != len(set(features["metadata_columns"])):
        raise ValueError("Duplicate feature or metadata column")
    frozen = _keys(system["frozen"], {"model_version", "policy_version", "reference_version",
                                      "reference_template_rows"}, "frozen")
    if not all(isinstance(frozen[k], str) and frozen[k] for k in ("model_version", "policy_version", "reference_version")):
        raise ValueError("Invalid frozen version")
    _number(frozen["reference_template_rows"], "reference template rows", positive=True, integer=True)
    det = _keys(system["detector"], {"evidence_chunk_rows", "policy_constraints"}, "detector")
    _number(det["evidence_chunk_rows"], "evidence chunk rows", positive=True, integer=True)
    constraints = _keys(det["policy_constraints"], {"max_attack_leakage", "max_false_malicious_alerts",
                                                    "desired_benign_acceptance", "max_unresolved_rate",
                                                    "rule_memory_bytes"}, "policy constraints")
    for name, value in constraints.items():
        _number(value, name, integer=name == "rule_memory_bytes")
        if name != "rule_memory_bytes" and value > 1:
            raise ValueError("Invalid policy rate")
    pattern = _keys(system["patterns"], {"feature_names"}, "patterns")
    if (not isinstance(pattern["feature_names"], list) or not pattern["feature_names"] or
            not all(isinstance(v, str) for v in pattern["feature_names"]) or
            len(pattern["feature_names"]) != len(set(pattern["feature_names"])) or
            not set(pattern["feature_names"]) <= set(names)):
        raise ValueError("Invalid pattern grouping")
    memory = _keys(system["memory"], {"verification_delay_windows", "verification_cost_usd",
                                      "max_verification_cost_usd", "min_independent_verifications",
                                      "memory_ttl_windows", "memory_max_entries"}, "memory")
    for name, value in memory.items():
        _number(value, name, positive=name in {"min_independent_verifications", "memory_ttl_windows", "memory_max_entries"},
                integer=name.endswith("windows") or name in {"min_independent_verifications", "memory_max_entries"})
    replay = _keys(system["simulation"], {"window_rows", "start_window", "max_windows"}, "simulation")
    _number(replay["window_rows"], "window rows", positive=True, integer=True)
    _number(replay["start_window"], "start window", integer=True)
    if replay["max_windows"] is not None:
        _number(replay["max_windows"], "max windows", positive=True, integer=True)
    output = _keys(system["output"], {"reviews_path", "memory_input_path"}, "output")
    for name, value in output.items():
        if value is not None:
            _path(value, name)
            if Path(value).suffix not in (".yaml", ".yml"):
                raise ValueError(f"{name} must be YAML")
    if cfg["provider"] not in ("mock", "openrouter"):
        raise ValueError("Unsupported agent provider")
    model_cfg = _keys(cfg["model"], {"model_env", "api_mode", "base_url_env",
                                     "decisions_url_env", "family_model_env", "api_key_env"}, "agent model")
    if model_cfg["api_mode"] not in ("chat", "decisions"):
        raise ValueError("Unsupported OpenRouter API mode")
    environment = _keys(cfg["environment"], {"file", "request_limit_env", "output_token_limit_env"}, "environment")
    for section, settings in (("agent model", model_cfg), ("environment", environment)):
        for name, value in settings.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"Invalid {section} {name}")
    generation = _keys(cfg["generation"], {"temperature", "max_output_tokens", "request_timeout_seconds"}, "generation")
    budgets = _keys(cfg["budgets"], {"max_requests", "max_tokens", "max_tool_calls", "max_cost_usd",
                                    "max_seconds", "max_input_tokens_reserve", "max_cost_per_request_usd"}, "budgets")
    values = read_environment(environment["file"])
    for target, key, name in ((budgets, "max_requests", environment["request_limit_env"]),
                              (generation, "max_output_tokens", environment["output_token_limit_env"])):
        if name in values and values[name]:
            try:
                target[key] = int(values[name])
            except ValueError as exc:
                raise ValueError(f"Invalid environment limit: {name}") from exc
    prompts = _keys(cfg["instructions"], {"system_prompt", "task_prompt"}, "instructions")
    if not all(isinstance(v, str) and v for v in prompts.values()):
        raise ValueError("Empty agent instruction")
    _number(generation["temperature"], "temperature")
    if generation["temperature"] > 2:
        raise ValueError("Unsupported generation option")
    investigation = _keys(cfg["investigation"], {"max_reasoning_steps", "max_tools_per_flow",
                                                   "max_investigations_per_window", "max_rationale_chars",
                                                   "input_framing_bytes"}, "investigation")
    _number(investigation["max_investigations_per_window"], "investigation cap", integer=True)
    decisions = _keys(cfg["decisions"], {"min_verdict_probability", "tool_instructions",
                                         "finalize_description", "verdict_instructions",
                                         "verdict_criteria"}, "decisions")
    _number(decisions["min_verdict_probability"], "decision probability", positive=True)
    if (decisions["min_verdict_probability"] > 1 or
            not all(isinstance(decisions[name], str) and decisions[name]
                    for name in ("tool_instructions", "finalize_description", "verdict_instructions")) or
            not isinstance(decisions["verdict_criteria"], dict) or
            set(decisions["verdict_criteria"]) != {"MALICIOUS", "BENIGN", "ABSTAIN"} or
            not all(isinstance(value, str) and value for value in decisions["verdict_criteria"].values())):
        raise ValueError("Invalid decision model settings")
    family = _keys(cfg["attack_family"], {"max_output_tokens", "max_reference_families",
                                           "system_prompt", "task_prompt"}, "attack family")
    _number(family["max_output_tokens"], "family output tokens", positive=True, integer=True)
    _number(family["max_reference_families"], "family source references", positive=True, integer=True)
    if not all(isinstance(family[name], str) and family[name] for name in ("system_prompt", "task_prompt")):
        raise ValueError("Invalid attack-family instructions")
    tools = _keys(cfg["tools"], {"enabled", "descriptions", "max_attack_families",
                                 "consistency_tolerance", "mock_first_tool"}, "tools")
    if (not isinstance(tools["enabled"], list) or not tools["enabled"] or
            not all(isinstance(v, str) for v in tools["enabled"]) or
            len(tools["enabled"]) != len(set(tools["enabled"])) or
            not set(tools["enabled"]) <= set(agent.TOOL_NAMES) or
            not isinstance(tools["descriptions"], dict) or set(tools["descriptions"]) != set(agent.TOOL_NAMES) or
            not all(isinstance(v, str) and v for v in tools["descriptions"].values()) or
            tools["mock_first_tool"] not in tools["enabled"]):
        raise ValueError("Invalid evidence tools")
    _number(tools["max_attack_families"], "attack family limit", positive=True, integer=True)
    _number(tools["consistency_tolerance"], "consistency tolerance")
    if model_cfg["api_mode"] == "decisions" and not {"inspect_attack_evidence", "retrieve_source_reference"} <= set(tools["enabled"]):
        raise ValueError("Decision-based family checking requires source-reference tools")
    verdict = _keys(cfg["verdict"], {"schema_name", "allowed", "required_fields",
                                     "require_evidence_for_decision"}, "verdict")
    if (not isinstance(verdict["schema_name"], str) or not verdict["schema_name"] or
            verdict["allowed"] != ["MALICIOUS", "BENIGN", "ABSTAIN"] or
            verdict["required_fields"] != ["verdict", "evidence_refs", "rationale"] or
            verdict["require_evidence_for_decision"] is not True):
        raise ValueError("Unsupported verdict protocol")
    agent.Limits(**limits(cfg))
    return system, cfg


def simulate(system: dict, cfg: dict, *, allow_paid: bool = False,
             ledger: agent.Ledger | None = None) -> Path:
    dataset = system["dataset"]
    paths = system["datasets"][dataset]
    source = paths["source_year"]
    output = Path(paths["memory"])
    if output.exists():
        raise FileExistsError(output)
    if cfg["provider"] == "openrouter" and not allow_paid:
        raise ValueError("Live mode requires explicit --allow-paid")
    if cfg["provider"] == "mock":
        backend = agent.MockBackend(cfg["tools"]["mock_first_tool"])
    else:
        model_cfg = cfg["model"]
        values = read_environment(cfg["environment"]["file"])
        model_id = values.get(model_cfg["model_env"], "")
        url_key = "base_url_env" if model_cfg["api_mode"] == "chat" else "decisions_url_env"
        base_url = values.get(model_cfg[url_key], "")
        api_key = values.get(model_cfg["api_key_env"], "")
        parsed = urlparse(base_url)
        if not model_id or not api_key or parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Live mode requires model, HTTPS base URL, and API key in .env")
        if model_cfg["api_mode"] == "decisions":
            family_model = values.get(model_cfg["family_model_env"], "")
            family_url = values.get(model_cfg["base_url_env"], "")
            family_endpoint = urlparse(family_url)
            if not family_model or family_endpoint.scheme != "https" or not family_endpoint.netloc:
                raise ValueError("Live attack-family checking requires model and HTTPS base URL in .env")
            backend = agent.JevBackend(model_id, base_url, api_key)
        else:
            endpoint = base_url.rstrip("/")
            if not endpoint.endswith("/chat/completions"):
                endpoint += "/chat/completions"
            backend = agent.OpenRouterBackend(model_id, cfg["generation"]["temperature"],
                                             endpoint, api_key, agent.verdict_schema(cfg["verdict"]))
    features_cfg = system["features"]
    names = data.feature_names(features_cfg)
    fingerprint = data.feature_fingerprint(features_cfg)
    frozen = system["frozen"]
    model = detector.load_model(source, paths["model"], fingerprint, frozen["model_version"])
    if model["partition"]["seed"] != system["source_partitions"]["partition_seed"]:
        raise ValueError("Frozen source partition seed differs")
    preprocessing = model["family_model"]["preprocessing"]
    if (preprocessing["quantizer_bins"] != features_cfg["preprocessing"]["quantizer_bins"] or
            preprocessing["missing_policy"] != features_cfg["preprocessing"]["missing_policy"]):
        raise ValueError("Frozen preprocessing differs")
    policy = detector.load_policy(paths["policy"], paths["model"], model, fingerprint, frozen["policy_version"])
    if policy["constraints"] != system["detector"]["policy_constraints"]:
        raise ValueError("Frozen routing constraints differ")
    reference = agent.load_reference(paths["reference"], paths["model"], paths["policy"], model,
                                     fingerprint, names, frozen["reference_version"],
                                     frozen["reference_template_rows"])
    if isinstance(backend, agent.JevBackend):
        endpoint = family_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
        backend.family_backend = agent.OpenRouterBackend(
            family_model, cfg["generation"]["temperature"], endpoint, api_key,
            agent.family_schema(model["family_names"], model["benign_code"]))
    hashes = tuple(detector.hash_file(paths[name]) for name in ("model", "policy", "reference"))
    memory_path = system["output"]["memory_input_path"]
    memory = (PatternMemory.load(memory_path, source, system["memory"], hashes)
              if memory_path else PatternMemory(source, system["memory"]))
    replay = system["simulation"]
    if memory_path and any(entry["updated_window"] >= replay["start_window"] for entry in memory.entries.values()):
        raise ValueError("Memory windows must precede the requested replay")
    review_path = system["output"]["reviews_path"]
    reviews = yaml.safe_load(Path(review_path).read_text()) if review_path else []
    if not isinstance(reviews, list) or not all(isinstance(review, dict) for review in reviews):
        raise ValueError("Review file must contain a YAML list")
    review_ids = {str(review.get("sample_id")) for review in reviews}
    if memory_path:
        for entry in memory.entries.values():
            for review in entry["reviews"]:
                if not isinstance(review, dict):
                    raise ValueError("Stored review is invalid")
                review_ids.add(str(review.get("sample_id")))
    allowed = (data.review_allowlist(paths["review_source"], source, model, review_ids)
               if review_ids else {})
    if memory_path:
        for entry in memory.entries.values():
            for review in entry["reviews"]:
                if allowed.get(review.get("sample_id")) != review.get("source_partition"):
                    raise ValueError("Stored review is outside source training/calibration")
    sample_ids, features, parent = data.load_sample(source, paths["flows"], system["sampling"]["ratio"],
                                                    system["sampling"]["seed"], features_cfg)
    if parent != model["source_artifact"]["data_fingerprint_sha256"]:
        raise ValueError("Flow PKL differs from frozen detector source")
    ledger = ledger or agent.Ledger(agent.Limits(**limits(cfg)))
    for offset, start in enumerate(range(0, len(features), replay["window_rows"])):
        if replay["max_windows"] is not None and offset >= replay["max_windows"]:
            break
        window = replay["start_window"] + offset
        for review in reviews:
            if review.get("available_window") == window:
                memory.review(review, window, allowed)
        stop = min(start + replay["window_rows"], len(features))
        controller.process_window(source, window, sample_ids[start:stop], features[start:stop], model,
                                  policy, reference, memory, backend, ledger, system, cfg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        yaml.safe_dump(memory.document(*hashes), stream, sort_keys=True)
    return output
