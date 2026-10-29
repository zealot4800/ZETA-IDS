"""Run the adaptive data plane, binary LLM and deep investigation controller."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import yaml

from .adaptive_memory import AdaptiveMemory
from .binary_detector import binary_decision, combine_decision, probability, request_body
from .data import load_dataset
from .dataplane import DataPlane
from .evidence import build_source, digest, flow_records, investigation_facts
from .investigation import investigate, render
from .llm import Ollama
from .reporting import GroundTruthReport
from .support import ControllerSupport


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for stage in ("binary_llm", "investigation_llm"):
        options = cfg[stage]
        for key in ("model", "base_url"):
            if value := os.environ.get(options.get(key + "_env", "")):
                options[key] = value
    return cfg


def resolve(flow: dict, source: dict, cfg: dict, client, memory=None) -> dict:
    started = time.perf_counter()
    requests_before = sum(client.requests.values())
    record = {"flow_id": flow["flow_id"], "dataset": flow["dataset"],
              "dataplane_decision": flow["dataplane_decision"],
              "dataplane_evidence": flow["dataplane_evidence"],
              "decision_source": "dataplane", "final_decision": flow["dataplane_decision"],
              "binary_decision": "NOT_RUN", "p_attack": None, "attack_family": None,
              "pattern_id": None, "pattern_status": None, "memory_audit": False,
              "investigation": None, "deep_reused": False, "error": None}
    context = flow.get("controller_context") or {}
    record["controller_support"] = {key: value for key, value in context.items()
                                    if key != "source_examples"}
    record["decision_reason"] = "bootstrap_tree"
    if flow["dataplane_decision"] == "ATTACK":
        record["attack_family"] = cfg["family_map"][flow["dataset"]].get(flow["known_family"])
    elif flow["dataplane_decision"] == "UNRESOLVED":
        match = (memory.lookup(flow) if memory else
                 {"matched": False, "audit": False, "pattern_id": None, "selection": None})
        record.update(pattern_id=match["pattern_id"], memory_audit=match["audit"],
                      pattern_status=match.get("status"))
        cached_report = None
        if match["selection"]:
            try:
                cached_report = render(match["selection"], investigation_facts(flow, source, cfg), cfg)
            except (KeyError, TypeError, ValueError):
                match["selection"] = None
        if match["matched"] and not match["audit"]:
            record.update(decision_source="adaptive_memory", final_decision=match["decision"],
                          p_attack=match["p_attack"], decision_reason="corroborated_memory")
            if match["decision"] == "ATTACK" and cached_report:
                record.update(investigation=cached_report, deep_reused=True,
                              attack_family=cached_report["attack_family"])
        else:
            record["decision_source"] = "llm_controller"
            reply = client.call("binary_llm", request_body(flow, source, cfg),
                                use_cache=not match["audit"])
            if reply["error"]:
                record.update(binary_decision="ERROR", final_decision="ERROR", error=reply["error"],
                              decision_reason="llm_request_failed")
            else:
                try:
                    p_attack = probability(reply["body"])
                    decision = binary_decision(p_attack, cfg["binary_llm"]["thresholds"])
                    final, reason = combine_decision(decision, context)
                    record.update(binary_decision=decision, final_decision=final,
                                  decision_reason=reason, p_attack=p_attack)
                    if reason == "training_support_resolved_abstention":
                        record["decision_source"] = "training_support"
                except (KeyError, IndexError, TypeError, ValueError):
                    record.update(binary_decision="ERROR", final_decision="ERROR",
                                  error="malformed_binary_response", decision_reason="llm_response_unreadable")
            selection = None
            if record["final_decision"] in cfg["adaptive_memory"]["deep_investigation_decisions"]:
                if cached_report and not match["audit"]:
                    record.update(investigation=cached_report, deep_reused=True)
                    selection = match["selection"]
                else:
                    result = investigate(flow, source, cfg, client, fresh=match["audit"])
                    record.update(investigation=result["report"], error=result["error"])
                    selection = result["selection"]
                if record["investigation"]:
                    record["attack_family"] = record["investigation"]["attack_family"]
            if memory and record["binary_decision"] == "ERROR" and match["audit"]:
                memory.suspend(flow)
                record["pattern_status"] = "demoted"
            elif memory and record["binary_decision"] != "ERROR":
                learned = memory.observe(flow, record["binary_decision"], record["p_attack"], selection)
                record.update(pattern_id=learned["pattern_id"], pattern_status=learned["status"])
    record["llm_requests"] = sum(client.requests.values()) - requests_before
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    return record


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/controller.yaml")
    parser.add_argument("--dataset", required=True)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-flows", type=int)
    limits.add_argument("--all-flows", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="preview inputs without Ollama or state writes")
    args = parser.parse_args()
    cfg = load_config(args.config)
    datasets = yaml.safe_load((root / cfg["general"]["datasets_config"]).read_text())["datasets"]
    if args.dataset not in datasets:
        parser.error("Choose a configured dataset: " + ", ".join(datasets))
    limit = (None if args.all_flows else args.max_flows if args.max_flows is not None
             else cfg["general"]["max_flows"])
    if limit is not None and limit < 1:
        parser.error("--max-flows must be positive")
    data = load_dataset(args.dataset, datasets[args.dataset], cfg, root)
    source = build_source(data, data["train"], cfg)
    if args.dry_run:
        print(json.dumps({"mode": "preview", "dataset": args.dataset,
                          "feature_count": len(source["feature_names"]),
                          "bootstrap_rows": int(data["train"].sum()),
                          "incoming_rows": int(data["stream"].sum()),
                          "max_flows": limit, "api_requests": 0}))
        return
    dataplane = DataPlane(data, cfg)
    support = ControllerSupport(data, dataplane, cfg)
    with (root / datasets[args.dataset]["path"]).open("rb") as stream:
        dataset_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    version = digest([dataset_hash, cfg, datasets[args.dataset],
                      {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (root / "src").glob("*.py")}])
    client = Ollama(cfg)
    memory = (AdaptiveMemory(root / cfg["adaptive_memory"]["database"], cfg, source,
                             args.dataset, version) if cfg["adaptive_memory"]["enabled"] else None)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    directory = (args.output_dir or root / cfg["output"]["directory"]) / args.dataset / run_id
    directory.mkdir(parents=True, exist_ok=False)
    sources, decisions, errors = Counter(), Counter(), Counter()
    reasons = Counter()
    truth_report = GroundTruthReport()
    audits = deep_reused = processed = 0
    started = time.perf_counter()
    summary = {"dataset": args.dataset, "detector_version": version,
               "results_directory": str(directory)}
    positions = np.flatnonzero(data["stream"])
    if limit is not None:
        positions = positions[:limit]
    try:
        (directory / "manifest.json").write_text(json.dumps({
            **summary, "configuration": cfg, "incoming_rows": int(data["stream"].sum())}, indent=2))
        with (directory / "flows.jsonl").open("x") as output:
            for position, flow in zip(positions, flow_records(data, dataplane, positions, support=support)):
                record = resolve(flow, source, cfg, client, memory)
                record["detector_version"] = version
                # Labels enter accounting only after the controller and memory
                # update have finished. They are never attached to the flow.
                label, family = int(data["y"][position]), str(data["family"][position])
                truth_report.record(record, label, family)
                record["ground_truth"] = {"binary": "ATTACK" if label else "BENIGN", "family": family}
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                processed += 1
                sources[record["decision_source"]] += 1
                decisions[record["final_decision"]] += 1
                reasons[record["decision_reason"]] += 1
                audits += record["memory_audit"]
                deep_reused += record["deep_reused"]
                if record["error"]:
                    errors[record["error"]] += 1
                if processed % cfg["general"]["progress_every"] == 0:
                    print(json.dumps({"event": "progress", "dataset": args.dataset,
                                      "processed": processed, "requested": len(positions),
                                      "percent": round(100 * processed / len(positions), 1),
                                      "decision_sources": dict(sources),
                                      "elapsed_seconds": round(time.perf_counter() - started, 1)}),
                          file=sys.stderr, flush=True)
    finally:
        if memory:
            memory.close()
        summary.update(processed=processed, decision_sources=dict(sources),
                       decision_reasons=dict(reasons),
                       requested_flows=len(positions), completed=processed == len(positions),
                       ground_truth_metrics=truth_report.summary(),
                       final_decisions=dict(decisions), errors=dict(errors),
                       api_requests=dict(client.requests), tokens=dict(client.tokens),
                       cache_hits=client.cache_hits, memory_audits=audits,
                       deep_reports_reused=deep_reused,
                       elapsed_seconds=time.perf_counter() - started)
        (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
