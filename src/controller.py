"""Run the adaptive data plane, binary LLM and deep investigation controller."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

from .adaptive_memory import AdaptiveMemory
from .binary_detector import binary_decision, combine_decision, probability, request_body
from .data import dataset_specs, load_dataset
from .dataplane import DataPlane
from .evidence import build_source, digest, file_sha256, flow_records, investigation_facts
from .investigation import investigate, render
from .llm import Ollama
from .reporting import GroundTruthReport, log_event, progress_heartbeat
from .results import atomic_json, new_run_id, optional_log, print_result, update_results
from .support import ControllerSupport


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for stage in ("binary_llm", "investigation_llm"):
        options = cfg[stage]
        for key in ("model", "base_url"):
            if value := os.environ.get(options.get(key + "_env", "")):
                options[key] = value
    return cfg


def new_record(flow: dict) -> dict:
    context = flow.get("controller_context") or {}
    return {"flow_id": flow["flow_id"], "dataset": flow["dataset"],
            "dataplane_decision": flow["dataplane_decision"],
            "dataplane_evidence": flow["dataplane_evidence"],
            "decision_source": "dataplane", "final_decision": flow["dataplane_decision"],
            "binary_decision": "NOT_RUN", "p_attack": None, "attack_family": None,
            "pattern_id": None, "pattern_status": None, "memory_audit": False,
            "investigation": None, "deep_reused": False, "error": None,
            "controller_support": {key: value for key, value in context.items()
                                   if key != "source_examples"},
            "decision_reason": "bootstrap_tree"}


def reusable_report(match: dict, flow: dict, source: dict, cfg: dict) -> dict | None:
    """Rebuild a remembered report from this flow's facts, if it is still valid."""
    if not match["selection"]:
        return None
    try:
        return render(match["selection"], investigation_facts(flow, source, cfg), cfg, flow["dataset"])
    except (KeyError, TypeError, ValueError):
        return None


def adjudicate(record: dict, flow: dict, source: dict, cfg: dict, client, fresh: bool) -> None:
    """Ask the binary LLM, then combine its answer with training support."""
    record["decision_source"] = "llm_controller"
    context = flow.get("controller_context") or {}
    if cfg["binary_llm"].get("support_first", False) and not context.get("supported"):
        # Without supported context the gate abstains whatever the LLM says,
        # so the opt-in support-first ordering skips the binary request.
        record.update(final_decision="UNCERTAIN", decision_reason="insufficient_training_support")
        return
    reply = client.call("binary_llm", request_body(flow, source, cfg), use_cache=not fresh)
    if reply["error"]:
        record.update(binary_decision="ERROR", final_decision="ERROR", error=reply["error"],
                      decision_reason="llm_request_failed")
        return
    try:
        p_attack = probability(reply["body"])
        decision = binary_decision(p_attack, cfg["binary_llm"]["thresholds"])
        final, reason = combine_decision(decision, flow.get("controller_context") or {})
    except (KeyError, IndexError, TypeError, ValueError):
        record.update(binary_decision="ERROR", final_decision="ERROR",
                      error="malformed_binary_response", decision_reason="llm_response_unreadable")
        return
    record.update(binary_decision=decision, final_decision=final,
                  decision_reason=reason, p_attack=p_attack)
    if reason == "training_support_resolved_abstention":
        record["decision_source"] = "training_support"


def attach_investigation(record: dict, flow: dict, source: dict, cfg: dict, client,
                         match: dict, cached_report: dict | None) -> dict | None:
    """Add a deep report when required; return the evidence selection behind it."""
    if record["final_decision"] not in cfg["adaptive_memory"]["deep_investigation_decisions"]:
        return None
    if cached_report and not match["audit"]:
        record.update(investigation=cached_report, deep_reused=True)
        selection = match["selection"]
    else:
        result = investigate(flow, source, cfg, client, fresh=match["audit"])
        record.update(investigation=result["report"], error=result["error"])
        selection = result["selection"]
    if record["investigation"]:
        record["attack_family"] = record["investigation"]["attack_family"]
    return selection


def resolve_deferred(record: dict, flow: dict, source: dict, cfg: dict, client, memory) -> None:
    match = (memory.lookup(flow) if memory else
             {"matched": False, "audit": False, "pattern_id": None, "selection": None})
    record.update(pattern_id=match["pattern_id"], memory_audit=match["audit"],
                  pattern_status=match.get("status"))
    cached_report = reusable_report(match, flow, source, cfg)
    if match["matched"] and not match["audit"]:
        record.update(decision_source="adaptive_memory", final_decision=match["decision"],
                      p_attack=match["p_attack"], decision_reason="corroborated_memory")
    else:
        adjudicate(record, flow, source, cfg, client, fresh=match["audit"])
    selection = attach_investigation(record, flow, source, cfg, client, match, cached_report)
    if not memory:
        return
    if record["binary_decision"] == "ERROR" and match["audit"]:
        memory.suspend(flow)
        record["pattern_status"] = "demoted"
    elif record["binary_decision"] not in ("ERROR", "NOT_RUN"):
        learned = memory.observe(flow, record["binary_decision"], record["p_attack"], selection)
        record.update(pattern_id=learned["pattern_id"], pattern_status=learned["status"])


def resolve(flow: dict, source: dict, cfg: dict, client, memory=None) -> dict:
    started = time.perf_counter()
    requests_before = sum(client.requests.values())
    record = new_record(flow)
    if flow["dataplane_decision"] == "ATTACK":
        record["attack_family"] = cfg["family_map"][flow["dataset"]].get(flow["known_family"])
    elif flow["dataplane_decision"] == "UNRESOLVED":
        resolve_deferred(record, flow, source, cfg, client, memory)
    record["llm_requests"] = sum(client.requests.values()) - requests_before
    record["latency_ms"] = (time.perf_counter() - started) * 1000
    return record


class RunStats:
    """Running totals for one AEGIS replay."""

    def __init__(self):
        self.truth = GroundTruthReport()
        self.sources, self.decisions, self.reasons, self.errors = Counter(), Counter(), Counter(), Counter()
        self.audits = self.deep_reused = self.processed = 0
        self.latency_total = self.latency_max = 0.0

    def add(self, record: dict) -> None:
        self.latency_total += record["latency_ms"]
        self.latency_max = max(self.latency_max, record["latency_ms"])
        self.processed += 1
        self.sources[record["decision_source"]] += 1
        self.decisions[record["final_decision"]] += 1
        self.reasons[record["decision_reason"]] += 1
        self.audits += record["memory_audit"]
        self.deep_reused += record["deep_reused"]
        if record["error"]:
            self.errors[record["error"]] += 1


def parse_args(root: Path) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/controller.yaml")
    parser.add_argument("--dataset", required=True)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument("--max-flows", type=int, help="Optionally limit evaluation flows")
    limits.add_argument("--all-flows", action="store_true", help="Use every evaluation flow (default)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--save-details", action="store_true",
                        help="Also save per-flow logs and a run manifest")
    parser.add_argument("--dry-run", action="store_true", help="preview inputs without Ollama or state writes")
    return parser, parser.parse_args()


def detector_version(root: Path, dataset_hash: str, cfg: dict, spec: dict) -> str:
    return digest([dataset_hash, cfg, spec,
                   {path.name: file_sha256(path) for path in (root / "src").glob("*.py")}])


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser, args = parse_args(root)
    cfg = load_config(args.config)
    datasets = dataset_specs(cfg, root)
    if args.dataset not in datasets:
        parser.error("Choose a configured dataset: " + ", ".join(datasets))
    limit = args.max_flows
    if limit is not None and limit < 1:
        parser.error("--max-flows must be positive")
    spec = datasets[args.dataset]
    started = time.perf_counter()
    data = load_dataset(args.dataset, spec, cfg, root)
    source = build_source(data, cfg)
    if args.dry_run:
        print(json.dumps({"mode": "preview", "dataset": args.dataset,
                          "dataset_size": len(data['x']),
                          "feature_count": len(source["feature_names"]),
                          "bootstrap_rows": int(data["train"].sum()),
                          "incoming_rows": int(data["stream"].sum()),
                          "max_flows": limit, "api_requests": 0}))
        return
    dataplane = DataPlane(data, cfg)
    support = ControllerSupport(data, dataplane, cfg)
    dataset_hash = file_sha256(root / spec["path"])
    version = detector_version(root, dataset_hash, cfg, spec)
    client = Ollama(cfg)
    memory = (AdaptiveMemory(root / cfg["adaptive_memory"]["database"], cfg, source,
                             args.dataset, version) if cfg["adaptive_memory"]["enabled"] else None)
    run_id = new_run_id()
    output_root = args.output_dir or root / cfg['output']['directory']
    directory = output_root / 'details' / 'aegis' / args.dataset / run_id if args.save_details else None
    if directory:
        directory.mkdir(parents=True, exist_ok=False)
    stats = RunStats()
    summary = {"method": "aegis", "dataset": args.dataset, "dataset_size": len(data['x']),
               "task": "binary", "detector_version": version, "run_id": run_id,
               "model": cfg['binary_llm']['model'], "seed": cfg['general']['seed'],
               "investigation_model": cfg['investigation_llm']['model'],
               "parameters": {"tree": cfg['dataplane']['tree'],
                              "train_fraction": cfg['dataplane']['train_fraction'],
                              "binary_thresholds": cfg['binary_llm']['thresholds']},
               "dataset_sha256": dataset_hash, "dataset_path": spec['path'],
               "configuration_sha256": digest(cfg), "latency_scope": "flow_resolution",
               "training_flows": int(data['train'].sum()),
               "incoming_flows": int(data['stream'].sum()),
               "results_file": str(output_root / 'summary.json')}
    failure = None
    positions = np.flatnonzero(data["stream"])
    if limit is not None:
        positions = positions[:limit]

    def progress_snapshot():
        active = client.active_request
        waiting = ({key: value for key, value in active.items() if key != 'started'} |
                   {'elapsed_seconds': round(time.perf_counter() - active['started'], 1)}
                   if active else None)
        return {'event': 'progress', 'method': 'aegis', 'dataset': args.dataset,
                'processed': stats.processed, 'requested': len(positions),
                'percent': round(100 * stats.processed / len(positions), 1) if len(positions) else 100,
                'elapsed_seconds': round(time.perf_counter() - started, 1),
                'api_requests': dict(client.requests), 'waiting_for_llm': waiting}

    def write_summary(completed=False):
        elapsed = time.perf_counter() - started
        processed = stats.processed
        summary.update(processed=processed, decision_sources=dict(stats.sources),
                       decision_reasons=dict(stats.reasons), requested_flows=len(positions),
                       completed=completed, failure=failure,
                       ground_truth_metrics=stats.truth.summary(),
                       final_decisions=dict(stats.decisions), errors=dict(stats.errors),
                       api_requests=dict(client.requests), tokens=dict(client.tokens),
                       cache_hits=client.cache_hits, memory_audits=stats.audits,
                       deep_reports_reused=stats.deep_reused, elapsed_seconds=elapsed,
                       mean_latency_ms=stats.latency_total / processed if processed else None,
                       max_latency_ms=stats.latency_max if processed else None,
                       flows_per_second=processed / elapsed if elapsed else None)
        if directory:
            atomic_json(directory / 'summary.json', summary)
        update_results(output_root, summary)

    write_summary()
    last_saved = time.perf_counter()
    progress_seconds = cfg['general'].get('progress_seconds', 15)
    try:
        if directory:
            atomic_json(directory / 'manifest.json', {**summary, 'configuration': cfg})
        with progress_heartbeat(progress_snapshot, progress_seconds), \
                optional_log(directory / 'flows.jsonl' if directory else None) as output:
            for position, flow in zip(positions, flow_records(data, dataplane, positions, support)):
                record = resolve(flow, source, cfg, client, memory)
                record["detector_version"] = version
                # Labels enter accounting only after the controller and memory
                # update have finished. They are never attached to the flow.
                label, family = int(data["y"][position]), str(data["family"][position])
                stats.truth.record(record, label, family)
                record["ground_truth"] = {"binary": "ATTACK" if label else "BENIGN", "family": family}
                if output:
                    output.write(json.dumps(record, allow_nan=False) + "\n")
                stats.add(record)
                now = time.perf_counter()
                if (stats.processed == 1 or stats.processed == len(positions) or
                        stats.processed % cfg["general"]["progress_every"] == 0 or
                        now - last_saved >= progress_seconds):
                    write_summary()
                    last_saved = now
                    log_event(progress_snapshot())
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        if memory:
            memory.close()
        write_summary(completed=failure is None and stats.processed == len(positions))
    print_result(summary)


if __name__ == "__main__":
    main()
