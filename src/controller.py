"""Route a window and investigate eligible unresolved patterns."""

from __future__ import annotations

import numpy as np

from . import agent, data, detector, patterns
from .memory import PatternMemory


def process_window(source: str, window: int, sample_ids: np.ndarray, features: np.ndarray,
                   model: dict, policy: dict, reference: dict, memory: PatternMemory,
                   backend, ledger: agent.Ledger, system: dict, cfg: dict) -> None:
    chunk_rows = system["detector"]["evidence_chunk_rows"]
    feature_names = data.feature_names(system["features"])
    routes, observed = detector.route(model, policy, features, chunk_rows)
    unresolved = np.flatnonzero(routes == detector.UNRESOLVED)
    if not len(unresolved):
        return
    signatures = patterns.signatures(model, features[unresolved],
                                     system["patterns"]["feature_names"], feature_names, source)
    seen = set()
    investigations = 0
    cap = cfg["investigation"]["max_investigations_per_window"]
    for position, pattern in zip(unresolved, signatures):
        if pattern in seen:
            continue
        seen.add(pattern)
        if (memory.resolve(pattern, window) or investigations >= cap or ledger.exceeded or
                ledger.requests >= ledger.limits.max_requests or ledger.remaining_seconds() <= 0 or
                ledger.cost_usd + ledger.limits.max_cost_per_request_usd > ledger.limits.max_cost_usd):
            continue
        tools = agent.EvidenceTools(model, reference, features[position], cfg["tools"],
                                    feature_names, chunk_rows, observed, int(position))
        result = agent.investigate(tools, backend, ledger, cfg)
        investigations += 1
        if result["verdict"] in ("MALICIOUS", "BENIGN"):
            sample_id = str(sample_ids[position])
            memory.provisional(pattern, sample_id, result["verdict"],
                               result["evidence_refs"], window,
                               result.get("attack_family_candidate"),
                               result.get("attack_family_evidence_refs"),
                               result.get("family_failure_reason"))
