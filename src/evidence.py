"""Flow observations and references used by the controller."""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd


def scalar(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None if math.isnan(value) else str(value)
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("Unsupported flow feature type")
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def feature_batch(frame: pd.DataFrame, positions):
    """Extract native columns once, then serialize Python row tuples."""
    names = list(frame.columns)
    return [dict(zip(names, (scalar(value) for value in row)))
            for row in frame.iloc[positions].itertuples(index=False, name=None)]


def build_source(data: dict, train: np.ndarray, cfg: dict) -> dict:
    raw = data["raw_x"]
    baseline_rows = raw.loc[train & (data["y"] == 0)]
    options = cfg["evidence"]
    statistics = {}
    for name in raw.columns:
        column = baseline_rows[name]
        if name in data["numeric"]:
            finite = pd.to_numeric(column, errors="raise").replace([np.inf, -np.inf], np.nan)
            clean = finite.dropna()
            statistics[name] = {
                "mean": scalar(clean.mean()),
                "quantiles": {str(q): scalar(clean.quantile(q)) for q in options["quantiles"]}}
        else:
            statistics[name] = {}
    return {"feature_names": list(raw.columns),
            "baseline": statistics,
            "limitations": options["limitations"],
            "missing_feature_handling": options["missing_feature_handling"]}


def flow_records(data: dict, dataplane, positions, batch_size: int = 256, support=None):
    """Stream preprocessed positions without per-flow pandas indexing."""
    positions = np.asarray(positions, dtype=int)
    raw_values = data["raw_x"].to_numpy(dtype=object)
    names = list(data["raw_x"].columns)
    for start in range(0, len(positions), batch_size):
        batch = positions[start:start + batch_size]
        decisions, families, leaves, probabilities = dataplane.predict(batch)
        contexts = [None] * len(batch)
        if support is not None:
            advice = dataplane.support_probabilities[batch]
            # A strong second-model contradiction punts a formerly confident
            # tree match to the controller, including rare attacks in benign leaves.
            conflict = (((decisions == "BENIGN") & (advice >= support.options["attack_min_probability"])) |
                        ((decisions == "ATTACK") & (advice <= support.options["benign_max_probability"])))
            decisions[conflict] = "UNRESOLVED"
            deferred = np.flatnonzero(decisions == "UNRESOLVED")
            if len(deferred):
                prepared = support.contexts(batch[deferred], leaves[deferred], advice[deferred])
                for offset, context in zip(deferred, prepared):
                    contexts[int(offset)] = context
        features = [dict(zip(names, (scalar(value) for value in row))) for row in raw_values[batch]]
        for offset, position in enumerate(batch):
            position, leaf = int(position), int(leaves[offset])
            yield {
                "flow_id": digest([data["name"], int(data["row_positions"][position])]),
                "dataset": data["name"],
                "features": features[offset],
                "controller_context": contexts[offset],
                "dataplane_decision": str(decisions[offset]),
                "known_family": str(families[offset]),
                "dataplane_evidence": {"tree_leaf": leaf,
                                       "p_attack": float(probabilities[offset])},
            }


def investigation_facts(flow: dict, source: dict, cfg: dict) -> dict:
    """Calculate immutable observations; the LLM selects IDs, never writes facts."""
    semantics = cfg["feature_semantics"]["datasets"].get(flow.get("dataset"), {})
    facts = {}
    for index, name in enumerate(source["feature_names"]):
        value = flow["features"][name]
        if value is None or value in ("inf", "-inf", "nan"):
            continue
        fact = {"feature": name, "value": value,
                "meaning": semantics.get(name, cfg["feature_semantics"]["default"])}
        baseline = source["baseline"][name]
        if type(value) in (int, float) and math.isfinite(value):
            mean = baseline.get("mean")
            if type(mean) in (int, float) and math.isfinite(mean):
                fact["baseline_mean"] = mean
                fact["direction"] = "above" if value > mean else "below" if value < mean else "equal"
                fact["comparison"] = f"Observed {value} is {fact['direction']} source benign mean {mean}."
            fact["source_quantiles"] = baseline.get("quantiles", {})
        if name == "flag" and flow.get("dataset") == "kdd":
            fact["meaning"] = {"SF": "Normal establishment and termination; not SYN flood or scan.",
                               "S0": "Connection attempt with no reply.",
                               "REJ": "Connection attempt rejected."}.get(value, fact["meaning"])
        facts[f"E{index + 1}"] = fact
    return facts
