"""Label-free flow records and references built only from permitted source rows."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math

import numpy as np
import pandas as pd

from .loao import _path_conditions
from .loao_policies import _benign_region_scores, _conditional_pass, _predict


def scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if not isinstance(value, (str, int, float, bool)):
        raise ValueError("Unsupported flow feature type")
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def feature_values(frame: pd.DataFrame, position: int) -> dict:
    return {name: scalar(frame[name].iloc[position]) for name in frame.columns}


def source_sample(x: pd.DataFrame, y: np.ndarray, indices: np.ndarray,
                  per_class: int, seed: int) -> np.ndarray:
    """One row per feature pattern, excluding ambiguous patterns; source input only."""
    groups = pd.util.hash_pandas_object(x.iloc[indices], index=False).to_numpy()
    rows = pd.DataFrame({"group": groups, "label": y[indices], "position": indices})
    unambiguous = rows.groupby("group")["label"].transform("nunique") == 1
    rows = rows.loc[unambiguous].drop_duplicates("group")
    rng = np.random.default_rng(seed)
    selected = []
    for label in (0, 1):
        pool = rows.loc[rows["label"] == label, "position"].to_numpy()
        selected.extend(rng.permutation(pool)[:per_class].tolist())
    return rng.permutation(selected).astype(int)


def build_source(data: dict, train: np.ndarray, cfg: dict) -> dict:
    raw = data["raw_x"]
    if raw is None:
        raise ValueError("Controller requires original permitted features")
    banned = {str(name).strip().lower() for name in cfg["evidence"]["forbidden_features"]}
    # These fixed label fields must remain excluded even if a config is weakened.
    banned |= {"label", "original_label", "binary_label", "mapped_label", "mapped_class_id",
               "attack_cat", "class_id", "target", "heldout_attack", "split", "difficulty"}
    if any(str(name).strip().lower() in banned for name in raw.columns):
        raise ValueError("Label or evaluation metadata in controller feature schema")
    selected = source_sample(data["x"], data["y"], np.flatnonzero(train),
                             cfg["jev"]["few_shot_per_class"], cfg["general"]["seed"])
    examples = [{"features": feature_values(raw, int(i)),
                 "binary": "ATTACK" if data["y"][i] else "BENIGN"} for i in selected]
    baseline_rows = raw.loc[train & (data["y"] == 0)]
    if baseline_rows.empty:
        raise ValueError("Source baseline requires benign source-training rows")
    options = cfg["evidence"]
    statistics = {}
    for name in raw.columns:
        column = baseline_rows[name]
        if name in data["numeric"]:
            finite = pd.to_numeric(column, errors="raise").replace([np.inf, -np.inf], np.nan)
            clean = finite.dropna()
            values = {"count": int(clean.size), "missing": int(finite.isna().sum()),
                      "min": scalar(clean.min()), "max": scalar(clean.max()),
                      "mean": scalar(clean.mean()), "std": scalar(clean.std()),
                      "quantiles": {str(q): scalar(clean.quantile(q)) for q in options["quantiles"]}}
            statistics[name] = {key: values[key] for key in options["numeric_statistics"]}
        else:
            counts = column.value_counts().head(options["categorical_top_k"])
            statistics[name] = {"count": int(column.notna().sum()),
                                "missing": int(column.isna().sum()),
                                "top_values": [{"value": scalar(k), "count": int(v)}
                                               for k, v in counts.items()]}
    groups = {name: {"description": spec["description"], "features": [
        feature for feature in raw.columns if any(fnmatch.fnmatchcase(feature, pattern)
                                                 for pattern in spec["patterns"])]}
              for name, spec in options["feature_groups"].items()}
    return {"feature_names": list(raw.columns), "examples": examples,
            "baseline": statistics, "feature_groups": groups,
            "limitations": options["limitations"],
            "missing_feature_handling": options["missing_feature_handling"]}


def unresolved_indices(data: dict, model: dict, mask: np.ndarray) -> np.ndarray:
    positions = np.flatnonzero(mask)
    if not positions.size:
        return positions
    decisions = _predict(model, data["x"].iloc[positions])[0]
    return positions[decisions == "UNRESOLVED"]


def flow_record(data: dict, model: dict, position: int) -> dict:
    x = data["x"].iloc[[position]]
    decisions, _, _, leaves, inside, raw_inside, membership = _predict(model, x)
    leaf = int(leaves[0])
    cap = model.get("benign_acceptance_max_score", np.inf)
    evidence = {"tree_leaf": leaf, "attack_rule_match": leaf in model["attack_rules"],
                "benign_rule_match": leaf in model["benign_rules"],
                "inside_benign_region": bool(inside[0]),
                "inside_raw_region": bool(raw_inside[0]),
                "membership_pass": bool(membership[0]),
                "conditional_pass": bool(_conditional_pass(model, x, leaves)[0]),
                "acceptance_limit_pass": bool(_benign_region_scores(model, x, leaves)[0] <= cap),
                "rule_conditions": _path_conditions(model, leaf)}
    return {"flow_id": digest([data["name"], int(data["row_positions"][position])]),
            "dataset": data["name"], "features": feature_values(data["raw_x"], position),
            "first_layer_decision": str(decisions[0]), "first_layer_evidence": evidence}


def public_flow(flow: dict, source: dict) -> dict:
    """Explicit allowlist: IDs, ground truth, fold names and row metadata stay local."""
    if set(flow["features"]) != set(source["feature_names"]):
        raise ValueError("Flow schema differs from permitted source schema")
    return {"features": flow["features"], "first_layer_evidence": flow["first_layer_evidence"],
            "limitations": source["limitations"],
            "missing_feature_handling": source["missing_feature_handling"]}


def evidence_tool(name: str, arguments: dict, flow: dict, source: dict, cfg: dict) -> dict:
    if name not in cfg["advanced_llm"]["allowed_tools"]:
        raise ValueError("Tool is not permitted")
    if name == "retrieve_attack_knowledge":
        families = arguments.get("families")
        knowledge = cfg["attack_knowledge"]
        if not families:
            return {"families": list(knowledge["families"]),
                    "unknown_handling": knowledge["unknown_handling"]}
        if len(families) > cfg["evidence"]["max_families_per_tool"]:
            raise ValueError("Too many requested families")
        return {family: knowledge["families"].get(family) for family in families}
    features = arguments.get("features")
    if features is None:
        features = source["feature_names"] if name == "inspect_flow" else []
    if name != "inspect_flow" or arguments.get("features") is not None:
        if len(features) > cfg["evidence"]["max_features_per_tool"]:
            raise ValueError("Too many requested features")
    if not set(features) <= set(source["feature_names"]):
        raise ValueError("Unavailable or forbidden feature")
    if name == "inspect_flow":
        result = public_flow(flow, source)
        result["features"] = {feature: flow["features"][feature] for feature in features}
        return result
    if name == "compare_source_baseline":
        return {feature: {"observed": flow["features"][feature],
                          "source_benign_training": source["baseline"][feature]}
                for feature in features}
    raise ValueError("Unknown evidence tool")
