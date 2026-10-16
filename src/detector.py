"""Read-only Phase 3.1 detector and Phase 7 routing policy."""

from __future__ import annotations

import hashlib

import joblib
import numpy as np

KNOWN_MALICIOUS, TRUSTED_BENIGN, UNRESOLVED = 0, 1, 2
RULE_DTYPE = np.dtype([("key", "<i4"), ("action", "u1")], align=False)


def hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        digest.update(stream.read())
    return digest.hexdigest()


def load_model(year: str, path: str, fingerprint: str, version: str) -> dict:
    model = joblib.load(path)
    if (not isinstance(model, dict) or
            model.get("artifact_version") != version or
            model.get("source_dataset") != year or
            model.get("feature_schema_fingerprint_sha256") != fingerprint):
        raise ValueError("Frozen detector or source feature schema differs")
    return model


def _numeric_bytes(model: dict) -> int:
    arrays = []
    for tree in (model["binary_tree"], model["family_model"]["tree"]):
        state = tree.tree_.__getstate__()
        arrays.extend((state["nodes"], state["values"]))
    arrays.extend(model["family_model"]["preprocessing"]["quantile_edges"])
    arrays.extend(v for v in model["family_model"]["attack_support"].values()
                  if isinstance(v, np.ndarray))
    arrays.extend(v for v in model["family_model"]["benign_regions"].values()
                  if isinstance(v, np.ndarray))
    return int(sum(array.nbytes for array in arrays))


def load_policy(path: str, model_path: str, model: dict,
                fingerprint: str, version: str) -> dict:
    policy = joblib.load(path)
    if (not isinstance(policy, dict) or
            policy.get("artifact_version") != version or
            policy.get("feature_schema_fingerprint_sha256") != fingerprint or
            policy.get("model_sha256") != hash_file(model_path) or
            policy.get("source_dataset") != model["source_dataset"] or
            policy.get("source_fingerprint_sha256") !=
            model["source_artifact"]["data_fingerprint_sha256"] or
            policy.get("partition_index_sha256") != model["partition"]["partition_index_sha256"]):
        raise ValueError("Frozen routing policy differs from detector")
    rules = policy.get("rules")
    size = _numeric_bytes(model)
    if (not isinstance(rules, np.ndarray) or rules.dtype != RULE_DTYPE or
            np.any(np.diff(rules["key"].astype(np.int64)) <= 0) or
            not np.isin(rules["action"], [KNOWN_MALICIOUS, TRUSTED_BENIGN]).all() or
            policy.get("rule_memory_bytes") != rules.nbytes or
            policy.get("base_model_numeric_bytes") != size or
            policy.get("total_numeric_bytes") != size + rules.nbytes):
        raise ValueError("Frozen routing rule table differs")
    return policy


def _pack_codes(q: np.ndarray, columns: np.ndarray, bins: int) -> np.ndarray:
    packed = np.zeros(len(q), dtype=np.uint64)
    for column in columns:
        packed = packed * np.uint64(bins) + q[:, column].astype(np.uint64)
    return packed


def _lookup(sorted_keys: np.ndarray, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position = np.searchsorted(sorted_keys, keys)
    found = (position < len(sorted_keys)) & (
        sorted_keys[np.minimum(position, len(sorted_keys) - 1)] == keys)
    return position, found


def _family_evidence(model: dict, safe: np.ndarray, valid: np.ndarray,
                     chunk_rows: int) -> dict:
    q = np.empty(safe.shape, dtype=np.uint8)
    for column, edge in enumerate(model["preprocessing"]["quantile_edges"]):
        q[:, column] = np.searchsorted(edge, safe[:, column], side="right")
    support = model["attack_support"]
    leaf = model["tree"].apply(safe)
    bins = int(model["preprocessing"]["quantizer_bins"])
    keys = leaf.astype(np.uint64) * np.uint64(support["radix"]) + _pack_codes(q, support["support_columns"], bins)
    position, found = _lookup(support["keys"], keys)
    safe_position = np.minimum(position, len(support["keys"]) - 1)
    regions = model["benign_regions"]
    position, matched = _lookup(regions["codes"], _pack_codes(q, regions["anchor_columns"], bins))
    region = np.full(len(q), -1, dtype=np.int16)
    for start in range(0, len(q), chunk_rows):
        stop = min(start + chunk_rows, len(q))
        local = np.minimum(position[start:stop], len(regions["codes"]) - 1)
        inside = (matched[start:stop] & (q[start:stop] >= regions["lower"][local]).all(axis=1) &
                  (q[start:stop] <= regions["upper"][local]).all(axis=1))
        region[start:stop][inside] = local[inside].astype(np.int16)
    region[~valid] = -1
    return {"valid": valid, "predicted_family_code": support["leaf_family_code"][leaf],
            "prototype_attack_count": np.where(found, support["attack_count"][safe_position], 0),
            "prototype_benign_count": np.where(found, support["benign_count"][safe_position], 0),
            "region": region}


def evidence(model: dict, features: np.ndarray, chunk_rows: int) -> dict:
    valid = np.isfinite(features).all(axis=1)
    safe = features if valid.all() else np.where(np.isfinite(features), features, 0).astype(np.float32)
    score = model["binary_tree"].predict_proba(safe)[:, 1].astype(np.float32)
    score[~valid] = np.nan
    family = _family_evidence(model["family_model"], safe, valid, chunk_rows)
    return {"valid": valid, "attack_score": score, "family": family,
            "predicted_family_code": family["predicted_family_code"]}


def decision_keys(model: dict, features: np.ndarray, observed: dict) -> np.ndarray:
    nodes = model["binary_tree"].tree_.node_count
    safe = features if observed["valid"].all() else np.where(np.isfinite(features), features, 0).astype(np.float32)
    leaf = model["binary_tree"].apply(safe)
    return ((observed["family"]["region"].astype(np.int32) + 1) * nodes + leaf).astype(np.int32)


def route(model: dict, policy: dict, features: np.ndarray,
          chunk_rows: int) -> tuple[np.ndarray, dict]:
    observed = evidence(model, features, chunk_rows)
    keys = decision_keys(model, features, observed)
    observed["decision_keys"] = keys
    rules = policy["rules"]
    result = np.full(len(keys), UNRESOLVED, dtype=np.uint8)
    if len(rules):
        position = np.searchsorted(rules["key"], keys)
        hit = position < len(rules)
        hit[hit] &= rules["key"][position[hit]] == keys[hit]
        hit &= observed["valid"]
        result[hit] = rules["action"][position[hit]]
    return result, observed
