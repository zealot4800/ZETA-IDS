"""Fit attack trees, certified benign regions, and membership gates."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from .loao_data import _partitions


def _wilson_upper(errors: int, count: int, confidence: float) -> float:
    if count == 0:
        return 1.0
    z = norm.ppf(confidence)
    p = errors / count
    z2 = z * z
    return (p + z2 / (2 * count) + z * math.sqrt(
        p * (1 - p) / count + z2 / (4 * count * count))) / (1 + z2 / count)


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = [("numeric", SimpleImputer(strategy="median", keep_empty_features=True), numeric)]
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def _distances(values: np.ndarray, median: np.ndarray, scale: np.ndarray, cap: float) -> np.ndarray:
    return np.mean(np.minimum(np.abs((values - median) / scale), cap), axis=1)


def _compact_distances(values: np.ndarray, median: np.ndarray, scale: np.ndarray,
                       cap: float, top_features: int) -> np.ndarray:
    deviations = np.minimum(np.abs((values - median) / scale), cap)
    count = min(top_features, deviations.shape[1])
    return np.mean(np.partition(deviations, -count, axis=1)[:, -count:], axis=1)


def _region_distances(model: dict, values: np.ndarray, median: np.ndarray,
                      scale: np.ndarray) -> np.ndarray:
    if model.get("top_features"):
        return _compact_distances(values, median, scale, model["distance_cap"],
                                  model["top_features"])
    return _distances(values, median, scale, model["distance_cap"])


def _score(model: dict, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pre, tree = model["preprocessor"], model["tree"]
    transformed = pre.transform(x)
    leaves = tree.apply(transformed)
    known_family = tree.predict(transformed)
    numbers = pre.named_transformers_["numeric"].transform(x[model["numeric"]])
    inside = np.zeros(len(x), dtype=bool)
    raw_inside = np.zeros(len(x), dtype=bool)
    for leaf, (median, scale, threshold) in model["raw_regions"].items():
        idx = np.flatnonzero(leaves == leaf)
        if len(idx):
            distances = _region_distances(model, numbers[idx], median, scale)
            raw_inside[idx] = distances <= threshold
            if leaf in model["regions"]:
                inside[idx] = distances <= model["regions"][leaf][2]
    return leaves, inside, raw_inside, known_family


def _fit_model(data: dict, train: np.ndarray, validation: np.ndarray, cfg: dict) -> tuple[dict, float]:
    start = time.perf_counter()
    x, y, family = data["x"], data["y"], data["family"]
    if not np.any(train & (y == 0)) or len(set(family[train & (y == 1)])) < 2:
        raise ValueError("Training requires benign and at least two known attack classes")
    pre = _preprocessor(data["numeric"], data["categorical"])
    pre.fit(x.loc[train])
    attack_train = train & (y == 1)
    tree = DecisionTreeClassifier(**cfg["tree"], random_state=cfg["seed"])
    tree.fit(pre.transform(x.loc[attack_train]), family[attack_train])
    benign_train = train & (y == 0)
    xb = x.loc[benign_train]
    benign_leaves = tree.apply(pre.transform(xb))
    numbers = pre.named_transformers_["numeric"].transform(xb[data["numeric"]])
    global_spread = np.percentile(numbers, 75, axis=0) - np.percentile(numbers, 25, axis=0)
    global_spread = np.where(global_spread > 0, global_spread, 1.0)
    regions = {}
    for leaf in np.unique(benign_leaves):
        subset = numbers[benign_leaves == leaf]
        if len(subset) < cfg["regions"]["min_train_benign"]:
            continue
        median = np.median(subset, axis=0)
        spread = np.percentile(subset, 75, axis=0) - np.percentile(subset, 25, axis=0)
        spread = np.where(spread > 0, spread, global_spread)
        distances = _distances(subset, median, spread, cfg["regions"]["distance_cap"])
        threshold = float(np.quantile(distances, cfg["regions"]["benign_quantile"]))
        regions[int(leaf)] = (median, spread, threshold)
    model = {"preprocessor": pre, "tree": tree, "numeric": data["numeric"],
             "regions": regions, "raw_regions": regions,
             "distance_cap": cfg["regions"]["distance_cap"]}
    val_leaves, val_inside, _, _ = _score(model, x.loc[validation])
    val_y = y[validation]
    rules_attack, rules_benign = set(), set()
    safety = cfg["calibration"]
    for leaf in np.unique(val_leaves):
        mask = val_leaves == leaf
        attack = int(np.sum(val_y[mask] == 1))
        benign = int(np.sum(val_y[mask] == 0))
        if (attack >= safety["min_validation_support"] and
                _wilson_upper(benign, attack + benign, safety["confidence"]) <=
                safety["max_attack_leaf_benign_share"]):
            rules_attack.add(int(leaf))
        region = mask & val_inside
        region_attack = int(np.sum(val_y[region] == 1))
        region_benign = int(np.sum(val_y[region] == 0))
        if (region_benign >= safety["min_validation_support"] and
                _wilson_upper(region_attack, region_attack + region_benign,
                              safety["confidence"]) <=
                safety["max_benign_region_attack_share"]):
            rules_benign.add(int(leaf))
    model["attack_rules"] = rules_attack
    model["benign_rules"] = rules_benign
    return model, time.perf_counter() - start


def _split_validation(data: dict, validation: np.ndarray, cfg: dict,
                      salt: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    indices = np.flatnonzero(validation)
    fraction = cfg["safe_regions"]["certification_fraction"]
    if salt is None:
        salt = cfg["source_validation"]["certification_salt"]
    if not 0 < fraction < 1:
        raise ValueError("certification_fraction must be between zero and one")
    _, parts = _partitions(data["x"].iloc[indices], cfg["seed"] + salt,
                           {"train": 1 - fraction, "validation": fraction, "test": 0},
                           cfg["partition_buckets"])
    tuning = np.zeros(len(validation), dtype=bool)
    certification = np.zeros(len(validation), dtype=bool)
    tuning[indices[parts == 0]] = True
    certification[indices[parts == 1]] = True
    if not tuning.any() or not certification.any():
        raise ValueError("Source validation split is empty")
    return tuning, certification


def _fit_safe_model(data: dict, train: np.ndarray, tuning: np.ndarray,
                    certification: np.ndarray, cfg: dict, base: dict) -> tuple[dict, float]:
    """Select on tuning data, then certify the complete frozen policy once."""
    start = time.perf_counter()
    x, y = data["x"], data["y"]
    pre, tree = base["preprocessor"], base["tree"]
    numeric = data["numeric"]
    options = cfg["safe_regions"]
    quantiles = options["quantiles"]
    if not quantiles or sorted(set(quantiles)) != list(quantiles) or not all(
            0 < q <= 1 for q in quantiles):
        raise ValueError("safe_regions.quantiles must be increasing and unique in (0, 1]")
    if options["top_features"] < 1:
        raise ValueError("safe_regions.top_features must be positive")
    cap = cfg["regions"]["distance_cap"]
    benign_train = train & (y == 0)
    xb = x.loc[benign_train]
    benign_leaves = tree.apply(pre.transform(xb))
    numbers = pre.named_transformers_["numeric"].transform(xb[numeric])
    global_spread = np.percentile(numbers, 75, axis=0) - np.percentile(numbers, 25, axis=0)
    global_spread = np.where(global_spread > 0, global_spread, 1.0)
    raw_regions, thresholds, train_support = {}, {}, {}
    for leaf in np.unique(benign_leaves):
        subset = numbers[benign_leaves == leaf]
        if len(subset) < cfg["regions"]["min_train_benign"]:
            continue
        median = np.median(subset, axis=0)
        spread = np.percentile(subset, 75, axis=0) - np.percentile(subset, 25, axis=0)
        spread = np.where(spread > 0, spread, global_spread)
        distances = _compact_distances(subset, median, spread, cap, options["top_features"])
        cutoffs = np.quantile(distances, quantiles)
        key = int(leaf)
        raw_regions[key] = (median, spread, float(cutoffs[-1]))
        thresholds[key] = cutoffs
        train_support[key] = len(subset)
    model = {"preprocessor": pre, "tree": tree, "numeric": numeric,
             "regions": {}, "raw_regions": raw_regions, "distance_cap": cap,
             "top_features": options["top_features"],
             "attack_rules": base["attack_rules"].copy(), "benign_rules": set(),
             "region_audit": {}, "policy_status": "infeasible"}

    tune_leaves, _, _, _ = _score(model, x.loc[tuning])
    tune_y = y[tuning]
    tune_numbers = pre.named_transformers_["numeric"].transform(x.loc[tuning, numeric])
    cert_y = y[certification]
    attack_train = train & (y == 1)
    xa = x.loc[attack_train]
    attack_train_leaves = tree.apply(pre.transform(xa))
    attack_train_numbers = pre.named_transformers_["numeric"].transform(xa[numeric])
    attack_tune_count = int((tune_y == 1).sum())
    attack_cert_count = int((cert_y == 1).sum())
    confidence = cfg["calibration"]["confidence"]
    budget = options["known_attack_leakage_budget"]
    if not 0 < budget < 1 or not attack_tune_count or not attack_cert_count:
        raise ValueError("Safe calibration needs a positive budget and known attacks in both subsets")
    max_tune_errors = -1
    for errors in range(attack_tune_count + 1):
        if _wilson_upper(errors, attack_tune_count, confidence) > budget:
            break
        max_tune_errors = errors
    model["tuning_known_attack_count"] = attack_tune_count
    model["cert_known_attack_count"] = attack_cert_count
    model["tuning_max_errors"] = max_tune_errors
    model["budget"] = budget
    model["cert_upper"] = _wilson_upper(0, attack_cert_count, confidence)
    model["cert_errors"] = 0
    model["cert_benign_accepted"] = 0
    if max_tune_errors < 0:
        model["policy_status"] = "infeasible_insufficient_source_support"
        return model, time.perf_counter() - start

    candidate_data = {}
    for leaf, (median, scale, _) in raw_regions.items():
        tune_mask = tune_leaves == leaf
        if not tune_mask.any():
            continue
        tune_dist = _compact_distances(tune_numbers[tune_mask], median, scale, cap,
                                       options["top_features"])
        tune_labels = tune_y[tune_mask]
        train_mask = attack_train_leaves == leaf
        train_dist = _compact_distances(attack_train_numbers[train_mask], median, scale,
                                        cap, options["top_features"])
        choices = []
        for q, threshold in zip(quantiles, thresholds[leaf]):
            tune_inside = tune_dist <= threshold
            tune_benign = int(np.sum(tune_inside & (tune_labels == 0)))
            tune_attack = int(np.sum(tune_inside & (tune_labels == 1)))
            choices.append({"quantile": q, "threshold": float(threshold),
                            "tune_benign": tune_benign, "tune_attack": tune_attack,
                            "train_attack": int(np.sum(train_dist <= threshold))})
        candidate_data[leaf] = choices
        model["region_audit"][leaf] = {"train_benign": int(train_support[leaf]),
                                       "candidate_count": len(choices),
                                       "raw_tune_attack": choices[-1]["tune_attack"],
                                       "raw_tune_benign": choices[-1]["tune_benign"],
                                       "raw_train_attack": choices[-1]["train_attack"]}

    # Leaves are disjoint, so this small dynamic program exactly maximizes tuning
    # benign acceptance for the predeclared source leakage budget.
    states = {0: (0, {})}
    for leaf, choices in candidate_data.items():
        if leaf in model["attack_rules"]:
            continue
        next_states = states.copy()
        for cost, (benefit, selected) in states.items():
            for choice in choices:
                if choice["tune_benign"] < cfg["calibration"]["min_validation_support"]:
                    continue
                if choice["train_attack"] > options["max_training_attack_overlap"]:
                    continue
                new_cost = cost + choice["tune_attack"]
                if new_cost > max_tune_errors:
                    continue
                new_benefit = benefit + choice["tune_benign"]
                if new_benefit > next_states.get(new_cost, (-1, None))[0]:
                    next_states[new_cost] = (new_benefit, {**selected, leaf: choice})
        states = next_states
    _, selected = max(states.values(), key=lambda state: state[0])
    tune_errors = sum(choice["tune_attack"] for choice in selected.values())
    # Score certification data only after the complete policy is frozen.
    cert_leaves, _, _, _ = _score(model, x.loc[certification])
    cert_numbers = pre.named_transformers_["numeric"].transform(x.loc[certification, numeric])
    for leaf, choice in selected.items():
        median, scale, _ = raw_regions[leaf]
        mask = cert_leaves == leaf
        distances = _compact_distances(cert_numbers[mask], median, scale, cap,
                                       options["top_features"])
        inside = distances <= choice["threshold"]
        choice["cert_benign"] = int(np.sum(inside & (cert_y[mask] == 0)))
        choice["cert_attack"] = int(np.sum(inside & (cert_y[mask] == 1)))
    cert_errors = sum(choice["cert_attack"] for choice in selected.values())
    cert_benign = sum(choice["cert_benign"] for choice in selected.values())
    upper = _wilson_upper(cert_errors, attack_cert_count, confidence)
    model.update(tuning_errors=tune_errors, tuning_benign_accepted=sum(
        choice["tune_benign"] for choice in selected.values()),
        cert_errors=cert_errors, cert_upper=upper, cert_benign_accepted=cert_benign)
    if selected and upper <= budget:
        model["regions"] = {leaf: (*raw_regions[leaf][:2], choice["threshold"])
                            for leaf, choice in selected.items()}
        model["benign_rules"] = set(selected)
        model["policy_status"] = "certified"
    elif not selected:
        model["policy_status"] = "infeasible_no_supported_region"
    else:
        model["policy_status"] = "infeasible_certification_failed"
    for leaf, choice in selected.items():
        model["region_audit"][leaf]["selected"] = choice
    return model, time.perf_counter() - start


def _membership_distances(model: dict, x: pd.DataFrame, leaves: np.ndarray,
                          inside: np.ndarray) -> np.ndarray:
    distances = np.full(len(x), np.inf)
    if not model.get("membership"):
        return distances
    numbers = model["preprocessor"].named_transformers_["numeric"].transform(
        x[model["numeric"]])
    for leaf, profile in model["membership"].items():
        idx = np.flatnonzero(inside & (leaves == leaf))
        if len(idx):
            transformed = np.arcsinh((numbers[idx] - profile["median"]) / profile["scale"])
            nearest, _ = profile["tree"].query(transformed, k=profile["neighbours"], workers=1)
            distances[idx] = nearest[:, -1] if nearest.ndim == 2 else nearest
    return distances


def _membership_pass(model: dict, x: pd.DataFrame, leaves: np.ndarray,
                     inside: np.ndarray) -> np.ndarray:
    quantile = model.get("membership_quantile")
    if quantile is None:
        return np.ones(len(x), dtype=bool)
    distances = _membership_distances(model, x, leaves, inside)
    passed = np.zeros(len(x), dtype=bool)
    for leaf, profile in model["membership"].items():
        mask = leaves == leaf
        passed[mask] = distances[mask] <= profile["radii"][quantile]
    return passed


def _build_membership(data: dict, model: dict, train: np.ndarray,
                      tuning: np.ndarray, cfg: dict) -> tuple[dict, float]:
    """Represent supported benign modes with bounded source-only prototypes."""
    start = time.perf_counter()
    options = cfg["membership"]
    if not 1 <= options["neighbours"] <= options["prototypes_per_region"]:
        raise ValueError("membership needs at least as many prototypes as neighbours")
    quantiles = options["radius_quantiles"]
    if (not quantiles or list(quantiles) != sorted(set(quantiles)) or
            not all(0 < q < 1 for q in quantiles)):
        raise ValueError("membership.radius_quantiles must be increasing in (0, 1)")
    x, y = data["x"], data["y"]
    pre, tree = model["preprocessor"], model["tree"]
    numeric = data["numeric"]
    benign_train = train & (y == 0)
    xb = x.loc[benign_train]
    benign_leaves = tree.apply(pre.transform(xb))
    train_numbers = pre.named_transformers_["numeric"].transform(xb[numeric])
    profiles = {}
    for leaf in model["benign_rules"]:
        subset = train_numbers[benign_leaves == leaf]
        if len(subset) < options["neighbours"]:
            continue
        median, scale, _ = model["regions"][leaf]
        rng = np.random.default_rng(cfg["seed"] + int(leaf))
        chosen = rng.choice(len(subset), min(len(subset), options["prototypes_per_region"]),
                            replace=False)
        points = np.arcsinh((subset[chosen] - median) / scale)
        profiles[leaf] = {"median": median, "scale": scale,
                          "tree": cKDTree(points), "neighbours": options["neighbours"],
                          "prototype_count": len(points), "radii": {}}
    candidate = {**model, "membership": profiles}
    benign_tune = tuning & (y == 0)
    x_tune = x.loc[benign_tune]
    leaves, inside, _, _ = _score(model, x_tune)
    distances = _membership_distances(candidate, x_tune, leaves, inside)
    for leaf, profile in profiles.items():
        local = distances[inside & (leaves == leaf)]
        if len(local) < cfg["calibration"]["min_validation_support"]:
            del profile["tree"]
            continue
        profile["radii"] = {q: float(np.quantile(local, q))
                            for q in options["radius_quantiles"]}
    candidate["membership"] = {leaf: profile for leaf, profile in profiles.items()
                               if "tree" in profile}
    candidate["membership_prototype_count"] = sum(
        p["prototype_count"] for p in candidate["membership"].values())
    candidate["membership_memory_mb"] = sum(
        p["tree"].data.nbytes + p["median"].nbytes + p["scale"].nbytes
        for p in candidate["membership"].values()) / (1024 * 1024)
    return candidate, time.perf_counter() - start
