"""Select source-only membership and conditional benign policies."""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.metrics import mutual_info_score

from .loao_data import _partitions
from .loao_model import (
    _build_membership, _fit_model, _fit_safe_model, _membership_distances,
    _membership_pass, _region_distances, _score, _split_validation,
)


def _select_membership(data: dict, cfg: dict, outer_attack: str,
                       outer_train: np.ndarray, outer_validation: np.ndarray) -> tuple[float | None, dict, float]:
    """Choose a global strictness using only source proxy-unknown families."""
    start = time.perf_counter()
    family, y, x = data["family"], data["y"], data["x"]
    options = cfg["membership"]
    if not 0 < options["min_benign_retention"] <= 1 or options["min_proxy_support"] < 1:
        raise ValueError("Invalid membership source-selection constraints")
    candidates = [None, *options["radius_quantiles"]]
    proxy_records = []
    for proxy in cfg["expected_attacks"][data["name"]]:
        if proxy == outer_attack:
            continue
        inner_train = outer_train & (family != proxy)
        inner_validation = outer_validation & (family != proxy)
        tuning, certification = _split_validation(data, inner_validation, cfg)
        base, _ = _fit_model(data, inner_train, tuning, cfg)
        safe, _ = _fit_safe_model(data, inner_train, tuning, certification, cfg, base)
        member, _ = _build_membership(data, safe, inner_train, tuning, cfg)
        evaluation = outer_validation & ((y == 0) | (family == proxy))
        truth = family[evaluation]
        x_eval = x.loc[evaluation]
        leaves, inside, _, _ = _score(safe, x_eval)
        distances = _membership_distances(member, x_eval, leaves, inside)
        allowed = inside & np.isin(leaves, list(safe["benign_rules"])) & ~np.isin(
            leaves, list(safe["attack_rules"]))
        benign = y[evaluation] == 0
        unknown = truth == proxy
        if not unknown.any() or not benign.any():
            raise ValueError(f"Inner proxy {proxy} lacks evaluation support")
        scores = {}
        for quantile in candidates:
            if quantile is None:
                accepted = allowed
            else:
                passed = np.zeros(len(leaves), dtype=bool)
                for leaf, profile in member["membership"].items():
                    mask = leaves == leaf
                    passed[mask] = distances[mask] <= profile["radii"][quantile]
                accepted = allowed & passed
            scores[str(quantile)] = {"proxy_bypassed": int(np.sum(accepted & unknown)),
                                     "proxy_uabr_pct": 100 * np.mean(accepted[unknown]),
                                     "benign_accepted": int(np.sum(accepted & benign)),
                                     "benign_acceptance_pct": 100 * np.mean(accepted[benign])}
        proxy_records.append({"proxy": proxy, "proxy_count": int(unknown.sum()),
                              "benign_count": int(benign.sum()),
                              "safe_policy_status": safe["policy_status"],
                              "safe_source_cert_upper_pct": 100 * safe["cert_upper"],
                              "candidate_scores": scores})
    supported = [r for r in proxy_records if r["proxy_count"] >= options["min_proxy_support"]]
    if not supported:
        supported = proxy_records
    candidate_summary = []
    baseline_benign = float(np.mean([r["candidate_scores"]["None"]["benign_acceptance_pct"]
                                     for r in proxy_records]))
    for quantile in candidates:
        key = str(quantile)
        worst = max(r["candidate_scores"][key]["proxy_uabr_pct"] for r in supported)
        mean_proxy = float(np.mean([r["candidate_scores"][key]["proxy_uabr_pct"]
                                    for r in supported]))
        mean_benign = float(np.mean([r["candidate_scores"][key]["benign_acceptance_pct"]
                                     for r in proxy_records]))
        retention = float(mean_benign / baseline_benign) if baseline_benign else 0.0
        candidate_summary.append({"quantile": quantile, "worst_proxy_uabr_pct": worst,
                                  "mean_proxy_uabr_pct": mean_proxy,
                                  "mean_benign_acceptance_pct": mean_benign,
                                  "benign_retention": retention,
                                  "meets_retention": retention >= options["min_benign_retention"]})
    feasible = [r for r in candidate_summary if r["meets_retention"]]
    if not feasible:
        chosen = candidate_summary[0]  # The no-gate policy preserves coverage.
        status = "no_policy_meets_retention"
    else:
        chosen = min(feasible, key=lambda r: (r["worst_proxy_uabr_pct"],
                                              r["mean_proxy_uabr_pct"],
                                              -r["mean_benign_acceptance_pct"]))
        status = "selected"
    return chosen["quantile"], {"status": status, "chosen": chosen,
                                "candidates": candidate_summary,
                                "proxy_folds": proxy_records,
                                "supported_proxies": [r["proxy"] for r in supported]}, time.perf_counter() - start


def _conditional_profiles(data: dict, model: dict, train: np.ndarray,
                          tuning: np.ndarray, cfg: dict) -> dict[int, dict]:
    """Describe joint source-benign modes in already certified, accepted regions."""
    options = cfg["conditional"]
    cuts = options["cut_quantiles"]
    if (list(cuts) != sorted(set(cuts)) or not cuts or
            not all(0 < q < 1 for q in cuts)):
        raise ValueError("conditional.cut_quantiles must be increasing in (0, 1)")
    if options["max_subregions_per_region"] < 1:
        raise ValueError("conditional.max_subregions_per_region must be positive")
    numeric = data["numeric"]
    pre = model["preprocessor"].named_transformers_["numeric"]
    x_train, x_tune = data["x"].loc[train], data["x"].loc[tuning]
    train_dec, _, _, train_leaves, _, _, _ = _predict(model, x_train)
    tune_dec, _, _, tune_leaves, _, _, _ = _predict(model, x_tune)
    train_y, tune_y = data["y"][train], data["y"][tuning]
    train_values = pre.transform(x_train[numeric])
    tune_values = pre.transform(x_tune[numeric])
    profiles = {}
    for leaf in sorted(model["benign_rules"]):
        source = (train_leaves == leaf) & (train_dec == "BENIGN") & (train_y == 0)
        support = train_values[source]
        summaries, features = {}, []
        if not len(support):
            profiles[int(leaf)] = {"source_support": 0, "dispersion": {},
                                   "reason": "no source-benign membership support"}
            continue
        for j, name in enumerate(numeric):
            column = support[:, j]
            edges = np.unique(np.quantile(column, cuts))
            q05, q25, q50, q75, q95 = np.quantile(column, [0.05, 0.25, 0.5, 0.75, 0.95])
            summaries[name] = {"q05": float(q05), "median": float(q50),
                               "q95": float(q95), "iqr": float(q75 - q25)}
            if len(edges) < 2:
                continue
            bins = np.searchsorted(edges, column, side="right")
            probabilities = np.bincount(bins, minlength=len(edges) + 1) / len(bins)
            positive = probabilities[probabilities > 0]
            entropy = float(-np.sum(positive * np.log(positive)))
            if entropy >= options["min_feature_entropy"]:
                features.append((entropy, j, edges, bins))
        if len(support) < options["min_source_region_support"]:
            profiles[int(leaf)] = {
                "source_support": len(support), "dispersion": summaries,
                "reason": "insufficient source-benign support"}
            continue
        features.sort(key=lambda item: (-item[0], numeric[item[1]]))
        features = features[:options["max_pair_features"]]
        pair = None
        for i, first in enumerate(features):
            for second in features[i + 1:]:
                information = float(mutual_info_score(first[3], second[3]))
                key = (information, -first[1], -second[1])
                if pair is None or key > pair[0]:
                    pair = (key, first, second)
        if pair is None or pair[0][0] < options["min_joint_information"]:
            profiles[int(leaf)] = {
                "source_support": len(support), "dispersion": summaries,
                "reason": "no sufficiently supported joint feature relationship"}
            continue
        _, first, second = pair
        indices = (first[1], second[1])
        edges = (first[2], second[2])
        width = len(edges[1]) + 1

        def cells(values: np.ndarray) -> np.ndarray:
            return (np.searchsorted(edges[0], values[:, indices[0]], side="right") * width +
                    np.searchsorted(edges[1], values[:, indices[1]], side="right"))

        train_cell, tune_cell = cells(train_values), cells(tune_values)
        count = (len(edges[0]) + 1) * width
        train_region = (train_leaves == leaf) & (train_dec == "BENIGN")
        tune_region = (tune_leaves == leaf) & (tune_dec == "BENIGN")
        counts = {}
        for label, cell, region, binary in (
                ("train_benign", train_cell, train_region, train_y == 0),
                ("train_attack", train_cell, train_region, train_y == 1),
                ("tune_benign", tune_cell, tune_region, tune_y == 0),
                ("tune_attack", tune_cell, tune_region, tune_y == 1)):
            counts[label] = np.bincount(cell[region & binary], minlength=count)
        profiles[int(leaf)] = {"features": indices, "edges": edges,
                               "information": pair[0][0], "source_support": len(support),
                               "dispersion": summaries, "counts": counts,
                               "cell_count": count}
    return profiles


def _with_conditional(model: dict, profiles: dict[int, dict],
                      minimum_support: int | None, cfg: dict) -> dict:
    if minimum_support is None:
        return model
    options = cfg["conditional"]
    selected = {}
    for leaf, profile in profiles.items():
        if "counts" not in profile:
            continue
        counts = profile["counts"]
        valid = np.flatnonzero(
            (counts["train_benign"] >= minimum_support) &
            (counts["tune_benign"] >= options["min_tuning_cell_support"]) &
            (counts["train_attack"] == 0) & (counts["tune_attack"] == 0))
        ranked = sorted(valid, key=lambda cell: (-counts["train_benign"][cell], int(cell)))
        selected[leaf] = {"features": profile["features"], "edges": profile["edges"],
                          "keep": np.asarray(
                              ranked[:options["max_subregions_per_region"]], dtype=np.int16)}
    return {**model, "conditional": selected, "conditional_minimum_support": minimum_support}


def _conditional_pass(model: dict, x: pd.DataFrame, leaves: np.ndarray) -> np.ndarray:
    profiles = model.get("conditional")
    passed = np.ones(len(x), dtype=bool)
    if not profiles:
        return passed
    numbers = model["preprocessor"].named_transformers_["numeric"].transform(
        x[model["numeric"]])
    for leaf, profile in profiles.items():
        positions = np.flatnonzero(leaves == leaf)
        if not len(positions):
            continue
        first, second = profile["features"]
        low, high = profile["edges"]
        cell = (np.searchsorted(low, numbers[positions, first], side="right") *
                (len(high) + 1) +
                np.searchsorted(high, numbers[positions, second], side="right"))
        passed[positions] = np.isin(cell, profile["keep"])
    return passed


def _select_conditional(data: dict, cfg: dict, outer_attack: str,
                        outer_train: np.ndarray, source_tuning: np.ndarray,
                        membership_quantile: float | None) -> tuple[int | None, dict, float]:
    """Use nested proxy-unknown folds; never consult the outer test partition."""
    start = time.perf_counter()
    options = cfg["conditional"]
    supports = options["min_train_cell_support_candidates"]
    if (not supports or list(supports) != sorted(set(supports)) or
            any(s < 1 for s in supports) or
            not 0 < options["min_benign_retention"] <= 1):
        raise ValueError("Invalid conditional source-selection constraints")
    candidates = [None, *supports]
    family, y, x = data["family"], data["y"], data["x"]
    source_indices = np.flatnonzero(outer_train)
    split_cfg = cfg["source_validation"]
    train_fraction = split_cfg["conditional_proxy_train_fraction"]
    _, proxy_parts = _partitions(
        x.iloc[source_indices], cfg["seed"] + split_cfg["conditional_proxy_salt"],
        {"train": train_fraction, "validation": 1 - train_fraction, "test": 0},
        cfg["partition_buckets"])
    proxy_train = np.zeros(len(y), dtype=bool)
    proxy_validation = np.zeros(len(y), dtype=bool)
    proxy_train[source_indices[proxy_parts == 0]] = True
    proxy_validation[source_indices[proxy_parts == 1]] = True
    if not proxy_train.any() or not proxy_validation.any() or not source_tuning.any():
        raise ValueError("Conditional proxy train, validation, or evaluation is empty")
    proxy_records = []
    for proxy in cfg["expected_attacks"][data["name"]]:
        if proxy == outer_attack:
            continue
        inner_train = proxy_train & (family != proxy)
        inner_validation = proxy_validation & (family != proxy)
        proxy_evaluation = source_tuning
        truth = family[proxy_evaluation]
        benign = y[proxy_evaluation] == 0
        unknown = truth == proxy
        if not unknown.any() or not benign.any():
            proxy_records.append({"proxy": proxy, "proxy_count": int(unknown.sum()),
                                  "benign_count": int(benign.sum()),
                                  "status": "insufficient_disjoint_evaluation_support"})
            continue
        tuning, certification = _split_validation(
            data, inner_validation, cfg,
            salt=split_cfg["conditional_certification_salt"])
        base, _ = _fit_model(data, inner_train, tuning, cfg)
        safe, _ = _fit_safe_model(data, inner_train, tuning, certification, cfg, base)
        member, _ = _build_membership(data, safe, inner_train, tuning, cfg)
        member["membership_quantile"] = membership_quantile
        profiles = _conditional_profiles(data, member, inner_train, tuning, cfg)
        x_eval = x.loc[proxy_evaluation]
        baseline, _, _, leaves, _, _, _ = _predict(member, x_eval)
        scores = {}
        for support in candidates:
            candidate = _with_conditional(member, profiles, support, cfg)
            passed = _conditional_pass(candidate, x_eval, leaves)
            accepted = (baseline == "BENIGN") & passed
            scores[str(support)] = {
                "proxy_bypassed": int(np.sum(accepted & unknown)),
                "proxy_uabr_pct": 100 * np.mean(accepted[unknown]),
                "benign_accepted": int(np.sum(accepted & benign)),
                "benign_acceptance_pct": 100 * np.mean(accepted[benign]),
                "known_attack_bypassed": int(np.sum(
                    accepted & (y[proxy_evaluation] == 1) & ~unknown)),
                "additional_unresolved": int(np.sum((baseline == "BENIGN") & ~passed)),
                "subregions": (sum(len(p["keep"]) for p in candidate.get(
                    "conditional", {}).values()) if support is not None else 0)}
        proxy_records.append({"proxy": proxy, "proxy_count": int(unknown.sum()),
                              "benign_count": int(benign.sum()), "scores": scores})
    scored = [r for r in proxy_records if "scores" in r]
    if not scored:
        return None, {"status": "no_disjoint_proxy_support",
                      "chosen": {"worst_proxy_uabr_pct": math.nan,
                                 "benign_retention": 1.0},
                      "candidates": [], "proxy_folds": proxy_records,
                      "supported_proxies": []}, time.perf_counter() - start
    supported = [r for r in scored if r["proxy_count"] >= options["min_proxy_support"]]
    if not supported:
        supported = scored
    baseline_benign = np.mean([
        r["scores"]["None"]["benign_acceptance_pct"] for r in scored])
    summary = []
    for support in candidates:
        scores = [r["scores"][str(support)] for r in scored]
        worst = max(r["scores"][str(support)]["proxy_uabr_pct"] for r in supported)
        mean_benign = float(np.mean([s["benign_acceptance_pct"] for s in scores]))
        retention = float(mean_benign / baseline_benign) if baseline_benign else 0.0
        summary.append({"minimum_support": support, "worst_proxy_uabr_pct": worst,
                        "mean_proxy_uabr_pct": float(np.mean([
                            r["scores"][str(support)]["proxy_uabr_pct"] for r in supported])),
                        "mean_benign_acceptance_pct": mean_benign,
                        "benign_retention": retention,
                        "mean_additional_unresolved": float(np.mean([
                            s["additional_unresolved"] for s in scores])),
                        "mean_subregions": float(np.mean([s["subregions"] for s in scores])),
                        "max_known_attack_bypassed": max(
                            s["known_attack_bypassed"] for s in scores),
                        "meets_retention": bool(
                            retention >= options["min_benign_retention"])})
    feasible = [r for r in summary if r["meets_retention"]]
    if not feasible:
        chosen, status = summary[0], "no_policy_meets_retention"
    else:
        chosen = min(feasible, key=lambda r: (r["worst_proxy_uabr_pct"],
                                              r["mean_proxy_uabr_pct"],
                                              -r["mean_benign_acceptance_pct"],
                                              r["mean_additional_unresolved"],
                                              r["mean_subregions"]))
        status = "selected" if chosen["minimum_support"] is not None else "no_proxy_gain"
    return chosen["minimum_support"], {
        "status": status, "chosen": chosen, "candidates": summary,
        "proxy_folds": proxy_records,
        "supported_proxies": [r["proxy"] for r in supported]}, time.perf_counter() - start


def _benign_region_scores(model: dict, x: pd.DataFrame,
                          leaves: np.ndarray) -> np.ndarray:
    """Lower scores are closer to the center of a certified benign region."""
    scores = np.full(len(x), np.inf)
    numbers = model["preprocessor"].named_transformers_["numeric"].transform(
        x[model["numeric"]])
    for leaf, (median, scale, threshold) in model["regions"].items():
        positions = np.flatnonzero(leaves == leaf)
        if len(positions):
            distances = _region_distances(model, numbers[positions], median, scale)
            scores[positions] = distances / max(threshold, np.finfo(float).tiny)
    return scores


def _cap_benign_acceptance(model: dict, data: dict,
                           calibration: np.ndarray, max_fraction: float) -> tuple[dict, dict]:
    """Set a source-only cutoff that accepts at most the requested benign share."""
    if not 0 <= max_fraction <= 1:
        raise ValueError("benign_acceptance.max_fraction must be in [0, 1]")
    if (calibration.shape != data["validation"].shape or calibration.dtype != np.bool_ or
            np.any(calibration & (~data["validation"] | data["test"]))):
        raise ValueError("Acceptance-cap fitting requires source-validation rows, never test rows")
    # Every exploratory cap starts from the same uncapped detector.
    model = {key: value for key, value in model.items() if key != "benign_acceptance_max_score"}
    x = data["x"].loc[calibration]
    benign = data["y"][calibration] == 0
    count = int(benign.sum())
    if not count:
        raise ValueError("Benign acceptance calibration needs benign source flows")
    decisions, _, _, leaves, _, _, _ = _predict(model, x)
    accepted = benign & (decisions == "BENIGN")
    scores = np.sort(_benign_region_scores(model, x, leaves)[accepted])
    allowance = int(np.floor(max_fraction * count))
    if len(scores) <= allowance:
        cutoff = np.inf
    elif allowance == 0:
        cutoff = -np.inf
    else:
        cutoff = scores[allowance - 1]
        if np.searchsorted(scores, cutoff, side="right") > allowance:
            cutoff = np.nextafter(cutoff, -np.inf)
    retained = int(np.searchsorted(scores, cutoff, side="right"))
    return ({**model, "benign_acceptance_max_score": float(cutoff)},
            {"source_benign_count": count,
             "source_benign_accepted_before": int(accepted.sum()),
             "source_benign_accepted_after": retained,
             "source_benign_acceptance_pct": 100 * retained / count,
             "target_benign_acceptance_pct": 100 * max_fraction})


def _predict(model: dict, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float,
                                                   np.ndarray, np.ndarray, np.ndarray,
                                                   np.ndarray]:
    start = time.perf_counter()
    leaves, inside, raw_inside, known_family = _score(model, x)
    attack = np.isin(leaves, list(model["attack_rules"]))
    membership_pass = _membership_pass(model, x, leaves, inside)
    benign = (inside & membership_pass & _conditional_pass(model, x, leaves) &
              np.isin(leaves, list(model["benign_rules"])))
    if "benign_acceptance_max_score" in model and benign.any():
        benign &= (_benign_region_scores(model, x, leaves) <=
                   model["benign_acceptance_max_score"])
    decisions = np.full(len(x), "UNRESOLVED", dtype="<U12")
    decisions[attack & ~benign] = "KNOWN_ATTACK"
    decisions[benign & ~attack] = "BENIGN"
    return (decisions, known_family, time.perf_counter() - start,
            leaves, inside, raw_inside, membership_pass)


def _acceptance_candidates(model: dict, data: dict, tuning: np.ndarray,
                           certification: np.ndarray, cfg: dict, excluded: str) -> tuple[list, float | None]:
    """Freeze source-fitted caps, certify on reserved rows, then rank by tuning coverage."""
    options = cfg["benign_acceptance"]["sensitivity"]
    caps, reference = options["caps"], options["reference_fraction"]
    if (not caps or caps != sorted(set(caps)) or reference not in caps or
            any(not 0 < cap <= 1 for cap in caps)):
        raise ValueError("Sensitivity caps must be unique, increasing, in (0, 1], and include the reference")
    if np.any(tuning & certification):
        raise ValueError("Cap fitting and certification must be disjoint")
    for mask in (tuning, certification):
        if np.any(mask & (~data["validation"] | data["test"] | (data["family"] == excluded))):
            raise ValueError("Cap analysis cannot use test rows or the withheld attack")
    # Simultaneous one-sided bounds permit selecting among the frozen candidates
    # without pretending each repeated certification check has 95% familywise coverage.
    confidence = 1 - (1 - cfg["calibration"]["confidence"]) / len(caps)
    budget = cfg["safe_regions"]["known_attack_leakage_budget"]
    y = data["y"][certification]
    attacks, benign = int(np.sum(y == 1)), int(np.sum(y == 0))
    candidates = []
    fitted = [(cap, *_cap_benign_acceptance(model, data, tuning, cap)) for cap in caps]
    for cap, candidate, audit in fitted:
        decisions = _predict(candidate, data["x"].loc[certification])[0]
        accepted = decisions == "BENIGN"
        errors = int(np.sum(accepted & (y == 1)))
        covered = int(np.sum(accepted & (y == 0)))
        upper = float(beta.ppf(confidence, errors + 1, attacks - errors)) if attacks > errors else 1.0
        safe = attacks >= cfg["calibration"]["min_validation_support"] and upper <= budget
        nondegenerate = audit["source_benign_accepted_after"] > 0 and covered > 0
        audit.update(acceptance_cap_fraction=cap, reference_baseline=cap == reference,
                     source_gate_attack_count=attacks, source_gate_attack_bypassed=errors,
                     source_gate_attack_upper_pct=100 * upper,
                     source_gate_confidence=confidence, source_gate_budget_pct=100 * budget,
                     source_gate_benign_count=benign, source_gate_benign_accepted=covered,
                     source_gate_benign_acceptance_pct=100 * covered / benign if benign else None,
                     source_gate_rows=len(y), source_gate_escalated=int(np.sum(~accepted)),
                     source_gate_escalation_pct=100 * np.mean(~accepted) if len(y) else None,
                     candidate_usable=bool(safe and nondegenerate),
                     source_gate_status=("rejected_safety" if not safe else
                                         "degenerate_no_benign_coverage" if not nondegenerate else "certified"))
        candidates.append((candidate, audit))
    eligible = [audit for _, audit in candidates if audit["candidate_usable"]]
    chosen = (max(eligible, key=lambda row: (row["source_benign_accepted_after"],
                                           -row["acceptance_cap_fraction"]))["acceptance_cap_fraction"]
              if eligible else None)
    return candidates, chosen
