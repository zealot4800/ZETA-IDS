"""Run and report independent leave-one-attack-out evaluation."""

from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .loao_data import _read_dataset
from .loao_model import (
    _build_membership, _fit_model, _fit_safe_model, _split_validation, _wilson_upper,
)
from .loao_policies import (
    _conditional_pass, _conditional_profiles, _predict, _select_conditional,
    _select_membership, _with_conditional,
)

DECISIONS = ("BENIGN", "KNOWN_ATTACK", "UNRESOLVED")


def _counts(decisions: np.ndarray, mask: np.ndarray) -> dict:
    n = int(mask.sum())
    result = {name.lower(): int(np.sum(decisions[mask] == name)) for name in DECISIONS}
    if sum(result.values()) != n:
        raise AssertionError("Decision counts do not sum to the evaluated population")
    return {"count": n, **result,
            **{f"{name}_pct": (100 * count / n if n else math.nan)
               for name, count in result.items()}}


def _path_conditions(model: dict, leaf: int) -> list[str]:
    tree = model["tree"].tree_
    names = model["preprocessor"].get_feature_names_out()
    stack = [(0, [])]
    while stack:
        node, path = stack.pop()
        if node == leaf:
            return path
        if tree.children_left[node] < 0:
            continue
        feature = str(names[tree.feature[node]])
        threshold = float(tree.threshold[node])
        stack.append((int(tree.children_right[node]), path + [f"{feature} > {threshold:.5g}"]))
        stack.append((int(tree.children_left[node]), path + [f"{feature} <= {threshold:.5g}"]))
    raise ValueError(f"Leaf {leaf} not found in attack tree")


def _overlap_diagnostics(data: dict, cfg: dict, model: dict, x_test: pd.DataFrame,
                         unknown_mask: np.ndarray, benign_mask: np.ndarray, leaves: np.ndarray,
                         inside: np.ndarray, raw_inside: np.ndarray,
                         membership_pass: np.ndarray,
                         conditional_pass: np.ndarray) -> dict:
    attack_evidence = np.isin(leaves, list(model["attack_rules"]))
    region_evidence = inside & np.isin(leaves, list(model["benign_rules"]))
    benign_evidence = region_evidence & membership_pass & conditional_pass
    raw = unknown_mask & raw_inside
    certified = unknown_mask & benign_evidence
    groups = {}
    for attack, benign, label in ((True, False, "attack_only"),
                                  (False, True, "benign_only"),
                                  (True, True, "both"), (False, False, "neither")):
        mask = unknown_mask & (attack_evidence == attack) & (benign_evidence == benign)
        groups[label] = {"count": int(mask.sum()), "decision":
                         "UNRESOLVED" if attack == benign else
                         ("KNOWN_ATTACK" if attack else "BENIGN")}
    records = []
    numbers = None
    for leaf in np.unique(leaves[raw]):
        leaf = int(leaf)
        mask = raw & (leaves == leaf)
        audit = model.get("region_audit", {}).get(leaf, {})
        median, scale, threshold = model["raw_regions"][leaf]
        if numbers is None:
            numbers = model["preprocessor"].named_transformers_["numeric"].transform(
                x_test[model["numeric"]])
        deviations = np.minimum(np.abs((numbers[mask] - median) / scale),
                                 model["distance_cap"])
        mean_dev = np.mean(deviations, axis=0)
        top = np.argsort(mean_dev)[-cfg["diagnostics"]["top_deviation_features"]:][::-1]
        selected = audit.get("selected", {})
        records.append({"leaf": leaf, "raw_attack_matches": int(mask.sum()),
                        "pre_membership_attack_matches": int(np.sum(mask & region_evidence)),
                        "membership_rejected_attacks": int(np.sum(mask & region_evidence & ~membership_pass)),
                        "conditional_rejected_attacks": int(np.sum(
                            mask & region_evidence & membership_pass & ~conditional_pass)),
                        "test_benign_pre_membership": int(np.sum(benign_mask & region_evidence & (leaves == leaf))),
                        "test_benign_membership_rejected": int(np.sum(
                            benign_mask & region_evidence & ~membership_pass & (leaves == leaf))),
                        "test_benign_conditional_rejected": int(np.sum(
                            benign_mask & region_evidence & membership_pass &
                            ~conditional_pass & (leaves == leaf))),
                        "certified_attack_matches": int(np.sum(mask & benign_evidence)),
                        "bypassed": int(np.sum(mask & benign_evidence & ~attack_evidence)),
                        "raw_threshold": threshold,
                        "selected_threshold": selected.get("threshold"),
                        "selected_quantile": selected.get("quantile"),
                        "source_train_benign_support": audit.get("train_benign"),
                        "source_train_attack_overlap": selected.get("train_attack",
                                                                       audit.get("raw_train_attack")),
                        "source_tune_benign_support": selected.get("tune_benign",
                                                                      audit.get("raw_tune_benign")),
                        "source_tune_attack_overlap": selected.get("tune_attack",
                                                                      audit.get("raw_tune_attack")),
                        "source_cert_benign_support": selected.get("cert_benign"),
                        "source_cert_attack_overlap": selected.get("cert_attack"),
                        "attack_rule_conflict": leaf in model["attack_rules"],
                        "path_conditions": _path_conditions(model, leaf),
                        "largest_normalized_deviations": [
                            {"feature": model["numeric"][j], "mean_deviation": float(mean_dev[j])}
                            for j in top]})
    n = int(unknown_mask.sum())
    return {"raw_benign_overlap": int(raw.sum()),
            "raw_benign_overlap_pct": 100 * raw.sum() / n if n else math.nan,
            "pre_membership_overlap": int(np.sum(unknown_mask & region_evidence)),
            "pre_membership_overlap_pct": 100 * np.sum(unknown_mask & region_evidence) / n if n else math.nan,
            "membership_rejected_unknown": int(np.sum(unknown_mask & region_evidence & ~membership_pass)),
            "conditional_subregion_overlap": int(np.sum(
                unknown_mask & region_evidence & membership_pass & conditional_pass)),
            "conditional_subregion_overlap_pct": 100 * np.sum(
                unknown_mask & region_evidence & membership_pass & conditional_pass) / n if n else math.nan,
            "conditional_rejected_unknown": int(np.sum(
                unknown_mask & region_evidence & membership_pass & ~conditional_pass)),
            "certified_benign_overlap": int(certified.sum()),
            "certified_benign_overlap_pct": 100 * certified.sum() / n if n else math.nan,
            "overlapping_benign_regions": len(records),
            "attack_flows_matching_multiple_benign_regions": 0,
            "evidence_groups_json": json.dumps(groups, sort_keys=True),
            "overlap_regions_json": json.dumps(records, sort_keys=True)}


def _evaluate_model(data: dict, cfg: dict, excluded: str, model: dict,
                    train: np.ndarray, tuning: np.ndarray, certification: np.ndarray,
                    train_seconds: float) -> dict:
    family, y = data["family"], data["y"]
    test = data["test"]
    x_test = data["x"].loc[test]
    (decisions, known_prediction, infer_seconds, leaves, inside, raw_inside,
     membership_pass) = _predict(model, x_test)
    conditional_pass = _conditional_pass(model, x_test, leaves)
    truth = family[test]
    test_y = y[test]
    benign = _counts(decisions, test_y == 0)
    known_mask = (test_y == 1) & (truth != excluded)
    known = _counts(decisions, known_mask)
    unknown = _counts(decisions, truth == excluded)
    all_counts = _counts(decisions, np.ones(len(decisions), dtype=bool))
    known_classes = sorted(set(truth[known_mask]))
    per_class = {label: _counts(decisions, truth == label) for label in known_classes}
    predicted_known = pd.Series(known_prediction[truth == excluded]).value_counts().to_dict()
    row = {"dataset": data["name"], "configuration": "conditional_loao",
           "heldout_attack": excluded, "status": "complete",
           "split_strategy": data["strategy"],
           "source_selection": data["source_selection"],
           "source_rows": data["source_rows"],
           "source_rows_excluded": data["source_rows_excluded"],
           "train_rows": int(train.sum()),
           "validation_rows": int((tuning | certification).sum()),
           "tuning_rows": int(tuning.sum()), "certification_rows": int(certification.sum()),
           "test_rows": int(test.sum()),
           "test_group_duplicates": data["duplicate_rows"],
           "official_overlap_removed": data["official_overlap_removed"],
           "attack_tree_train_classes": len(set(family[train & (y == 1)])),
           "attack_rules": len(model["attack_rules"]), "benign_rules": len(model["benign_rules"]),
           "unknown_count": unknown["count"],
           "unknown_to_benign": unknown["benign"],
           "unknown_to_known_attack": unknown["known_attack"],
           "unknown_to_unresolved": unknown["unresolved"],
           "unknown_to_benign_pct": unknown["benign_pct"],
           "unknown_to_known_attack_pct": unknown["known_attack_pct"],
           "unknown_to_unresolved_pct": unknown["unresolved_pct"],
           "confirmed_malicious_detection_pct": unknown["known_attack_pct"],
           "candidate_detection_pct": 100 - unknown["benign_pct"],
           "known_attack_count": known["count"], "known_attack_detected": known["known_attack"],
           "known_attack_bypassed": known["benign"], "known_attack_unresolved": known["unresolved"],
           "known_attack_recall_pct": known["known_attack_pct"],
           "known_attack_to_benign_pct": known["benign_pct"],
           "known_attack_to_unresolved_pct": known["unresolved_pct"],
           "known_per_class_json": json.dumps(per_class, sort_keys=True),
           "unknown_predicted_known_family_json": json.dumps(predicted_known, sort_keys=True),
           "benign_count": benign["count"], "benign_accepted": benign["benign"],
           "benign_false_positive": benign["known_attack"],
           "benign_escalated": benign["unresolved"],
           "benign_acceptance_pct": benign["benign_pct"],
           "benign_false_positive_pct": benign["known_attack_pct"],
           "benign_escalation_pct": benign["unresolved_pct"],
           "overall_unresolved_count": all_counts["unresolved"],
           "overall_unresolved_pct": all_counts["unresolved_pct"],
           "controller_escalation_workload": all_counts["unresolved"],
           "automatic_coverage_pct": 100 - all_counts["unresolved_pct"],
           "training_seconds": train_seconds, "inference_seconds": infer_seconds,
           "inference_ms_per_flow": 1000 * infer_seconds / len(decisions),
           "throughput_flows_per_second": len(decisions) / infer_seconds,
           "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "policy_status": model.get("policy_status", "existing_policy"),
           "source_known_attack_leakage_budget_pct": 100 * cfg["safe_regions"]["known_attack_leakage_budget"],
           "source_tuning_known_attack_count": model.get("tuning_known_attack_count", math.nan),
           "source_tuning_benign_accepted": model.get("tuning_benign_accepted", math.nan),
           "source_tuning_attack_bypassed": model.get("tuning_errors", math.nan),
           "source_cert_known_attack_count": model.get("cert_known_attack_count", math.nan),
           "source_cert_attack_bypassed": model.get("cert_errors", math.nan),
           "source_cert_attack_leakage_upper_pct": 100 * model.get("cert_upper", math.nan),
           "source_cert_benign_accepted": model.get("cert_benign_accepted", math.nan),
           "membership_quantile": model.get("membership_quantile", math.nan),
           "membership_prototype_count": model.get("membership_prototype_count", 0),
           "membership_memory_mb": model.get("membership_memory_mb", 0.0),
           "conditional_minimum_support": model.get("conditional_minimum_support", math.nan),
           "conditional_region_count": len(model.get("conditional", {})),
           "conditional_subregion_count": sum(
               len(p["keep"]) for p in model.get("conditional", {}).values()),
           "conditional_memory_kb": sum(
               p["edges"][0].nbytes + p["edges"][1].nbytes + p["keep"].nbytes
               for p in model.get("conditional", {}).values()) / 1024,
           "conditional_affected_flows": int(np.sum(
               inside & membership_pass & np.isin(leaves, list(model["benign_rules"])) &
               ~conditional_pass & ~np.isin(leaves, list(model["attack_rules"])))),
           "membership_rejected_benign": int(np.sum(
               (test_y == 0) & inside & np.isin(leaves, list(model["benign_rules"]))
               & ~membership_pass & ~np.isin(leaves, list(model["attack_rules"]))))}
    original = data.get("original_label")
    if original is not None:
        test_original = original[test]
        breakdown = {}
        for label in sorted(set(test_original[truth == excluded])):
            mask = (truth == excluded) & (test_original == label)
            details = _counts(decisions, mask)
            distinct_mask = ~x_test.loc[mask].duplicated().to_numpy()
            unique_decisions = decisions[mask][distinct_mask]
            distinct = _counts(unique_decisions,
                               np.ones(len(unique_decisions), dtype=bool))
            details.update({f"distinct_{key}": value
                            for key, value in distinct.items()})
            breakdown[label] = details
        if sum(item["count"] for item in breakdown.values()) != unknown["count"]:
            raise AssertionError("Original-label counts disagree with held-out total")
        for decision in ("benign", "known_attack", "unresolved"):
            if sum(item[decision] for item in breakdown.values()) != unknown[decision]:
                raise AssertionError("Original-label decisions disagree with held-out total")
        row["unknown_by_original_label_json"] = json.dumps(breakdown, sort_keys=True)
        row["heldout_original_label_population_json"] = json.dumps({
            label: {"sample_rows": data["original_counts"][label],
                    "sample_distinct_patterns": data[
                        "sample_pattern_counts"].get(label),
                    "source_rows_before_sampling": data[
                        "source_original_counts"].get(label)}
            for label in breakdown}, sort_keys=True)
        row["sampling_method"] = data["sampling_method"]
    row.update(_overlap_diagnostics(data, cfg, model, x_test, truth == excluded, test_y == 0,
                                    leaves, inside, raw_inside, membership_pass,
                                    conditional_pass))
    return row


def _evaluate_conditional_fold(data: dict, cfg: dict, excluded: str) -> dict:
    family, y = data["family"], data["y"]
    train = data["train"] & (family != excluded)
    validation = data["validation"] & (family != excluded)
    tuning, certification = _split_validation(data, validation, cfg)
    base, base_seconds = _fit_model(data, train, tuning, cfg)
    safe, safe_seconds = _fit_safe_model(data, train, tuning, certification, cfg, base)
    quantile, proxy_audit, proxy_seconds = _select_membership(
        data, cfg, excluded, train, validation)
    refined, membership_seconds = _build_membership(data, safe, train, tuning, cfg)
    refined["membership_quantile"] = quantile
    selected, conditional_audit, selection_seconds = _select_conditional(
        data, cfg, excluded, train, tuning, quantile)
    conditional_start = time.perf_counter()
    profiles = _conditional_profiles(data, refined, train, tuning, cfg)
    candidate = _with_conditional(refined, profiles, selected, cfg)
    fit_seconds = time.perf_counter() - conditional_start
    total_training_seconds = (base_seconds + safe_seconds + proxy_seconds +
                              membership_seconds + selection_seconds + fit_seconds)
    row = _evaluate_model(
        data, cfg, excluded, candidate, train, tuning, certification,
        total_training_seconds)
    cert_decisions = _predict(candidate, data["x"].loc[certification])[0]
    cert_attack = y[certification] == 1
    cert_errors = int(np.sum((cert_decisions == "BENIGN") & cert_attack))
    cert_upper = _wilson_upper(cert_errors, int(cert_attack.sum()),
                                cfg["calibration"]["confidence"])
    if safe["policy_status"] == "certified" and cert_upper > cfg[
            "safe_regions"]["known_attack_leakage_budget"]:
        raise AssertionError("Conditional gate breached source certification budget")
    region_audit = []
    for leaf, profile in sorted(profiles.items()):
        if "counts" not in profile:
            region_audit.append({
                "leaf": leaf, "source_benign_support": profile["source_support"],
                "feature_dispersion": profile["dispersion"],
                "reason": profile["reason"], "retained_cells": [],
                "source_train_attack_in_retained": 0,
                "source_tune_attack_in_retained": 0})
            continue
        selected_profile = candidate.get("conditional", {}).get(leaf)
        kept = (selected_profile["keep"] if selected_profile else np.array([], dtype=int))
        counts = profile["counts"]
        region_audit.append({
            "leaf": leaf, "source_benign_support": profile["source_support"],
            "features": [data["numeric"][j] for j in profile["features"]],
            "edges": [edge.tolist() for edge in profile["edges"]],
            "joint_information": profile["information"],
            "feature_dispersion": profile["dispersion"],
            "grid_cells": profile["cell_count"],
            "empty_cells": int(np.sum(counts["train_benign"] == 0)),
            "weak_cells": int(np.sum((counts["train_benign"] > 0) &
                                     (counts["train_benign"] <
                                      cfg["conditional"]["min_train_cell_support_candidates"][0]))),
            "retained_cells": kept.tolist(),
            "source_train_attack_in_retained": int(counts["train_attack"][kept].sum()),
            "source_tune_attack_in_retained": int(counts["tune_attack"][kept].sum()),
            "cells": [{
                "cell": cell,
                "train_benign": int(counts["train_benign"][cell]),
                "train_attack": int(counts["train_attack"][cell]),
                "tune_benign": int(counts["tune_benign"][cell]),
                "tune_attack": int(counts["tune_attack"][cell])}
                for cell in range(profile["cell_count"])]})
    row.update({
        "proxy_status": proxy_audit["status"],
        "proxy_selected_quantile": quantile,
        "proxy_worst_uabr_pct": proxy_audit["chosen"]["worst_proxy_uabr_pct"],
        "proxy_mean_uabr_pct": proxy_audit["chosen"]["mean_proxy_uabr_pct"],
        "proxy_benign_retention": proxy_audit["chosen"]["benign_retention"],
        "proxy_selection_seconds": proxy_seconds,
        "membership_fit_seconds": membership_seconds,
        "proxy_audit_json": json.dumps(proxy_audit, sort_keys=True),
        "conditional_status": conditional_audit["status"],
        "conditional_selected_minimum_support": selected,
        "conditional_proxy_worst_uabr_pct":
            conditional_audit["chosen"]["worst_proxy_uabr_pct"],
        "conditional_proxy_benign_retention":
            conditional_audit["chosen"]["benign_retention"],
        "conditional_selection_seconds": selection_seconds,
        "conditional_fit_seconds": fit_seconds,
        "conditional_proxy_audit_json": json.dumps(conditional_audit, sort_keys=True),
        "conditional_region_audit_json": json.dumps(region_audit, sort_keys=True),
        "source_cert_final_attack_bypassed": cert_errors,
        "source_cert_final_attack_leakage_upper_pct": 100 * cert_upper})
    return row


def run(config_path: Path, dataset: str | None = None,
        attack: str | None = None) -> dict[str, list[dict]]:
    """Print conditional-policy LOAO results without saving reports."""
    with config_path.open() as stream:
        cfg = yaml.safe_load(stream)
    root = config_path.resolve().parent.parent
    selected = [dataset] if dataset else list(cfg["datasets"])
    if any(name not in cfg["datasets"] for name in selected):
        raise ValueError(f"Unknown dataset; choose from {list(cfg['datasets'])}")
    if attack is not None and len(selected) != 1:
        raise ValueError("--attack requires --dataset")
    completed = {}
    for name in selected:
        data = _read_dataset(name, cfg["datasets"][name], cfg, root)
        if data["source_rows_excluded"]:
            print(f"{name}: LOAO uses {len(data['family']):,}/{data['source_rows']:,} "
                  f"rows ({data['source_selection']}); "
                  f"excludes {data['source_rows_excluded']:,} source rows", flush=True)
        families = cfg["expected_attacks"][name]
        if attack is not None:
            if attack not in families:
                raise ValueError(f"{attack!r} is not configured for {name}")
            families = [attack]
        completed[name] = []
        for excluded in families:
            row = _evaluate_conditional_fold(data, cfg, excluded)
            completed[name].append(row)
            print(f"{name}/{excluded}: C held-out {row['unknown_count']}; "
                  f"bypass {row['unknown_to_benign']} "
                  f"({row['unknown_to_benign_pct']:.2f}%); "
                  f"attack {row['unknown_to_known_attack']}; "
                  f"unresolved {row['unknown_to_unresolved']}; "
                  f"benign acceptance {row['benign_acceptance_pct']:.2f}%; "
                  f"known recall {row['known_attack_recall_pct']:.2f}%; "
                  f"policy {row['conditional_status']}", flush=True)
            if "unknown_by_original_label_json" in row:
                for label, details in json.loads(row["unknown_by_original_label_json"]).items():
                    print(f"  {label}: bypass {details['benign']}/{details['count']} "
                          f"({details['benign_pct']:.2f}%); distinct patterns "
                          f"{details['distinct_benign']}/{details['distinct_count']}",
                          flush=True)
    return completed
