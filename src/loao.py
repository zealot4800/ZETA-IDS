"""Run and report independent leave-one-attack-out evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

from .loao_data import _read_dataset
from .loao_model import (
    _build_membership, _fit_model, _fit_safe_model, _split_validation, _wilson_upper,
)
from .loao_policies import (
    _acceptance_candidates, _benign_region_scores, _cap_benign_acceptance, _conditional_pass,
    _conditional_profiles, _predict, _select_conditional,
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
    before_cap = region_evidence & membership_pass & conditional_pass
    cap_pass = np.ones(len(leaves), dtype=bool)
    if "benign_acceptance_max_score" in model:
        cap_pass = (_benign_region_scores(model, x_test, leaves) <=
                    model["benign_acceptance_max_score"])
    benign_evidence = before_cap & cap_pass
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
            "acceptance_cap_rejected_unknown": int(np.sum(
                unknown_mask & before_cap & ~cap_pass)),
            "acceptance_cap_rejected_benign": int(np.sum(
                benign_mask & before_cap & ~cap_pass)),
            "certified_benign_overlap": int(certified.sum()),
            "certified_benign_overlap_pct": 100 * certified.sum() / n if n else math.nan,
            "overlapping_benign_regions": len(records),
            "attack_flows_matching_multiple_benign_regions": 0,
            "gate_evidence_overlap_count": int(np.sum(attack_evidence & benign_evidence)),
            "gate_rejection_reasons_json": json.dumps({
                "attack_rule_match": int(np.sum(attack_evidence)),
                "no_certified_region_match": int(np.sum(~region_evidence)),
                "membership_rejected_in_region": int(np.sum(region_evidence & ~membership_pass)),
                "conditional_rejected_after_membership": int(np.sum(
                    region_evidence & membership_pass & ~conditional_pass)),
                "acceptance_cap_rejected_after_gates": int(np.sum(before_cap & ~cap_pass))}, sort_keys=True),
            "evidence_groups_json": json.dumps(groups, sort_keys=True),
            "overlap_regions_json": json.dumps(records, sort_keys=True)}


def _evaluate_model(data: dict, cfg: dict, excluded: str, model: dict,
                    train: np.ndarray, tuning: np.ndarray, certification: np.ndarray,
                    train_seconds: float, prediction: tuple | None = None) -> dict:
    family, y = data["family"], data["y"]
    test = data["test"]
    x_test = data["x"].loc[test]
    (decisions, known_prediction, infer_seconds, leaves, inside, raw_inside,
     membership_pass) = _predict(model, x_test) if prediction is None else prediction
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
    # Legacy fields above describe the internal three-way classifier. Gate
    # routing forwards every non-benign decision, including known-attack rules.
    routing = {}
    for label, counts in (("all", all_counts), ("benign", benign), ("known_attack", known), ("unknown_attack", unknown)):
        escalated = counts["count"] - counts["benign"]
        routing[label] = {"count": counts["count"], "accepted_benign": counts["benign"],
                          "escalated": escalated, "accepted_benign_pct": counts["benign_pct"],
                          "escalated_pct": 100 * escalated / counts["count"] if counts["count"] else None}
    row["gate_routing_json"] = json.dumps(routing, sort_keys=True)
    row["gate_overall_escalated"] = routing["all"]["escalated"]
    row["gate_overall_escalation_pct"] = routing["all"]["escalated_pct"]
    return row


def fit_fold(data: dict, cfg: dict, excluded: str) -> tuple[dict, dict]:
    """Build the existing first-layer fold once; downstream users only read its model."""
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
    candidate, cap_audit = _cap_benign_acceptance(
        candidate, data, tuning, cfg["benign_acceptance"]["max_fraction"])
    fit_seconds = time.perf_counter() - conditional_start
    total_training_seconds = (base_seconds + safe_seconds + proxy_seconds +
                              membership_seconds + selection_seconds + fit_seconds)
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
    metrics = {
        **cap_audit,
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
        "source_cert_final_attack_leakage_upper_pct": 100 * cert_upper}
    return candidate, {"train": train, "validation": validation, "tuning": tuning,
                       "certification": certification, "training_seconds": total_training_seconds,
                       "metrics": metrics}


def _evaluate_conditional_fold(data: dict, cfg: dict, excluded: str) -> dict:
    model, fold = fit_fold(data, cfg, excluded)
    row = _evaluate_model(data, cfg, excluded, model, fold["train"], fold["tuning"],
                          fold["certification"], fold["training_seconds"])
    row.update(fold["metrics"])
    return row


def _evaluate_acceptance_sensitivity(data: dict, cfg: dict, excluded: str) -> list[dict]:
    """Exploratory caps share one detector; an extra source holdout certifies the gate."""
    options = cfg["benign_acceptance"]["sensitivity"]
    source = data["validation"] & (data["family"] != excluded)
    split_cfg = {**cfg, "safe_regions": {**cfg["safe_regions"],
                                        "certification_fraction": options["certification_fraction"]}}
    detector_validation, gate_certification = _split_validation(
        data, source, split_cfg, salt=options["certification_salt"])
    detector_data = {**data, "validation": detector_validation}
    uncapped_cfg = {**cfg, "benign_acceptance": {**cfg["benign_acceptance"], "max_fraction": 1.0}}
    model, fold = fit_fold(detector_data, uncapped_cfg, excluded)
    candidates, chosen = _acceptance_candidates(model, data, fold["tuning"], gate_certification, cfg, excluded)
    # Selection is finished before any test prediction or test-label evaluation.
    rows = []
    for candidate, audit in candidates:
        row = _evaluate_model(data, cfg, excluded, candidate, fold["train"], fold["tuning"],
                              fold["certification"], fold["training_seconds"])
        row.update({key: value for key, value in fold["metrics"].items()
                    if not key.startswith("source_cert_final_")})
        row.update(audit)
        row.update(configuration="acceptance_cap_sensitivity", source_recommended_cap=chosen,
                   source_recommended=audit["acceptance_cap_fraction"] == chosen,
                   gate_certification_rows=int(gate_certification.sum()))
        rows.append(row)
    return rows


def run(config_path: Path, dataset: str | None = None,
        attack: str | None = None, method: str | None = None,
        acceptance_sensitivity: bool = False) -> dict[str, list[dict]]:
    """Keep the original terminal workflow; explicit methods use shared comparison."""
    if method is not None:
        if acceptance_sensitivity:
            raise ValueError("Acceptance sensitivity is a first-layer diagnostic; omit --method")
        return _run_comparison(config_path, dataset, attack, method)
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
            rows = (_evaluate_acceptance_sensitivity(data, cfg, excluded) if acceptance_sensitivity
                    else [_evaluate_conditional_fold(data, cfg, excluded)])
            completed[name].extend(rows)
            for row in rows:
                _print_first_layer(row)
    return completed


def _print_first_layer(row: dict) -> None:
    name, excluded = row["dataset"], row["heldout_attack"]
    print(f"{name}/{excluded}: internal C held-out {row['unknown_count']}; "
          f"bypass {row['unknown_to_benign']} "
          f"({row['unknown_to_benign_pct']:.2f}%); "
          f"attack {row['unknown_to_known_attack']}; "
          f"unresolved {row['unknown_to_unresolved']}; "
          f"benign acceptance {row['benign_acceptance_pct']:.2f}%; "
          f"source cap {row['target_benign_acceptance_pct']:.2f}%; "
          f"known recall {row['known_attack_recall_pct']:.2f}%; "
          f"policy {row['conditional_status']}", flush=True)
    for group, counts in json.loads(row["gate_routing_json"]).items():
        percentage = f"{counts['escalated_pct']:.2f}" if counts['escalated_pct'] is not None else "n/a"
        print(f"  gate {group}: accept {counts['accepted_benign']}/{counts['count']}; "
              f"escalate {counts['escalated']}/{counts['count']} "
              f"({percentage}%)", flush=True)
    print(f"  gate overlap={row['gate_evidence_overlap_count']}; "
          f"latency={row['inference_ms_per_flow']:.5f} ms/flow; "
          f"rejections={row['gate_rejection_reasons_json']}", flush=True)
    if "source_gate_status" in row:
        print(f"  source cap={row['acceptance_cap_fraction']:.0%}; "
              f"reference={row['reference_baseline']}; recommended={row['source_recommended']}; "
              f"status={row['source_gate_status']}; "
              f"cert attacks bypassed={row['source_gate_attack_bypassed']}/{row['source_gate_attack_count']}; "
              f"upper={row['source_gate_attack_upper_pct']:.4f}% "
              f"budget={row['source_gate_budget_pct']:.4f}%; "
              f"cert benign accepted={row['source_gate_benign_accepted']}/{row['source_gate_benign_count']}; "
              f"cert escalated={row['source_gate_escalated']}/{row['source_gate_rows']}", flush=True)
        if row["source_recommended_cap"] is None and row["reference_baseline"]:
            print("  No nondegenerate cap certified; no optimization approved and no configuration changed.", flush=True)
    if "unknown_by_original_label_json" in row:
        for label, details in json.loads(row["unknown_by_original_label_json"]).items():
            print(f"  {label}: bypass {details['benign']}/{details['count']} "
                  f"({details['benign_pct']:.2f}%); distinct patterns "
                  f"{details['distinct_benign']}/{details['distinct_count']}",
                  flush=True)



def _comparison_metrics(data: dict, excluded: str, positions: np.ndarray, result: dict,
                        training_seconds: float, inference_seconds: float, cfg: dict) -> dict:
    """Identical attack-only binary novelty evaluation for either native method."""
    truth = data["family"][positions]
    if not np.all(data["y"][positions] == 1):
        raise ValueError("Primary novelty metrics require attack-only rows")
    positive = truth == excluded
    candidate = np.isin(result["decisions"], ("UNRESOLVED", "ZERO_DAY_CANDIDATE"))
    tp, fp = int(np.sum(candidate & positive)), int(np.sum(candidate & ~positive))
    fn, tn = int(np.sum(~candidate & positive)), int(np.sum(~candidate & ~positive))
    zero = cfg["comparison"]["metric_zero_division"]
    precision = tp / (tp + fp) if tp + fp else zero
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else zero
    known_count = int((~positive).sum())
    correct = result["known_family"] == truth
    score = result["score"]
    has_score = score is not None and len(np.unique(positive)) == 2
    if score is not None and (np.shape(score) != positive.shape or not np.isfinite(score).all()):
        raise ValueError("Novelty scores must be finite and aligned with the shared test rows")
    return {"primary_attack_rows": len(positions), "withheld_attack_rows": int(positive.sum()),
            "known_attack_rows": known_count, "zero_day_tp": tp, "zero_day_fp": fp,
            "zero_day_fn": fn, "zero_day_tn": tn, "candidate_count": int(candidate.sum()),
            "zero_day_precision_pct": 100 * precision,
            "zero_day_recall_pct": None if recall is None else 100 * recall,
            "zero_day_f1_pct": 100 * f1,
            "auroc": float(roc_auc_score(positive, score)) if has_score else None,
            "auprc_average_precision": float(average_precision_score(positive, score)) if has_score else None,
            "known_attack_false_novelty_pct": 100 * fp / known_count if known_count else None,
            "known_attack_classification_accuracy_pct": 100 * float(np.mean(correct[~positive])) if known_count else None,
            "known_attack_correctly_routed_pct": 100 * float(np.mean(
                (correct & (result["decisions"] == "KNOWN_ATTACK"))[~positive])) if known_count else None,
            "fit_wall_seconds": training_seconds, "primary_inference_seconds": inference_seconds,
            "primary_latency_ms_per_flow": 1000 * inference_seconds / len(positions),
            "primary_throughput_flows_per_second": len(positions) / inference_seconds if inference_seconds > 0 else None,
            "process_peak_rss_mb_cumulative": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                if cfg["comparison"]["measure_memory"] else None}


def _limit_rows(data: dict, excluded: str, cfg: dict) -> dict:
    """Optional reproducible subsets; never move a row across an existing partition."""
    limited = dict(data)
    for split in ("train", "validation", "test"):
        mask = data[split].copy()
        if split != "test":
            mask &= data["family"] != excluded
        limit = cfg["comparison"][split + "_row_limit"]
        if limit is not None:
            if type(limit) is not int or limit < 1:
                raise ValueError("Comparison row limits must be positive integers or null")
            positions = np.flatnonzero(mask)
            if len(positions) > limit:
                rng = np.random.default_rng(cfg["comparison"]["sampling_seed"])
                shuffled = rng.permutation(positions)
                if split == "test":
                    keep = shuffled[:limit]  # Test selection does not inspect family labels.
                else:
                    _, first = np.unique(data["family"][shuffled], return_index=True)
                    if len(first) > limit:
                        raise ValueError("Source row limit is too small to retain every permitted class")
                    required = shuffled[first]
                    keep = np.r_[required, shuffled[~np.isin(shuffled, required)][:limit - len(required)]]
                mask[:] = False
                mask[keep] = True
        limited[split] = mask
    return limited


def _zeta_comparison(data: dict, cfg: dict, excluded: str, positions: np.ndarray) -> tuple[dict, dict, dict]:
    start = time.perf_counter()
    model, fold = fit_fold(data, cfg, excluded)
    train_seconds = time.perf_counter() - start
    start = time.perf_counter()
    attack_prediction = _predict(model, data["x"].iloc[positions])
    inference_seconds = time.perf_counter() - start
    result = {"decisions": attack_prediction[0], "known_family": attack_prediction[1], "score": None}
    metrics = _comparison_metrics(data, excluded, positions, result, train_seconds, inference_seconds, cfg)
    # Evaluate benign rows separately, then reuse the original full-traffic metrics.
    test_positions = np.flatnonzero(data["test"])
    is_attack = data["y"][test_positions] == 1
    combined = [np.empty(len(test_positions), dtype=value.dtype) if i != 2 else attack_prediction[2]
                for i, value in enumerate(attack_prediction)]
    for i in range(len(combined)):
        if i != 2:
            combined[i][is_attack] = attack_prediction[i]
    if np.any(~is_attack):
        benign_prediction = _predict(model, data["x"].iloc[test_positions[~is_attack]])
        for i in range(len(combined)):
            if i == 2:
                combined[i] += benign_prediction[i]
            else:
                combined[i][~is_attack] = benign_prediction[i]
    original = _evaluate_model(data, cfg, excluded, model, fold["train"], fold["tuning"],
                               fold["certification"], fold["training_seconds"], prediction=tuple(combined))
    original.update(fold["metrics"])
    original.update(metrics)
    result.update(all_positions=test_positions, all_decisions=combined[0], all_known_family=combined[1])
    metadata = {"reproduction_status": "original_zeta_unchanged", "class_mapping": {
        str(label): i for i, label in enumerate(model["tree"].classes_)}}
    return result, original, metadata


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _split_metadata(data: dict) -> dict:
    return {split: {"rows": int(data[split].sum()), "row_positions_sha256": hashlib.sha256(
        np.asarray(data["row_positions"][data[split]], dtype="<i8").tobytes()).hexdigest()}
        for split in ("train", "validation", "test")}


def _comparison_table(records: list[dict], cfg: dict) -> list[dict]:
    table = {}
    for row in records:
        key = (row["dataset"], row["heldout_attack"])
        item = table.setdefault(key, {"Dataset": key[0], "Withheld Attack": key[1]})
        label = "ZETA-IDS" if row["method"] == "zeta" else "Paper"
        for metric, title in (("zero_day_recall_pct", "Zero-Day Recall (%)"),
                              ("zero_day_f1_pct", "F1 (%)"), ("primary_latency_ms_per_flow", "Latency (ms/flow)")):
            item[f"{label} {title}"] = row[metric]
    columns = ["Dataset", "Withheld Attack", "ZETA-IDS Zero-Day Recall (%)", "Paper Zero-Day Recall (%)",
               "ZETA-IDS F1 (%)", "Paper F1 (%)", "ZETA-IDS Latency (ms/flow)", "Paper Latency (ms/flow)"]
    result = [{column: row.get(column) for column in columns} for row in table.values()]
    if cfg["comparison"]["print_table"] and result:
        print(pd.DataFrame(result).to_string(index=False, na_rep="N/A", float_format=lambda v:
              f"{v:.{cfg['comparison']['percentage_decimals']}f}"), flush=True)
    return result


def _run_comparison(config_path: Path, dataset: str | None, attack: str | None,
                    method: str) -> dict[str, list[dict]]:
    cfg = yaml.safe_load(config_path.read_text())
    options = cfg["comparison"]
    if method not in ("zeta", "paper", "both") or set(options["methods"]) != {"zeta", "paper"} or len(options["methods"]) != 2:
        raise ValueError("Comparison has exactly two methods: zeta and paper")
    if options["metric_zero_division"] not in (0, 1):
        raise ValueError("metric_zero_division must be 0 or 1")
    methods = options["methods"] if method == "both" else [method]
    if "paper" in methods:
        from . import paper
        paper.validate_config(cfg)
    root = config_path.resolve().parent.parent
    datasets = [dataset] if dataset else options["datasets"]
    if not datasets or len(datasets) != len(set(datasets)) or any(name not in cfg["datasets"] for name in datasets):
        raise ValueError("Select nonempty, unique configured comparison datasets")
    if attack is not None and len(datasets) != 1:
        raise ValueError("--attack requires one selected dataset")
    selected = {}
    for name in datasets:
        families = [attack] if attack is not None else options["attacks"].get(name, cfg["expected_attacks"][name])
        if not families or len(families) != len(set(families)) or not set(families) <= set(cfg["expected_attacks"][name]):
            raise ValueError(f"Invalid withheld attack selection for {name}")
        selected[name] = families
    output = None
    if options["save_results"]:
        output = root / options["output_directory"] / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        output.mkdir(parents=True, exist_ok=False)
        filenames = list(options["filenames"].values())
        if len(filenames) != len(set(filenames)) or any(Path(name).name != name for name in filenames):
            raise ValueError("Comparison filenames must be unique simple filenames")
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "shap", "tensorflow-cpu", "xgboost-cpu")
    versions = {}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    metadata = {"status": "running", "configurations": cfg, "methods": methods,
                "python": sys.version, "packages": versions, "datasets": {}, "folds": [],
                "metric_definitions": {
                    "population": "Identical attack-only frozen test rows; withheld family is positive.",
                    "candidate": "ZETA UNRESOLVED or paper ZERO_DAY_CANDIDATE. UNRESOLVED does not establish maliciousness or novelty.",
                    "precision": "TP / all primary candidates", "recall": "TP / all held-out attack test rows",
                    "f1": "2 TP / (2 TP + FP + FN); positive-class F1, not the paper's macro F1",
                    "auprc_average_precision": "Noninterpolated average precision; no fabricated ZETA novelty score.",
                    "known_false_novelty": "Known attacks marked candidates / all known attack test rows",
                    "known_classification_accuracy": "Underlying multiclass prediction correct / all known attack test rows, irrespective of final routing",
                    "known_correctly_routed": "Correct family AND KNOWN_ATTACK / all known attack test rows",
                    "benign_and_bypass": "ZETA original full-traffic metrics; denominators are benign, known-attack or held-out attack counts as named",
                    "timing": "Primary attack-only batch inference, including preprocessing; excludes reporting and supplemental benign inference",
                    "memory": "Cumulative process peak RSS, not isolated per-method memory"}}
    records, completed = [], {}
    try:
        for name, families in selected.items():
            data = _read_dataset(name, cfg["datasets"][name], cfg, root)
            with (root / cfg["datasets"][name]["path"]).open("rb") as stream:
                data_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            metadata["datasets"][name] = {"path": cfg["datasets"][name]["path"], "sha256": data_hash,
                "source_rows": data["source_rows"], "source_rows_excluded": data["source_rows_excluded"],
                "source_selection": data["source_selection"], "split_strategy": data["strategy"],
                "splits": _split_metadata(data), "feature_schema": {column: str(dtype) for column, dtype in data["x"].dtypes.items()},
                "numeric": data["numeric"], "categorical": data["categorical"], "attack_families": cfg["expected_attacks"][name],
                "family_map": cfg["datasets"][name].get("family_map")}
            completed[name] = []
            for excluded in families:
                fold_data = _limit_rows(data, excluded, cfg)
                positions = np.flatnonzero(fold_data["test"] & (fold_data["y"] == 1))
                truth = fold_data["family"][positions]
                if not np.any(truth == excluded) or not np.any(truth != excluded):
                    raise ValueError("Comparison needs held-out and known attack test rows; increase test_row_limit")
                split_info = _split_metadata(fold_data)
                for selected_method in methods:
                    if selected_method == "zeta":
                        result, row, method_meta = _zeta_comparison(fold_data, cfg, excluded, positions)
                    else:
                        start = time.perf_counter()
                        model = paper.fit(fold_data, fold_data["train"], fold_data["validation"], excluded, cfg)
                        training = time.perf_counter() - start
                        start = time.perf_counter()
                        result = paper.predict(model, fold_data["x"].iloc[positions])
                        seconds = time.perf_counter() - start
                        row = _comparison_metrics(fold_data, excluded, positions, result, training, seconds, cfg)
                        method_meta = model["metadata"]
                        del model
                    row.update(dataset=name, heldout_attack=excluded, method=selected_method,
                               reproduction_status=method_meta["reproduction_status"])
                    records.append(row)
                    completed[name].append(row)
                    metadata["folds"].append({"dataset": name, "heldout_attack": excluded, "method": selected_method,
                        "splits": split_info, "primary_test_rows_sha256": hashlib.sha256(
                            np.asarray(data["row_positions"][positions], dtype="<i8").tobytes()).hexdigest(), **method_meta})
                    if output:
                        exported = result.get("all_positions", positions)
                        predictions = pd.DataFrame({"dataset": name, "heldout_attack": excluded, "method": selected_method,
                            "row_position": data["row_positions"][exported], "split": "test",
                            "test_split_id": split_info["test"]["row_positions_sha256"],
                            "primary_attack_population": data["y"][exported] == 1, "true_family": data["family"][exported],
                            "is_heldout_attack": data["family"][exported] == excluded,
                            "decision": result.get("all_decisions", result["decisions"]),
                            "predicted_known_family": result.get("all_known_family", result["known_family"])})
                        for key in ("score", "reconstruction_error", "epistemic_uncertainty"):
                            predictions[key] = result.get(key) if selected_method == "paper" else None
                        predictions["novelty_threshold"] = method_meta.get("novelty_threshold")
                        target = output / options["filenames"]["predictions"]
                        predictions.to_csv(target, mode="a", header=not target.exists(), index=False)
                        pd.DataFrame(_json_value(records)).to_csv(output / options["filenames"]["metrics"], index=False)
        metadata["status"] = "complete"
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        table = _comparison_table(records, cfg)
        if output:
            summary = {"status": metadata["status"], "output_directory": str(output),
                       "completed_method_folds": len(records), "comparison_table": table, "metrics": records}
            (output / options["filenames"]["summary"]).write_text(json.dumps(_json_value(summary), allow_nan=False, indent=2))
            (output / options["filenames"]["metadata"]).write_text(json.dumps(_json_value(metadata), allow_nan=False, indent=2))
            print(f"Comparison reports: {output}", flush=True)
    return completed
