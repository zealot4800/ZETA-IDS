"""Load and partition each dataset independently for LOAO evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _features(frame: pd.DataFrame, spec: dict) -> tuple[list[str], list[str], list[str]]:
    labels = {spec["family_column"], spec["binary_column"], "original_label",
              "mapped_label", "mapped_class_id", "label", "binary_label"}
    if spec["feature_source"] == "metadata":
        names = list(frame.attrs.get("feature_names", []))
        if not names:
            raise ValueError("Dataset has no feature_names metadata")
    elif spec["feature_source"] == "columns":
        names = [name for name in frame.columns if name not in labels | set(spec["exclude"])]
    else:
        raise ValueError("Unknown feature_source")
    if len(names) != len(set(names)) or any(name not in frame for name in names):
        raise ValueError("Feature list has duplicates or absent columns")
    if labels.intersection(names) or set(spec["exclude"]).intersection(names):
        raise ValueError("Labels or excluded fields would enter model features")
    categorical = list(spec["categorical"])
    if not set(categorical) <= set(names):
        raise ValueError("Categorical feature is absent")
    numeric = [name for name in names if name not in categorical]
    if not all(pd.api.types.is_numeric_dtype(frame[name]) for name in numeric):
        raise ValueError("Non-numeric feature omitted from categorical config")
    return names, numeric, categorical


def _prepare(frame: pd.DataFrame, names: list[str], numeric: list[str],
             categorical: list[str]) -> pd.DataFrame:
    x = frame[names].copy()
    for name in numeric:
        x[name] = pd.to_numeric(x[name], errors="raise").replace([np.inf, -np.inf], np.nan)
    for name in categorical:
        x[name] = x[name].astype("string").fillna("<MISSING>").astype(str)
    return x


def _partitions(x: pd.DataFrame, seed: int, fractions: dict,
                buckets: int) -> tuple[np.ndarray, np.ndarray]:
    """Hash full model inputs so identical inputs share a partition."""
    h = pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64, copy=True)
    h ^= np.uint64(seed)
    h ^= h >> np.uint64(30)
    h *= np.uint64(0xbf58476d1ce4e5b9)
    h ^= h >> np.uint64(27)
    h *= np.uint64(0x94d049bb133111eb)
    h ^= h >> np.uint64(31)
    u = (h % np.uint64(buckets)).astype(np.int32)
    train_end = round(fractions["train"] * buckets)
    val_end = round((fractions["train"] + fractions["validation"]) * buckets)
    part = np.where(u < train_end, 0, np.where(u < val_end, 1, 2))
    return h, part


def _read_dataset(name: str, spec: dict, cfg: dict, root: Path) -> dict:
    frame = pd.read_pickle(root / spec["path"])
    source_rows = len(frame)
    selection = spec.get("row_selection")
    if selection is not None:
        column = selection["column"]
        if frame[column].isna().any():
            raise ValueError(f"{name}: missing source split values")
        values = set(frame[column].astype(str))
        expected_values = set(selection["expected_values"])
        if values != expected_values:
            raise ValueError(f"{name}: source split values differ: {values} != {expected_values}")
        frame = frame.loc[frame[column] == selection["include"]]
        if (len(frame) != selection["expected_included_rows"] or
                source_rows - len(frame) != selection["expected_excluded_rows"]):
            raise ValueError(f"{name}: source split row counts differ from config")
    excluded_rows = source_rows - len(frame)
    features, numeric, categorical = _features(frame, spec)
    if not numeric:
        raise ValueError(f"{name}: benign regions require numeric features")
    if frame[[spec["family_column"], spec["binary_column"]]].isna().any().any():
        raise ValueError(f"{name}: missing labels")
    detailed = frame[spec["family_column"]].astype(str).str.strip()
    if "family_map" in spec:
        mapping = spec["family_map"]
        labels = set(detailed)
        if labels != set(mapping):
            raise ValueError(f"{name}: unmapped={sorted(labels - set(mapping))}, "
                             f"unused mappings={sorted(set(mapping) - labels)}")
        family = detailed.map(mapping).to_numpy()
    else:
        family = detailed.to_numpy()
    binary = frame[spec["binary_column"]].to_numpy()
    if np.isin(binary, [0, 1]).all():
        y = binary.astype(np.uint8)
    elif np.isin(binary, ["BENIGN", "ATTACK"]).all():
        y = (binary == "ATTACK").astype(np.uint8)
    else:
        raise ValueError(f"{name}: unsupported binary labels")
    if not np.array_equal(y == 0, family == spec["benign_value"]):
        raise ValueError(f"{name}: binary and family labels disagree")
    expected = set(cfg["expected_attacks"][name])
    actual = set(family) - {spec["benign_value"]}
    if actual != expected:
        raise ValueError(f"{name}: attack taxonomy differs: missing={sorted(expected-actual)}, "
                         f"unexpected={sorted(actual-expected)}")
    if "class_id_column" in spec:
        ids = frame[spec["class_id_column"]].to_numpy()
        class_names = frame.attrs.get("harmonization", {}).get("classes")
        if (not class_names or len(class_names) != len(expected) + 1 or
                set(class_names) != expected | {spec["benign_value"]} or
                set(ids) != set(range(len(class_names))) or
                not np.array_equal(family, np.asarray(class_names, dtype=object)[ids])):
            raise ValueError(f"{name}: mapped class IDs disagree")
    original = (frame["original_label"].astype(str).str.strip().to_numpy()
                if "original_label" in frame else None)
    original_counts = (pd.Series(original).value_counts().to_dict()
                       if original is not None else {})
    sampling = frame.attrs.get("sampling", {})
    source_original_counts = sampling.get("source_label_counts", {})
    x = _prepare(frame, features, numeric, categorical)
    buckets = cfg["partition_buckets"]
    groups, part = _partitions(x, cfg["seed"], cfg["split"], buckets)
    strategy = "feature-group " + "/".join(
        str(round(100 * cfg["split"][part])) for part in ("train", "validation", "test"))
    overlap_removed = 0
    if "official_split" in spec:
        official = spec["official_split"]
        ids = frame[official["id_column"]].to_numpy()
        reset = np.flatnonzero(np.diff(ids) < 0) + 1
        boundary = official["reset_index"]
        if (not np.array_equal(reset, [boundary]) or ids[0] != official["start_id"] or
                ids[boundary] != official["start_id"]):
            raise ValueError(f"{name}: id reset does not match the configured official split")
        part = np.full(len(x), 2, dtype=np.int8)
        part[:boundary] = np.where(groups[:boundary] % buckets <
                                   round(official["train_fraction"] * buckets), 0, 1)
        shared = np.isin(groups[:boundary], groups[boundary:])
        overlap_removed = int(shared.sum())
        part[:boundary][shared] = 3
        strategy = official["description"]
    train, val, test = (part == i for i in (0, 1, 2))
    if np.intersect1d(groups[train], groups[val]).size or np.intersect1d(
            groups[train | val], groups[test]).size:
        raise ValueError(f"{name}: identical model inputs cross split boundaries")
    for attack in expected:
        if not np.any(test & (family == attack)):
            raise ValueError(f"{name}: fixed test partition lacks {attack}")
    return {"name": name, "x": x, "family": family, "y": y,
            "source_rows": source_rows, "source_rows_excluded": excluded_rows,
            "source_selection": selection["description"] if selection else "all rows",
            "original_label": original, "original_counts": original_counts,
            "source_original_counts": source_original_counts,
            "sample_pattern_counts": sampling.get(
                "sample_distinct_feature_patterns_by_label", {}),
            "sampling_method": sampling.get("method", "unspecified"),
            "numeric": numeric, "categorical": categorical,
            "train": train, "validation": val, "test": test,
            "strategy": strategy, "duplicate_rows": len(groups) - len(np.unique(groups)),
            "official_overlap_removed": overlap_removed,
            "test_counts": {k: int(v) for k, v in pd.Series(family[test]).value_counts().items()}}
