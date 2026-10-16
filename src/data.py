"""Read standardized flow PKLs and select a reproducible source subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def feature_names(config: dict) -> list[str]:
    return [column["name"] for column in config["columns"]]


def feature_schema(config: dict) -> list[dict]:
    return [{"name": column["name"], "unit": column["unit"], "dtype": config["dtype"]}
            for column in config["columns"]]


def feature_fingerprint(config: dict) -> str:
    return hashlib.sha256(json.dumps({"schema_version": config["compact_schema_version"],
                                     "feature_schema": feature_schema(config)},
                                    sort_keys=True).encode()).hexdigest()


def load_sample(year: str, path: str, fraction: float, seed: int,
                config: dict) -> tuple[np.ndarray, np.ndarray, str]:
    """Return sampled IDs and features in original order, without carrying labels."""
    if not 0 < fraction <= 1 or type(seed) is not int:
        raise ValueError("Sampling needs a fraction in (0, 1] and an integer seed")
    frame = pd.read_pickle(Path(path))
    if not isinstance(frame, pd.DataFrame) or "schema" not in frame.attrs:
        raise ValueError("Expected a standardized flow DataFrame")
    schema = frame.attrs["schema"]
    metadata = config["metadata_columns"]
    names = feature_names(config)
    expected_schema = feature_schema(config)
    if schema.get("dataset") != f"CIC-IDS{year}" or list(frame.columns[:len(metadata)]) != metadata:
        raise ValueError("Dataset or metadata schema differs")
    if schema.get("schema_version") == config["compact_schema_version"]:
        if (schema.get("feature_names") != names or
                schema.get("feature_schema") != expected_schema or
                frame.attrs.get("feature_schema_fingerprint_sha256") != feature_fingerprint(config)):
            raise ValueError("Compact feature schema differs")
        columns = names
    elif schema.get("schema_version") == config["full_schema_version"]:
        columns = [item[f"cic{year}"] for item in config["columns"]]
        source_schema = {item["name"]: item for item in schema["feature_schema"]}
        for expected, column in zip(expected_schema, columns):
            actual = source_schema.get(column)
            if actual is None or actual["unit"] != expected["unit"] or actual["dtype"] != config["dtype"]:
                raise ValueError(f"Full feature schema differs: {column}")
    else:
        raise ValueError("Unsupported flow schema")
    if (len(frame) != frame.attrs["provenance"]["total_rows"] or
            any(not pd.api.types.is_numeric_dtype(frame[name]) for name in columns)):
        raise ValueError("Flow count or numeric feature schema differs")
    fingerprint = frame.attrs.get("data_fingerprint_sha256")
    if not isinstance(fingerprint, str):
        raise ValueError("Missing source data fingerprint")
    count = round(len(frame) * fraction)
    if count < 1:
        raise ValueError("Sampling fraction selects no flows")
    indices = np.sort(np.random.default_rng(seed).choice(len(frame), count, replace=False))
    sample_ids = frame["sample_id"].iloc[indices].to_numpy(copy=True)
    if not pd.Index(sample_ids).is_unique:
        raise ValueError("Selected sample IDs are not unique")
    features = frame[columns].iloc[indices].to_numpy(
        dtype=np.dtype(config["preprocessing"]["model_input_dtype"]), copy=True)
    return sample_ids, features, fingerprint


def review_allowlist(path: str, year: str, model: dict,
                     requested_ids: set[str] | None = None) -> dict[str, str]:
    """Only frozen source training/calibration rows may corroborate memory."""
    frame = pd.read_pickle(Path(path))
    info = frame.attrs.get("subset")
    if (not isinstance(info, dict) or info.get("year") != year or
            frame.attrs.get("schema", {}).get("dataset") != f"CIC-IDS{year}" or
            info.get("parent_fingerprint_sha256") !=
            model["source_artifact"]["data_fingerprint_sha256"] or
            info.get("parent_partition_index_sha256") !=
            model["partition"]["partition_index_sha256"]):
        raise ValueError("Review source is not bound to the frozen model")
    indices = np.asarray(info["original_indices"], dtype=np.int64)
    tags = np.asarray(info["frozen_partition_codes"], dtype=np.uint8)
    digest = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
    if (len(indices) != len(frame) or len(tags) != len(frame) or
            np.any(np.diff(indices) <= 0) or np.any(tags > 3) or
            digest != info["original_index_sha256"] or not frame["sample_id"].is_unique):
        raise ValueError("Review source indices or partition tags differ")
    return {str(sample_id): ("training" if tag == 0 else "calibration")
            for sample_id, tag in zip(frame["sample_id"], tags)
            if tag in (0, 1) and (requested_ids is None or str(sample_id) in requested_ids)}
