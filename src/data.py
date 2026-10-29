"""Load flow features for bootstrap training and incoming replay."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(name: str, spec: dict, cfg: dict, root: Path) -> dict:
    frame = pd.read_pickle(root / spec["path"])
    positions = np.arange(len(frame))
    if selection := spec.get("row_selection"):
        selected = frame[selection["column"]] == selection["include"]
        positions = positions[selected.to_numpy()]
        frame = frame.loc[selected]
    frame = frame.reset_index(drop=True)
    labels = {spec["family_column"], spec["binary_column"], "original_label",
              "mapped_label", "mapped_class_id", "label", "binary_label"}
    excluded = labels | set(spec["exclude"]) | set(cfg["evidence"]["forbidden_features"])
    candidates = (frame.attrs["feature_names"] if spec["feature_source"] == "metadata"
                  else frame.columns)
    names = [name for name in candidates if name not in excluded]
    categorical = list(spec["categorical"])
    numeric = [name for name in names if name not in categorical]
    raw = frame[names].copy()
    x = raw.copy()
    for column in numeric:
        x[column] = pd.to_numeric(x[column], errors="raise").replace([np.inf, -np.inf], np.nan)
    for column in categorical:
        x[column] = x[column].astype("string").fillna("<MISSING>").astype(str)
    family = frame[spec["family_column"]].astype(str).str.strip()
    if mapping := spec.get("family_map"):
        family = family.map(lambda value: mapping.get(value, value))
    y = (family.to_numpy() != spec["benign_value"]).astype(np.uint8)
    # Identical inputs share a side of the bootstrap/replay split.
    groups = pd.util.hash_pandas_object(x, index=False).to_numpy(dtype=np.uint64, copy=True)
    groups ^= np.uint64(cfg["general"]["seed"])
    groups ^= groups >> np.uint64(30)
    groups *= np.uint64(0xbf58476d1ce4e5b9)
    groups ^= groups >> np.uint64(27)
    train = groups % 10000 < round(cfg["dataplane"]["train_fraction"] * 10000)
    stream = ~train
    if official := spec.get("official_split"):
        boundary = official["reset_index"]
        train = np.arange(len(x)) < boundary
        stream = ~train
        train &= ~np.isin(groups, groups[stream])
    return {"name": name, "x": x, "raw_x": raw, "family": family.to_numpy(),
            "y": y, "row_positions": positions, "numeric": numeric,
            "categorical": categorical, "train": train, "stream": stream}
