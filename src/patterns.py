"""Compact, source-bound signatures for recurring unresolved flows."""

import numpy as np

def signatures(model: dict, features: np.ndarray, names: list[str],
               feature_names: list[str], source: str) -> list[str]:
    indices = [feature_names.index(name) for name in names]
    edges = model["family_model"]["preprocessing"]["quantile_edges"]
    bins = np.column_stack([np.searchsorted(edges[index], features[:, index], side="right")
                            for index in indices])
    return [source + ":" + ":".join(map(str, row)) for row in bins]
