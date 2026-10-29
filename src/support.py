"""Bootstrap-only local references and model support for controller decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import feature_batch


class ControllerSupport:
    def __init__(self, data: dict, dataplane, cfg: dict):
        self.options = cfg["controller_support"]
        self.numeric = data["numeric"]
        self.categorical = data["categorical"]
        train = np.flatnonzero(data["train"])
        frame = data["x"].iloc[train]
        # Exact duplicates must not supply spurious neighbour support. Conflicting
        # labels for the same features cannot be trusted as examples either.
        hashes = pd.util.hash_pandas_object(frame, index=False).to_numpy()
        rows = pd.DataFrame({"hash": hashes, "label": data["y"][train], "position": train})
        rows = rows.loc[rows.groupby("hash")["label"].transform("nunique") == 1]
        train = rows.drop_duplicates("hash")["position"].to_numpy()
        frame = data["x"].iloc[train]
        numbers = frame[self.numeric].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            self.median = np.nanmedian(numbers, axis=0)
            spread = np.nanquantile(numbers, .75, axis=0) - np.nanquantile(numbers, .25, axis=0)
        self.median = np.nan_to_num(self.median)
        self.scale = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
        self.all_numbers, self.all_categories = self._encode(data["x"])
        leaves = dataplane.leaves[train]
        rng = np.random.default_rng(cfg["general"]["seed"] + 101)
        self.pools = {}
        self.reference_features = self._important_features(dataplane, data)
        for leaf in np.unique(leaves):
            local = train[leaves == leaf]
            selected = []
            # Preserve rare source families when bounding the reference table.
            for family in np.unique(data["family"][local]):
                candidates = local[data["family"][local] == family]
                selected.extend(rng.permutation(candidates)[:self.options["references_per_family"]])
            positions = np.asarray(selected, dtype=int)
            numeric, categories = self.all_numbers[positions], self.all_categories[positions]
            distances = self._distances(numeric, categories, numeric, categories)
            np.fill_diagonal(distances, np.inf)
            k = self.options["neighbours"]
            radius = (float(np.quantile(np.sort(distances, axis=1)[:, k - 1], .95))
                      if len(positions) > k else 0.0)
            self.pools[int(leaf)] = {
                "numeric": numeric, "categories": categories,
                "labels": data["y"][positions], "radius": radius,
                "families": [cfg["family_map"][data["name"]].get(str(family))
                             for family in data["family"][positions]],
                "examples": feature_batch(data["raw_x"][self.reference_features], positions)}

    def _important_features(self, dataplane, data):
        importance = dataplane.support_model.feature_importances_
        weights = dict(zip(self.numeric, importance[:len(self.numeric)]))
        if self.categorical:
            encoder = dataplane.preprocessor.named_transformers_["categorical"]
            offset = len(self.numeric)
            for name, categories in zip(self.categorical, encoder.categories_):
                weights[name] = float(importance[offset:offset + len(categories)].sum())
                offset += len(categories)
        ranked = sorted(data["x"].columns, key=lambda name: (-weights.get(name, 0), name))
        return ranked[:self.options["example_features"]]

    def _encode(self, frame):
        values = frame[self.numeric].to_numpy(dtype=float)
        values = np.where(np.isfinite(values), values, self.median)
        numbers = np.clip(np.arcsinh((values - self.median) / self.scale), -8, 8)
        return numbers, frame[self.categorical].to_numpy(dtype=str)

    @staticmethod
    def _distances(numbers, categories, other_numbers, other_categories):
        distances = ((numbers * numbers).sum(axis=1)[:, None] +
                     (other_numbers * other_numbers).sum(axis=1)[None, :] -
                     2 * numbers @ other_numbers.T) / max(numbers.shape[1], 1)
        distances = np.maximum(distances, 0)
        for index in range(categories.shape[1]):
            distances += (categories[:, index, None] != other_categories[None, :, index])
        return np.sqrt(distances)

    def contexts(self, positions, leaves, probabilities):
        numbers, categories = self.all_numbers[positions], self.all_categories[positions]
        result = []
        for i, (leaf, probability) in enumerate(zip(leaves, probabilities)):
            pool = self.pools.get(int(leaf))
            context = {"decision": "UNCERTAIN", "p_attack": float(probability),
                       "supported": False, "neighbour_agreement": 0.0,
                       "local_match": False, "source_examples": [], "candidate_families": []}
            if pool is None:
                result.append(context)
                continue
            distances = self._distances(numbers[i:i+1], categories[i:i+1],
                                        pool["numeric"], pool["categories"])[0]
            nearest = np.argsort(distances, kind="stable")
            k = self.options["neighbours"]
            neighbours = nearest[:k]
            attack = probability >= self.options["attack_min_probability"]
            benign = probability <= self.options["benign_max_probability"]
            vote = float(np.mean(pool["labels"][neighbours] == int(attack)))
            local = len(neighbours) == k and distances[neighbours[-1]] <= max(pool["radius"], 1e-8)
            supported = (bool(attack or benign) and bool(local) and
                         vote >= self.options["min_neighbour_agreement"])
            # Examples from both classes explain conflicts rather than hiding them.
            for label in (0, 1):
                selected = nearest[pool["labels"][nearest] == label]
                for index in selected[:self.options["examples_per_class"]]:
                    context["source_examples"].append({
                        "features": pool["examples"][index],
                        "binary": "ATTACK" if label else "BENIGN",
                        "family": pool["families"][index],
                        "distance": round(float(distances[index]), 6)})
            context.update(
                decision=("ATTACK" if attack else "BENIGN") if supported else "UNCERTAIN",
                supported=bool(supported), neighbour_agreement=vote, local_match=bool(local),
                neighbours=int(len(neighbours)),
                candidate_families=list(dict.fromkeys(
                    pool["families"][index] for index in neighbours
                    if pool["labels"][index] == 1 and pool["families"][index])))
            result.append(context)
        return result
