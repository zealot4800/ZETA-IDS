"""Fast bootstrap attack/benign rules with a separate family tree."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


class DataPlane:
    def __init__(self, data: dict, cfg: dict):
        self.options = cfg["dataplane"]
        transformers = [("numeric", SimpleImputer(strategy="median", keep_empty_features=True),
                         data["numeric"])]
        if data["categorical"]:
            transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"),
                                 data["categorical"]))
        self.preprocessor = ColumnTransformer(transformers, sparse_threshold=1.0)
        train = data["train"]
        values = self.preprocessor.fit_transform(data["x"].loc[train])
        self.binary = DecisionTreeClassifier(**self.options["tree"],
                                             random_state=cfg["general"]["seed"])
        self.binary.fit(values, data["y"][train])
        self.attack_column = list(self.binary.classes_).index(1)
        attacks = data["y"][train] == 1
        self.family = DecisionTreeClassifier(**self.options["tree"],
                                             random_state=cfg["general"]["seed"] + 1)
        self.family.fit(values[attacks], data["family"][train][attacks])
        # Controller support uses a different model, trained on bootstrap rows
        # only. Its probabilities are model scores, not calibrated confidence.
        self.support_model = ExtraTreesClassifier(
            **cfg["controller_support"]["ensemble"],
            random_state=cfg["general"]["seed"] + 2, n_jobs=1)
        self.support_model.fit(values, data["y"][train])
        self.support_attack_column = list(self.support_model.classes_).index(1)
        # The replay cohort is already resident in memory. Transform features
        # once, avoiding repeated pandas/sklearn native indexing in a long run.
        # No replay labels are used by any prediction here.
        all_values = self.preprocessor.transform(data["x"])
        self.probabilities = self.binary.predict_proba(all_values)[:, self.attack_column]
        self.support_probabilities = self.support_model.predict_proba(all_values)[:, self.support_attack_column]
        self.families = self.family.predict(all_values)
        self.leaves = self.binary.apply(all_values)

    def predict(self, positions):
        probabilities = self.probabilities[positions]
        decisions = np.full(len(positions), "UNRESOLVED", dtype="<U12")
        decisions[probabilities <= self.options["benign_max_probability"]] = "BENIGN"
        decisions[probabilities >= self.options["attack_min_probability"]] = "ATTACK"
        return decisions, self.families[positions], self.leaves[positions], probabilities
