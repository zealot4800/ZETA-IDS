"""MA-IDS and SERA-IDS: training-error libraries frozen before replay."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .common import (VectorLibrary, admission_confidence, class_diverse,
                     condition_body, text, verify_conditions)


class TrainingProxyTools:
    """Four source-only statistical proxies, NOT SERA's external ToolSuite.

    The source flow's identical-feature group is excluded when scoring that flow.
    Thus rule admission never obtains a perfect score just by querying itself.
    """
    def __init__(self, values, labels, groups, seed):
        self.values, self.labels, self.groups = values, np.asarray(labels), groups
        self.seed = seed
        self.models = None

    def score(self, vector, label, group):
        # Cross-fit by a deterministic two-fold partition of unique feature groups.
        # Models are cached by fold, avoiding a new fit for every construction error.
        if self.models is None:
            rng = np.random.default_rng(self.seed)
            unique = rng.permutation(np.unique(self.groups))
            assignments = {key: i % 2 for i, key in enumerate(unique)}
            self.folds = np.asarray([assignments[key] for key in self.groups])
            self.group_fold = assignments
            self.models = {}
            for fold in (0, 1):
                mask = self.folds != fold
                if not mask.any() or len(np.unique(self.labels[mask])) < 2:
                    self.models[fold] = None
                    continue
                models = [
                    KNeighborsClassifier(n_neighbors=min(7, int(mask.sum()))),
                    ExtraTreesClassifier(n_estimators=32, min_samples_leaf=2,
                                         random_state=self.seed, n_jobs=1),
                    RandomForestClassifier(n_estimators=32, min_samples_leaf=2,
                                           random_state=self.seed + 1, n_jobs=1),
                    LogisticRegression(max_iter=500, random_state=self.seed)]
                for model in models:
                    model.fit(self.values[mask], self.labels[mask])
                self.models[fold] = models
        models = self.models[self.group_fold[group]]
        if models is None:
            return [0., 0., 0., 0.]
        return [float(model.predict_proba(vector[None])[0, list(model.classes_).index(label)])
                if label in model.classes_ else 0. for model in models]


class ExperienceIDS:
    def __init__(self, method, reasoner, embedder, options, training, labels,
                 values, groups, seed):
        self.method, self.reasoner, self.embedder = method, reasoner, embedder
        self.options, self.library = options, VectorLibrary()
        self.training, self.labels = training, labels
        self.vectors = embedder.encode([text(flow) for flow in training])
        numeric = pd.DataFrame(training).select_dtypes(include='number')
        self.medians = numeric.median().fillna(0).to_dict()
        self.tools = (TrainingProxyTools(values, labels, groups, seed)
                      if method == 'sera_ids' else None)
        self.values, self.groups = values, groups
        self.construction = {'processed': 0, 'errors': 0, 'misclassified': 0,
                             'rules_admitted': 0, 'rules_queued': 0}
        self.queue = []

    def context(self, flow, vector):
        if self.method == 'ma_ids':
            selected = self.library.nearest(vector, 1)
            return [{'rule': entry, 'similarity': score} for entry, score in selected
                    if score >= self.options['similarity_threshold']]
        pool = self.library.nearest(vector, self.options['candidate_pool'])
        selected = class_diverse(pool, self.options['retrieval_k'])
        return [{'rule': entry, 'similarity': score,
                 'condition_evidence': verify_conditions(flow, entry['conditions'])}
                for entry, score in selected]

    def fit(self, directory):
        # Streaming journal retains construction failures and rules for audit.
        with (directory / 'construction.jsonl').open('x') as output:
            for index, (flow, label, vector) in enumerate(zip(self.training, self.labels, self.vectors)):
                context = self.context(flow, vector)
                event = {'construction_index': index, 'true_label': label}
                try:
                    prediction = self.reasoner.classify(flow, context)
                    event.update(prediction)
                    if prediction['label'] != label:
                        self.construction['misclassified'] += 1
                        description = self.reasoner.explain_error(flow, prediction['label'], label, context)
                        entry = {'label': label, 'description': description,
                                 'provenance': {'construction_index': index,
                                                'confused_class': prediction['label'],
                                                'phase': 'bootstrap_construction'}}
                        admit = True
                        if self.method == 'sera_ids':
                            entry['conditions'] = condition_body(flow, self.medians)
                            signals = self.tools.score(self.values[index], label, self.groups[index])
                            confidence = admission_confidence(signals, self.options['signal_threshold'])
                            entry['provenance'].update(confidence_backend='training_proxy',
                                                       signals=signals, confidence=confidence)
                            admit = confidence >= self.options['admission_threshold']
                        if admit:
                            key = (vector if self.method == 'ma_ids' else
                                   self.embedder.encode([description])[0])
                            self.library.add(key, entry)
                            self.construction['rules_admitted'] += 1
                        else:
                            self.queue.append(entry)
                            self.construction['rules_queued'] += 1
                        event.update(rule=entry, admitted=admit)
                except (ValueError, KeyError, TypeError, IndexError) as exc:
                    self.construction['errors'] += 1
                    event['error'] = str(exc)
                self.construction['processed'] += 1
                output.write(json.dumps(event, allow_nan=False) + '\n')
                output.flush()
                if self.construction['processed'] >= 3 and self.construction['errors'] == self.construction['processed']:
                    raise RuntimeError('All initial construction calls failed; inspect construction.jsonl and Ollama')
        self.library.frozen = True
        self.save(directory)

    def predict(self, flow, vector=None):
        vector = self.embedder.encode([text(flow)])[0] if vector is None else vector
        context = self.context(flow, vector)
        result = self.reasoner.classify(flow, context)
        return {**result, 'retrieved_rules': len(context), 'context': context}

    def save(self, directory):
        self.library.save(directory)
        (directory / 'administrative_queue.json').write_text(json.dumps(self.queue, indent=2))

    def statistics(self):
        return {'construction': self.construction, 'active_rules': len(self.library.entries),
                'library_frozen': self.library.frozen}
