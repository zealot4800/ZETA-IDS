"""Post-decision ground-truth accounting; never feeds routing or learning."""
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
import sys
from threading import Event, Thread


@contextmanager
def progress_heartbeat(snapshot, interval=15):
    """Keep status visible during slow provider calls without touching model state."""
    stopped = Event()

    def report():
        while not stopped.wait(interval):
            print(json.dumps(snapshot()), file=sys.stderr, flush=True)

    print(json.dumps(snapshot()), file=sys.stderr, flush=True)
    thread = Thread(target=report, name='run-progress', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def metrics(counts: Counter) -> dict:
    total = sum(counts.values())
    tp, tn = counts[(1, "ATTACK")], counts[(0, "BENIGN")]
    fp, fn = counts[(0, "ATTACK")], counts[(1, "BENIGN")]
    attacks = sum(n for (label, _), n in counts.items() if label == 1)
    benign = total - attacks
    resolved = tp + tn + fp + fn
    recall = tp / attacks if attacks else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "flows": total, "true_attacks": attacks, "true_benign": benign,
        "true_positive": tp, "true_negative": tn, "false_positive": fp,
        "false_negative_benign": fn,
        "uncertain_attacks": counts[(1, "UNCERTAIN")],
        "uncertain_benign": counts[(0, "UNCERTAIN")],
        "error_attacks": counts[(1, "ERROR")], "error_benign": counts[(0, "ERROR")],
        "correct_over_all": (tp + tn) / total if total else None,
        "resolved_accuracy": (tp + tn) / resolved if resolved else None,
        "coverage": resolved / total if total else None,
        "attack_precision": precision, "attack_recall": recall,
        "attack_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_positive_rate": fp / benign if benign else None,
    }


class GroundTruthReport:
    def __init__(self):
        self.overall = Counter()
        self.sources = defaultdict(Counter)
        self.families = defaultdict(Counter)
        self.raw_llm = Counter()

    def record(self, result: dict, binary: int, family: str):
        prediction = result["final_decision"]
        self.overall[(binary, prediction)] += 1
        self.sources[result["decision_source"]][(binary, prediction)] += 1
        self.families[family][(binary, prediction)] += 1
        if result["binary_decision"] != "NOT_RUN":
            self.raw_llm[(binary, result["binary_decision"])] += 1

    def summary(self):
        return {"overall": metrics(self.overall),
                "by_source": {key: metrics(counts) for key, counts in self.sources.items()},
                "by_family": {key: metrics(counts) for key, counts in self.families.items()},
                "raw_llm": metrics(self.raw_llm),
                "definition": "Ratios in [0,1]. Uncertain/error outputs are not correct; "
                              "attack recall includes all true attacks. Family metrics measure "
                              "binary detection within each true family, not family attribution."}


class ClassificationReport:
    """Macro class metrics using bounded counters instead of per-flow lists."""
    def __init__(self):
        self.actual = Counter()
        self.predicted = Counter()
        self.correct = Counter()

    def record(self, actual, predicted):
        self.actual[actual] += 1
        self.predicted[predicted] += 1
        if actual == predicted:
            self.correct[actual] += 1

    def summary(self):
        if not self.actual:
            return None
        labels = sorted(set(self.actual) | (set(self.predicted) - {'ERROR', 'UNCERTAIN'}))
        precision, recall, f1 = [], [], []
        for label in labels:
            true = self.correct[label]
            p = true / self.predicted[label] if self.predicted[label] else 0.0
            r = true / self.actual[label] if self.actual[label] else 0.0
            precision.append(p)
            recall.append(r)
            f1.append(2 * p * r / (p + r) if p + r else 0.0)
        return {'labels': labels, 'macro_precision': sum(precision) / len(labels),
                'macro_recall': sum(recall) / len(labels), 'macro_f1': sum(f1) / len(labels),
                'definition': 'Macro over observed true/predicted traffic classes; errors and abstentions are not target classes.'}
