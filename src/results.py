"""One bounded comparison report shared by every architecture."""
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import tempfile


@contextmanager
def optional_log(path):
    """Avoid serializing flow records unless detailed output was requested."""
    if path is None:
        yield None
    else:
        with path.open('a') as stream:
            yield stream


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def update_results(directory, summary):
    """Keep the latest attempt per method/dataset/size/task, without lost writes."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'summary.json'
    # Lock the directory itself: no extra persistent lock file is needed.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        report = json.loads(path.read_text()) if path.exists() else {
            'schema_version': 1,
            'policy': 'Latest attempt per method/dataset/size/task; incomplete attempts are marked.',
            'metric_definition': 'Ratios in [0,1]; errors and abstentions count against coverage and attack recall.',
            'runs': {},
        }
        key = '/'.join(str(summary[field]) for field in ('method', 'dataset', 'dataset_size', 'task'))
        entry = {**summary, 'updated_at': datetime.now(timezone.utc).isoformat()}
        # Overall metrics retain all confusion counts. Family breakdown keeps
        # only the values needed to spot rare-family regressions.
        metrics = entry.pop('ground_truth_metrics', {})
        entry['metrics'] = metrics.get('overall', {})
        entry['by_family'] = {
            family: {name: values.get(name) for name in
                     ('flows', 'correct_over_all', 'attack_recall', 'coverage')}
            for family, values in metrics.get('by_family', {}).items()
        }
        report['runs'][key] = entry
        atomic_json(path, report)
    finally:
        os.close(descriptor)
    return path


def print_result(summary):
    """Print a short completion notice; detailed metrics live in the report."""
    metrics = summary.get('ground_truth_metrics', {}).get('overall', {})
    print(json.dumps({key: summary[key] for key in
                      ('method', 'dataset', 'dataset_size', 'processed', 'completed', 'results_file')}
                     | {'accuracy': metrics.get('correct_over_all'),
                        'attack_f1': metrics.get('attack_f1'),
                        'elapsed_seconds': round(summary['elapsed_seconds'], 3)}))
