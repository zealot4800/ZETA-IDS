#!/usr/bin/env python3
"""Export comparable metric definitions; does not certify matched protocols."""
import argparse
import csv
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results/baselines')
    args = parser.parse_args()
    rows = []
    for path in sorted(args.results_dir.glob('*/*/*/summary.json')):
        summary = json.loads(path.read_text())
        metrics = summary.get('ground_truth_metrics', {}).get('overall', {})
        statistics = summary.get('method_statistics', {})
        rows.append({
            'method': summary['method'], 'dataset': summary['dataset'], 'task': summary['task'],
            'run_id': path.parent.name, 'implementation': summary['implementation'],
            'completed': summary['completed'], 'embedding': summary['embedding'],
            'model': summary['model'], 'construction_flows': summary['construction_flows'],
            'replay_flows': summary['processed'], 'attack_f1': metrics.get('attack_f1'),
            'attack_precision': metrics.get('attack_precision'), 'attack_recall': metrics.get('attack_recall'),
            'coverage': metrics.get('coverage'), 'false_positive_rate': metrics.get('false_positive_rate'),
            'correct_over_all': metrics.get('correct_over_all'),
            'macro_class_f1': (summary.get('classification_metrics') or {}).get('macro_f1'),
            'llm_requests_total': sum(summary.get('api_requests', {}).values()),
            'llm_requests_construction': sum(summary.get('construction_api_requests', {}).values()),
            'input_tokens': summary.get('tokens', {}).get('input', 0),
            'output_tokens': summary.get('tokens', {}).get('output', 0),
            'elapsed_seconds': summary.get('elapsed_seconds'),
            'construction_errors': statistics.get('construction', {}).get('errors', 0),
            'replay_errors': sum(summary.get('errors', {}).values()),
            'summary_path': str(path)})
    if not rows:
        parser.error('No baseline summaries found')
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


if __name__ == '__main__':
    main()
