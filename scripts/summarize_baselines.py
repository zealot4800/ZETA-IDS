#!/usr/bin/env python3
"""Display the shared comparison report, with support for older run folders."""
import argparse
import csv
import json
from pathlib import Path
import sys


def load_summaries(results_dir):
    compact = results_dir / 'summary.json'
    if compact.exists():
        return json.loads(compact.read_text())['runs'].values()
    return [json.loads(path.read_text()) for path in sorted(results_dir.rglob('summary.json'))]


def summary_row(summary):
    metrics = summary.get('metrics', summary.get('ground_truth_metrics', {}).get('overall', {}))
    return {
        'method': summary.get('method', 'aegis'), 'dataset': summary['dataset'],
        'size': summary.get('dataset_size', 'legacy'), 'task': summary.get('task', 'binary'),
        'complete': summary.get('completed', False), 'flows': summary['processed'],
        'accuracy': metrics.get('correct_over_all'), 'precision': metrics.get('attack_precision'),
        'recall': metrics.get('attack_recall'), 'f1': metrics.get('attack_f1'),
        'coverage': metrics.get('coverage'), 'fpr': metrics.get('false_positive_rate'),
        'seconds': summary.get('elapsed_seconds'),
        'llm_requests': sum(summary.get('api_requests', {}).values()),
        'errors': sum(summary.get('errors', {}).values()),
    }


def print_csv(rows):
    writer = csv.DictWriter(sys.stdout, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def print_table(rows):
    headers = list(rows[0])
    rendered = [[f'{value:.4f}' if isinstance(value, float) else str(value) for value in row.values()]
                for row in rows]
    widths = [max(len(header), *(len(row[i]) for row in rendered)) for i, header in enumerate(headers)]
    for row in [headers, *rendered]:
        print('  '.join(value.ljust(width) for value, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'results')
    parser.add_argument('--format', choices=['table', 'csv'], default='table')
    args = parser.parse_args()
    rows = [summary_row(summary) for summary in load_summaries(args.results_dir)]
    if not rows:
        parser.error('No results yet. Run an architecture first; dry runs do not create results.')
    rows.sort(key=lambda row: (row['method'], row['dataset'], str(row['size']), row['task']))
    (print_csv if args.format == 'csv' else print_table)(rows)


if __name__ == '__main__':
    main()
