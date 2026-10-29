"""Run explicitly labeled paper adaptations against repository datasets."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_fscore_support

from ..controller import load_config
from ..data import load_dataset
from ..evidence import digest
from ..llm import Ollama
from ..reporting import GroundTruthReport
from .common import (Features, LocalReasoner, TextEmbedding, binary_label,
                     raw_records, select_source)

METHODS = ('rlfe_ids', 'scala_nids', 'ma_ids', 'sera_ids')
DATASETS = ('kdd', 'cicids2018', 'ton_iot', 'unsw_nb15')


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def package_versions():
    result = {'python': sys.version.split()[0]}
    for name in ('numpy', 'pandas', 'scikit-learn', 'scipy', 'torch',
                 'sentence-transformers', 'transformers', 'PyYAML'):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def validate(options, method):
    if not 0 < options['general']['validation_fraction'] < 1:
        raise ValueError('validation_fraction must be strictly between 0 and 1')
    for key in ('batch_size', 'epochs', 'prototypes_per_class', 'references_per_class', 'retrieval_k'):
        if options['rlfe_ids'][key] <= 0:
            raise ValueError(f'rlfe_ids.{key} must be positive')
    if options['rlfe_ids']['distance'] not in ('euclidean', 'mahalanobis'):
        raise ValueError('RLFE distance must be euclidean or mahalanobis')
    if options['sera_ids']['confidence_backend'] != 'training_proxy':
        raise ValueError('Only explicitly labeled training_proxy ToolSuite is implemented')
    for key in ('batch_size', 'chunk_size', 'buffer_size', 'inspection_period'):
        if options['scala_nids'][key] <= 0:
            raise ValueError(f'scala_nids.{key} must be positive')
    if method == 'scala_nids' and options['general']['task'] != 'binary':
        raise ValueError('SCALA-NIDS implements binary anomaly screening; use --task binary')


def run_one(args, method, dataset):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / 'config/controller.yaml')
    options = yaml.safe_load(args.config.read_text())
    options['general']['task'] = args.task or options['general']['task']
    if args.construction_flows is not None:
        options['general']['construction_limit'] = args.construction_flows
    if options['embedding'].get('cache_folder'):
        options['embedding']['cache_folder'] = str(root / options['embedding']['cache_folder'])
    if args.epochs is not None:
        options['rlfe_ids']['epochs'] = args.epochs
        for key in ('domain_epochs', 'reconstruction_epochs', 'joint_epochs'):
            options['scala_nids'][key] = args.epochs
    validate(options, method)
    spec = yaml.safe_load((root / cfg['general']['datasets_config']).read_text())['datasets'][dataset]
    data = load_dataset(dataset, spec, cfg, root)
    train, validation = select_source(data, options['general']['construction_limit'],
                                      cfg['general']['seed'], options['general']['validation_fraction'])
    replay = np.flatnonzero(data['stream'])
    if args.max_flows is not None:
        replay = replay[:args.max_flows]
    if not len(train) or not len(validation) or not len(replay):
        raise ValueError('Construction, validation, and replay cohorts must be nonempty')
    task = options['general']['task']
    labels = np.where(data['y'][train] == 0, 'BENIGN',
                      'ATTACK' if task == 'binary' else data['family'][train]).tolist()
    if len(set(labels)) < 2:
        raise ValueError('Construction cohort needs at least two classes; increase --construction-flows')
    cfg['baseline_llm'] = {**copy.deepcopy(cfg['investigation_llm']), **options['llm']}
    for key, env in [('model', 'OLLAMA_BASELINE_MODEL'), ('base_url', 'OLLAMA_BASE_URL')]:
        if os.environ.get(env):
            cfg['baseline_llm'][key] = os.environ[env]
    if method == 'scala_nids':
        cfg['baseline_llm']['request_retries'] = 0  # Paper uses immediate fallback.
    preview = {'method': method, 'dataset': dataset, 'implementation': 'paper_adaptation',
               'task': task, 'construction_flows': len(train), 'validation_flows': len(validation),
               'requested_flows': len(replay), 'classes': sorted(set(labels)),
               'model': cfg['baseline_llm']['model'],
               'embedding': options['embedding']['model'] if method in ('ma_ids', 'sera_ids') else None,
               'stream_order': 'repository_row_order_not_certified_temporal'}
    if args.dry_run:
        print(json.dumps({'mode': 'preview', **preview, 'api_requests': 0}))
        return
    if method in ('rlfe_ids', 'scala_nids'):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError('Install requirements-baselines.txt (CPU PyTorch recommended)') from exc
        torch.set_num_threads(int(os.environ.get('ZETA_NATIVE_THREADS', '1')))
    output_root = args.output_dir or root / 'results/baselines'
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    directory = output_root / method / dataset / run_id
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    client, report = Ollama(cfg), GroundTruthReport()
    reasoner = LocalReasoner(client, cfg, sorted(set(labels)))
    features = Features().fit(data['x'].iloc[train], data['numeric'], data['categorical'])
    values = features.transform(data['x'].iloc[train])
    raw = raw_records(data, train)
    manifest = {**preview, 'configuration': options, 'llm_configuration': cfg['baseline_llm'],
                'package_versions': package_versions(),
                'dataset_specification': spec, 'seed': cfg['general']['seed'],
                'dataset_sha256': sha256(root / spec['path']),
                'source_sha256': {str(p.relative_to(root)): sha256(p)
                                  for p in sorted((root / 'src').rglob('*.py'))},
                'protocol': 'bootstrap_only_construction; no_replay_label_feedback',
                'adaptations': {'llm_backend': 'local_ollama', 'data': 'repository_schemas',
                                'confidence_backend': options['sera_ids']['confidence_backend']
                                if method == 'sera_ids' else None}}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    np.savez_compressed(directory / 'cohorts.npz', construction=train, validation=validation, replay=replay)
    with (directory / 'preprocessor.pkl').open('wb') as stream:
        pickle.dump(features, stream)
    algorithm = None
    phase = 'construction'
    processed, errors = 0, Counter()
    true_classes, predicted_classes = [], []
    construction_seconds = None
    try:
        if method in ('ma_ids', 'sera_ids'):
            from .experience import ExperienceIDS
            embedder = TextEmbedding(options['embedding'])
            groups = pd.util.hash_pandas_object(data['x'].iloc[train], index=False).to_numpy()
            algorithm = ExperienceIDS(method, reasoner, embedder, options[method], raw,
                                      labels, values, groups, cfg['general']['seed'])
        elif method == 'rlfe_ids':
            from .rlfe import RLFEIDS
            algorithm = RLFEIDS(reasoner, options[method], values, raw, labels, cfg['general']['seed'])
        else:
            from .scala import SCALANIDS
            algorithm = SCALANIDS(reasoner, options[method], values, data['y'][train],
                                  features.transform(data['x'].iloc[validation]), data['y'][validation],
                                  len(data['numeric']), cfg['general']['seed'])
        algorithm.fit(directory)
        construction_seconds = time.perf_counter() - started
        construction_requests = dict(client.requests)
        phase = 'replay'
        chunk_size = options['scala_nids']['chunk_size'] if method == 'scala_nids' else 1
        with (directory / 'flows.jsonl').open('x') as stream:
            for start in range(0, len(replay), chunk_size):
                positions = replay[start:start + chunk_size]
                flows = raw_records(data, positions)
                encoded = features.transform(data['x'].iloc[positions])
                before_requests = sum(client.requests.values())
                tick = time.perf_counter()
                if method == 'scala_nids':
                    # Batch failure is fatal: do not skip unknown partial updates.
                    results = algorithm.predict_chunk(encoded)
                else:
                    try:
                        result = algorithm.predict(flows[0], encoded[0]) if method == 'rlfe_ids' else algorithm.predict(flows[0])
                        results = [result]
                    except (ValueError, KeyError, TypeError, IndexError) as exc:
                        results = [{'label': 'ERROR', 'error': str(exc)}]
                duration = (time.perf_counter() - tick) * 1000
                requests = sum(client.requests.values()) - before_requests
                for position, result in zip(positions, results):
                    decision = binary_label(result['label'], task)
                    record = {'flow_id': digest([dataset, int(data['row_positions'][position])]),
                              'row_position': int(data['row_positions'][position]),
                              'final_decision': decision, 'binary_decision': decision,
                              'decision_source': method, 'method_output': result,
                              'latency_ms': duration / len(positions),
                              'latency_scope': 'chunk_prediction_and_adaptation_amortized' if method == 'scala_nids' else 'flow_inference',
                              'chunk_llm_requests': requests, 'chunk_index': start // chunk_size}
                    # Ground truth first enters here, after inference and adaptation.
                    truth = int(data['y'][position])
                    family = str(data['family'][position])
                    report.record(record, truth, family)
                    true_class = ('BENIGN' if not truth else 'ATTACK') if task == 'binary' else ('BENIGN' if not truth else family)
                    true_classes.append(true_class); predicted_classes.append(result['label'])
                    record['ground_truth'] = {'binary': 'ATTACK' if truth else 'BENIGN', 'family': family}
                    if result.get('error'):
                        errors[result['error']] += 1
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                    processed += 1
                if processed % 100 == 0 or processed == len(replay):
                    print(json.dumps({'event': 'baseline_progress', 'method': method, 'dataset': dataset,
                                      'processed': processed, 'requested': len(replay)}), file=sys.stderr, flush=True)
        phase = 'complete'
    finally:
        if algorithm is not None:
            algorithm.save(directory)
        class_metrics = None
        if true_classes:
            classes = sorted(set(true_classes) | (set(predicted_classes) - {'ERROR', 'UNCERTAIN'}))
            precision, recall, f1, _ = precision_recall_fscore_support(
                true_classes, predicted_classes, labels=classes, average='macro', zero_division=0)
            class_metrics = {'labels': classes, 'macro_precision': float(precision),
                             'macro_recall': float(recall), 'macro_f1': float(f1),
                             'definition': 'Macro over observed true/predicted traffic classes; ERROR is not a target class.'}
        summary = {**preview, 'processed': processed, 'completed': phase == 'complete', 'phase': phase,
                   'ground_truth_metrics': report.summary(), 'classification_metrics': class_metrics,
                   'errors': dict(errors), 'api_requests': dict(client.requests),
                   'tokens': dict(client.tokens), 'cache_hits': client.cache_hits,
                   'construction_seconds': construction_seconds,
                   'construction_api_requests': locals().get('construction_requests', {}),
                   'elapsed_seconds': time.perf_counter() - started,
                   'method_statistics': algorithm.statistics() if algorithm is not None else {},
                   'results_directory': str(directory)}
        (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, allow_nan=False))


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=[*METHODS, 'all'], required=True)
    parser.add_argument('--dataset', choices=[*DATASETS, 'all'], required=True)
    parser.add_argument('--config', type=Path, default=root / 'config/baselines.yaml')
    parser.add_argument('--task', choices=['binary', 'family'])
    parser.add_argument('--construction-flows', type=int)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--max-flows', type=int)
    limits.add_argument('--all-flows', action='store_true')
    parser.add_argument('--epochs', type=int, help='Override neural training epochs')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--dry-run', action='store_true', help='Preview cohorts without models, writes or API calls')
    args = parser.parse_args()
    if args.task == 'family' and args.method in ('all', 'scala_nids'):
        parser.error('SCALA-NIDS is binary only; run family-capable methods individually')
    for key in ('construction_flows', 'max_flows', 'epochs'):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.max_flows is None and not args.all_flows:
        args.max_flows = 50
    for method in METHODS if args.method == 'all' else [args.method]:
        for dataset in DATASETS if args.dataset == 'all' else [args.dataset]:
            run_one(args, method, dataset)


if __name__ == '__main__':
    main()
