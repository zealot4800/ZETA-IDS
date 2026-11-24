"""Run explicitly labeled paper adaptations against repository datasets."""
from __future__ import annotations

import argparse
import copy
from collections import Counter
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

from ..controller import load_config
from ..data import dataset_specs, load_dataset
from ..evidence import digest, file_sha256, flow_id
from ..llm import Ollama
from ..reporting import ClassificationReport, GroundTruthReport, log_event
from ..results import atomic_json, new_run_id, optional_log, print_result, update_results
from .common import (Features, LocalReasoner, ReplayInputs, TextEmbedding, binary_label,
                     raw_records, select_source)
from .recovery import copy_journal, journal

METHODS = ('rlfe_ids', 'scala_nids', 'ma_ids', 'sera_ids')
EXPERIENCE_METHODS = ('ma_ids', 'sera_ids')
NEURAL_METHODS = ('rlfe_ids', 'scala_nids')
DATASETS = ('kdd', 'cicids2018', 'ton_iot', 'unsw_nb15')
MAX_CONSECUTIVE_ERRORS = 3


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


def load_options(args, method, root):
    """Read the baseline configuration and apply command-line overrides."""
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
    return options


def configure_llm(cfg, options, method):
    cfg['baseline_llm'] = {**copy.deepcopy(cfg['investigation_llm']), **options['llm']}
    for key, env in [('model', 'OLLAMA_BASELINE_MODEL'), ('base_url', 'OLLAMA_BASE_URL')]:
        if os.environ.get(env):
            cfg['baseline_llm'][key] = os.environ[env]
    if method == 'scala_nids':
        cfg['baseline_llm']['request_retries'] = 0  # Paper uses immediate fallback.
        cfg['baseline_llm']['response_retries'] = 0


def configure_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('Install requirements.txt and a CUDA-enabled PyTorch build for GPU training') from exc
    torch.set_num_threads(int(os.environ.get('ZETA_NATIVE_THREADS', '1')))


def latency_scope(method):
    return 'chunk_prediction_and_adaptation_amortized' if method == 'scala_nids' else 'flow_inference'


def ground_truth(truth, family):
    return {'binary': 'ATTACK' if truth else 'BENIGN', 'family': family}


def check_resume(source, manifest, train, validation, replay):
    """Reject a resume that differs from its run; return recoverable record counts."""
    previous = json.loads((source / 'manifest.json').read_text())
    for key in ('method', 'dataset', 'task', 'configuration', 'llm_configuration',
                'dataset_sha256', 'dataset_specification', 'seed', 'classes'):
        if previous.get(key) != manifest.get(key):
            raise ValueError(f'Resume rejected: {key} differs from the original run')
    with np.load(source / 'cohorts.npz') as cohorts:
        for key, selected in [('construction', train), ('validation', validation), ('replay', replay)]:
            if not np.array_equal(cohorts[key], selected):
                raise ValueError(f'Resume rejected: {key} cohort differs; use the original flow limits')
    manifest['resumed_from'] = str(source)
    manifest['previous_source_sha256'] = previous['source_sha256']
    manifest['accounting_scope'] = 'API requests, tokens and elapsed time cover this invocation only'
    construction_count = sum(1 for _ in journal(source / 'construction.jsonl'))
    replay_count = sum(1 for _ in journal(source / 'flows.jsonl'))
    if construction_count > len(train) or (replay_count and construction_count != len(train)):
        raise ValueError('Resume rejected: inconsistent construction and replay journals')
    if replay_count > len(replay):
        raise ValueError('Resume rejected: replay journal exceeds the requested cohort')
    return construction_count, replay_count


def build_algorithm(method, reasoner, options, cfg, data, features, values, raw, labels, train, validation):
    seed = cfg['general']['seed']
    if method in EXPERIENCE_METHODS:
        from .experience import ExperienceIDS
        embedder = TextEmbedding(options['embedding'])
        groups = pd.util.hash_pandas_object(data['x'].iloc[train], index=False).to_numpy()
        return ExperienceIDS(method, reasoner, embedder, options[method], raw, labels, values, groups, seed)
    if method == 'rlfe_ids':
        from .rlfe import RLFEIDS
        return RLFEIDS(reasoner, options[method], values, raw, labels, seed)
    from .scala import SCALANIDS
    return SCALANIDS(reasoner, options[method], values, data['y'][train],
                     features.transform(data['x'].iloc[validation]), data['y'][validation],
                     len(data['numeric']), seed)


class ReplayTally:
    """Metrics and counters accumulated over the replay cohort."""

    def __init__(self, task):
        self.task = task
        self.report = GroundTruthReport()
        self.classification = ClassificationReport()
        self.errors = Counter()
        self.processed = 0
        self.latency_total = self.latency_max = 0.0

    def score(self, record, truth, family):
        self.report.record(record, truth, family)
        result = record['method_output']
        attack_class = 'ATTACK' if self.task == 'binary' else family
        self.classification.record(attack_class if truth else 'BENIGN', result['label'])
        if result.get('error'):
            self.errors[result['error']] += 1

    def count(self, record):
        self.latency_total += record['latency_ms']
        self.latency_max = max(self.latency_max, record['latency_ms'])
        self.processed += 1


def restore_replay(path, tally, data, replay, dataset, method):
    """Re-score journaled replay records from an interrupted run."""
    for index, record in enumerate(journal(path)):
        if index >= len(replay):
            raise ValueError('Replay journal exceeds the requested cohort')
        position = replay[index]
        truth, family = int(data['y'][position]), str(data['family'][position])
        if (record['flow_id'] != flow_id(dataset, data['row_positions'][position])
                or record['decision_source'] != method
                or record['ground_truth'] != ground_truth(truth, family)):
            raise ValueError('Replay journal does not match the selected cohort')
        tally.score(record, truth, family)
        tally.count(record)


def predict_positions(algorithm, method, inputs, start, stop):
    """Predict one replay step; only SCALA-NIDS works on whole chunks."""
    encoded = inputs.encoded[start:stop] if inputs.encoded is not None else None
    if method == 'scala_nids':
        # Batch failure is fatal: do not skip unknown partial updates.
        return algorithm.predict_chunk(encoded)
    flow = inputs.records(start, stop)[0]
    try:
        return [algorithm.predict(flow, encoded[0]) if method == 'rlfe_ids' else algorithm.predict(flow)]
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        return [{'label': 'ERROR', 'error': str(exc)}]


def stream_replay(algorithm, method, dataset, data, replay, inputs, tally, client, chunk_size,
                  log_path, write_summary):
    task = tally.task
    replay_started = last_progress = time.monotonic()
    restored = tally.processed
    consecutive_errors = 0
    with optional_log(log_path) as stream:
        for start in range(tally.processed, len(replay), chunk_size):
            positions = replay[start:start + chunk_size]
            before_requests = sum(client.requests.values())
            tick = time.perf_counter()
            results = predict_positions(algorithm, method, inputs, start, start + len(positions))
            duration = (time.perf_counter() - tick) * 1000
            requests = sum(client.requests.values()) - before_requests
            for position, result in zip(positions, results):
                decision = binary_label(result['label'], task)
                record = {'flow_id': flow_id(dataset, data['row_positions'][position]),
                          'row_position': int(data['row_positions'][position]),
                          'final_decision': decision, 'binary_decision': decision,
                          'decision_source': method, 'method_output': result,
                          'latency_ms': duration / len(positions),
                          'latency_scope': latency_scope(method),
                          'chunk_llm_requests': requests, 'chunk_index': start // chunk_size}
                # Ground truth first enters here, after inference and adaptation.
                truth, family = int(data['y'][position]), str(data['family'][position])
                tally.score(record, truth, family)
                record['ground_truth'] = ground_truth(truth, family)
                if stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
                    stream.flush()
                tally.count(record)
                consecutive_errors = consecutive_errors + 1 if result.get('error') else 0
                if method in EXPERIENCE_METHODS and consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise RuntimeError('Three consecutive replay calls failed; '
                                       'Check Ollama; --save-details enables recoverable journals.')
            now = time.monotonic()
            if tally.processed % 100 == 0 or tally.processed == len(replay) or now - last_progress >= 30:
                write_summary()
                log_event({'event': 'baseline_progress', 'method': method, 'dataset': dataset,
                           'processed': tally.processed, 'requested': len(replay),
                           'eta_seconds': round((now - replay_started) / (tally.processed - restored)
                                                * (len(replay) - tally.processed))})
                last_progress = now


def run_one(args, method, dataset):
    root = Path(__file__).resolve().parents[2]
    cfg = load_config(root / 'config/controller.yaml')
    options = load_options(args, method, root)
    spec = dataset_specs(cfg, root)[dataset]
    started = time.perf_counter()
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
    configure_llm(cfg, options, method)
    preview = {'method': method, 'dataset': dataset, 'implementation': 'paper_adaptation',
               'dataset_size': len(data['x']),
               'task': task, 'construction_flows': len(train), 'validation_flows': len(validation),
               'requested_flows': len(replay), 'classes': sorted(set(labels)),
               'model': cfg['baseline_llm']['model'],
               'embedding': options['embedding']['model'] if method in EXPERIENCE_METHODS else None,
               'stream_order': 'repository_row_order_not_certified_temporal'}
    # A preview needs only cohort selection; avoid fitting preprocessors,
    # importing torch, or hashing all sources until execution is requested.
    if args.dry_run and not args.resume_from:
        print(json.dumps({'mode': 'preview', **preview, 'api_requests': 0}))
        return
    if method in NEURAL_METHODS:
        configure_torch()
    output_root = args.output_dir or root / cfg['output']['directory']
    run_id = new_run_id()
    directory = output_root / 'details' / method / dataset / run_id if args.save_details else None
    client, tally = Ollama(cfg), ReplayTally(task)
    reasoner = LocalReasoner(client, cfg, sorted(set(labels)))
    features = Features().fit(data['x'].iloc[train], data['numeric'], data['categorical'])
    values = features.transform(data['x'].iloc[train])
    raw = raw_records(data, train)
    manifest = {**preview, 'configuration': options, 'llm_configuration': cfg['baseline_llm'],
                'package_versions': package_versions(),
                'dataset_specification': spec, 'seed': cfg['general']['seed'],
                'dataset_sha256': file_sha256(root / spec['path']),
                'source_sha256': {str(p.relative_to(root)): file_sha256(p)
                                  for p in sorted((root / 'src').rglob('*.py'))},
                'protocol': 'bootstrap_only_construction; no_replay_label_feedback',
                'adaptations': {'llm_backend': 'local_ollama', 'data': 'repository_schemas',
                                'confidence_backend': options['sera_ids']['confidence_backend']
                                if method == 'sera_ids' else None}}
    source = args.resume_from.resolve() if args.resume_from else None
    construction_count = replay_count = 0
    if source:
        construction_count, replay_count = check_resume(source, manifest, train, validation, replay)
    if args.dry_run:
        print(json.dumps({'mode': 'preview', **preview, 'api_requests': 0,
                          'resume_from': str(args.resume_from) if args.resume_from else None,
                          'recoverable_construction': construction_count,
                          'recoverable_replay': replay_count}))
        return
    if directory:
        directory.mkdir(parents=True, exist_ok=False)
        if source:
            for name in ('construction.jsonl', 'flows.jsonl'):
                copy_journal(source / name, directory / name)
        atomic_json(directory / 'manifest.json', manifest)
        np.savez_compressed(directory / 'cohorts.npz', construction=train, validation=validation, replay=replay)
        with (directory / 'preprocessor.pkl').open('wb') as stream:
            pickle.dump(features, stream)
    algorithm = None
    phase = 'construction'
    construction_seconds = None
    construction_requests = {}
    failure = None

    def write_summary():
        processed = tally.processed
        summary = {**preview, 'processed': processed, 'completed': phase == 'complete', 'phase': phase,
                   'ground_truth_metrics': tally.report.summary(),
                   'classification_metrics': tally.classification.summary(),
                   'errors': dict(tally.errors), 'api_requests': dict(client.requests),
                   'tokens': dict(client.tokens), 'cache_hits': client.cache_hits,
                   'construction_seconds': construction_seconds,
                   'construction_api_requests': construction_requests,
                   'elapsed_seconds': time.perf_counter() - started,
                   'method_statistics': algorithm.statistics() if algorithm is not None else {},
                   'results_file': str(output_root / 'summary.json'), 'run_id': run_id,
                   'failure': failure,
                   'seed': manifest['seed'], 'dataset_sha256': manifest['dataset_sha256'],
                   'dataset_path': spec['path'],
                   'package_versions': manifest['package_versions'],
                   'parameters': {key: value for key, value in options[method].items()
                                  if not isinstance(value, (dict, list))},
                   'configuration_sha256': digest([options, cfg]),
                   'source_sha256': digest(manifest['source_sha256']),
                   'mean_latency_ms': tally.latency_total / processed if processed else None,
                   'max_latency_ms': tally.latency_max if processed else None,
                   'latency_scope': latency_scope(method)}
        elapsed = summary['elapsed_seconds']
        summary['flows_per_second'] = processed / elapsed if elapsed else None
        if args.resume_from:
            summary['resumed_from'] = manifest['resumed_from']
            summary['accounting_scope'] = manifest['accounting_scope']
        if directory:
            atomic_json(directory / 'summary.json', summary)
        update_results(output_root, summary)
        return summary

    write_summary()
    try:
        log_event({'event': 'baseline_phase', 'method': method, 'dataset': dataset,
                   'phase': 'construction', 'flows': len(train),
                   'results_file': str(output_root / 'summary.json')})
        algorithm = build_algorithm(method, reasoner, options, cfg, data, features, values, raw,
                                    labels, train, validation)
        if method in EXPERIENCE_METHODS:
            algorithm.fit(directory, progress=write_summary)
        else:
            algorithm.fit()
        construction_seconds = time.perf_counter() - started
        construction_requests = dict(client.requests)
        phase = 'preparing_replay'
        write_summary()
        inputs = ReplayInputs(data, replay, features if method in NEURAL_METHODS else None)
        if directory:
            restore_replay(directory / 'flows.jsonl', tally, data, replay, dataset, method)
        phase = 'replay'
        write_summary()
        log_event({'event': 'baseline_phase', 'method': method, 'dataset': dataset,
                   'phase': phase, 'flows': len(replay)})
        chunk_size = options['scala_nids']['chunk_size'] if method == 'scala_nids' else 1
        stream_replay(algorithm, method, dataset, data, replay, inputs, tally, client, chunk_size,
                      directory / 'flows.jsonl' if directory else None, write_summary)
        phase = 'complete'
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Keep a final incomplete summary even if saving the model itself fails.
        try:
            if phase == 'complete':
                phase = 'saving'
            if algorithm is not None:
                algorithm.save(directory)
            if phase == 'saving':
                phase = 'complete'
        except BaseException as exc:
            failure = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            summary = write_summary()
    print_result(summary)


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=[*METHODS, 'all'], required=True)
    parser.add_argument('--dataset', choices=[*DATASETS, 'all'], required=True)
    parser.add_argument('--config', type=Path, default=root / 'config/baselines.yaml')
    parser.add_argument('--task', choices=['binary', 'family'])
    parser.add_argument('--construction-flows', type=int)
    limits = parser.add_mutually_exclusive_group()
    limits.add_argument('--max-flows', type=int, help='Optionally limit evaluation flows')
    limits.add_argument('--all-flows', action='store_true', help='Use every evaluation flow (default)')
    parser.add_argument('--epochs', type=int, help='Override neural training epochs')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--save-details', action='store_true',
                        help='Save journals and model artifacts for debugging and recovery')
    parser.add_argument('--resume-from', type=Path,
                        help='Recover MA-IDS/SERA-IDS journals into a new run, preserving the source')
    parser.add_argument('--dry-run', action='store_true', help='Preview cohorts without models, writes or API calls')
    args = parser.parse_args()
    if args.resume_from and (args.method not in EXPERIENCE_METHODS or args.dataset == 'all'):
        parser.error('--resume-from requires one ma_ids/sera_ids method and one dataset')
    if args.resume_from:
        args.save_details = True
    if args.task == 'family' and args.method in ('all', 'scala_nids'):
        parser.error('SCALA-NIDS is binary only; run family-capable methods individually')
    for key in ('construction_flows', 'max_flows', 'epochs'):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    for method in METHODS if args.method == 'all' else [args.method]:
        for dataset in DATASETS if args.dataset == 'all' else [args.dataset]:
            run_one(args, method, dataset)


if __name__ == '__main__':
    main()
