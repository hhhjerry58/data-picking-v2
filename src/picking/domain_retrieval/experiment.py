from __future__ import annotations

import fcntl
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from picking.common import REPO, emit, file_hash, rank_key, read_json, utc_now
from .config import freeze_json, method_file, run_dir, runtime_cfg, write_json
from .data import download_assets, prepare_data
from .metrics import paired_bootstrap, rank_documents, retrieval_metrics, weighted_nll
from .training import model_identity, score_documents, train


@contextmanager
def run_lock(cfg):
    folder = run_dir(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    scratch = Path(runtime_cfg(cfg)['runtime_root'])
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch / 'writer.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        owner = folder / 'OWNER.json'
        hostname = socket.gethostname()
        if owner.exists():
            previous = read_json(owner)
            if previous['hostname'] != hostname:
                raise RuntimeError(f'Run belongs to node {previous["hostname"]}; verify its writer has stopped before explicitly transferring OWNER.json')
        else:
            # Exclusive create, followed by a persistent node guard. A run is
            # never automatically taken over just because a PID is absent here.
            with owner.open('x') as stream:
                json.dump({'hostname': hostname, 'created_at': utc_now()}, stream)
        yield


def progress(cfg, stage, **fields):
    value = {'stage': stage, 'pid': os.getpid(), 'hostname': socket.gethostname(), 'time': utc_now(), **fields}
    write_json(run_dir(cfg) / 'status.json', value)
    emit('domain_experiment_stage', **value)


def select_arm(documents, rankings, arm, cfg):
    candidates = documents[documents.split == 'candidate']
    count = cfg['continuation']['selected_documents']
    if arm in ('oracle_target', 'other'):
        eligible = candidates[candidates.is_target == (arm == 'oracle_target')]
        ids = sorted(eligible.id, key=lambda identifier: rank_key(f'control:{identifier}', cfg['selection_seed']))[:count]
    else:
        ids = rankings[arm].sort_values('rank').id.tolist()[:count]
    if len(ids) != count or len(set(ids)) != count or not set(ids) <= set(candidates.id):
        raise RuntimeError('Selected training IDs must be unique members of the candidate pool')
    selected = candidates[candidates.id.isin(ids)].sort_values('id')
    if len(selected) != count:
        raise RuntimeError('Selection coverage differs')
    return selected


def continue_one(cfg, arm, seed, device='cuda', documents=None):
    """One independent arm/seed; the parent owns the node/run writer lock."""
    folder = run_dir(cfg)
    if documents is None:
        documents = pd.read_parquet(folder / 'data/documents.parquet')
    selected_ids = pd.read_csv(folder / 'selection' / arm / 'documents.csv').id.tolist()
    candidates = documents[documents.split == 'candidate']
    if len(set(selected_ids)) != cfg['continuation']['selected_documents'] or not set(selected_ids) <= set(candidates.id):
        raise RuntimeError('Worker selection must be a fixed subset of candidates')
    selected = candidates[candidates.id.isin(selected_ids)].sort_values('id')
    tests = documents[documents.split.isin(['target_test', 'other_test'])]
    base = read_json(folder / 'M0_identity.json')
    checkpoint = train(cfg, f'{arm}-seed-{seed}', selected, seed,
                       cfg['continuation']['prediction_tokens'], base, device)
    identity = read_json(checkpoint / 'manifest.json')['fingerprint']
    return score_documents(cfg, tests, folder / 'evaluation' / f'{arm}-seed-{seed}.parquet', identity,
                           checkpoint=checkpoint, device=device)


def continuation_gpu_ids(gpu_ids):
    from picking.runtime import free_gpu_ids
    available = set(free_gpu_ids()) & set(gpu_ids)
    # M0/probe scoring can leave a CUDA context in the coordinator even after
    # every tensor is freed. That context is ours, not a competing job.
    first = gpu_ids[0]
    if first not in available:
        try:
            output = subprocess.check_output(
                ['nvidia-smi', f'--id={first}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                text=True, timeout=15)
            pids = {int(line.strip()) for line in output.splitlines() if line.strip()}
            if pids == {os.getpid()}:
                available.add(first)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return available


def parallel_continuations(cfg, gpu_ids):
    """Separate processes/GPU contexts, distinct arm folders, one parent writer."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    pending = [(arm, seed) for seed in cfg['continuation']['seeds'] for arm in cfg['continuation']['arms']]
    active = []
    logs = Path(runtime_cfg(cfg)['runtime_root']) / 'worker_logs'
    logs.mkdir(parents=True, exist_ok=True)
    worker = ('import sys; from picking.common import read_json; '
              'from picking.domain_retrieval.experiment import continue_one; '
              'continue_one(read_json(sys.argv[1]), sys.argv[2], int(sys.argv[3]))')
    try:
        while pending or active:
            for job in list(active):
                code = job['process'].poll()
                if code is None:
                    continue
                job['stream'].close()
                destination = run_dir(cfg) / 'logs' / job['log'].name
                destination.parent.mkdir(exist_ok=True)
                shutil.copyfile(job['log'], destination)
                active.remove(job)
                if code:
                    raise RuntimeError(f'Continuation {job["arm"]}/{job["seed"]} exited {code}; see {job["log"]}')
                progress(cfg, 'continuation_finished', arm=job['arm'], seed=job['seed'], remaining=len(pending)+len(active))
            occupied = {job['gpu'] for job in active}
            available = continuation_gpu_ids(gpu_ids)
            for gpu in gpu_ids:
                if not pending or gpu in occupied or gpu not in available:
                    continue
                arm, seed = pending.pop(0)
                path = logs / f'{arm}-seed-{seed}.log'
                stream = path.open('ab', buffering=0)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'DATA_PICKING_GPU_IDS': str(gpu),
                       'PYTHONPATH': str(REPO / 'src'), 'PYTHONUNBUFFERED': '1'}
                process = subprocess.Popen([sys.executable, '-c', worker, str(run_dir(cfg) / 'effective_config.json'), arm, str(seed)],
                                           cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
                active.append({'arm': arm, 'seed': seed, 'gpu': gpu, 'process': process, 'stream': stream, 'log': path})
                progress(cfg, 'continue_from_M0', active=[{'arm': j['arm'], 'seed': j['seed'], 'gpu': j['gpu']} for j in active],
                         pending=len(pending))
            if pending or active:
                time.sleep(1)
    finally:
        # Only children started by this run are interrupted if the parent fails.
        for job in active:
            if job['process'].poll() is None:
                job['process'].terminate()
        for job in active:
            try:
                job['process'].wait(timeout=20)
            except subprocess.TimeoutExpired:
                job['process'].kill()
                job['process'].wait()
            job['stream'].close()


def run(cfg, device='cuda', gpu_ids=None):
    folder = run_dir(cfg)
    freeze_json(folder / 'effective_config.json', cfg)
    progress(cfg, 'prepare')
    download_assets(cfg)
    documents = prepare_data(cfg)
    base = model_identity(cfg)
    freeze_json(folder / 'M0_identity.json', base)
    candidates = documents[documents.split == 'candidate']
    tests = documents[documents.split.isin(['target_test', 'other_test'])]
    probe = documents[documents.split == 'probe']
    progress(cfg, 'score_M0')
    before = score_documents(cfg, candidates, folder / 'scores/M0_candidates.parquet', base, device=device)
    progress(cfg, 'probe')
    checkpoint = train(cfg, 'probe', probe, cfg['probe']['seed'], cfg['probe']['prediction_tokens'], base, device)
    probe_identity = read_json(checkpoint / 'manifest.json')['fingerprint']
    progress(cfg, 'score_M1')
    after = score_documents(cfg, candidates, folder / 'scores/M1_candidates.parquet', probe_identity,
                            checkpoint=checkpoint, device=device)
    paired = before.merge(after, on='id', suffixes=('_before', '_after'), validate='one_to_one')
    if len(paired) != len(candidates) or not np.array_equal(paired.prediction_tokens_before, paired.prediction_tokens_after):
        raise RuntimeError('Before/after scoring coverage or token masks differ')
    losses = paired[['id', 'loss_before', 'loss_after']].copy()
    losses['prediction_tokens'] = paired.prediction_tokens_before
    losses.to_csv(folder / 'scores/paired_losses.csv', index=False)
    truth = candidates[['id', 'category', 'is_target']]
    truth.to_csv(folder / 'data/candidate_ground_truth.csv', index=False)
    rankings, retrieval = {}, {}
    progress(cfg, 'retrieval')
    for name in cfg['methods']:
        destination = folder / 'methods' / name
        destination.mkdir(parents=True, exist_ok=True)
        freeze_json(destination / 'identity.json', {'implementation': file_hash(method_file(name)),
                                                  'losses': file_hash(folder / 'scores/paired_losses.csv'),
                                                  'selection_seed': cfg['selection_seed']})
        ranked = rank_documents(name, losses, cfg['selection_seed'])
        joined, curve, metrics = retrieval_metrics(ranked, truth, ks=[100, 500, cfg['continuation']['selected_documents']])
        ranked.to_csv(destination / 'ranking.csv', index=False)
        joined.to_csv(destination / 'ranking_with_ground_truth.csv', index=False)
        curve.to_csv(destination / 'pr_curve.csv', index=False)
        write_json(destination / 'metrics.json', metrics)
        rankings[name], retrieval[name] = ranked, metrics
    evaluations, results = {}, []

    def collect_evaluation(arm, seed, losses):
        annotated = losses.merge(tests[['id', 'split']], on='id', validate='one_to_one')
        evaluations[(arm, seed)] = annotated
        metrics = {'arm': arm, 'seed': seed,
                   'target_nll': weighted_nll(annotated[annotated.split == 'target_test']),
                   'other_nll': weighted_nll(annotated[annotated.split == 'other_test'])}
        results.append(metrics)
        return metrics

    def evaluate(arm, seed, checkpoint=None, identity=None):
        tag = arm if seed is None else f'{arm}-seed-{seed}'
        progress(cfg, 'evaluate', arm=arm, seed=seed)
        losses = score_documents(cfg, tests, folder / 'evaluation' / f'{tag}.parquet', identity,
                                 checkpoint=checkpoint, device=device)
        return collect_evaluation(arm, seed, losses)

    baseline = evaluate('M0', None, identity=base)
    evaluate('M1_probe', cfg['probe']['seed'], checkpoint=checkpoint, identity=probe_identity)
    selections = {}
    for arm in cfg['continuation']['arms']:
        selected = select_arm(documents, rankings, arm, cfg)
        target = folder / 'selection' / arm
        target.mkdir(parents=True, exist_ok=True)
        selected[['id', 'category', 'is_target', 'prediction_tokens']].to_csv(target / 'documents.csv', index=False)
        selections[arm] = {'documents': len(selected), 'target_documents': int(selected.is_target.sum()),
                           'unique_prediction_tokens': int(selected.prediction_tokens.sum()),
                           'training_prediction_tokens': cfg['continuation']['prediction_tokens']}
    if device == 'cuda' and gpu_ids and len(gpu_ids) > 1:
        parallel_continuations(cfg, gpu_ids)
        for seed in cfg['continuation']['seeds']:
            for arm in cfg['continuation']['arms']:
                collect_evaluation(arm, seed, pd.read_parquet(folder / 'evaluation' / f'{arm}-seed-{seed}.parquet'))
    else:
        for seed in cfg['continuation']['seeds']:
            for arm in cfg['continuation']['arms']:
                progress(cfg, 'continue_from_M0', arm=arm, seed=seed)
                collect_evaluation(arm, seed, continue_one(cfg, arm, seed, device, documents))
    table = pd.DataFrame(results)
    table['target_improvement_over_M0'] = baseline['target_nll'] - table.target_nll
    table['other_improvement_over_M0'] = baseline['other_nll'] - table.other_nll
    table.to_csv(folder / 'metrics.csv', index=False)
    comparisons = []
    for seed in cfg['continuation']['seeds']:
        other = evaluations[('other', seed)]
        target = evaluations[('oracle_target', seed)]
        estimate = paired_bootstrap(other[other.split == 'target_test'], target[target.split == 'target_test'],
                                    seed=cfg['evaluation']['bootstrap_seed'], samples=cfg['evaluation']['bootstrap_samples'])
        comparisons.append({'seed': seed, 'other_minus_oracle_target': estimate})
    differences = [row['other_minus_oracle_target']['difference'] for row in comparisons]
    summary = {'config': cfg, 'retrieval': retrieval, 'selections': selections,
               'oracle_vs_other': comparisons, 'mean_other_minus_oracle': float(np.mean(differences)),
               'std_other_minus_oracle_across_seeds': float(np.std(differences, ddof=1)) if len(differences) > 1 else None,
               'all_seeds_oracle_better': bool(all(value > 0 for value in differences)),
               'limitations': ['Source category is retrieval ground truth, not a document-level causal usefulness label.',
                               'Held out from this experiment; original pretraining exposure is unknown.',
                               'One probe model; paired continuation seeds share fixed splits and selections.',
                               'First 512 tokens only in the default pilot; token budgets may repeat selected documents.']}
    write_json(folder / 'summary.json', summary)
    progress(cfg, 'archive')
    from .reporting import report
    archive = report(cfg)
    write_json(folder / 'EXPERIMENT_COMPLETED.json', {'time': utc_now(), 'archive': str(archive),
                                                     'seeds': cfg['continuation']['seeds'],
                                                     'arms': cfg['continuation']['arms']})
    progress(cfg, 'complete', archive=str(archive))
    return archive
