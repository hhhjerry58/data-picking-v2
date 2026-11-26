from __future__ import annotations

import heapq
import json
import shutil
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from picking.common import REPO, digest, file_hash, rank_key
from ..config import atomic_json, freeze_json, identity, run_dir
from ..modeling.masks import render_trajectory
from .pr import encode_pr
from .splits import referenced_pr_ids, split_tasks, task_id


def prepare(cfg, tokenizer):
    folder = run_dir(cfg) / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    data_files = [*Path(__file__).parent.glob('*.py'), Path(__file__).parents[1]/'modeling/masks.py', REPO/'src/picking/common.py']
    signature = digest({'data': cfg['data'], 'model': {k: cfg['model'][k] for k in ('name', 'revision', 'max_length')},
                        'implementation': {str(p.relative_to(REPO)): file_hash(p) for p in data_files},
                        'tokenizer_version': version('tokenizers'), 'transformers_version': version('transformers')})
    receipt = {'identity': identity(cfg), 'data_signature': signature,
               'trajectory_sha256': file_hash(cfg['data']['trajectory_file']),
               'pr_index_sha256': file_hash(cfg['data']['pr_index'])}
    freeze_json(folder / 'identity.json', receipt)
    if (folder / 'COMPLETED.json').exists():
        return
    if cfg.get('prepared_from_run'):
        parent = Path(cfg['artifact_root'])/'runs'/cfg['prepared_from_run']/'data'
        previous = json.loads((parent/'COMPLETED.json').read_text())
        for key in ('data_signature', 'trajectory_sha256', 'pr_index_sha256'):
            if previous[key] != receipt[key]:
                raise ValueError(f'Prepared dataset {key} differs; cannot reuse')
        for name in ('guidance.parquet', 'candidates.parquet', 'split_manifest.csv', 'test_manifest.json', 'source_shards.json'):
            shutil.copyfile(parent/name, folder/name)
            if file_hash(parent/name) != file_hash(folder/name):
                raise RuntimeError('Prepared dataset copy failed integrity check')
        atomic_json(folder/'COMPLETED.json', {**previous, **receipt, 'prepared_from_run': cfg['prepared_from_run']})
        return
    spec = cfg['data']
    with open(spec['trajectory_file']) as source:
        records = [json.loads(line) for line in source if line.strip()]
    assignments = split_tasks(records, spec['q_tasks'], spec['split_seed'], spec['expected_tasks'])
    if len(assignments) != len(records):
        raise ValueError('Repeated trajectory ID')
    q, manifest = [], []
    excluded = referenced_pr_ids(records)
    if spec.get('exclude_pr_ids_file'):
        excluded |= {line.strip().lower() for line in Path(spec['exclude_pr_ids_file']).read_text().splitlines() if line.strip()}
    for row in records:
        group = task_id(row)
        manifest.append({'id': row['id'], 'task_id': group, 'split': assignments[row['id']]})
        if assignments[row['id']] == 'Q':
            encoded = render_trajectory(tokenizer, row)
            if encoded['input_tokens'] + 1 > cfg['model']['max_length']:
                raise ValueError(f'Guidance trajectory exceeds context: {row["id"]}')
            q.append({**encoded, 'task_id': group})
    pd.DataFrame(manifest).to_csv(folder / 'split_manifest.csv', index=False)
    pd.DataFrame(q).to_parquet(folder / 'guidance.parquet', index=False)
    # T is only an identity manifest. The scoring stages never load T tensors/losses.
    atomic_json(folder / 'test_manifest.json', [r for r in manifest if r['split'] == 'T'])
    index = json.loads(Path(spec['pr_index']).read_text())
    root = Path(spec['pr_root'])
    eligible_paths = []
    for value in index:
        path = Path(value)
        if not path.is_relative_to(root):
            raise ValueError('PR index points outside authorized dataset root')
        subset = '/'.join(path.relative_to(root).parts[:3])
        if subset in spec['pr_subsets']:
            eligible_paths.append(path)
    paths = sorted(set(eligible_paths), key=lambda p: rank_key(str(p), spec['sample_seed']))[:spec['max_source_shards']]
    if not paths:
        raise ValueError('No matching PR shards')
    heap, seen_pr, seen_query, seen_text = [], set(), set(), set()
    counts = Counter()
    receipts = []
    for path in paths:
        # Pin only shards actually used, rather than hashing the entire data pool.
        receipts.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': file_hash(path)})
        with path.open() as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                counts['rows_scanned'] += 1
                pr, query = row['id'].lower(), row['query_id']
                text_id = digest(row['text'])
                if pr in excluded:
                    counts['explicit_benchmark_overlap'] += 1
                    continue
                if pr in seen_pr or query in seen_query or text_id in seen_text:
                    counts['duplicate'] += 1
                    continue
                try:
                    encoded = encode_pr(tokenizer, row, spec['max_candidate_tokens'], spec['min_answer_tokens'])
                except ValueError as exc:
                    counts[str(exc)] += 1
                    continue
                seen_pr.add(pr); seen_query.add(query); seen_text.add(text_id)
                key = int(rank_key(pr, spec['sample_seed']), 16)
                encoded['source_file'] = str(path)
                encoded['text_identity'] = text_id
                counts['eligible_unique'] += 1
                item = (-key, pr, encoded)
                if len(heap) < spec['candidates']:
                    heapq.heappush(heap, item)
                elif key < -heap[0][0]:
                    heapq.heapreplace(heap, item)
        print('PR_SHARD', len(receipts), len(paths), dict(counts), flush=True)
    if len(heap) < spec['candidates']:
        raise ValueError(f'Only {len(heap)} eligible PRs; increase fixed shard budget in a NEW run')
    candidates = sorted((item[2] for item in heap), key=lambda r: r['id'])
    pd.DataFrame(candidates).to_parquet(folder / 'candidates.parquet', index=False)
    atomic_json(folder / 'source_shards.json', receipts)
    atomic_json(folder / 'COMPLETED.json', {
        **receipt, 'q_documents': len(q), 'q_tasks': spec['q_tasks'],
        'test_tasks': spec['expected_tasks']-spec['q_tasks'], 'candidate_documents': len(candidates),
        'q_prediction_tokens': sum(row['prediction_tokens'] for row in q),
        'counts': dict(counts), 'explicit_excluded_pr_ids': sorted(excluded),
        'sampling': 'Uniform ID-hash reservoir of eligible unique PRs in a fixed, seeded shard subset.',
        'contamination_limit': 'Task grouping and explicit GitHub PR/issue IDs are checked; semantic patch overlap and pretraining exposure remain unverified.'})
