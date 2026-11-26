from __future__ import annotations

import heapq
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH
from transformers import AutoTokenizer

from picking.common import digest, emit, file_hash, rank_key, read_json
from picking.download import fetch_many
from .config import freeze_json, run_dir, write_json


def model_assets(cfg):
    model = cfg['model']
    if model.get('local_path'):
        path = Path(model['local_path']).expanduser().resolve()
        if not (path / 'config.json').is_file():
            raise ValueError('local_path must be a saved HuggingFace checkpoint')
        return path
    return Path(cfg['artifact_root']) / 'cache/domain-retrieval' / model['name'].split('/')[-1] / model['revision']


def source_paths(cfg):
    spec = cfg['data']
    if spec.get('local_files'):
        return [Path(path).expanduser().resolve() for path in spec['local_files']]
    root = Path(cfg['artifact_root']) / 'cache/domain-retrieval' / spec['dataset'].split('/')[-1].lower() / spec['revision']
    return [root / Path(name).name for name in spec['source_files']]


def download_assets(cfg):
    items = []
    model = cfg['model']
    if not model.get('local_path'):
        items.extend((f'https://huggingface.co/{model["name"]}/resolve/{model["revision"]}/{name}',
                      model_assets(cfg) / name) for name in model['files'])
    spec = cfg['data']
    if not spec.get('local_files'):
        items.extend((f'https://huggingface.co/datasets/{spec["dataset"]}/resolve/{spec["revision"]}/{name}', path)
                     for name, path in zip(spec['source_files'], source_paths(cfg)))
    fetch_many(items, workers=4)


def nested(row, key):
    for part in key.split('.'):
        row = row[part]
    return row


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


class Deduplicator:
    """Deduplicate scored prefixes across all splits before assigning labels.

    Whole-document identity and exact token prefixes are exact checks; LSH is an
    additional approximate search for 5-token-shingle Jaccard >= threshold.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.shingles = {}
        self.identities = set()
        self.prefixes = set()

    def accept(self, identifier, tokens):
        prefix = digest(tokens)
        if identifier in self.identities or prefix in self.prefixes:
            return False
        width = min(5, len(tokens))
        shingles = {tuple(tokens[i:i+width]) for i in range(len(tokens)-width+1)}
        signature = MinHash(num_perm=128, seed=1)
        signature.update_batch([np.asarray(item, dtype='<u4').tobytes() for item in sorted(shingles)])
        for existing in self.lsh.query(signature):
            other = self.shingles[existing]
            if len(shingles & other) / len(shingles | other) >= self.threshold:
                return False
        self.lsh.insert(identifier, signature)
        self.shingles[identifier] = shingles
        self.identities.add(identifier)
        self.prefixes.add(prefix)
        return True


def candidate_reservoir(cfg):
    """Seeded bounded reservoirs over every row of the frozen source shards."""
    spec = cfg['data']
    sizes = {True: spec['probe_documents'] + spec['positive_documents'] + spec['target_test_documents'],
             False: spec['negative_documents'] + spec['other_test_documents']}
    heaps, ids, counts = {True: [], False: []}, {True: set(), False: set()}, Counter()
    for path in source_paths(cfg):
        if path.suffix == '.parquet':
            batches = (batch.to_pylist() for batch in pq.ParquetFile(path).iter_batches(batch_size=1024))
        elif path.suffix == '.jsonl':
            def lines():
                with path.open() as stream:
                    for line in stream:
                        if line.strip():
                            yield [json.loads(line)]
            batches = lines()
        else:
            raise ValueError('Use parquet or JSONL input with text and category columns')
        for batch in batches:
            for row in batch:
                text, category = nested(row, spec['text_column']), nested(row, spec['category_column'])
                if not isinstance(text, str) or not isinstance(category, str) or not category:
                    raise ValueError('Every source row needs text and a nonempty category')
                counts[category] += 1
                canonical = normalized(text)
                if len(canonical) < spec['min_prediction_tokens']:
                    continue
                identifier = digest(canonical)
                positive = category == spec['target_category']
                heap, seen = heaps[positive], ids[positive]
                if identifier in seen:
                    continue
                key = int(rank_key(identifier, spec['split_seed']), 16)
                capacity = sizes[positive] * 2
                if len(heap) < capacity or key < -heap[0][0]:
                    value = (-key, identifier, category, text)
                    if len(heap) == capacity:
                        seen.remove(heapq.heapreplace(heap, value)[1])
                    else:
                        heapq.heappush(heap, value)
                    seen.add(identifier)
        emit('domain_source_scanned', file=path.name, categories=dict(counts))
    values = [item for heap in heaps.values() for item in heap]
    values.sort(key=lambda item: (-item[0], item[1]))
    return values, sizes, counts


def prepare_data(cfg):
    folder = run_dir(cfg) / 'data'
    folder.mkdir(parents=True, exist_ok=True)
    identity = {'data': cfg['data'], 'model': cfg['model'], 'implementation': file_hash(__file__)}
    # Pin local inputs too; remote inputs have immutable revisions and verified
    # download receipts. No broad repository or artifact hash inventory.
    if cfg['data'].get('local_files'):
        identity['local_inputs'] = {str(path): file_hash(path) for path in source_paths(cfg)}
    freeze_json(folder / 'identity.json', identity)
    marker = folder / 'COMPLETED.json'
    if marker.exists():
        record = read_json(marker)
        if record['documents_sha256'] != file_hash(folder / 'documents.parquet'):
            raise RuntimeError('Frozen data changed')
        return pd.read_parquet(folder / 'documents.parquet')
    reservoir, sizes, categories = candidate_reservoir(cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_assets(cfg), local_files_only=True)
    spec = cfg['data']
    dedup = Deduplicator(spec['near_duplicate_jaccard'])
    accepted, rejected = {True: [], False: []}, Counter()
    for _, identifier, category, text in reservoir:
        positive = category == spec['target_category']
        if len(accepted[positive]) >= sizes[positive]:
            continue
        tokens = tokenizer(text, add_special_tokens=False, truncation=True,
                           max_length=spec['max_length'])['input_ids']
        if len(tokens)-1 < spec['min_prediction_tokens']:
            rejected['short'] += 1
            continue
        if not dedup.accept(identifier, tokens):
            rejected['duplicate'] += 1
            continue
        accepted[positive].append({'id': identifier, 'category': category, 'text': text,
                                   'input_ids': tokens, 'prediction_tokens': len(tokens)-1})
    for positive, required in sizes.items():
        if len(accepted[positive]) < required:
            raise RuntimeError(f'Insufficient deduplicated {"target" if positive else "other"} documents: '
                               f'{len(accepted[positive])} < {required}; add source shards or reduce the pilot counts')
    rows = []
    assignments = {True: [('probe', spec['probe_documents']), ('candidate', spec['positive_documents']),
                           ('target_test', spec['target_test_documents'])],
                   False: [('candidate', spec['negative_documents']), ('other_test', spec['other_test_documents'])]}
    for positive, chunks in assignments.items():
        ordered = sorted(accepted[positive], key=lambda row: rank_key('split:' + row['id'], spec['split_seed']))
        start = 0
        for split, count in chunks:
            for row in ordered[start:start+count]:
                rows.append({**row, 'split': split, 'is_target': positive})
            start += count
    frame = pd.DataFrame(rows).sort_values('id').reset_index(drop=True)
    if frame.id.duplicated().any():
        raise RuntimeError('Duplicate document crossed partitions')
    frame.to_parquet(folder / 'documents.parquet', index=False)
    frame[['id', 'category', 'split', 'is_target', 'prediction_tokens']].to_csv(folder / 'split_manifest.csv', index=False)
    summary = {'source_category_counts': dict(categories), 'rejected': dict(rejected),
               'splits': frame.groupby(['split', 'is_target']).size().rename('documents').reset_index().to_dict('records'),
               'documents_sha256': file_hash(folder / 'documents.parquet'),
               'deduplication': 'NFKC/casefold/whitespace document identity; exact scored token prefix; '
                                'MinHash128 LSH plus verified token-5gram Jaccard before all splits',
               'evaluation_scope': 'Held out from this experiment; prior exposure during original pretraining is unknown',
               'scoring_unit': f'First at most {spec["max_length"]} tokens per document; no appended EOS or chat template'}
    write_json(marker, summary)
    emit('domain_data_prepared', **summary)
    return frame
