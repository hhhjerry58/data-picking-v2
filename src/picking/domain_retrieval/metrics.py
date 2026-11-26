from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from picking.common import rank_key
from .config import method_file


def rank_documents(name, losses, seed):
    fields = ['id', 'loss_before', 'loss_after', 'prediction_tokens']
    rows = losses[fields].to_dict('records')
    if not rows or losses.id.duplicated().any():
        raise ValueError('Candidate IDs must be unique and nonempty')
    numeric = losses[['loss_before', 'loss_after', 'prediction_tokens']].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric[:, 2] < 1).any():
        raise ValueError('Invalid paired losses')
    module_spec = importlib.util.spec_from_file_location(f'domain_method_{name}', method_file(name))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    scores = np.asarray(module.score(rows, seed=seed), dtype=float)
    if scores.shape != (len(rows),) or not np.isfinite(scores).all():
        raise ValueError('A method must return one finite score per document')
    ranked = losses[fields].copy()
    ranked['score'] = scores
    ranked['tie_key'] = [rank_key(f'tie:{identifier}', seed) for identifier in ranked.id]
    ranked = ranked.sort_values(['score', 'tie_key', 'id'], ascending=[False, True, True]).reset_index(drop=True)
    ranked['rank'] = np.arange(1, len(ranked) + 1)
    return ranked.drop(columns='tie_key')


def retrieval_metrics(ranked, truth, ks=(100, 500, 1000)):
    if truth.id.duplicated().any() or set(ranked.id) != set(truth.id):
        raise ValueError('Candidate/ground-truth IDs must match one to one')
    joined = ranked.merge(truth[['id', 'is_target']], on='id', validate='one_to_one').sort_values('rank')
    labels = joined.is_target.to_numpy(dtype=int)
    if set(labels) != {0, 1}:
        raise ValueError('PR evaluation needs both positives and negatives')
    precision, recall, thresholds = precision_recall_curve(labels, joined.score, drop_intermediate=False)
    # The last (recall=0, precision=1) point has no finite score threshold.
    curve = pd.DataFrame({'precision': precision, 'recall': recall,
                          'threshold': np.r_[thresholds, np.nan]})
    result = {'documents': len(labels), 'positives': int(labels.sum()),
              'prevalence': float(labels.mean()),
              'average_precision': float(average_precision_score(labels, joined.score)),
              'pr_definition': 'All candidates; ties grouped at a common threshold; AP is step-integrated precision'}
    for k in sorted(set(min(int(k), len(labels)) for k in ks)):
        result[f'precision_at_{k}'] = float(labels[:k].mean())
        result[f'recall_at_{k}'] = float(labels[:k].sum() / labels.sum())
    return joined, curve, result


def weighted_nll(rows):
    tokens = int(rows.prediction_tokens.sum())
    if tokens <= 0 or not np.isfinite(rows.sum_nll).all():
        raise ValueError('Invalid evaluation rows')
    return float(rows.sum_nll.sum() / tokens)


def paired_bootstrap(left, right, *, seed, samples):
    if left.id.duplicated().any() or right.id.duplicated().any() or set(left.id) != set(right.id):
        raise ValueError('Paired evaluation IDs differ')
    left = left.sort_values('id')
    right = right.sort_values('id')
    counts = left.prediction_tokens.to_numpy()
    if not np.array_equal(counts, right.prediction_tokens.to_numpy()):
        raise ValueError('Paired evaluation token masks differ')
    differences = left.sum_nll.to_numpy() - right.sum_nll.to_numpy()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        indexes = rng.integers(len(counts), size=len(counts))
        estimates.append(float(differences[indexes].sum() / counts[indexes].sum()))
    return {'difference': float(differences.sum() / counts.sum()),
            'ci95': np.quantile(estimates, [0.025, 0.975]).tolist(),
            'resampling_unit': 'document; conditional on this pair of trained models'}
