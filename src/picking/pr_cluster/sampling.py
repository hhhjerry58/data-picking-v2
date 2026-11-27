"""冻结百分比子集和按全局输入 token 预算循环采样的 100-step 清单。"""
import math

import numpy as np


def experiments():
    jobs = []
    for objective in ('full_text', 'answer_only'):
        for score in ('absolute_loss_drop', 'relative_loss_drop'):
            for label, fraction, descending in [('top10', .1, True), ('top20', .2, True),
                                                 ('top50', .5, True), ('bottom10', .1, False)]:
                jobs.append({'name': f'{objective}-{score}-{label}-s42', 'objective': objective,
                             'score': score, 'fraction': fraction, 'descending': descending, 'seed': 42})
        for seed in range(42, 50):
            jobs.append({'name': f'{objective}-random-s{seed}', 'objective': objective,
                         'score': None, 'fraction': 1., 'descending': True, 'seed': seed})
    return jobs


def subset_indices(candidates, scores, job):
    lookup = {identifier: i for i, identifier in enumerate(candidates.id)}
    if len(lookup) != len(candidates): raise ValueError('Repeated candidate IDs')
    group = scores[scores.objective == job['objective']]
    if group.id.duplicated().any() or set(group.id) != set(lookup): raise ValueError('Incomplete score coverage')
    if job['score'] is None: return list(range(len(candidates)))
    chosen = group.sort_values([job['score'], 'id'], ascending=[not job['descending'], True])
    n = int(len(candidates)*job['fraction'])
    return [lookup[identifier] for identifier in chosen.head(n).id]


def step_plan(candidates, indexes, seed, *, steps=100, input_tokens_per_step=256000):
    """Whole records nearest to each step budget; report actual tokens explicitly.

    The maximum per-step discrepancy is at most half the longest record.
    Epochs cycle only when the shuffled subset is exhausted.
    """
    if not indexes or len(indexes) != len(set(indexes)): raise ValueError('Invalid training subset')
    lengths = candidates.input_tokens.to_numpy()
    rng = np.random.default_rng(seed); order = []; cursor = 0; epoch = -1
    def next_item():
        nonlocal order, cursor, epoch
        if cursor == len(order):
            order = rng.permutation(indexes).tolist(); cursor = 0; epoch += 1
        value = (int(order[cursor]), epoch); cursor += 1
        return value
    pending = None; plan = []
    for step in range(steps):
        selected = []; total = 0
        while True:
            item = pending if pending is not None else next_item(); pending = None
            index, item_epoch = item
            length = int(lengths[index])
            if total+length >= input_tokens_per_step:
                if not selected or total+length-input_tokens_per_step <= input_tokens_per_step-total:
                    selected.append({'index': index, 'epoch': item_epoch}); total += length
                else: pending = item
                break
            selected.append({'index': index, 'epoch': item_epoch}); total += length
        plan.append({'step': step+1, 'samples': selected, 'input_tokens': total})
    return plan


def rank_batches(step, candidates, rank, world_size):
    """Same number of FSDP collectives on all ranks; None is a zero-loss dummy."""
    if not 0 <= rank < world_size: raise ValueError('Invalid distributed rank')
    indexes = [sample['index'] for sample in step['samples']]
    indexes.sort(key=lambda i: (-int(candidates.iloc[i].input_tokens), i))
    return [indexes[j+rank] if j+rank < len(indexes) else None for j in range(0, len(indexes), world_size)]
