from __future__ import annotations

import re

from picking.common import rank_key


def task_id(record):
    trajectory = record['source']['trajectory']
    trial = trajectory['trial_name']
    if record['id'] != f"{trial}:{trajectory['target_step_id']}":
        raise ValueError('Trajectory/source identity mismatch')
    if '__' not in trial:
        raise ValueError('DeepSWE trial name lacks its task separator')
    return trial.rsplit('__', 1)[0]


def split_tasks(records, q_tasks, seed, expected_tasks):
    tasks = {task_id(row) for row in records}
    if len(tasks) != expected_tasks:
        raise ValueError(f'Expected {expected_tasks} tasks, found {len(tasks)}; verify task mapping')
    chosen = set(sorted(tasks, key=lambda x: rank_key(x, seed))[:q_tasks])
    return {row['id']: ('Q' if task_id(row) in chosen else 'T') for row in records}


def referenced_pr_ids(records):
    """Explicit GitHub references only; this is not a complete leakage oracle."""
    found = set()
    pattern = re.compile(r'https?://github\.com/([\w.-]+)/([\w.-]+)/(?:pull|issues)/(\d+)')
    for row in records:
        for message in row['messages'] + row['target_messages']:
            for owner, repo, number in pattern.findall(message.get('content') or ''):
                found.add(f'{owner}/{repo}#{number}'.lower())
    return found
