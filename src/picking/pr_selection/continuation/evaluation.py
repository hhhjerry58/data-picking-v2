from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..artifacts import ScoreJournal
from ..config import atomic_json


def evaluate(engine, test, output, identity, runtime, persist_rows):
    output, runtime = Path(output), Path(runtime)
    journal = ScoreJournal(output/'losses.jsonl', identity, mirror_path=runtime/'losses.jsonl', persist_every=persist_rows)
    expected = {row['id'] for row in test}
    if not set(journal.rows) <= expected:
        raise ValueError('Evaluation cache has foreign T records')
    try:
        for i, row in enumerate(test):
            if row['id'] in journal.rows: continue
            start = time.monotonic()
            loss = engine.losses(row, ['guidance'])['guidance']
            journal.append({'id': row['id'], 'task_id': row['task_id'], 'loss': loss,
                            'prediction_tokens': int(row['prediction_tokens']), 'input_tokens': int(row['input_tokens'])})
            print('TEST', output.name, i+1, len(test), loss, time.monotonic()-start, flush=True)
            atomic_json(output/'progress.json', {'completed': len(journal.rows), 'records': len(test),
                                                'updated_at_unix': time.time()})
    finally:
        journal.persist()
    frame = pd.DataFrame(journal.rows.values())
    if set(frame.id) != expected or not np.isfinite(frame.loss).all():
        raise ValueError('Incomplete/nonfinite T evaluation')
    frame['nll_sum'] = frame.loss * frame.prediction_tokens
    tasks = frame.groupby('task_id')[['nll_sum', 'prediction_tokens']].sum()
    tasks['loss'] = tasks.nll_sum / tasks.prediction_tokens
    tasks.to_csv(output/'tasks.csv')
    result = {'token_mean_nll': float(frame.nll_sum.sum()/frame.prediction_tokens.sum()),
              'task_mean_nll': float(tasks.loss.mean()), 'records': len(frame), 'tasks': len(tasks),
              'prediction_tokens': int(frame.prediction_tokens.sum()), 'identity': identity}
    atomic_json(output/'COMPLETED.json', result)
    return result
