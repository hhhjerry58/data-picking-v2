from __future__ import annotations

import copy
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from picking.pr_selection.artifacts import ScoreJournal
from picking.pr_selection.data.pr import answer_start, encode_pr
from picking.pr_selection.data.splits import split_tasks
from picking.pr_selection.modeling.loss import chunked_head_loss
from picking.pr_selection.modeling.qwen35 import Engine


class Characters:
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text], 'offset_mapping': [(i, i+1) for i in range(len(text))]}


class TinyText(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(19, 7)
        self.proj = torch.nn.Linear(7, 7)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, inputs_embeds, use_cache=False):
        # Every target depends on earlier context; a detached-history bug would
        # fail the embedding-gradient comparison below.
        return SimpleNamespace(last_hidden_state=self.dropout(torch.tanh(self.proj(inputs_embeds.cumsum(1)))))


class TinyLM(torch.nn.Module):
    def __init__(self, dropout=0.):
        super().__init__()
        self.model = TinyText(dropout)
        self.lm_head = torch.nn.Linear(7, 19, bias=False)


def test_frozen_json_preserves_scientific_notation(tmp_path):
    from picking.pr_selection.config import load_config
    original = load_config('configs/pr_selection.yaml')
    frozen = tmp_path/'effective_config.json'
    frozen.write_text(json.dumps(original))
    restored = load_config(frozen)
    assert restored == original
    assert isinstance(restored['probe']['learning_rate'], float)
    assert isinstance(restored['probe']['adam_epsilon'], float)


def test_head_chunking_preserves_complete_context_gradient():
    torch.manual_seed(7)
    model = TinyLM()
    reference = copy.deepcopy(model)
    ids = [2, 4, 7, 3, 5, 6, 8]
    record = {'input_ids': ids, 'answer_only_runs': [[4, 7]]}
    engine = Engine(model, chunk_size=1)
    actual = engine.backward(record, 'answer_only')
    x = reference.model.embed_tokens(torch.tensor([ids]))
    hidden = reference.model(inputs_embeds=x).last_hidden_state
    expected = F.cross_entropy(reference.lm_head(hidden[0, 3:6]), torch.tensor(ids[4:]))
    expected.backward()
    assert actual == pytest.approx(float(expected.detach()), rel=2e-6)
    for (name, p), (other_name, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == other_name
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=2e-5)
    assert model.model.embed_tokens.weight.grad[2].abs().sum() > 0


def test_four_score_artifacts_end_to_end(tmp_path, monkeypatch):
    import pandas as pd
    from picking.common import file_hash
    from picking.pr_selection.artifacts import CheckpointStore
    from picking.pr_selection.config import load_config, run_dir
    from picking.pr_selection.scoring import score_m0, score_m1, combine
    from picking.pr_selection.training import train_probe
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO', 'filesystem')
    cfg = load_config('configs/pr_selection.yaml')
    cfg.update(run_id='tiny-end-to-end', artifact_root=str(tmp_path/'artifacts'), runtime_root=str(tmp_path/'runtime'))
    cfg.pop('probe_from_run', None)
    cfg['model']['name'] = 'test/TinyLM'
    cfg['probe'].update(epochs=2, documents_per_step=2, learning_rate=.003)
    q = [{'id': 'q', 'input_ids': [1, 2, 3, 4], 'target_runs': [[2, 4]], 'prediction_tokens': 2}]
    candidates = [{'id': f'fixture/repo#{i}', 'query_id': f'query-{i}', 'source_category': 'fixture', 'format': 'git_diff',
                   'input_ids': [i+1, 2, 3, 4, 5], 'input_tokens': 5,
                   'full_text_runs': [[1, 5]], 'answer_only_runs': [[3, 5]],
                   'full_text_tokens': 4, 'answer_only_tokens': 2} for i in range(3)]
    folder = run_dir(cfg)
    (folder/'data').mkdir(parents=True)
    pd.DataFrame(q).to_parquet(folder/'data/guidance.parquet', index=False)
    pd.DataFrame(candidates).to_parquet(folder/'data/candidates.parquet', index=False)
    torch.manual_seed(3)
    engine = Engine(TinyLM(), chunk_size=2)
    score_m0(engine, candidates, cfg)
    probe_store = CheckpointStore(cfg, 'probe', file_hash(folder/'data/guidance.parquet'))
    train_probe(engine, q, probe_store, cfg)
    score_m1(engine, candidates, cfg)
    frame = combine(cfg, candidates)
    assert len(frame) == 6
    assert set(frame.objective) == {'full_text', 'answer_only'}
    assert json.loads((folder/'SCORING_COMPLETED.json').read_text())['distributions'] == 4
    for objective, group in frame.groupby('objective'):
        assert set(group.id) == {f'fixture/repo#{i}' for i in range(3)}
        for name in cfg['methods']:
            saved = pd.read_parquet(folder/'methods'/name/objective/'scores.parquet')
            assert len(saved) == 3 and saved.score.notna().all()
    assert frame.absolute_loss_drop.abs().max() > 0


def test_pr_objectives_have_identical_inputs_and_unicode_boundary():
    text = '## Required answer\nReturn diff.\n\n## Answer\né中 diff\n'
    row = {'id': 'r/p#1', 'query_id': 'a', 'format': 'git_diff', 'text': text,
           'metadata': {'source': {'domain': 'base'}}}
    encoded = encode_pr(Characters(), row, 1000, 1)
    assert encoded['input_ids'] == [ord(c) for c in text]
    assert encoded['answer_only_runs'] == [[answer_start(text), len(text)]]
    assert encoded['full_text_runs'] == [[1, len(text)]]
    assert encoded['answer_only_tokens'] < encoded['full_text_tokens']


def test_ambiguous_answer_and_truncation_are_rejected():
    with pytest.raises(ValueError, match='ambiguous'):
        answer_start('## Required answer\nx\n## Answer\ny\n## Answer\nz')
    row = {'id': 'r/p#1', 'query_id': 'a', 'format': 'git_diff',
           'text': '## Required answer\nx\n## Answer\nyyyyy', 'metadata': {'source': {'domain': 'base'}}}
    with pytest.raises(ValueError, match='over_candidate_length'):
        encode_pr(Characters(), row, 5, 1)


def test_task_split_keeps_trials_together():
    rows = [{'id': f'{task}__{trial}:9', 'source': {'trajectory': {'trial_name': f'{task}__{trial}', 'target_step_id': 9}}}
            for task in ('a', 'b', 'c', 'd') for trial in ('one', 'two')]
    split = split_tasks(rows, 2, 42, 4)
    for task in ('a', 'b', 'c', 'd'):
        assert split[f'{task}__one:9'] == split[f'{task}__two:9']
    assert list(split.values()).count('Q') == 4


def test_score_resume_recovers_only_uncommitted_tail(tmp_path):
    path = tmp_path/'scores.jsonl'
    journal = ScoreJournal(path, {'model': 'M0'})
    journal.append({'id': 'one', 'score': -1.})
    with path.open('ab') as stream:
        stream.write(b'{"id":"two"')
    restored = ScoreJournal(path, {'model': 'M0'})
    assert list(restored.rows) == ['one']
    restored.append({'id': 'two', 'score': 2.})
    assert len(ScoreJournal(path, {'model': 'M0'}).rows) == 2
    with pytest.raises(RuntimeError, match='Identity changed'):
        ScoreJournal(path, {'model': 'M1'})


def test_optimizer_resume_matches_uninterrupted_probe(tmp_path, monkeypatch):
    from picking.pr_selection.artifacts import CheckpointStore
    from picking.pr_selection.config import load_config
    from picking.pr_selection.training import train_probe
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO', 'filesystem')
    cfg = load_config('configs/pr_selection.yaml')
    cfg.update(artifact_root=str(tmp_path/'artifacts'), runtime_root=str(tmp_path/'runtime'))
    cfg['probe'].update(epochs=2, documents_per_step=2, learning_rate=.003, save_steps=2)
    records = [{'id': str(i), 'input_ids': [i+1, 2, 4, 6], 'target_runs': [[2, 4]], 'prediction_tokens': 2}
               for i in range(4)]
    torch.manual_seed(100)
    original = TinyLM(dropout=.2)
    full = Engine(copy.deepcopy(original), chunk_size=2)
    cfg['run_id'] = 'uninterrupted'
    full_store = CheckpointStore(cfg, 'probe', 'tiny-data')
    train_probe(full, records, full_store, cfg)
    cfg['run_id'] = 'resumed'
    partial = Engine(copy.deepcopy(original), chunk_size=2)
    partial_store = CheckpointStore(cfg, 'probe', 'tiny-data')
    train_probe(partial, records, partial_store, cfg, stop_after_steps=1)
    with (partial_store.folder/'train.jsonl').open('ab') as stream:
        stream.write(b'{"step":2')
    resumed = Engine(copy.deepcopy(original), chunk_size=2)
    train_probe(resumed, records, partial_store, cfg)
    for (name, p), (other, q) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    full_state, resumed_state = full_store.load(), partial_store.load()
    assert full_state['trainer'] == resumed_state['trainer']
    assert full_state['scheduler'] == resumed_state['scheduler']
    for key, state in full_state['optimizer']['state'].items():
        for field, value in state.items():
            torch.testing.assert_close(value, resumed_state['optimizer']['state'][key][field], rtol=0, atol=0)


def test_score_journal_mirror_survives_process_and_node_loss(tmp_path):
    path, mirror = tmp_path/'remote/scores.jsonl', tmp_path/'local/scores.jsonl'
    journal = ScoreJournal(path, {'model': 'M0'}, mirror_path=mirror, persist_every=2)
    journal.append({'id': 'a', 'score': 1.})
    assert not path.exists()
    journal = ScoreJournal(path, {'model': 'M0'}, mirror_path=mirror, persist_every=2)
    assert list(journal.rows) == ['a']
    journal.append({'id': 'b', 'score': 2.})
    assert path.exists()
    mirror.unlink()
    journal = ScoreJournal(path, {'model': 'M0'}, mirror_path=mirror, persist_every=2)
    assert list(journal.rows) == ['a', 'b']


@pytest.mark.skipif(os.environ.get('PR_SCORING_GPU_TEST') != '1', reason='Explicit idle GPU required')
def test_cpu_optimizer_offload_resumes_cuda_model(tmp_path, monkeypatch):
    assert os.environ.get('DATA_PICKING_GPU_IDS') and os.environ.get('CUDA_VISIBLE_DEVICES')
    from picking.pr_selection.artifacts import CheckpointStore
    from picking.pr_selection.config import load_config
    from picking.pr_selection.training import train_probe
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO', 'filesystem')
    cfg = load_config('configs/pr_selection.yaml')
    cfg.update(artifact_root=str(tmp_path/'artifacts'), runtime_root=str(tmp_path/'runtime'))
    cfg['probe'].update(epochs=2, documents_per_step=2, learning_rate=.003, save_steps=2)
    records = [{'id': str(i), 'input_ids': [i+1, 2, 4, 6], 'target_runs': [[2, 4]], 'prediction_tokens': 2}
               for i in range(4)]
    torch.manual_seed(100)
    initial = TinyLM(dropout=.2).cuda()
    full = Engine(copy.deepcopy(initial), chunk_size=2, compute_dtype=torch.bfloat16)
    cfg['run_id'] = 'cuda-full'
    store = CheckpointStore(cfg, 'probe', 'tiny-cuda-data')
    train_probe(full, records, store, cfg)
    cfg['run_id'] = 'cuda-resumed'
    partial = Engine(copy.deepcopy(initial), chunk_size=2, compute_dtype=torch.bfloat16)
    store = CheckpointStore(cfg, 'probe', 'tiny-cuda-data')
    train_probe(partial, records, store, cfg, stop_after_steps=1)
    resumed = Engine(copy.deepcopy(initial), chunk_size=2, compute_dtype=torch.bfloat16)
    train_probe(resumed, records, store, cfg)
    for (name, p), (other, q) in zip(full.model.named_parameters(), resumed.model.named_parameters()):
        assert name == other
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert any(not torch.equal(p, q) for p, q in zip(initial.parameters(), resumed.model.parameters()))
