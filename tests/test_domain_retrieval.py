import os

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from picking.common import read_json
from picking.domain_retrieval.config import load_config
from picking.domain_retrieval.data import Deduplicator, normalized
from picking.domain_retrieval.experiment import continuation_gpu_ids, select_arm
from picking.domain_retrieval.metrics import paired_bootstrap, rank_documents, retrieval_metrics
from picking.domain_retrieval.training import TokenStream, batch_tensors, token_nll, train


def losses():
    return pd.DataFrame({'id': ['large', 'small', 'worse', 'zero'],
                         'loss_before': [10., 2., 4., 4.], 'loss_after': [8., 1., 5., 4.],
                         'prediction_tokens': [8, 8, 8, 8]})


def test_absolute_relative_rankings_and_negative_drops():
    absolute = rank_documents('absolute_loss_drop', losses(), 5)
    relative = rank_documents('relative_loss_drop', losses(), 5)
    assert absolute.id.tolist() == ['large', 'small', 'zero', 'worse']
    assert relative.id.tolist() == ['small', 'large', 'zero', 'worse']
    assert relative.score.tolist() == [.5, .2, 0., -.25]
    random = rank_documents('random', losses(), 9)
    changed = losses().assign(loss_before=100., loss_after=0.)
    assert random.id.tolist() == rank_documents('random', changed, 9).id.tolist()


def test_method_receives_no_ground_truth(tmp_path, monkeypatch):
    method = tmp_path / 'method.py'
    method.write_text("def score(rows, *, seed):\n    assert all(set(r) == {'id', 'loss_before', 'loss_after', 'prediction_tokens'} for r in rows)\n    return [1.0] * len(rows)\n")
    monkeypatch.setattr('picking.domain_retrieval.metrics.method_file', lambda name: method)
    ranked = rank_documents('custom', losses().assign(is_target=True, category='secret'), 3)
    assert 'is_target' not in ranked and 'category' not in ranked


def test_pr_groups_ties_and_covers_all_candidates():
    frame = losses().assign(loss_before=2., loss_after=1.)
    truth = pd.DataFrame({'id': frame.id, 'is_target': [1, 0, 1, 0]})
    ranked = rank_documents('absolute_loss_drop', frame, 1)
    _, curve, metrics = retrieval_metrics(ranked, truth, ks=[2])
    assert metrics['average_precision'] == .5
    assert curve.recall.tolist() == [1., 0.]
    assert curve.precision.tolist() == [.5, 1.]
    assert retrieval_metrics(rank_documents('absolute_loss_drop', frame, 29), truth)[2]['average_precision'] == .5
    with pytest.raises(ValueError, match='match'):
        retrieval_metrics(ranked, truth.iloc[:-1])


def test_normalized_and_scored_prefix_deduplication():
    assert normalized('Ａ BC\nDef') == normalized('a bc   def')
    dedup = Deduplicator(.8)
    tokens = list(range(160))
    assert dedup.accept('first', tokens)
    assert not dedup.accept('first', list(range(200, 360)))
    assert not dedup.accept('different_document_same_prefix', tokens)
    assert not dedup.accept('near_duplicate', tokens[:-1] + [900])
    assert dedup.accept('different', list(range(1000, 1160)))


def test_partial_token_stream_resume_does_not_skip_or_duplicate_targets():
    rows = [{'id': 'a', 'input_ids': [1, 2, 3, 4]}, {'id': 'b', 'input_ids': [5, 6, 7]}]
    stream = TokenStream(rows, 42)
    first, count = stream.next_batch(2, 8)
    assert count == 2
    saved = dict(stream.state)
    continued, count = stream.next_batch(3, 8)
    resumed, resumed_count = TokenStream(rows, 42, saved).next_batch(3, 8)
    assert continued == resumed and count == resumed_count == 3
    targets = [token for row in first+continued for token in row['input_ids'][row['loss_start']:]]
    assert sorted(targets) == [2, 3, 4, 6, 7]
    assert stream.state == {'epoch': 1, 'document': 0, 'offset': 0}


def test_causal_mask_excludes_padding_and_reused_context():
    rows = [{'input_ids': [1, 2, 3, 4], 'loss_start': 3}, {'input_ids': [2, 3], 'loss_start': 1}]
    ids, attention, labels = batch_tensors(rows, 0, 'cpu')
    assert labels.tolist() == [[-100, -100, -100, 4], [-100, 3, -100, -100]]
    assert attention.tolist() == [[1, 1, 1, 1], [1, 1, 0, 0]]
    logits = torch.zeros(2, 4, 5)
    values = token_nll(logits, labels).sum(dim=1)
    assert torch.allclose(values, torch.tensor([np.log(5), np.log(5)], dtype=torch.float32))


def test_selection_only_uses_candidates_and_matches_document_count():
    docs = pd.DataFrame({'id': ['p', 't', 'a', 'b', 'c', 'd'],
                         'split': ['probe', 'target_test', 'candidate', 'candidate', 'candidate', 'candidate'],
                         'is_target': [True, True, True, True, False, False]})
    cfg = {'continuation': {'selected_documents': 2}, 'selection_seed': 3}
    assert set(select_arm(docs, {}, 'oracle_target', cfg).id) == {'a', 'b'}
    assert set(select_arm(docs, {}, 'other', cfg).id) == {'c', 'd'}
    rankings = {'bad': pd.DataFrame({'id': ['p', 't'], 'rank': [1, 2]})}
    with pytest.raises(RuntimeError, match='candidate'):
        select_arm(docs, rankings, 'bad', cfg)


def test_paired_bootstrap_token_weighted_sign_and_matching_masks():
    left = pd.DataFrame({'id': ['a', 'b'], 'sum_nll': [3., 12.], 'prediction_tokens': [1, 4]})
    right = pd.DataFrame({'id': ['b', 'a'], 'sum_nll': [8., 2.], 'prediction_tokens': [4, 1]})
    result = paired_bootstrap(left, right, seed=3, samples=20)
    assert result['difference'] == 1. and result['ci95'] == [1., 1.]
    with pytest.raises(ValueError, match='token masks'):
        paired_bootstrap(left, right.assign(prediction_tokens=1), seed=3, samples=2)


@pytest.mark.parametrize('device', [
    'cpu', pytest.param('cuda', marks=pytest.mark.skipif(
        os.environ.get('DOMAIN_RETRIEVAL_TEST_CUDA') != '1',
        reason='Opt in with an explicitly allowed CUDA_VISIBLE_DEVICES / DATA_PICKING_GPU_IDS'))])
def test_full_state_resume_matches_uninterrupted_continuation(tmp_path, monkeypatch, device):
    monkeypatch.setenv('DATA_PICKING_CHECKPOINT_IO', 'filesystem')
    torch.set_num_threads(1)
    torch.manual_seed(7)
    # A tiny saved checkpoint is a test fixture, never an experiment result.
    base = tmp_path / 'base'
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                                           num_attention_heads=2, intermediate_size=24,
                                           max_position_embeddings=32, eos_token_id=0,
                                           attention_dropout=.1, hidden_dropout=.1))
    model.save_pretrained(base)
    cfg = {'run_id': 'resume-test', 'artifact_root': str(tmp_path / 'persistent'),
           'runtime_root': str(tmp_path / 'runtime'), 'model': {'local_path': str(base)},
           'training': {'save_optimizer': True, 'auto_resume': True, 'learning_rate': .001,
                        'min_learning_rate': .0001, 'warmup_steps': 1, 'prediction_tokens_per_step': 13,
                        'micro_batch_size': 2, 'weight_decay': .01, 'betas': [.9, .95],
                        'adam_epsilon': 1e-8, 'max_grad_norm': 1., 'autocast_dtype': 'float32',
                        'save_steps': 2, 'keep_checkpoints': 2}}
    docs = pd.DataFrame({'id': ['a', 'b', 'c'], 'input_ids': [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11], [12, 13, 14]]})
    identity = {'fixture': 'saved_checkpoint'}
    full = train(cfg, 'full', docs, 42, 45, identity, device=device)
    partial = train(cfg, 'resumed', docs, 42, 45, identity, device=device, stop_after_steps=2)
    partial_state = torch.load(partial / 'state.pt', weights_only=False)
    assert partial_state['trainer']['step'] == 2 and partial_state['trainer']['seen_tokens'] == 26
    assert partial_state['optimizer']['state'] and partial_state['scheduler'] and partial_state['rng']
    resumed = train(cfg, 'resumed', docs, 42, 45, identity, device=device)
    final = torch.load(full / 'state.pt', weights_only=False)
    actual = torch.load(resumed / 'state.pt', weights_only=False)
    assert final['trainer'] == actual['trainer']
    assert actual['trainer']['seen_tokens'] == 45 and actual['trainer']['step'] == 4
    assert final['scheduler'] == actual['scheduler']
    for key in final['model']:
        assert torch.equal(final['model'][key], actual['model'][key]), key
    for key, state in final['optimizer']['state'].items():
        for field in state:
            assert torch.equal(state[field], actual['optimizer']['state'][key][field])
    assert read_json(resumed.parent.parent / 'TRAINING_COMPLETED.json')['prediction_tokens'] == 45
    modified = docs.copy(deep=True)
    modified.at[0, 'input_ids'] = [1, 2, 3]
    with pytest.raises(RuntimeError, match='identity differs'):
        train(cfg, 'resumed', modified, 42, 45, identity, device=device)


def test_default_config_requires_M0_and_optimizer_resume(tmp_path):
    from picking.common import REPO
    import yaml
    cfg = load_config(REPO / 'configs/domain_retrieval.yaml')
    cfg['training']['auto_resume'] = False
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='auto_resume'):
        load_config(path)


def test_duplicate_arms_cannot_start_concurrent_writers(tmp_path):
    from picking.common import REPO
    import yaml
    cfg = load_config(REPO / 'configs/domain_retrieval.yaml')
    cfg['continuation']['arms'].append('other')
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match='concurrent writers'):
        load_config(path)


def test_gpu_scheduler_reuses_only_its_own_idle_context(monkeypatch):
    monkeypatch.setattr('picking.runtime.free_gpu_ids', lambda: [1, 3])
    monkeypatch.setattr('picking.domain_retrieval.experiment.os.getpid', lambda: 123)
    monkeypatch.setattr('picking.domain_retrieval.experiment.subprocess.check_output', lambda *a, **k: '123\n')
    assert continuation_gpu_ids([0, 1, 2]) == {0, 1}
    monkeypatch.setattr('picking.domain_retrieval.experiment.subprocess.check_output', lambda *a, **k: '123\n999\n')
    assert continuation_gpu_ids([0, 1, 2]) == {1}
