"""Read completed-stage receipts so restart schedules only outstanding work."""
from pathlib import Path
from .common import paths, read_json, digest, file_hash, REPO


def receipt(path):
    path = Path(path)
    return read_json(path) if path.exists() else None


def training_complete(cfg, arm, seed):
    folder = paths(cfg)['run']/'pretraining'/f'{arm}-seed-{seed}'
    completed = receipt(folder/'TRAINING_COMPLETED.json')
    if completed is None:
        return False
    effective = read_json(folder/'effective_config.json')
    checkpoint = Path(completed['checkpoint'])
    manifest = read_json(checkpoint/'manifest.json')
    marker = read_json(checkpoint/'COMPLETED.json')
    if marker['manifest_hash'] != digest(manifest):
        raise RuntimeError('Completed training checkpoint manifest changed')
    if completed['fingerprint'] != effective['fingerprint'] or manifest['fingerprint'] != effective['fingerprint']:
        raise RuntimeError('Completed training fingerprint differs')
    if effective['identity']['training'] != cfg['pretraining']:
        raise RuntimeError('Completed model used a different training configuration')
    if effective['identity']['implementation_sha256'] != file_hash(REPO/'src/picking/pretraining.py'):
        raise RuntimeError('Training implementation changed across paired runs')
    if completed['seen_tokens'] != cfg['pretraining']['total_prediction_tokens'] or manifest['seen_tokens'] != completed['seen_tokens']:
        raise RuntimeError('Completed training token budget differs')
    if not manifest.get('save_optimizer') or not manifest.get('auto_resume'):
        raise RuntimeError('Full optimizer state and auto-resume are required')
    state = manifest['files']['state.pt']
    if (checkpoint/'state.pt').stat().st_size != state['size']:
        raise RuntimeError('Completed model state is missing or incomplete')
    return True


def evaluation_complete(cfg, arm, seed):
    folder = paths(cfg)['run']/'evaluation'/f'{arm}-seed-{seed}'
    completed = receipt(folder/'COMPLETED.json')
    if completed is None:
        return False
    if file_hash(folder/'metrics.json') != completed['metrics_sha256']:
        raise RuntimeError('Completed evaluation metrics changed')
    metrics = read_json(folder/'metrics.json')
    trained = read_json(paths(cfg)['run']/'pretraining'/f'{arm}-seed-{seed}'/'TRAINING_COMPLETED.json')
    if metrics['identity']['training_fingerprint'] != trained['fingerprint']:
        raise RuntimeError('Evaluation refers to a different training run')
    return True


def completed_job(cfg, args):
    p = paths(cfg)
    command = args[0]
    def option(name):
        return args[args.index(name)+1]
    simple = {'prepare-label-cache': p['data']/'label_documents.parquet.complete.json',
              'aggregate-scores': p['run']/'label_scores.parquet.complete.json',
              'scan-candidates': p['run']/'candidate_scores/COMPLETED.json',
              'select': p['run']/'selection/COMPLETED.json',
              'prepare-evaluation': p['data']/'evaluation/COMPLETED.json'}
    if command in simple:
        return receipt(simple[command]) is not None
    if command == 'score':
        shard, total = int(option('--shard')), int(option('--num-shards'))
        folder = p['run']/'scores/fp32_sorted_short_single_v2'
        return all(receipt(folder/label/f'shard-{shard:02d}-of-{total:02d}'/'COMPLETED.json') is not None for label in ['reference','sft'])
    if command == 'prepare-fasttext':
        return all(receipt(p['run']/'fasttext'/metric/'train.txt.complete.json') is not None
                   and receipt(p['run']/'fasttext'/metric/'label_summary.json') is not None
                   and receipt(p['run']/'fasttext'/metric/'threshold.json') is not None
                   and (p['run']/'fasttext'/metric/'labels.parquet').is_file()
                   for metric in cfg['fasttext']['metrics'])
    if command == 'train-fasttext':
        folder = p['run']/'fasttext'/option('--metric')
        record = receipt(folder/'TRAINING_COMPLETED.json')
        if not record or not record['trainer']['finished'] or receipt(folder/'heldout_metrics.json') is None:
            return False
        effective = read_json(folder/'effective_config.json')
        model = receipt(folder/'model.bin.complete.json')
        if model is None:
            return False
        if effective['fingerprint'] != record['fingerprint'] or model['fingerprint'] != record['fingerprint']:
            raise RuntimeError('Completed classifier identity differs')
        if effective['parameters'] != cfg['fasttext']:
            raise RuntimeError('Completed classifier configuration differs')
        return True
    if command == 'build-token-stores':
        return all(receipt(p['run']/'selection'/arm/'TOKEN_STORE_COMPLETED.json') is not None for arm in cfg['pretraining']['arms'])
    if command == 'pack':
        return receipt(p['run']/'selection'/option('--arm')/f"packed-seed-{int(option('--seed'))}.bin.complete.json") is not None
    if command == 'pretrain':
        return training_complete(cfg, option('--arm'), int(option('--seed')))
    if command == 'evaluate':
        return evaluation_complete(cfg, option('--arm'), int(option('--seed')))
    return False
