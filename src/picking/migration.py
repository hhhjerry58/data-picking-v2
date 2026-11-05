"""Read-only compatibility checks before continuing on a different devnode."""
import importlib.metadata
import shutil
import socket
import sys
import uuid
from pathlib import Path

from .common import config,paths,read_json,digest,file_hash,utc_now,REPO
from .workflow import training_complete
from .runtime import gpu_inventory,allowed_gpu_ids
from .storage import checkpoint_transport,verified_checkpoint_mirror


def check_resume(cfg=None, require_gpu=True, full_checkpoint=False):
    cfg=cfg or config();p=paths(cfg)
    expected=read_json(REPO/'runtime_versions.json')
    import torch
    versions={'python_major_minor':f'{sys.version_info.major}.{sys.version_info.minor}',
              'torch':torch.__version__,'torch_git_version':torch.version.git_version,'cuda_build':torch.version.cuda,
              'transformers':importlib.metadata.version('transformers'),'numpy':importlib.metadata.version('numpy')}
    for name,value in versions.items():
        if value!=expected[name]:raise RuntimeError(f'{name} differs from the checkpoint environment: expected {expected[name]}, found {value}')
    implementation=file_hash(REPO/'src/picking/pretraining.py')
    if implementation!=expected['pretraining_implementation_sha256']:
        raise RuntimeError('Numerical training implementation differs from existing checkpoints')
    preflight=read_json(p['run']/'preflight/training_resume_verification.json')
    if not preflight['passed'] or preflight['pretraining_implementation_sha256']!=implementation:
        raise RuntimeError('Verified training configuration is unavailable')
    from . import _fasttext_native
    if not p['run'].is_dir():raise RuntimeError('Mount the original artifact store at the path in configs/experiment.yaml')
    for filename in ['config.json','tokenizer.json','tokenizer_config.json']:
        if not (p['assets']/'pythia160m'/filename).is_file():raise RuntimeError(f'Missing model configuration/tokenizer: {filename}')
    for path in [p['data']/'CORPUS_COMPLETED.json',p['data']/'evaluation/COMPLETED.json',
                 p['run']/'candidate_scores/COMPLETED.json',p['run']/'selection/COMPLETED.json']:
        read_json(path)
    p['scratch'].mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(p['scratch']).free
    if free<32*1024**3:raise RuntimeError('Runtime storage needs at least 32 GiB free for one job; set DATA_PICKING_RUNTIME_ROOT to a larger POSIX filesystem')
    probe=p['run']/('.migration-probe-'+uuid.uuid4().hex)
    probe.write_text('persistent-read-write-check\n')
    try:
        if probe.read_text()!='persistent-read-write-check\n':raise RuntimeError('Persistent storage read/write check failed')
    finally:probe.unlink(missing_ok=True)
    jobs=[]
    for seed in cfg['pretraining']['seeds']:
        for arm in cfg['pretraining']['arms']:
            if training_complete(cfg,arm,seed):
                jobs.append({'arm':arm,'seed':seed,'state':'trained'})
                continue
            folder=p['run']/'pretraining'/f'{arm}-seed-{seed}'
            candidates=[]
            for marker in (folder/'checkpoints').glob('step-*/COMPLETED.json'):
                manifest=read_json(marker.parent/'manifest.json')
                if read_json(marker)['manifest_hash']!=digest(manifest):raise RuntimeError('Checkpoint completion hash mismatch')
                candidates.append((manifest['global_step'],marker.parent,manifest))
            store=read_json(p['run']/'selection'/arm/'TOKEN_STORE_COMPLETED.json')
            if (p['run']/'selection'/arm/'tokens.bin').stat().st_size!=store['tokens']*4:
                raise RuntimeError('Selected token store is incomplete')
            if candidates:
                step,checkpoint,manifest=max(candidates,key=lambda item:item[0])
                effective=read_json(folder/'effective_config.json')
                if effective['identity']['training']!=cfg['pretraining'] or manifest['fingerprint']!=effective['fingerprint']:
                    raise RuntimeError('Saved optimizer/training configuration differs')
                if not manifest.get('save_optimizer') or not manifest.get('auto_resume'):
                    raise RuntimeError('Checkpoint does not declare full optimizer state and auto-resume')
                metadata=manifest['files']['state.pt']
                if (checkpoint/'state.pt').stat().st_size!=metadata['size']:raise RuntimeError('Incomplete checkpoint file')
                packed=p['run']/'selection'/arm/f'packed-seed-{seed}.bin'
                if read_json(str(packed)+'.complete.json')['sha256']!=effective['identity']['input_sha256']:
                    raise RuntimeError('Training input fingerprint changed')
                if full_checkpoint:
                    source=verified_checkpoint_mirror(folder,checkpoint,metadata['sha256'],cfg) or checkpoint/'state.pt'
                    if file_hash(source)!=metadata['sha256']:raise RuntimeError('Checkpoint SHA-256 mismatch')
                    payload=torch.load(source,map_location='cpu',mmap=False,weights_only=False)
                    optimizer_steps={int(value['step']) for value in payload['optimizer']['state'].values()}
                    if optimizer_steps!={step} or payload['scheduler']['last_epoch']!=step or payload['trainer']['global_step']!=step:
                        raise RuntimeError('Optimizer/scheduler/trainer state disagree')
                    if payload['trainer']['seen_tokens']!=manifest['seen_tokens'] or not all(key in payload['rng'] for key in ['python','numpy','torch','cuda']):
                        raise RuntimeError('Token cursor or RNG state missing')
                    del payload
                jobs.append({'arm':arm,'seed':seed,'state':'resume','checkpoint':str(checkpoint),'step':step,
                             'seen_tokens':manifest['seen_tokens'],'save_optimizer':True,'auto_resume':True})
            else:
                jobs.append({'arm':arm,'seed':seed,'state':'not_started'})
    pending=[job for job in jobs if job['state']!='trained']
    transport=checkpoint_transport(cfg,p['run']/'pretraining'/f"{pending[0]['arm']}-seed-{pending[0]['seed']}" if pending else p['run']/'pretraining/random-seed-44')
    transport_kind=type(transport).__name__
    inventory=[]
    if require_gpu:
        inventory=gpu_inventory()
        allowed=allowed_gpu_ids(row['index'] for row in inventory)
        available=[row for row in inventory if row['index'] in allowed and row['memory_used_mib']<768 and row['utilization']<5]
        if not available:raise RuntimeError('No selected GPU is free; use --wait-for-gpu when starting, or choose another node/GPU')
        if any(row['memory_total_mib']<70000 for row in available):
            raise RuntimeError('The frozen micro-batch configuration expects 80GB GPUs; select compatible GPUs')
    return {'checked_at_utc':utc_now(),'hostname':socket.gethostname(),'versions':versions,'jobs':jobs,
            'artifact_root':str(p['root']),'runtime_root':str(p['scratch']),'runtime_free_gib':round(free/1024**3,1),
            'checkpoint_transport':transport_kind,'numerical_training_sha256':implementation,'full_checkpoint_verified':full_checkpoint,
            'gpu_inventory':inventory,'save_optimizer':cfg['pretraining']['save_optimizer'],'auto_resume':cfg['pretraining']['auto_resume']}
