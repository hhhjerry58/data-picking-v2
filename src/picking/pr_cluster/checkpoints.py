"""FSDP rank shards include model, Adam, scheduler, RNG and the exact step cursor."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP, StateDictType,
    ShardedStateDictConfig, ShardedOptimStateDictConfig)


@contextmanager
def sharded_state(model):
    with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT,
                             ShardedStateDictConfig(offload_to_cpu=True),
                             ShardedOptimStateDictConfig(offload_to_cpu=True)):
        yield


def save_local(model, optimizer, scheduler, step, identity, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with sharded_state(model):
        payload = {'identity': identity, 'rank': dist.get_rank(), 'world_size': dist.get_world_size(), 'step': step,
                   'model': model.state_dict(), 'optimizer': FSDP.optim_state_dict(model, optimizer),
                   'scheduler': scheduler.state_dict(),
                   'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                           'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary); os.replace(temporary, path)


def restore_local(model, optimizer, scheduler, identity, path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload['identity'] != identity or payload['rank'] != dist.get_rank()
            or payload['world_size'] != dist.get_world_size()):
        raise ValueError('Checkpoint configuration/rank/world size differs')
    with sharded_state(model):
        model.load_state_dict(payload['model'], strict=True)
        state = FSDP.optim_state_dict_to_load(model, optimizer, payload['optimizer'])
        optimizer.load_state_dict(state)
    scheduler.load_state_dict(payload['scheduler'])
    random.setstate(payload['rng']['python']); np.random.set_state(payload['rng']['numpy'])
    torch.set_rng_state(payload['rng']['torch']); torch.cuda.set_rng_state(payload['rng']['cuda'])
    return int(payload['step'])
