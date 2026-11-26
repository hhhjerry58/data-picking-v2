from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from picking.pr_selection.config import atomic_json
from picking.pr_selection.modeling.qwen35 import load_engine
from .backend import gradient_statistics
from .parent import checkpoint_descriptor, open_verified_checkpoint, parent_config


def benchmark(parent, preview, output, gpu_count=2):
    parent, preview, output = Path(parent), Path(preview), Path(output)
    cfg = parent_config(parent)
    descriptor = checkpoint_descriptor(parent, 'target-gradient')
    frame = pd.read_parquet(preview/'candidates.parquet').sort_values('input_tokens')
    rows = frame.iloc[sorted({0,len(frame)//2,len(frame)-1})].to_dict('records')
    old = pd.read_parquet(preview/'m0.parquet').set_index(['id','objective'])
    engine = load_engine(cfg, list(range(gpu_count)))
    state = open_verified_checkpoint(descriptor)
    if state['cursor'] != 133:
        raise ValueError('Expected the completed parent guidance gradient')
    reference = {name: tensor.to(engine.parameters[name].device) for name,tensor in state['gradients'].items()}
    del state
    engine.install_gradients(reference)
    _, reference_norm = gradient_statistics(engine, reference)
    engine.zero_grad()
    measurements = []
    for row in rows:
        for objective in cfg['objectives']:
            start = time.perf_counter()
            loss = engine.backward(row, objective)
            backward_seconds = time.perf_counter()-start
            start = time.perf_counter()
            legacy = engine.gradient_statistics(reference)
            legacy_seconds = time.perf_counter()-start
            start = time.perf_counter()
            fast = gradient_statistics(engine, reference)
            fast_seconds = time.perf_counter()-start
            if abs(legacy[0]-fast[0]) > 1e-8+1e-10*abs(legacy[0]) or abs(legacy[1]-fast[1]) > 1e-8+1e-10*legacy[1]:
                raise RuntimeError('Accelerated FP64 reduction differs from original')
            parent_row = old.loc[(row['id'],objective)]
            result = {'id':row['id'], 'objective':objective, 'input_tokens':int(row['input_tokens']),
                      'loss':loss, 'parent_loss':float(parent_row.loss_before),
                      'dot':fast[0], 'parent_dot':float(parent_row.gradient_dot),
                      'norm':fast[1], 'parent_norm':float(parent_row.gradient_norm),
                      'dot_relative_difference':abs(fast[0]-parent_row.gradient_dot)/max(1e-12,abs(parent_row.gradient_dot)),
                      'dot_difference_over_norm_product':abs(fast[0]-parent_row.gradient_dot)/max(1e-12,reference_norm*parent_row.gradient_norm),
                      'gradient_norm_relative_difference':abs(fast[1]-parent_row.gradient_norm)/max(1e-12,parent_row.gradient_norm),
                      'backward_seconds':backward_seconds, 'old_reduction_seconds':legacy_seconds,
                      'new_reduction_seconds':fast_seconds,
                      'gpu_peak_bytes':{str(i):torch.cuda.max_memory_allocated(i) for i in range(gpu_count)}}
            result['strict_placement_parity'] = bool(abs(loss-parent_row.loss_before)<=2e-5*max(1.,abs(parent_row.loss_before))
                and abs(fast[0]-parent_row.gradient_dot)<=1e-5+2e-5*abs(parent_row.gradient_dot))
            measurements.append(result)
            atomic_json(output, {'parent':str(parent),'gpu_count':gpu_count,'reference_gradient_norm':reference_norm,
                                'all_strict_placement_checks_passed':all(r['strict_placement_parity'] for r in measurements),
                                'measurements':measurements})
            print('BENCHMARK',json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--parent',required=True)
    parser.add_argument('--preview',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu-count',type=int,default=2)
    args=parser.parse_args()
    benchmark(args.parent,args.preview,args.output,args.gpu_count)
