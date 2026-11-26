from __future__ import annotations

import math

import torch


def gradient_statistics(engine, reference, chunk_elements=8*1024*1024):
    """Same full FP64 dot/norm, with GPU-local sums and one readback per GPU.

    The original implementation read back two scalars for every chunk, forcing
    thousands of CPU/GPU synchronization points for each PR. This changes only
    FP64 summation grouping; no projection, layer selection or cosine scaling.
    """
    if set(reference) != set(engine.parameters):
        raise ValueError('Reference parameter names differ')
    totals = {}
    for name, parameter in engine.parameters.items():
        if parameter.grad is None:
            raise RuntimeError(f'Missing full-model gradient: {name}')
        gradient = parameter.grad.detach().reshape(-1)
        target = reference[name].reshape(-1)
        if gradient.shape != target.shape:
            raise ValueError('Reference gradient shape differs')
        device = gradient.device
        if device not in totals:
            totals[device] = torch.zeros(2, device=device, dtype=torch.float64)
        for offset in range(0, gradient.numel(), chunk_elements):
            g = gradient[offset:offset+chunk_elements].double()
            q = target[offset:offset+chunk_elements].to(device=device, dtype=torch.float64)
            totals[device][0].add_(torch.dot(g, q))
            totals[device][1].add_(torch.dot(g, g))
    values = [value.cpu().tolist() for value in totals.values()]
    dot = math.fsum(value[0] for value in values)
    squared = math.fsum(value[1] for value in values)
    if not math.isfinite(dot) or not math.isfinite(squared) or squared < 0:
        raise FloatingPointError('Nonfinite gradient statistic')
    return dot, math.sqrt(squared)
