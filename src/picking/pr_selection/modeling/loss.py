from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

from .masks import indices


def chunked_head_loss(hidden, head, token_ids, target_positions, *, chunk_size=128,
                      denominator=None, backward=False, autocast_dtype=None):
    """Exact chain rule, without keeping sequence × vocabulary logits alive.

    Accumulate the head and hidden-state gradients chunk by chunk, then bridge
    the hidden adjoint into the FULL context graph once. No context is detached
    from the resulting parameter gradient.
    """
    if not target_positions or min(target_positions) < 1:
        raise ValueError('Targets require a preceding causal position')
    denominator = len(target_positions) if denominator is None else denominator
    if denominator <= 0:
        raise ValueError('Invalid normalization')
    device = next(head.parameters()).device
    activation = hidden.detach().requires_grad_(True) if backward else hidden
    ids = torch.as_tensor(token_ids, device=device, dtype=torch.long)
    total = 0.0
    for start in range(0, len(target_positions), chunk_size):
        positions = torch.tensor(target_positions[start:start+chunk_size], device=activation.device)
        x = activation[0].index_select(0, positions-1).to(device)
        ctx = torch.autocast('cuda', dtype=autocast_dtype) if device.type == 'cuda' and autocast_dtype else nullcontext()
        with ctx:
            logits = head(x)
        loss = F.cross_entropy(logits.float(), ids[positions.to(device)], reduction='sum')
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite token NLL')
        total += float(loss.detach().double())
        if backward:
            (loss / denominator).backward()
    if backward:
        if activation.grad is None:
            raise RuntimeError('Missing hidden-state adjoint')
        hidden.backward(activation.grad)
    return total / len(target_positions)


def positions(record, objective):
    key = 'target_runs' if objective == 'guidance' else f'{objective}_runs'
    return indices(record[key])
