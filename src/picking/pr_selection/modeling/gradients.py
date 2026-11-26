from __future__ import annotations

import torch

from ..artifacts import rng_state, restore_rng


def target_gradient(engine, guidance, checkpoint_store, config):
    """Exactly sum guidance token gradients, normalized once by all Q targets."""
    denominator = sum(int(row['prediction_tokens']) for row in guidance)
    if denominator <= 0:
        raise ValueError('Guidance needs prediction tokens')
    state = checkpoint_store.load()
    cursor = 0
    engine.zero_grad()
    if state is not None:
        if state['denominator'] != denominator:
            raise RuntimeError('Guidance normalization changed')
        cursor = state['cursor']
        engine.install_gradients(state['gradients'])
        restore_rng(state['rng'])
        del state
    for index in range(cursor, len(guidance)):
        row = guidance[index]
        loss = engine.backward(row, 'guidance', denominator=denominator, accumulate=True)
        print('TARGET_GRADIENT', index+1, len(guidance), row['id'], loss, flush=True)
        if (index+1) % config['scoring']['gradient_save_documents'] == 0 or index+1 == len(guidance):
            checkpoint_store.save({'cursor': index+1, 'denominator': denominator,
                                   'gradients': engine.copy_gradients(), 'rng': rng_state()}, index+1)
    reference = {}
    for name, parameter in engine.parameters.items():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f'Invalid guidance gradient: {name}')
        reference[name] = parameter.grad.detach()
    engine.zero_grad()
    return reference
