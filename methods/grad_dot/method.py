"""Full-language-model GradDot at M0; no cosine normalization or projection."""


def prepare(engine, guidance, checkpoint_store, config):
    from picking.pr_selection.modeling.gradients import target_gradient
    return target_gradient(engine, guidance, checkpoint_store, config)


def score_example(engine, record, objective, reference):
    loss = engine.backward(record, objective)
    dot, norm = engine.gradient_statistics(reference)
    return {'loss': loss, 'gradient_dot': dot, 'gradient_norm': norm}


def score(rows, *, seed):
    return [row['gradient_dot'] for row in rows]
