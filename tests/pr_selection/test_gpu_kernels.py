"""Opt-in Hopper kernel check against the differentiable PyTorch reference."""
import inspect
import os

import pytest
import torch


@pytest.mark.skipif(os.environ.get('PR_SCORING_GPU_TEST') != '1', reason='Explicit idle GPU required')
def test_gated_delta_backward_matches_reference():
    assert os.environ.get('DATA_PICKING_GPU_IDS') and os.environ.get('CUDA_VISIBLE_DEVICES')
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    reference = inspect.unwrap(torch_chunk_gated_delta_rule)
    torch.manual_seed(42)
    originals = [torch.randn(1, 128, 4, 128, device='cuda', dtype=torch.bfloat16),
                 torch.randn(1, 128, 4, 128, device='cuda', dtype=torch.bfloat16),
                 torch.randn(1, 128, 8, 128, device='cuda', dtype=torch.bfloat16),
                 -torch.rand(1, 128, 8, device='cuda', dtype=torch.float32),
                 torch.rand(1, 128, 8, device='cuda', dtype=torch.float32)]
    sets = [[x.clone().requires_grad_() for x in originals] for _ in range(2)]
    outputs = []
    for fn, (q, k, v, g, beta) in zip((reference, chunk_gated_delta_rule), sets):
        out, _ = fn(q, k, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        out.float().square().mean().backward()
        outputs.append(out.detach().float())
    relative = (outputs[1]-outputs[0]).norm()/outputs[0].norm()
    assert float(relative) < .02
    for name, a, b in zip(('q', 'k', 'v', 'g', 'beta'), sets[0], sets[1]):
        assert a.grad is not None and b.grad is not None and torch.isfinite(b.grad).all()
        error = (a.grad.float()-b.grad.float()).norm()/a.grad.float().norm().clamp_min(1e-12)
        assert float(error) < .03, (name, float(error))
