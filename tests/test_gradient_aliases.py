import torch
from picking.pretraining import separate_gradient_aliases


def test_equal_compiled_bias_gradients_have_independent_accumulators():
    model = torch.nn.Module()
    model.register_parameter('attention_bias',torch.nn.Parameter(torch.zeros(4)))
    model.register_parameter('mlp_bias',torch.nn.Parameter(torch.zeros(4)))
    first = torch.tensor([1.,2.,3.,4.])
    model.attention_bias.grad = first
    model.mlp_bias.grad = first
    assert len(separate_gradient_aliases(model)) == 1
    model.attention_bias.grad.add_(torch.ones(4))
    model.mlp_bias.grad.add_(torch.ones(4))
    expected = torch.tensor([2.,3.,4.,5.])
    assert torch.equal(model.attention_bias.grad,expected)
    assert torch.equal(model.mlp_bias.grad,expected)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,foreach=False)
    assert torch.allclose(norm,torch.sqrt(2*expected.square().sum()))
    assert torch.equal(model.attention_bias.grad,model.mlp_bias.grad)
    assert separate_gradient_aliases(model) == []
