"""Same-device mathematical and distributed checks; no H100/A100 score mixing."""
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

from .checkpoints import restore_local, save_local
from .model import wrap


def check_linear_attention(device):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
    reference = inspect.unwrap(torch_chunk_gated_delta_rule)
    torch.manual_seed(42)
    tensors = [torch.randn(1,128,4,128,device=device,dtype=torch.bfloat16),
               torch.randn(1,128,4,128,device=device,dtype=torch.bfloat16),
               torch.randn(1,128,8,128,device=device,dtype=torch.bfloat16),
               -torch.rand(1,128,8,device=device), torch.rand(1,128,8,device=device)]
    inputs = [[x.clone().requires_grad_() for x in tensors] for _ in range(2)]
    outputs = []
    for function, values in zip([reference,chunk_gated_delta_rule], inputs):
        q,k,v,g,beta = values
        out,_ = function(q,k,v,g=g,beta=beta,use_qk_l2norm_in_kernel=True)
        out.float().square().mean().backward(); outputs.append(out.detach().float())
    forward_error = float((outputs[1]-outputs[0]).norm()/outputs[0].norm().clamp_min(1e-12))
    errors = [float((b.grad.float()-a.grad.float()).norm()/a.grad.float().norm().clamp_min(1e-12))
              for a,b in zip(inputs[0],inputs[1])]
    if forward_error >= .02 or any(not torch.isfinite(b.grad).all() for b in inputs[1]) or max(errors) >= .03:
        raise RuntimeError(f'Linear-attention reference check failed: forward={forward_error}, backward={errors}')
    return {'forward_relative_error':forward_error,'backward_relative_errors':errors}


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.proj=torch.nn.Linear(16,16)
    def forward(self,x): return x+torch.tanh(self.proj(x))


class TinyText(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens=torch.nn.Embedding(97,16)
        self.layers=torch.nn.ModuleList([TinyLayer() for _ in range(2)])
    def forward(self,inputs_embeds,use_cache=False):
        value=inputs_embeds.cumsum(1)
        for layer in self.layers: value=layer(value)
        return SimpleNamespace(last_hidden_state=value)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.model=TinyText(); self.lm_head=torch.nn.Linear(16,97,bias=False)


def check_fsdp(device, scratch):
    """Unequal target counts + a zero-weight rank must equal global token loss."""
    from .model import LossModel
    rank, world=dist.get_rank(),dist.get_world_size()
    torch.manual_seed(92); backbone=TinyLM()
    reference=LossModel(copy.deepcopy(backbone),1,compute_dtype=torch.float32).to(device)
    model=wrap(backbone,device,1); model.module.compute_dtype=torch.float32
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    total=sum(i+1 for i in range(world-1))
    ids=torch.tensor([[1,2,3,4,5,6,7,8,9,10]],device=device)
    ref_opt=torch.optim.AdamW(reference.parameters(),lr=.001,weight_decay=0.,foreach=False)
    ref_loss=sum(reference(ids+micro,torch.arange(1,i+2,device=device))
                 for micro in range(2) for i in range(world-1))/(2*total)
    ref_loss.backward()
    def distributed_step(compare_gradients=False):
        optimizer.zero_grad(set_to_none=True)
        positions=torch.arange(1,rank+2,device=device) if rank<world-1 else torch.tensor([1],device=device)
        for micro in range(2):
            value=model(ids+micro,positions)
            (value*(world/(2*total) if rank<world-1 else 0.)).backward()
        if compare_gradients:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            with FSDP.summon_full_params(model,writeback=False,recurse=True,with_grads=True):
                actual=dict(model.module.backbone.named_parameters())
                for name,parameter in reference.backbone.named_parameters():
                    normalized=name
                    found=next(p for key,p in actual.items() if key.replace('_fsdp_wrapped_module.','').replace('_checkpoint_wrapped_module.','')==normalized)
                    torch.testing.assert_close(found.grad,parameter.grad,rtol=2e-5,atol=2e-6)
        optimizer.step();scheduler.step()
    distributed_step(compare_gradients=True)
    ref_opt.step()
    path=Path(scratch)/f'tiny-rank-{rank}.pt'
    save_local(model,optimizer,scheduler,1,'tiny-check',path)
    distributed_step()
    expected={name:p.detach().clone() for name,p in model.named_parameters()}
    restored=restore_local(model,optimizer,scheduler,'tiny-check',path)
    if restored!=1:raise RuntimeError('Wrong restored step')
    distributed_step()
    for name,parameter in model.named_parameters():
        torch.testing.assert_close(parameter,expected[name],rtol=0,atol=0)
    # Independently check the first Adam update against an unsharded global loss.
    restored=restore_local(model,optimizer,scheduler,'tiny-check',path)
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    with FSDP.summon_full_params(model,writeback=False,recurse=True):
        actual=model.module.backbone.state_dict(); desired=reference.backbone.state_dict()
        for name,value in actual.items():torch.testing.assert_close(value,desired[name],rtol=2e-5,atol=2e-6)
    path.unlink();dist.barrier()
    return {'global_target_normalization':True,'optimizer_resume_exact':True,'world_size':world}
