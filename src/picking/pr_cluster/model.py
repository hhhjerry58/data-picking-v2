"""FP32 parameter/gradient/Adam state, BF16 compute, full-shard data parallelism."""
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LossModel(torch.nn.Module):
    def __init__(self, backbone, head_chunk_tokens=512, compute_dtype=torch.bfloat16):
        super().__init__()
        self.backbone = backbone
        self.head_chunk_tokens = head_chunk_tokens
        self.compute_dtype = compute_dtype

    def forward(self, input_ids, positions):
        context = torch.autocast('cuda', dtype=self.compute_dtype) if input_ids.is_cuda and self.compute_dtype != torch.float32 else nullcontext()
        with context:
            embedded = self.backbone.model.embed_tokens(input_ids)
            if input_ids.is_cuda: embedded = embedded.to(self.compute_dtype)
            hidden = self.backbone.model(inputs_embeds=embedded, use_cache=False).last_hidden_state
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        def chunk_loss(values, labels):
            ctx = torch.autocast('cuda', dtype=self.compute_dtype) if values.is_cuda and self.compute_dtype != torch.float32 else nullcontext()
            with ctx: logits = self.backbone.lm_head(values)
            return F.cross_entropy(logits.float(), labels, reduction='sum')
        for start in range(0, positions.numel(), self.head_chunk_tokens):
            pos = positions[start:start+self.head_chunk_tokens]
            values = hidden[0].index_select(0, pos-1)
            labels = input_ids[0].index_select(0, pos)
            if self.training and torch.is_grad_enabled():
                value = checkpoint(chunk_loss, values, labels, use_reentrant=False)
            else: value = chunk_loss(values, labels)
            total = total + value
        return total


def load_backbone(assets):
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    from picking.pr_selection.modeling.qwen35 import enable_fla
    assets = Path(assets)
    config = Qwen3_5TextConfig(**json.loads((assets/'config.json').read_text())['text_config'])
    config._attn_implementation = 'sdpa'; config.use_cache = False
    enable_fla()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    with init_empty_weights(): model = Qwen3_5ForCausalLM(config)
    expected = dict(model.named_parameters()); loaded = set()
    index = json.loads((assets/'model.safetensors.index.json').read_text())['weight_map']
    for filename in sorted(set(index.values())):
        with safe_open(assets/filename, framework='pt', device='cpu') as source:
            for key in source.keys():
                if key.startswith('model.language_model.'): name = 'model.'+key[len('model.language_model.'):]
                elif key == 'lm_head.weight': name = key
                else: continue
                value = source.get_tensor(key)
                if name not in expected or name in loaded or value.shape != expected[name].shape:
                    raise ValueError('M0 text parameter contract differs')
                set_module_tensor_to_device(model, name, 'cpu', value=value, dtype=torch.float32)
                loaded.add(name)
    if loaded != set(expected): raise ValueError('Incomplete M0 language-model parameters')
    return model


def wrap(backbone, device, head_chunk_tokens=512):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl)
    layer_type = type(backbone.model.layers[0])
    wrapper = LossModel(backbone, head_chunk_tokens)
    model = FSDP(wrapper, auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={layer_type}),
                 sharding_strategy=ShardingStrategy.FULL_SHARD, device_id=device,
                 mixed_precision=None, use_orig_params=True, sync_module_states=True, limit_all_gathers=True)
    apply_activation_checkpointing(model,
        checkpoint_wrapper_fn=partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
        check_fn=lambda module: isinstance(module, layer_type))
    return model


def example(row, objective, device):
    from picking.pr_selection.modeling.masks import indices
    if row is None:
        return torch.tensor([[1, 2]], device=device), torch.tensor([1], device=device), 0
    key = 'target_runs' if objective == 'guidance' else objective+'_runs'
    positions = indices(row[key])
    if not positions or min(positions) < 1: raise ValueError('Invalid target positions')
    return (torch.tensor([list(row['input_ids'])], device=device, dtype=torch.long),
            torch.tensor(positions, device=device, dtype=torch.long), len(positions))
