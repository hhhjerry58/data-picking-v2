from __future__ import annotations

import json
import hashlib
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import torch

from .loss import chunked_head_loss, positions


def asset_dir(cfg):
    if cfg['model']['name'] != 'Qwen/Qwen3.5-9B':
        raise ValueError('This backend is pinned to Qwen/Qwen3.5-9B')
    return Path(cfg['artifact_root']) / 'cache/pr-selection' / ('qwen35-9b-'+cfg['model']['revision'][:8])


def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg['model']['name'], revision=cfg['model']['revision'])


def ensure_assets(cfg):
    from huggingface_hub import snapshot_download
    assets = asset_dir(cfg)
    index = assets/'model.safetensors.index.json'
    needed = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja']
    if index.exists():
        needed += list(set(json.loads(index.read_text())['weight_map'].values()))
    if not index.exists() or any(not (assets/name).is_file() for name in needed):
        snapshot_download(cfg['model']['name'], revision=cfg['model']['revision'], local_dir=assets,
                          allow_patterns=['config.json', 'tokenizer*', 'chat_template.jinja', '*.safetensors',
                                          'model.safetensors.index.json', 'special_tokens_map.json'], max_workers=2)
    # Verify the small model configuration against its published Git blob ID.
    raw = (assets/'config.json').read_bytes()
    receipt = (assets/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()
    expected = receipt[1].strip().strip('"')
    actual = hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest() if len(expected)==40 else hashlib.sha256(raw).hexdigest()
    if receipt[0].strip() != cfg['model']['revision'] or actual != expected:
        raise ValueError('Model configuration differs from the fixed HF revision')
    return assets


def weight_mirror(cfg, filename):
    """Verify this specific checkpoint shard once, then mmap local RAM storage.

    Large safetensors mmaps through blobfuse stall on remote page faults. The
    authoritative weights remain on the configured persistent artifact mount.
    """
    from picking.common import file_hash
    from ..config import atomic_json
    source = asset_dir(cfg) / filename
    metadata = asset_dir(cfg) / '.cache/huggingface/download' / (filename+'.metadata')
    lines = metadata.read_text().splitlines()
    if lines[0].strip() != cfg['model']['revision']:
        raise ValueError('Model shard metadata refers to another revision')
    expected = lines[1].strip().strip('"')
    if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected immutable SHA256 for the HF checkpoint shard')
    directory = Path(cfg['runtime_root']) / 'model_mirrors' / cfg['model']['revision']
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename
    receipt = target.with_suffix('.verified.json')
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        stat = target.stat()
        if saved == {'sha256': expected, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}:
            return target
    temporary = target.with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    if file_hash(temporary) != expected:
        raise ValueError('Model shard copy differs from its fixed HF revision')
    os.replace(temporary, target)
    stat = target.stat()
    atomic_json(receipt, {'sha256': expected, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns})
    return target


def enable_fla():
    """Use the installed, pinned training kernel; no runtime remote kernel fetch."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from transformers.models.qwen3_5 import modeling_qwen3_5
    modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule


def load_engine(cfg, gpu_ids):
    from accelerate import dispatch_model, init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
    if not gpu_ids:
        raise ValueError('Explicit GPU allocation is required')
    assets = ensure_assets(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = Qwen3_5TextConfig(**json.loads((assets / 'config.json').read_text())['text_config'])
    config._attn_implementation = 'sdpa'
    config.use_cache = False
    if cfg['model']['linear_attention'] == 'fla':
        enable_fla()
    else:
        raise ValueError('Production full-context training requires the pinned FLA kernel')
    with init_empty_weights():
        model = Qwen3_5ForCausalLM(config)
    expected = dict(model.named_parameters())
    total = sum(p.numel() for p in expected.values())
    device_map = {'model.embed_tokens': gpu_ids[0], 'model.rotary_emb': gpu_ids[0],
                  'model.norm': gpu_ids[-1], 'lm_head': gpu_ids[-1]}
    load = sum(p.numel() for n, p in expected.items() if n.startswith('model.embed_tokens.'))
    slot = 0
    for i, layer in enumerate(model.model.layers):
        if load >= total / len(gpu_ids) and slot < len(gpu_ids)-1:
            slot += 1
            load = 0
        device_map[f'model.layers.{i}'] = gpu_ids[slot]
        load += sum(p.numel() for p in layer.parameters())
    index = json.loads((assets / 'model.safetensors.index.json').read_text())['weight_map']
    loaded = set()
    for filename in sorted(set(index.values())):
        with safe_open(weight_mirror(cfg, filename), framework='pt', device='cpu') as source:
            for key in source.keys():
                if key.startswith('model.language_model.'):
                    name = 'model.' + key[len('model.language_model.'):]
                elif key == 'lm_head.weight':
                    name = key
                else:
                    # The vision encoder and unused auxiliary heads are never
                    # loaded or trained. Every text-model parameter is verified.
                    continue
                if name not in expected or name in loaded:
                    raise ValueError(f'Unexpected/repeated language-model weight: {name}')
                owner = max((k for k in device_map if name.startswith(k + '.')), key=len)
                tensor = source.get_tensor(key)
                if tensor.shape != expected[name].shape:
                    raise ValueError(f'Checkpoint shape mismatch: {name}')
                set_module_tensor_to_device(model, name, device_map[owner], value=tensor, dtype=torch.float32)
                loaded.add(name)
        print('MODEL_SHARD_LOADED', filename, flush=True)
    if loaded != set(expected):
        raise ValueError(f'Missing language-model weights: {sorted(set(expected)-loaded)[:8]}')
    model = dispatch_model(model, device_map=device_map, main_device=gpu_ids[0], force_hooks=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for p in model.parameters():
        p.requires_grad_(True)
    engine = Engine(model, cfg['model']['lm_head_chunk_tokens'], torch.bfloat16,
                    cfg['model']['activation_offload_min_tokens'],
                    score_backend=cfg.get('scoring', {}).get('attention_backend', 'default'))
    engine.device_map = device_map
    return engine


class Engine:
    """One autograd graph, optionally pipelined across several explicitly allowed GPUs."""

    def __init__(self, model, chunk_size=128, compute_dtype=None, offload_min_tokens=32768, score_backend='default'):
        self.model = model
        self.score_backend = score_backend
        self.chunk_size = chunk_size
        self.compute_dtype = compute_dtype
        self.offload_min_tokens = offload_min_tokens
        self.parameters = dict(model.named_parameters())
        if any(not p.requires_grad for p in self.parameters.values()):
            raise ValueError('The entire loaded language model must be trainable')

    def zero_grad(self):
        self.model.zero_grad(set_to_none=True)

    def hidden(self, record, backward):
        self.model.train(backward)
        device = self.model.model.embed_tokens.weight.device
        ids = torch.as_tensor(record['input_ids'], device=device, dtype=torch.long).unsqueeze(0)
        context = torch.autocast('cuda', dtype=self.compute_dtype) if device.type == 'cuda' and self.compute_dtype else nullcontext()
        offload = torch.autograd.graph.save_on_cpu(pin_memory=True) if (
            backward and device.type == 'cuda' and len(record['input_ids']) >= self.offload_min_tokens
        ) else nullcontext()
        with torch.set_grad_enabled(backward), context, offload:
            embedded = self.model.model.embed_tokens(ids)
            if self.compute_dtype and device.type == 'cuda':
                embedded = embedded.to(self.compute_dtype)
            hidden = self.model.model(inputs_embeds=embedded, use_cache=False).last_hidden_state
        return hidden

    def losses(self, record, objectives):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        context = sdpa_kernel(SDPBackend.FLASH_ATTENTION) if self.score_backend == 'flash' and 'guidance' not in objectives else nullcontext()
        with torch.no_grad(), context:
            hidden = self.hidden(record, False)
            return {objective: chunked_head_loss(hidden, self.model.lm_head, record['input_ids'],
                       positions(record, objective), chunk_size=self.chunk_size,
                       autocast_dtype=self.compute_dtype) for objective in objectives}

    def backward(self, record, objective, *, denominator=None, accumulate=False):
        if not accumulate:
            self.zero_grad()
        hidden = self.hidden(record, True)
        return chunked_head_loss(hidden, self.model.lm_head, record['input_ids'], positions(record, objective),
                                 denominator=denominator, backward=True, chunk_size=self.chunk_size,
                                 autocast_dtype=self.compute_dtype)
