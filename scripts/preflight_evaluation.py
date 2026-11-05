#!/usr/bin/env python
"""Exercise scoring masks and HF generation on synthetic prompts only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer
from picking.common import config,paths,read_json,write_json,utc_now
from picking.storage import verified_checkpoint_mirror
from picking.evaluation import evaluate_batch

def main():
    cfg=config();p=paths(cfg)
    trained=p['run']/'pretraining/random-seed-42-preflight-v5-continuous'
    completed=read_json(trained/'TRAINING_COMPLETED.json');checkpoint=Path(completed['checkpoint'])
    meta=read_json(checkpoint/'manifest.json');sha=meta['files']['state.pt']['sha256']
    file=verified_checkpoint_mirror(trained,checkpoint,sha,cfg)
    payload=torch.load(file,map_location='cpu',weights_only=False)
    architecture=AutoConfig.from_pretrained(p['assets']/'pythia160m',local_files_only=True)
    model=AutoModelForCausalLM.from_config(architecture,dtype=torch.float32,attn_implementation='sdpa')
    model.load_state_dict(payload['model']);del payload
    model.to('cuda').eval()
    tokenizer=AutoTokenizer.from_pretrained(p['assets']/'pythia160m',local_files_only=True)
    rows=[]
    prompts=['Question: What is two plus three?\nAnswer: ','Question: Solve x plus seven equals twelve.\nAnswer: ']
    for i,(prompt,answer) in enumerate(zip(prompts,['Five.','The answer is five.'])):
        encoded=tokenizer(prompt+answer,add_special_tokens=False,return_offsets_mapping=True)
        mask=[end>len(prompt) for start,end in encoded['offset_mapping']]
        rows.append({'id':str(i),'window_index':0,'group_id':str(i),'subject':'synthetic','level':'synthetic',
                     'input_ids':encoded['input_ids'],'mask':mask,'scored_tokens':sum(mask)})
    scored=evaluate_batch(model,rows)
    assert all(r['sum_nll']>0 and r['scored_tokens']==rows[i]['scored_tokens'] for i,r in enumerate(scored))
    tokenizer.padding_side='left';tokenizer.pad_token=tokenizer.eos_token
    inputs=tokenizer(prompts,return_tensors='pt',padding=True,add_special_tokens=False).to('cuda')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        generated=model.generate(**inputs,max_new_tokens=4,do_sample=False,use_cache=True,
                                 pad_token_id=0,eos_token_id=0,logits_to_keep=1)
    assert generated.shape[0]==2 and generated.shape[1]>inputs['input_ids'].shape[1]
    write_json(p['run']/'preflight/evaluation_interface_verification.json',{'passed':True,'synthetic_prompts_only':True,
               'finite_answer_nll':True,'answer_mask_counts_verified':True,'padded_greedy_generation_verified':True,'time':utc_now()})
    print('Evaluation interface verified on synthetic prompts; no benchmark performance result.',flush=True)

if __name__=='__main__':main()
