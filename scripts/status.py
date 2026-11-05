#!/usr/bin/env python
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from picking.common import config,paths,read_json
from picking.control_state import read_control,process_alive

def optional(path):
    try:return read_json(path)
    except (OSError,json.JSONDecodeError):return None

def main():
    cfg=config();p=paths(cfg)
    result={'run_id':cfg['run_id'],'artifact_directory':str(p['run']),
            'pipeline':read_control(cfg,'pipeline_progress.json'),
            'data':optional(p['scratch']/'corpus_state/data_progress.json') or optional(p['run']/'data_progress.json'),
            'formal_label_scores':optional(p['run']/'label_score_summary.json'),
            'trainer_integrity':optional(p['run']/'preflight/training_resume_verification.json'),
            'latest_report':optional(p['run']/'latest_report.json')}
    result['watchdog']=read_control(cfg,'watchdog.json')
    result['needs_attention']=read_control(cfg,'WATCHDOG_NEEDS_ATTENTION.json')
    if result['pipeline']:
        result['pipeline']['process_alive']=process_alive(result['pipeline'].get('pid'),'run_pipeline.py',result['pipeline'].get('hostname'))
        if not result['pipeline']['process_alive'] and result['pipeline'].get('stage')!='complete':
            result['pipeline']['recorded_stage']=result['pipeline'].get('stage')
            result['pipeline']['stage']='not_running'
    if result['watchdog']:
        result['watchdog']['process_alive']=process_alive(result['watchdog'].get('pid'),'watch_pipeline.py',result['watchdog'].get('hostname'))
    result['pretraining']={}
    active={item['name']:item for item in (result['pipeline'] or {}).get('active',[])}
    pending=set((result['pipeline'] or {}).get('pending',[]))
    for seed in cfg['pretraining']['seeds']:
        for arm in cfg['pretraining']['arms']:
            name=f'{arm}-seed-{seed}';folder=p['run']/'pretraining'/name
            completed=optional(folder/'TRAINING_COMPLETED.json')
            progress=completed or optional(folder/'progress.json')
            row=dict(progress or {})
            job=f'pretrain_{arm}_{seed}'
            if completed:
                row['status']='complete'
            elif job in active and process_alive(active[job]['pid'],'experiment.py'):
                row['status']='running'
            elif job in pending:
                row['status']=(result['pipeline'] or {}).get('state','queued')
            else:
                row['status']='interrupted' if progress else 'not_started'
            if progress and not completed:
                checkpoints=[]
                try:
                    for file in (folder/'checkpoints').glob('step-*/manifest.json'):
                        saved=optional(file)
                        if saved and (file.parent/'COMPLETED.json').exists():checkpoints.append((saved['global_step'],file.parent,saved))
                    if checkpoints:
                        step,path,saved=max(checkpoints,key=lambda item:item[0])
                        row['resume_checkpoint']={'path':str(path),'step':step,'seen_tokens':saved['seen_tokens']}
                except OSError:
                    row['checkpoint_status']='storage_temporarily_unavailable'
            result['pretraining'][name]=row
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
