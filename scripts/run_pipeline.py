#!/usr/bin/env python
"""Resume the approved experiment stages; each child owns its output/logs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import os,subprocess,time,json,fcntl,signal
from picking.common import config,paths,read_json,write_json,file_hash,emit,utc_now,REPO
from picking.control_state import write_control, stop_requested
from picking.workflow import completed_job, receipt, training_complete, evaluation_complete
from picking.runtime import free_gpu_ids


class Runner:
    def __init__(self):
        self.cfg=config();self.p=paths(self.cfg)
        self.logs=self.p['scratch']/'logs';self.logs.mkdir(parents=True,exist_ok=True)
        self.python=str(REPO/'.venv/bin/python')
        self.entry=str(REPO/'scripts/experiment.py')
        self.progress_path=self.p['run']/'pipeline_progress.json'
        self.children=[]

    def progress(self,stage,**extra):
        if stop_requested(self.cfg):raise SystemExit(0)
        write_control(self.cfg,'pipeline_progress.json',{'stage':stage,'pid':os.getpid(),'time':utc_now(),**extra})
        emit('pipeline_stage',stage=stage,**extra)

    def remaining_jobs(self,jobs,stage):
        while True:
            if stop_requested(self.cfg):raise SystemExit(0)
            try:
                pending=[]
                for args,name in jobs:
                    if completed_job(self.cfg,args):emit('stage_job_cached',name=name)
                    else:pending.append((args,name))
                return pending
            except (OSError,json.JSONDecodeError) as error:
                self.progress(stage,state='waiting_for_storage',storage_error=type(error).__name__)
                time.sleep(20)

    def wait_for(self,path,stage):
        self.progress(stage,waiting_for=str(path))
        while True:
            if stop_requested(self.cfg):raise SystemExit(0)
            try:
                if Path(path).exists():
                    read_json(path)
                    return
            except (OSError,json.JSONDecodeError):pass
            time.sleep(20)

    def launch(self,args,name,gpu=None):
        log=self.logs/(name+'.log')
        env=dict(os.environ)
        env['CUDA_VISIBLE_DEVICES']='' if gpu is None else str(gpu)
        env['OMP_NUM_THREADS']='2'
        env['MKL_NUM_THREADS']='2'
        output=log.open('ab',buffering=0)
        process=subprocess.Popen([self.python,self.entry,*args],cwd=REPO,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        self.children.append(process)
        emit('stage_job_started',name=name,pid=process.pid,gpu=gpu,log=str(log))
        return {'process':process,'output':output,'name':name,'log':log,'gpu':gpu}

    def stop_children(self):
        for process in self.children:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for process in self.children:
            if process.poll() is None:
                try:process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
        self.children=[]

    def completed(self,job):
        status=job['process'].poll()
        if status is None:return False
        job['output'].close()
        try:
            import shutil
            shutil.copyfile(job['log'],self.p['run']/job['log'].name)
        except OSError:pass
        if status!=0:raise RuntimeError(f"{job['name']} failed with exit {status}; see {job['log']}")
        emit('stage_job_finished',name=job['name'],gpu=job['gpu'])
        return True

    def cpu_jobs(self,jobs,stage,workers=2):
        self.progress(stage)
        pending=self.remaining_jobs(jobs,stage);active=[]
        while pending or active:
            if stop_requested(self.cfg):raise SystemExit(0)
            while pending and len(active)<workers:
                args,name=pending.pop(0);active.append(self.launch(args,name))
            active=[job for job in active if not self.completed(job)]
            if pending or active:time.sleep(10)

    def free_gpus(self):
        return free_gpu_ids()

    def gpu_jobs(self,jobs,stage):
        self.progress(stage)
        pending=self.remaining_jobs(jobs,stage);active=[]
        while pending or active:
            if stop_requested(self.cfg):raise SystemExit(0)
            active=[job for job in active if not self.completed(job)]
            occupied={job['gpu'] for job in active}
            for gpu in self.free_gpus():
                if not pending:break
                if gpu in occupied:continue
                args,name=pending.pop(0);active.append(self.launch(args,name,gpu))
            write_control(self.cfg,'pipeline_progress.json',{'stage':stage,'pid':os.getpid(),'time':utc_now(),
                       'state':'running' if active else 'waiting_for_gpu',
                       'pending':[name for _,name in pending],
                       'active':[{'name':j['name'],'pid':j['process'].pid,'gpu':j['gpu']} for j in active]})
            if pending or active:time.sleep(20)

    def run(self):
        self.wait_for(self.p['data']/'CORPUS_COMPLETED.json','preparing_corpus')
        self.cpu_jobs([(['prepare-label-cache'],'prepare_label_cache')],'preparing_formal_labels',1)
        self.gpu_jobs([(['score','--shard',str(i),'--num-shards','3'],f'teacher_score_{i}') for i in range(3)],'scoring_100000_documents')
        self.cpu_jobs([(['aggregate-scores'],'aggregate_scores')],'aggregating_paired_scores',1)
        self.cpu_jobs([(['prepare-fasttext'],'prepare_fasttext')],'forming_two_frozen_label_sets',1)
        self.cpu_jobs([(['train-fasttext','--metric',m],f'fasttext_{m}') for m in self.cfg['fasttext']['metrics']],'training_two_fasttext_models',2)
        self.cpu_jobs([(['scan-candidates'],'scan_candidates')],'scoring_candidate_pool_with_fasttext',1)
        self.cpu_jobs([(['select'],'select_data')],'selecting_three_equal_token_budgets',1)
        self.cpu_jobs([(['build-token-stores'],'build_token_stores')],'building_selected_token_stores',1)
        self.cpu_jobs([(['prepare-evaluation'],'prepare_evaluation')],'freezing_shared_evaluation_inputs',1)
        check=self.p['run']/'preflight/training_resume_verification.json'
        self.wait_for(check,'waiting_for_trainer_integrity_verification')
        verified=read_json(check)
        if not verified['passed'] or verified['pretraining_implementation_sha256']!=file_hash(REPO/'src/picking/pretraining.py'):
            raise RuntimeError('Formal trainer differs from the verified full-state implementation')
        for seed in self.cfg['pretraining']['seeds']:
            arms=self.cfg['pretraining']['arms']
            if receipt(self.p['run']/f'SEED_{seed}_COMPLETED.json') is not None and all(training_complete(self.cfg,a,seed) and evaluation_complete(self.cfg,a,seed) for a in arms):
                emit('paired_seed_already_complete',seed=seed)
                continue
            self.cpu_jobs([(['pack','--arm',a,'--seed',str(seed)],f'pack_{a}_{seed}') for a in arms],f'packing_seed_{seed}',3)
            self.gpu_jobs([(['pretrain','--arm',a,'--seed',str(seed)],f'pretrain_{a}_{seed}') for a in arms],f'pretraining_seed_{seed}')
            self.gpu_jobs([(['evaluate','--arm',a,'--seed',str(seed)],f'evaluate_{a}_{seed}') for a in arms],f'evaluating_seed_{seed}')
            self.cpu_jobs([(['report'],f'report_seed_{seed}')],f'reporting_seed_{seed}',1)
            write_json(self.p['run']/f'SEED_{seed}_COMPLETED.json',{'seed':seed,'arms':arms,'time':utc_now()})
        write_json(self.p['run']/'EXPERIMENT_COMPLETED.json',{'seeds':self.cfg['pretraining']['seeds'],'arms':self.cfg['pretraining']['arms'],'time':utc_now()})
        self.progress('complete')


def main():
    runner=Runner()
    lock=(runner.p['scratch']/'pipeline.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def interrupted(signum,frame):raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    try:runner.run()
    except Exception as error:
        runner.progress('failed',error=str(error))
        raise
    finally:runner.stop_children()

if __name__=='__main__':main()
