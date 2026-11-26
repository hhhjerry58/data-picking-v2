"""集群部署身份：固定输入、数值代码和运行环境必须一起匹配。"""
from __future__ import annotations

import json
import argparse
import shutil
import tarfile
from importlib.metadata import version
from pathlib import Path

from picking.common import REPO, digest, file_hash


def read_deployment(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'manifest.json').read_text())
    deployment = json.loads((bundle / 'deployment.json').read_text())
    ready = json.loads((bundle / 'READY.json').read_text())
    if (deployment['bundle_identity'] != digest(manifest)
            or ready['bundle_identity'] != digest(manifest)
            or ready['deployment_identity'] != digest(deployment)):
        raise ValueError('Published bundle/deployment identity differs')
    if not 1 <= deployment['nodes'] <= 10 or deployment['workers_per_node'] != 4:
        raise ValueError('This deployment requires four two-GPU workers per node')
    return manifest, deployment


def verify_execution(deployment, repo=REPO):
    for relative, expected in deployment['source_files'].items():
        if file_hash(Path(repo) / relative) != expected:
            raise ValueError(f'Frozen execution source differs: {relative}')
    for package, expected in deployment['packages'].items():
        if version(package) != expected:
            raise ValueError(f'Frozen runtime package differs: {package}')


def publish(bundle, runtime, staging):
    """仅向尚未发布的导出包补充代码/环境；READY 不表示 A100 数值已通过。"""
    from picking.pr_selection.config import atomic_json, freeze_json
    from .parent import parent_config

    bundle,runtime,staging=Path(bundle),Path(runtime),Path(staging)
    if (bundle/'READY.json').exists():raise ValueError('Use a new deployment; this bundle is already published')
    manifest=json.loads((bundle/'manifest.json').read_text())
    if parent_config(manifest['parent_run'])!=manifest['parent_config']:
        raise ValueError('Exported parent configuration differs')
    for item in [manifest['candidate_file'],manifest['validation_file'],*manifest['m1_files'],*manifest['gq_files']]:
        if (bundle/item['file']).stat().st_size!=item['bytes']:
            raise ValueError(f'Incomplete exported asset: {item["file"]}')
    files=set()
    for folder in ['src/picking/pr_selection','src/picking/pr_parallel']:
        files.update(p.relative_to(REPO) for p in (REPO/folder).rglob('*.py'))
    for name in ['__init__','common','runtime','storage','blob_transport']:
        files.add(Path(f'src/picking/{name}.py'))
    files.update(Path(f'methods/{name}/method.py') for name in manifest['parent_config']['methods'])
    files.update(map(Path,['scripts/pr_cluster_node.sh','scripts/submit_pr_a100.py','scripts/pr_selection.py',
                           'requirements-pr.txt','requirements.txt','configs/pr_selection.yaml']))
    sources={str(p):file_hash(REPO/p) for p in sorted(files)}
    staging.mkdir(parents=True,exist_ok=True)
    archive=staging/'code.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for relative in sorted(files):tar.add(REPO/relative,arcname=str(relative),recursive=False)
    shutil.copyfile(archive,bundle/'code.tar.gz')
    runtime_hash=file_hash(runtime)
    target=bundle/'runtime.tar.gz'
    if not target.exists():shutil.copyfile(runtime,target)
    if file_hash(target)!=runtime_hash:raise ValueError('Published runtime archive differs')
    code_hash=file_hash(archive)
    if file_hash(bundle/'code.tar.gz')!=code_hash:raise ValueError('Published code archive differs')
    packages={name:version(name) for name in ['torch','transformers','tokenizers','huggingface-hub',
               'flash-linear-attention','tilelang','triton','causal-conv1d','accelerate','safetensors','numpy','pandas','pyarrow']}
    deployment={'schema_version':1,'nodes':10,'workers_per_node':4,'gpus_per_worker':2,'expected_gpu':'A100',
                'bundle_identity':digest(manifest),'source_files':sources,'packages':packages,
                'archives':{'runtime.tar.gz':runtime_hash,'code.tar.gz':code_hash},
                'reduction':'gpu-fp64-v1','partition':'round_robin_row_index',
                'loss':'token_mean_nll','parameter_dtype':'float32','compute_dtype':'bfloat16',
                'validation':{'nll_absolute_scaled':5e-4,'dot_over_norm_product':1e-3,'gradient_norm_relative':.02},
                'reuse_m1_and_gq':True,'recompute_all_candidates':True,'test_losses_computed':False}
    freeze_json(bundle/'deployment.json',deployment)
    (bundle/'archives.sha256').write_text(''.join(f'{checksum}  {name}\n' for name,checksum in deployment['archives'].items()))
    verify_execution(deployment)
    atomic_json(bundle/'READY.json',{'bundle_identity':digest(manifest),'deployment_identity':digest(deployment),
                                   'status':'ready_for_a100_validation','a100_validated':False})
    print('DEPLOYMENT_PUBLISHED',bundle,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',required=True);parser.add_argument('--runtime',required=True)
    parser.add_argument('--staging',default='/tmp/pr-a100-code-package')
    args=parser.parse_args();publish(args.bundle,args.runtime,args.staging)
