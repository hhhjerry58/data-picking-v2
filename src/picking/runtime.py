"""Node-local GPU selection; independent of frozen numerical training settings."""
import os
import subprocess


def allowed_gpu_ids(detected):
    raw=os.environ.get('DATA_PICKING_GPU_IDS')
    if raw is None:
        raw=os.environ.get('CUDA_VISIBLE_DEVICES')
    if raw is None:
        return set(detected)
    if not raw.strip():
        return set()
    try:
        chosen={int(item.strip()) for item in raw.split(',')}
    except ValueError:
        raise ValueError('Set DATA_PICKING_GPU_IDS to comma-separated physical GPU indices') from None
    if not chosen <= set(detected):
        raise ValueError('DATA_PICKING_GPU_IDS includes a GPU not present on this node')
    return chosen


def gpu_inventory():
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                                    '--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=[]
    for line in output.splitlines():
        index,name,total,used,util=[item.strip() for item in line.split(',')]
        rows.append({'index':int(index),'name':name,'memory_total_mib':int(total),
                     'memory_used_mib':int(used),'utilization':int(util)})
    return rows


def free_gpu_ids():
    try:
        rows=gpu_inventory()
    except (OSError,subprocess.SubprocessError):
        return []
    allowed=allowed_gpu_ids(row['index'] for row in rows)
    return [row['index'] for row in rows if row['index'] in allowed and row['memory_used_mib']<768 and row['utilization']<5]
