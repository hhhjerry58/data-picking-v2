"""Smoke validates the numerical/runtime recipe independently of pool membership."""
from picking.common import digest


def recipe_identity(manifest):
    cfg=manifest['protocol']
    return digest({'source_files':manifest['source_files'],'runtime':manifest['archives']['runtime.tar.gz'],
                   'packages':manifest['packages'],'model':manifest['model'],'world_size':cfg['world_size'],
                   'training':cfg['training'],'checkpoint':cfg['checkpoint'],'evaluation':cfg['evaluation'],
                   'model_files':sorted((a['local_path'],a['sha256']) for a in manifest['assets'] if a['local_path'].startswith('m0/')),
                   'test_file':next(a['sha256'] for a in manifest['assets'] if a['local_path']=='data/test.parquet')})
