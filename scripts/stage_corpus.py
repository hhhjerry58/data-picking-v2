#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from concurrent.futures import ThreadPoolExecutor
import time
from picking.common import config, paths, emit
from picking.corpus_io import prepared_shard

def main():
    cfg=config();p=paths(cfg);done=set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        while True:
            indexes=[int(path.name.split('-')[1].split('.')[0]) for path in (p['data']/'source_shards').glob('part-*.parquet.complete.json')]
            pending=sorted(set(indexes)-done)
            for start in range(0,len(pending),16):
                batch=pending[start:start+16]
                list(pool.map(lambda i:prepared_shard(cfg,i),batch))
                done.update(batch)
                emit('corpus_staged',shards=len(done))
            if (p['data']/'CORPUS_COMPLETED.json').exists():
                break
            time.sleep(20)

if __name__=='__main__':main()
