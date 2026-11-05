#!/usr/bin/env python
from pathlib import Path
import sys,argparse,json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from picking.migration import check_resume

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Check node/environment/data compatibility without starting training')
    parser.add_argument('--no-gpu',action='store_true',help='Validate storage/configuration only, even when no GPU is free')
    parser.add_argument('--full-checkpoint',action='store_true',help='Hash and load the pending checkpoint on CPU to verify optimizer/RNG/cursor state')
    args=parser.parse_args()
    print(json.dumps(check_resume(require_gpu=not args.no_gpu,full_checkpoint=args.full_checkpoint),ensure_ascii=False,indent=2))
