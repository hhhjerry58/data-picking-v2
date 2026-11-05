#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import argparse
import fcntl
from picking.common import config, paths, digest


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ['prepare-data','prepare-label-cache','aggregate-scores','prepare-fasttext','scan-candidates','select','build-token-stores','prepare-evaluation','report']:
        commands.add_parser(name)
    score = commands.add_parser('score')
    score.add_argument('--shard', type=int, required=True)
    score.add_argument('--num-shards', type=int, default=3)
    ft = commands.add_parser('train-fasttext')
    ft.add_argument('--metric', choices=['delta_loss','relative_gain'], required=True)
    ft.add_argument('--stop-after-epoch', type=int)
    for name in ['pack','pretrain','evaluate']:
        item = commands.add_parser(name)
        item.add_argument('--arm', choices=['random','delta_loss','relative_gain'], required=True)
        item.add_argument('--seed', type=int, required=True)
        if name=='pretrain':item.add_argument('--stop-after-steps', type=int)
    args = parser.parse_args()
    lock_root = paths(config())['scratch']/'job_locks'
    lock_root.mkdir(parents=True,exist_ok=True)
    lock = (lock_root/(digest(vars(args))+'.lock')).open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)  # Only one writer for a resumed stage/job.
    if args.command=='prepare-data':
        from picking.data import prepare_corpus
        prepare_corpus()
    elif args.command=='prepare-label-cache':
        from picking.scoring import prepare_label_cache
        prepare_label_cache()
    elif args.command=='score':
        from picking.scoring import score_shard
        score_shard(args.shard,args.num_shards)
    elif args.command=='aggregate-scores':
        from picking.scoring import aggregate_scores
        aggregate_scores()
    elif args.command=='prepare-fasttext':
        from picking.fasttext_pipeline import make_training_data
        make_training_data()
    elif args.command=='train-fasttext':
        from picking.fasttext_pipeline import train_classifier
        train_classifier(args.metric,stop_after_epoch=args.stop_after_epoch)
    elif args.command=='scan-candidates':
        from picking.fasttext_pipeline import scan_candidates
        scan_candidates()
    elif args.command=='select':
        from picking.selection import select
        select()
    elif args.command=='build-token-stores':
        from picking.selection import build_token_stores
        build_token_stores()
    elif args.command=='pack':
        from picking.selection import pack
        pack(args.arm,args.seed)
    elif args.command=='pretrain':
        from picking.pretraining import train
        train(args.arm,args.seed,stop_after_steps=args.stop_after_steps)
    elif args.command=='evaluate':
        from picking.evaluation import evaluate
        evaluate(args.arm,args.seed)
    elif args.command=='prepare-evaluation':
        from picking.evaluation import prepare_evaluation
        prepare_evaluation()
    elif args.command=='report':
        from picking.reporting import report
        report()


if __name__=='__main__':main()
