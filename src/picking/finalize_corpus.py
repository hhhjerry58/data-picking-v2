"""Typed corpus finalization and recovery from verified duplicate decisions."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from .common import config,paths,read_json,write_json,file_hash,rank_key,complete_file,emit,utc_now
from .corpus_io import prepared_shard

META_COLUMNS=['id','source_index','source_row','domain','crawl','text_hash','sample_key','pythia_tokens']
META_SCHEMA=pa.schema([('id',pa.string()),('source_index',pa.int32()),('source_row',pa.int32()),
                      ('domain',pa.string()),('crawl',pa.string()),('text_hash',pa.string()),
                      ('sample_key',pa.uint64()),('pythia_tokens',pa.int64())])


def freeze_table(cfg,table,results,counts):
    p=paths(cfg);spec=cfg['source']
    table=table.select(META_COLUMNS).cast(META_SCHEMA)
    keys=table['sample_key'].combine_chunks().to_numpy()
    tokens=table['pythia_tokens'].combine_chunks().to_numpy()
    order=np.argsort(keys,kind='stable')
    labels,general=spec['label_documents'],spec['general_eval_documents']
    remaining=order[labels+general:]
    cumulative=np.cumsum(tokens[remaining],dtype=np.int64)
    if not len(cumulative) or cumulative[-1]<spec['candidate_tokens']:return False
    ncandidate=int(np.searchsorted(cumulative,spec['candidate_tokens'])+1)
    names=['unused','label_train','label_dev','label_test','general_eval','candidate']
    codes=np.zeros(len(table),dtype=np.uint8)
    label_rows=table.take(pa.array(order[:labels])).select(['id','domain']).to_pylist()
    domain_sets={1:set(),2:set(),3:set()}
    for index,row in zip(order[:labels],label_rows):
        value=int(rank_key(row['domain'],spec['data_seed'])[:16],16)/2**64
        code=1 if value<.8 else (2 if value<.9 else 3)
        codes[index]=code;domain_sets[code].add(row['domain'])
    if domain_sets[1]&domain_sets[2] or domain_sets[1]&domain_sets[3] or domain_sets[2]&domain_sets[3]:
        raise RuntimeError('Classifier domain splits overlap')
    codes[order[labels:labels+general]]=4
    codes[remaining[:ncandidate]]=5
    table=table.append_column('partition',pc.take(pa.array(names),pa.array(codes)))
    table=table.append_column('group_id',table['text_hash'])
    target=p['data']/'assignments.parquet'
    pq.write_table(table,target,compression='zstd',row_group_size=100000)
    complete_file(target)
    document_counts={name:int((codes==code).sum()) for code,name in enumerate(names)}
    token_counts={name:int(tokens[codes==code].sum()) for code,name in enumerate(names)}
    if sum(document_counts[n] for n in names[1:4])!=labels:raise RuntimeError('Label set cardinality changed')
    summary={'document_counts':document_counts,'pythia_tokens_including_eos':token_counts,
             'source_shards':results,'deduplication':dict(counts),'source_config':spec,
             'dedup_rule':'Upstream duplicate IDs, normalized SHA256, upstream 9x13 MinHash candidates, verified punctuation-aware 5-token Jaccard >= 0.8. Seeded source order retains first representative; candidate search is approximate.',
             'assignment_sha256':file_hash(target),'created_at_utc':utc_now()}
    write_json(p['data']/'corpus_summary.json',summary)
    write_json(p['data']/'CORPUS_COMPLETED.json',{'assignment_sha256':summary['assignment_sha256'],
               'source_files':{r['path']:read_json(r['path']+'.complete.json')['sha256'] for r in results},'time':utc_now()})
    emit('corpus_frozen',documents=document_counts,tokens=token_counts)
    return True


def freeze_assignments(cfg,dedup,results):
    table=pa.Table.from_pylist(dedup.meta,schema=META_SCHEMA)
    return freeze_table(cfg,table,results,dedup.counts)


def recover_finalization(cfg=None):
    cfg=cfg or config();p=paths(cfg)
    progress=read_json(p['scratch']/'corpus_state/data_progress.json')
    edges=p['scratch']/'corpus_state/dedup_edges.jsonl'
    if file_hash(edges)!=file_hash(p['data']/'dedup_edges.jsonl'):
        raise RuntimeError('Duplicate decision log is not durably complete')
    dropped=set();edge_counts=Counter()
    with edges.open() as stream:
        for line in stream:
            record=json.loads(line)
            if record['id'] in dropped:raise RuntimeError('Repeated duplicate decision')
            dropped.add(record['id']);edge_counts[record['kind']]+=1
    if edge_counts['exact']!=progress['dedup_counts']['exact_duplicate'] or edge_counts['near']!=progress['dedup_counts']['near_duplicate']:
        raise RuntimeError('Duplicate decision coverage differs from completed coordinator')
    tables=[];results=[];raw_valid=0
    drop_values=pa.array(sorted(dropped))
    for index in range(progress['completed_source_shards']):
        source=prepared_shard(cfg,index)
        table=pq.read_table(source,columns=META_COLUMNS)
        raw_valid+=len(table)
        table=table.filter(pc.invert(pc.is_in(table['id'],value_set=drop_values)))
        tables.append(table)
        results.append(read_json(p['data']/'source_shards'/f'part-{index:05d}.stats.json'))
        if (index+1)%50==0:emit('recover_kept_metadata',shards=index+1)
    kept=pa.concat_tables(tables)
    if len(kept)!=progress['kept_documents'] or raw_valid-len(kept)!=len(dropped):
        raise RuntimeError('Reconstructed kept document coverage differs')
    if pc.sum(kept['pythia_tokens']).as_py()!=progress['kept_tokens']:
        raise RuntimeError('Reconstructed kept token budget differs')
    write_json(p['data']/'finalization_recovery.json',{'reason':'Explicit uint64 schema for document sampling hashes',
               'source_processor_sha256':progress['processor_sha256'],'kept_documents':len(kept),'kept_tokens':progress['kept_tokens'],
               'dedup_log_sha256':file_hash(edges),'filters_and_sampling_unchanged':True,'time':utc_now()})
    if not freeze_table(cfg,kept,results,progress['dedup_counts']):raise RuntimeError('Insufficient frozen pool')
