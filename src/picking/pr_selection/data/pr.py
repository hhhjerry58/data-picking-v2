from __future__ import annotations

import re

from ..modeling.masks import count_tokens, token_runs


def answer_start(text):
    required = list(re.finditer(r'^## Required answer[ \t]*\r?\n', text, re.M))
    answer = list(re.finditer(r'^## Answer[ \t]*\r?\n', text, re.M))
    if len(required) != 1 or len(answer) != 1 or required[0].start() >= answer[0].start():
        raise ValueError('ambiguous_answer_boundary')
    if not text[answer[0].end():].strip():
        raise ValueError('empty_answer')
    return answer[0].end()


def encode_pr(tokenizer, row, max_length, min_answer_tokens):
    text = row['text']
    start = answer_start(text)
    byte_start = len(text[:start].encode('utf-8'))
    ids, answer_runs = token_runs(tokenizer, text, [[byte_start, len(text.encode('utf-8'))]])
    if len(ids) > max_length:
        raise ValueError('over_candidate_length')
    if count_tokens(answer_runs) < min_answer_tokens:
        raise ValueError('short_answer')
    return {'id': row['id'], 'query_id': row['query_id'], 'format': row['format'],
            'source_category': row['metadata']['source']['domain'],
            'input_ids': ids, 'input_tokens': len(ids),
            'full_text_runs': [[1, len(ids)]], 'answer_only_runs': answer_runs,
            'full_text_tokens': len(ids)-1, 'answer_only_tokens': count_tokens(answer_runs)}
