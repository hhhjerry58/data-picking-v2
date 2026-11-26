from __future__ import annotations

import copy
import json
import re


def indices(runs):
    return [i for start, stop in runs for i in range(int(start), int(stop))]


def count_tokens(runs):
    return sum(int(stop) - int(start) for start, stop in runs)


def token_runs(tokenizer, text, byte_spans):
    """Wenhui containment rule: exclude boundary-crossing and zero-width tokens."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False, return_attention_mask=False)
    boundaries = [0]
    for char in text:
        boundaries.append(boundaries[-1] + len(char.encode('utf-8')))
    spans = sorted(byte_spans)
    if not spans or any(a < 0 or a >= b or b > boundaries[-1] or
                        (j and a < spans[j - 1][1]) for j, (a, b) in enumerate(spans)):
        raise ValueError('Invalid semantic byte spans')
    runs, j = [], 0
    for i, (start, stop) in enumerate(encoded['offset_mapping']):
        a, b = boundaries[start], boundaries[stop]
        while j < len(spans) and a >= spans[j][1]:
            j += 1
        if i and j < len(spans) and spans[j][0] <= a < b <= spans[j][1]:
            if runs and runs[-1][1] == i:
                runs[-1][1] = i + 1
            else:
                runs.append([i, i + 1])
    if not runs:
        raise ValueError('No causally scorable target tokens')
    return encoded['input_ids'], runs


def prepare_messages(messages):
    result = copy.deepcopy(messages)
    for message in result:
        if 'reasoning_content' in message:
            raise ValueError('Phase one requires without_reasoning trajectories')
        for call in message.get('tool_calls', []):
            args = call['function']['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            if not isinstance(args, dict):
                raise ValueError('Function arguments must be JSON objects')
            call['function']['arguments'] = args
    return result


def render_trajectory(tokenizer, record):
    """Native Qwen3.5 representation; score visible assistant semantic fields.

    Marker removal MUST reproduce the unmarked native template exactly. Counts
    are recomputed from native spans; frozen Agens byte/token counts are unused.
    """
    if record.get('variant') != 'without_reasoning':
        raise ValueError('Only without_reasoning is supported')
    history = prepare_messages(record['messages'])
    targets = prepare_messages(record['target_messages'])
    if len(targets) != 1 or targets[0]['role'] != 'assistant':
        raise ValueError('Guidance records must contain one reference assistant action')
    selected = {(int(f['message_index']), tuple(f['path'])) for f in record['score_fields']}
    render = lambda ms: tokenizer.apply_chat_template(
        ms, tools=record['tools'], tokenize=False, add_generation_prompt=False, enable_thinking=False)
    baseline = render(history + targets)
    marker = '\ue000PRSCORE'
    if marker in baseline:
        raise ValueError('Scoring marker collision')
    fields = []

    def mark(value):
        if not value:
            return value
        index = len(fields)
        fields.append(value)
        return f'{marker}S{index}\ue001{value}{marker}E{index}\ue001'

    marked = copy.deepcopy(targets)
    accounted = set()
    for mi, message in enumerate(marked):
        if (mi, ('content',)) in selected:
            content = (message.get('content') or '').strip()
            if not content:
                raise ValueError('Selected assistant content is empty after native trimming')
            message['content'] = mark(content)
            accounted.add((mi, ('content',)))
        for ci, call in enumerate(message.get('tool_calls', [])):
            prefix = ('tool_calls', ci, 'function')
            names = {(mi, prefix + ('name',)), (mi, prefix + ('arguments',))}
            if not names & selected:
                continue
            if not names <= selected:
                raise ValueError('A scored call must select name and arguments together')
            accounted |= names
            function = call['function']
            function['name'] = mark(function['name'])
            marked_args = {}
            for key, value in function['arguments'].items():
                # This is exactly the Qwen3.5 Jinja value rendering. The exact
                # round trip below rejects unexpected escaping/serialization.
                native = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
                marked_args[mark(key)] = mark(native)
            function['arguments'] = marked_args
    if accounted != selected:
        raise ValueError('Unknown or unaccounted score_fields')
    tagged = render(history + marked)
    pattern = re.compile(re.escape(marker) + r'([SE])(\d+)\ue001')
    pieces, opened, spans = [], {}, []
    last = offset = 0
    for match in pattern.finditer(tagged):
        part = tagged[last:match.start()]
        pieces.append(part)
        offset += len(part.encode('utf-8'))
        kind, index = match.group(1), int(match.group(2))
        if kind == 'S':
            if index in opened:
                raise ValueError('Duplicate marker')
            opened[index] = offset
        else:
            if index not in opened:
                raise ValueError('Missing start marker')
            spans.append((opened.pop(index), offset, index))
        last = match.end()
    pieces.append(tagged[last:])
    if ''.join(pieces) != baseline or opened or len(spans) != len(fields):
        raise ValueError('Native template changed/omitted a semantic field')
    raw = baseline.encode('utf-8')
    for start, stop, index in spans:
        if raw[start:stop].decode('utf-8') != fields[index]:
            raise ValueError('Semantic value differs from native rendering')
    byte_spans = [[a, b] for a, b, _ in sorted(spans)]
    ids, runs = token_runs(tokenizer, baseline, byte_spans)
    return {'id': record['id'], 'input_ids': ids, 'target_runs': runs,
            'prediction_tokens': count_tokens(runs), 'input_tokens': len(ids),
            'semantic_bytes': sum(b-a for a, b in byte_spans)}
