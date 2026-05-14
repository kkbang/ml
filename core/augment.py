# /home/ngseokim/code-killr/core/augment.py
import re
import random

try:
    from nltk.corpus import wordnet as _wn
    _HAS_WORDNET = True
except ImportError:
    _HAS_WORDNET = False

_ABBREV = {
    'i': ['idx', 'index', 'pos'], 'j': ['k', 'col', 'm'],
    'k': ['j', 'cnt', 'iter'], 'n': ['count', 'num', 'size'],
    'idx': ['i', 'index', 'pos', 'offset'], 'pos': ['idx', 'index', 'offset', 'loc'],
    'cnt': ['count', 'num', 'n', 'total'], 'val': ['value', 'result', 'v', 'out'],
    'res': ['result', 'output', 'ret', 'value'], 'ret': ['result', 'output', 'res', 'retval'],
    'tmp': ['temp', 'scratch', 'intermediate'], 'buf': ['buffer', 'storage', 'cache'],
    'img': ['image', 'picture', 'frame', 'photo'], 'pic': ['image', 'img', 'photo', 'frame'],
    'msg': ['message', 'text', 'content', 'info'], 'err': ['error', 'ex', 'exc', 'fault'],
    'ex': ['error', 'err', 'exc', 'e'], 'obj': ['node', 'item', 'entity', 'ref'],
    'lst': ['arr', 'items', 'collection', 'seq'], 'arr': ['lst', 'items', 'collection', 'seq'],
    'src': ['source', 'origin', 'inp'], 'dst': ['dest', 'target', 'sink'],
    'cfg': ['config', 'settings', 'conf', 'opts'], 'ctx': ['context', 'env', 'state', 'scope'],
    'env': ['context', 'ctx', 'state', 'settings'], 'req': ['request', 'query', 'call'],
    'resp': ['response', 'reply', 'result'], 'acc': ['accumulator', 'total', 'agg'],
    'cur': ['current', 'curr', 'present', 'active'], 'prev': ['previous', 'last', 'prior', 'old'],
    'num': ['count', 'n', 'total', 'amount'], 'len': ['length', 'size', 'count', 'n'],
    'fn': ['func', 'callback', 'handler', 'action'], 'cb': ['callback', 'handler', 'fn', 'hook'],
    'db': ['database', 'store', 'repo', 'storage'], 'doc': ['document', 'record', 'entry', 'item'],
    'uid': ['id', 'identifier', 'key', 'handle'], 'uri': ['url', 'endpoint', 'link', 'path'],
    'ref': ['pointer', 'handle', 'link', 'obj'], 'opt': ['option', 'choice', 'setting', 'flag'],
    'attr': ['attribute', 'field', 'prop', 'key'], 'prop': ['property', 'attr', 'field', 'key'],
    'col': ['column', 'field', 'key', 'j'], 'row': ['record', 'entry', 'line', 'item'],
    'out': ['output', 'result', 'ret'], 'dir': ['directory', 'folder', 'path'],
    'info': ['details', 'meta', 'desc'], 'first': ['initial', 'head', 'leading'],
    'last': ['final', 'tail', 'end'], 'forward': ['advance', 'propagate', 'next_step'],
    'next': ['following', 'subsequent', 'successor'], 'file': ['document', 'resource', 'asset'],
    'name': ['label', 'title', 'tag'], 'cache': ['store', 'pool', 'registry'],
    'stream': ['flow', 'channel', 'pipe'], 'listener': ['handler', 'observer', 'subscriber'],
    'manager': ['controller', 'handler', 'coordinator'], 'state': ['status', 'phase', 'condition'],
    'event': ['action', 'trigger', 'signal'], 'item': ['entry', 'element', 'record'],
    'list': ['collection', 'seq', 'arr'], 'key': ['field', 'attr', 'token'],
    'flag': ['marker', 'indicator', 'switch'], 'max': ['upper', 'ceiling', 'limit'],
    'min': ['lower', 'floor', 'bound'], 'step': ['phase', 'stage', 'iteration'],
    'index': ['pos', 'offset', 'rank'], 'size': ['count', 'num', 'capacity'],
    'score': ['rating', 'rank', 'measure'], 'weight': ['importance', 'factor', 'coeff'],
    'get': ['fetch', 'retrieve', 'load'], 'set': ['update', 'assign', 'store'],
    'run': ['execute', 'invoke', 'perform'], 'call': ['invoke', 'execute', 'trigger'],
    'do': ['perform', 'execute', 'run'], 'on': ['handle', 'process'],
    'is': ['has', 'contains'], 'check': ['verify', 'validate', 'test'],
    'make': ['create', 'build', 'construct'], 'build': ['construct', 'create', 'assemble'],
    'write': ['store', 'save', 'output'], 'read': ['load', 'fetch', 'retrieve'],
    'close': ['shutdown', 'terminate', 'finalize'], 'open': ['init', 'start', 'launch'],
    'find': ['search', 'locate', 'lookup'], 'send': ['transmit', 'dispatch', 'emit'],
    'parse': ['decode', 'process', 'analyze'], 'handle': ['process', 'manage', 'deal'],
    'dim': ['axis', 'd', 'ndim'], 'shape': ['dims', 'tensor_size', 'size'],
    'grad': ['gradient', 'grads'], 'emb': ['embedding', 'vec'],
    'attn': ['attention', 'weight'], 'logit': ['score', 'raw_score'],
    'prob': ['likelihood', 'score'], 'loss': ['cost', 'penalty'],
    'pred': ['output', 'forecast'], 'epoch': ['step', 'iteration'],
    'batch': ['chunk', 'group'], 'concat': ['merge', 'combine', 'join'],
}

_KEYWORDS = {
    'python': {'def','return','if','else','elif','for','while','in','not','and','or',
               'True','False','None','import','from','class','self','try','except',
               'with','as','pass','break','continue','lambda','yield','raise','del',
               'global','nonlocal'},
    'java':   {'public','private','protected','static','void','int','String','boolean',
               'return','if','else','for','while','new','this','class','true','false',
               'null','final','try','catch','throw'},
    'javascript': {'function','return','if','else','for','while','var','let','const',
                   'true','false','null','undefined','new','this','class','import',
                   'export','from','async','await'},
    'go':     {'func','return','if','else','for','range','var','type','struct',
               'interface','true','false','nil','new','make','package','import',
               'defer','go','chan','map'},
}


def extract_identifiers(code: str, language: str) -> list[str]:
    kw = _KEYWORDS.get(language, set())
    masked = re.sub(r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"[^\"]*\"|\'[^\']*\')', '__STR__', code)
    masked = re.sub(r'(#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/)', '__CMT__', masked)
    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', masked)
    return list({
        t for t in tokens
        if t not in kw
        and len(t) > 1
        and t not in ('__STR__', '__CMT__')
        and not t[0].isupper()
        and not all(c.isupper() or c == '_' for c in t)
    })


def _split_identifier(ident: str) -> tuple[list[str], str]:
    if '_' in ident:
        return [p for p in ident.split('_') if p], 'snake'
    parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', ident)
    parts = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', parts).split('_')
    parts = [p for p in parts if p]
    style = 'pascal' if ident[0].isupper() else 'camel'
    return parts, style


def _join_identifier(parts: list[str], style: str) -> str:
    if not parts:
        return ''
    if style == 'snake':
        return '_'.join(p.lower() for p in parts)
    clean = [re.sub(r'[_\s]+', '', p) for p in parts]
    clean = [p for p in clean if p]
    if not clean:
        return ''
    if style == 'pascal':
        return ''.join(p.capitalize() for p in clean)
    return clean[0].lower() + ''.join(p.capitalize() for p in clean[1:])


def _wordnet_synonyms(word: str) -> list[str]:
    if not _HAS_WORDNET:
        return []
    synsets = _wn.synsets(word.lower(), pos=[_wn.NOUN, _wn.VERB])
    candidates = set()
    for syn in synsets[:2]:
        for lemma in syn.lemmas()[:5]:
            name = lemma.name().lower()
            if (name != word.lower()
                    and '_' not in name
                    and name.isalpha()
                    and name.isidentifier()
                    and 2 <= len(name) <= len(word) + 3):
                candidates.add(name)
    return list(candidates)


def _rename_word(word: str) -> str:
    lower = word.lower()
    if lower in _ABBREV:
        return random.choice(_ABBREV[lower])
    syns = _wordnet_synonyms(lower)
    if syns:
        chosen = random.choice(syns)
        return chosen.capitalize() if word[0].isupper() else chosen
    patterns = [('get_','fetch_'),('set_','update_'),('is_','has_'),
                ('do_','perform_'),('on_','handle_'),('check_','verify_'),
                ('make_','create_'),('build_','construct_'),('parse_','decode_')]
    for s, d in patterns:
        if lower.startswith(s):
            return d + word[len(s):]
    if len(word) <= 3:
        return word + random.choice(['_n', '_v', '_r'])
    return random.choice(['new_', 'cur_', 'my_']) + word


def realistic_rename_identifiers(code: str, language: str, ratio: float) -> str:
    identifiers = extract_identifiers(code, language)
    if not identifiers:
        return code
    n_rename  = max(1, int(len(identifiers) * ratio))
    to_rename = random.sample(identifiers, min(n_rename, len(identifiers)))

    placeholders: dict[str, str] = {}
    counter = [0]

    def mask_literal(m):
        key = f'__LIT_{counter[0]}__'
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    masked = re.sub(
        r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"[^\"]*\"|\'[^\']*\'|#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/)',
        mask_literal, code
    )

    existing = set(identifiers)
    rename_map: dict[str, str] = {}
    for i, ident in enumerate(to_rename):
        parts, style = _split_identifier(ident)
        if len(parts) == 1:
            new_name = _join_identifier([_rename_word(parts[0])], style)
        else:
            idx_to_change = 0 if random.random() < 0.5 else -1
            new_parts = parts[:]
            new_parts[idx_to_change] = _rename_word(parts[idx_to_change])
            new_name = _join_identifier(new_parts, style)
        if new_name in existing:
            new_name += '_r' if style == 'snake' else 'R'
        rename_map[f'__TAG_{i}__'] = new_name
        masked = re.sub(r'\b' + re.escape(ident) + r'\b', f'__TAG_{i}__', masked)

    for tag, new_name in rename_map.items():
        masked = masked.replace(tag, new_name)
    for key, val in placeholders.items():
        masked = masked.replace(key, val)
    return masked
