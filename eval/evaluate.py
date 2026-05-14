"""
GraphCodeBERT 성능 평가
수정 사항:
  - GraphCodeBERTEncoder + ckpt.get('model', ckpt) 로드
  - evaluate() DFG 입력 복원 + 벡터화
  - PairDataset(path, tokenizer) 인자 추가
  - get_embedding() → chunked_embedding() 으로 교체 (512 토큰 초과 대응)
  - rename_identifiers() 문자열/주석 보호 + 충돌 방지 prefix
  - model_path GCB_dfg_stage2.pt 로 수정
"""

import os
import re
import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader

sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import PairDataset, encode_with_dfg, build_attn_mask, TOTAL_LENGTH
from model import GraphCodeBERTEncoder
from chunker import chunked_embedding
from augment import (realistic_rename_identifiers, extract_identifiers,
                     _ABBREV, _split_identifier, _join_identifier,
                     _wordnet_synonyms, _rename_word)


try:
    from nltk.corpus import wordnet as _wn
    _HAS_WORDNET = True
except ImportError:
    _HAS_WORDNET = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'test_path':  '/home/ngseokim/code-killr/data/test_v3.jsonl',
    'model_path': '/home/ngseokim/code-killr/model/GCB_dfg_stage3.pt',
    'batch_size': 64,
}

tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')


# ── 모델 로드 ──────────────────────────────────────────────────────────
def load_model(model_path: str) -> GraphCodeBERTEncoder:
    encoder = AutoModel.from_pretrained('microsoft/graphcodebert-base')
    model   = GraphCodeBERTEncoder(encoder)
    ckpt    = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(ckpt.get('model', ckpt))
    model = model.to(DEVICE)
    model.eval()
    return model


# ── 기본 평가 ──────────────────────────────────────────────────────────
def evaluate(model, test_loader) -> dict:
    model.eval()
    all_pos_sims, all_neg_sims = [], []
    fp_count = total_neg = total_pos = correct_pos = 0

    with torch.no_grad():
        for batch in test_loader:
            a_emb = F.normalize(model(
                batch['anchor_input_ids'].to(DEVICE),
                batch['anchor_position_ids'].to(DEVICE),
                batch['anchor_attn_mask'].to(DEVICE),
            ), dim=1)
            p_emb = F.normalize(model(
                batch['positive_input_ids'].to(DEVICE),
                batch['positive_position_ids'].to(DEVICE),
                batch['positive_attn_mask'].to(DEVICE),
            ), dim=1)

            B          = a_emb.shape[0]
            sim_matrix = torch.matmul(a_emb, p_emb.T)

            pos_sims = sim_matrix.diag()
            all_pos_sims.extend(pos_sims.cpu().tolist())
            total_pos   += B
            correct_pos += (pos_sims > 0.5).sum().item()

            mask     = ~torch.eye(B, dtype=torch.bool, device=DEVICE)
            neg_sims = sim_matrix[mask]
            all_neg_sims.extend(neg_sims.cpu().tolist())
            total_neg += neg_sims.numel()
            fp_count  += (neg_sims > 0.5).sum().item()

    return {
        'recall':   correct_pos / total_pos if total_pos > 0 else 0,
        'fpr':         fp_count / total_neg    if total_neg > 0 else 0,
        'avg_pos_sim': float(np.mean(all_pos_sims)),
        'avg_neg_sim': float(np.mean(all_neg_sims)),
        'pos_sim_std': float(np.std(all_pos_sims)),
        'neg_sim_std': float(np.std(all_neg_sims)),
        'total_pos':   total_pos,
    }


def save_tmp(pairs: list, name: str) -> str:
    path = f'_tmp_{name}.jsonl'
    with open(path, 'w') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    return path


def load_and_eval(model, pairs: list, name: str) -> dict:
    if not pairs:
        return {}
    path   = save_tmp(pairs, name)
    loader = DataLoader(
        PairDataset(path, tokenizer),
        batch_size=CONFIG['batch_size'], shuffle=False,
    )
    result = evaluate(model, loader)
    os.remove(path)
    return result


# ── 강건성 평가 ────────────────────────────────────────────────────────
def extract_identifiers(code: str, language: str) -> list[str]:
    """문자열 리터럴/주석 제거 후 식별자 추출"""
    keywords = {
        'python':     {'def', 'return', 'if', 'else', 'elif', 'for', 'while', 'in',
                       'not', 'and', 'or', 'True', 'False', 'None', 'import', 'from',
                       'class', 'self', 'try', 'except', 'with', 'as', 'pass', 'break',
                       'continue', 'lambda', 'yield', 'raise', 'del', 'global', 'nonlocal'},
        'java':       {'public', 'private', 'protected', 'static', 'void', 'int', 'String',
                       'boolean', 'return', 'if', 'else', 'for', 'while', 'new', 'this',
                       'class', 'true', 'false', 'null', 'final', 'try', 'catch', 'throw'},
        'javascript': {'function', 'return', 'if', 'else', 'for', 'while', 'var',
                       'let', 'const', 'true', 'false', 'null', 'undefined', 'new',
                       'this', 'class', 'import', 'export', 'from', 'async', 'await'},
        'go':         {'func', 'return', 'if', 'else', 'for', 'range', 'var', 'type',
                       'struct', 'interface', 'true', 'false', 'nil', 'new', 'make',
                       'package', 'import', 'defer', 'go', 'chan', 'map'},
    }
    kw = keywords.get(language, set())

    # 문자열 리터럴과 주석을 placeholder로 마스킹 후 식별자 추출
    masked = re.sub(r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"[^\"]*\"|\'[^\']*\')', '__STR__', code)
    masked = re.sub(r'(#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/)', '__CMT__', masked)

    tokens = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', masked)
    return list({
        t for t in tokens
        if t not in kw
        and len(t) > 1
        and t not in ('__STR__', '__CMT__')
        and not t[0].isupper()                          # PascalCase 클래스명 제외
        and not all(c.isupper() or c == '_' for c in t) # ALL_CAPS 상수 제외
    })

def rename_identifiers(code: str, language: str, ratio: float) -> str:
    """
    식별자를 ratio 비율만큼 치환.
    - 문자열/주석 내부는 치환하지 않음
    - 충돌 방지를 위해 __rnm_{i}__ prefix 사용
    """
    identifiers = extract_identifiers(code, language)
    if not identifiers:
        return code

    n_rename  = max(1, int(len(identifiers) * ratio))
    to_rename = random.sample(identifiers, min(n_rename, len(identifiers)))

    # 1단계: 문자열/주석을 placeholder로 치환 (보호)
    placeholders = {}
    counter      = [0]

    def mask_literal(m):
        key = f'__LITERAL_{counter[0]}__'
        placeholders[key] = m.group(0)
        counter[0] += 1
        return key

    masked = re.sub(
        r'(\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\'|\"[^\"]*\"|\'[^\']*\'|#[^\n]*|//[^\n]*|/\*[\s\S]*?\*/)',
        mask_literal, code
    )

    # 2단계: 식별자 치환 (충돌 방지 prefix)
    for i, ident in enumerate(to_rename):
        masked = re.sub(r'\b' + re.escape(ident) + r'\b', f'__rnm_{i}__', masked)

    # 3단계: placeholder 복원
    for key, val in placeholders.items():
        masked = masked.replace(key, val)

    return masked


# ── 현실적 rename: WordNet + 프로그래밍 약어 사전 ──────────────────────

# WordNet이 모르는 프로그래밍 약어 전용 사전
_ABBREV = {
    'i':    ['idx', 'index', 'pos'],
    'j':    ['k', 'col', 'm'],
    'k':    ['j', 'cnt', 'iter'],
    'n':    ['count', 'num', 'size'],
    'idx':  ['i', 'index', 'pos', 'offset'],
    'pos':  ['idx', 'index', 'offset', 'loc'],
    'cnt':  ['count', 'num', 'n', 'total'],
    'val':  ['value', 'result', 'v', 'out'],
    'res':  ['result', 'output', 'ret', 'value'],
    'ret':  ['result', 'output', 'res', 'retval'],
    'tmp':  ['temp', 'scratch', 'intermediate'],
    'buf':  ['buffer', 'storage', 'cache'],
    'img':  ['image', 'picture', 'frame', 'photo'],
    'pic':  ['image', 'img', 'photo', 'frame'],
    'str':  ['text', 'content', 's', 'raw'],
    'msg':  ['message', 'text', 'content', 'info'],
    'err':  ['error', 'ex', 'exc', 'fault'],
    'ex':   ['error', 'err', 'exc', 'e'],
    'obj':  ['node', 'item', 'entity', 'ref'],
    'lst':  ['arr', 'items', 'collection', 'seq'],
    'arr':  ['lst', 'items', 'collection', 'seq'],
    'src':  ['source', 'origin', 'inp', 'from_val'],
    'dst':  ['dest', 'target', 'sink', 'to_val'],
    'cfg':  ['config', 'settings', 'conf', 'opts'],
    'ctx':  ['context', 'env', 'state', 'scope'],
    'env':  ['context', 'ctx', 'state', 'settings'],
    'req':  ['request', 'query', 'call'],
    'resp': ['response', 'reply', 'result'],
    'acc':  ['accumulator', 'total', 'agg'],
    'cur':  ['current', 'curr', 'present', 'active'],
    'prev': ['previous', 'last', 'prior', 'old'],
    'num':  ['count', 'n', 'total', 'amount'],
    'len':  ['length', 'size', 'count', 'n'],
    'fn':   ['func', 'callback', 'handler', 'action'],
    'cb':   ['callback', 'handler', 'fn', 'hook'],
    'db':   ['database', 'store', 'repo', 'storage'],
    'doc':  ['document', 'record', 'entry', 'item'],
    'uid':  ['id', 'identifier', 'key', 'handle'],
    'uri':  ['url', 'endpoint', 'link', 'path'],
    'auth': ['credentials', 'token', 'identity'],
    'ref':  ['pointer', 'handle', 'link', 'obj'],
    'opt':  ['option', 'choice', 'setting', 'flag'],
    'attr': ['attribute', 'field', 'prop', 'key'],
    'prop': ['property', 'attr', 'field', 'key'],
    'col':  ['column', 'field', 'key', 'j'],
    'row':  ['record', 'entry', 'line', 'item'],
        # ML/tensor 전용 (WordNet이 틀리게 해석하는 것들)
    'dim':     ['axis', 'd', 'ndim'],
    'shape':   ['dims', 'size', 'tensor_size'],
    'grad':    ['gradient', 'grads'],
    'emb':     ['embedding', 'embed', 'vec'],
    'attn':    ['attention', 'attn_weight'],
    'hidden':  ['latent', 'feat', 'repr'],
    'feat':    ['feature', 'hidden', 'repr'],
    'logit':   ['score', 'raw_score', 'pred'],
    'prob':    ['likelihood', 'score', 'weight'],
    'loss':    ['cost', 'penalty', 'objective'],
    'pred':    ['output', 'forecast', 'logit'],
    'epoch':   ['step', 'iteration', 'round'],
    'batch':   ['chunk', 'group', 'subset'],
    # 코드에서 WordNet이 틀리게 해석하는 단어들
    'write':   ['store', 'save', 'output'],
    'read':    ['load', 'fetch', 'retrieve'],
    'close':   ['shutdown', 'terminate', 'finalize'],
    'open':    ['init', 'start', 'launch'],
    'get':     ['fetch', 'retrieve', 'load'],
    'set':     ['update', 'assign', 'store'],
    'run':     ['execute', 'invoke', 'perform'],
    'call':    ['invoke', 'execute', 'trigger'],
    'build':   ['construct', 'create', 'assemble'],
    'make':    ['create', 'build', 'construct'],
    'find':    ['search', 'locate', 'lookup'],
    'check':   ['verify', 'validate', 'test'],
    'parse':   ['decode', 'process', 'analyze'],
    'handle':  ['process', 'manage', 'deal'],
    'send':    ['transmit', 'dispatch', 'emit'],
    'receive': ['accept', 'get', 'fetch'],
    # 복합 식별자 부분 단어 (fallback에 떨어지는 것들)
    'out':      ['output', 'result', 'ret'],
    'dir':      ['directory', 'folder', 'path'],
    'info':     ['details', 'meta', 'desc'],
    'first':    ['initial', 'head', 'leading'],
    'last':     ['final', 'tail', 'end'],
    'forward':  ['advance', 'next_step', 'propagate'],
    'next':     ['following', 'subsequent', 'successor'],
    'prev':     ['prior', 'preceding', 'earlier'],
    'file':     ['document', 'resource', 'asset'],
    'name':     ['label', 'title', 'tag'],
    'cache':    ['store', 'pool', 'registry'],
    'crash':    ['fault', 'failure', 'incident'],
    'stream':   ['flow', 'channel', 'pipe'],
    'listener': ['handler', 'observer', 'watcher'],
    'manager':  ['controller', 'handler', 'coordinator'],
    'helper':   ['util', 'assistant', 'support'],
    'base':     ['root', 'parent', 'foundation'],
    'type':     ['kind', 'category', 'variant'],
    'mode':     ['style', 'method', 'approach'],
    'state':    ['status', 'phase', 'condition'],
    'event':    ['action', 'trigger', 'signal'],
    'item':     ['entry', 'element', 'record'],
    'list':     ['collection', 'seq', 'arr'],
    'map':      ['table', 'mapping', 'registry'],
    'key':      ['field', 'attr', 'token'],
    'flag':     ['marker', 'indicator', 'switch'],
    'max':      ['upper', 'ceiling', 'limit'],
    'min':      ['lower', 'floor', 'bound'],
    'step':     ['phase', 'stage', 'iteration'],
    'index':    ['pos', 'offset', 'rank'],
    'offset':   ['pos', 'start', 'shift'],
    'size':     ['count', 'num', 'capacity'],
    'score':    ['rating', 'rank', 'measure'],
    'weight':   ['importance', 'factor', 'coeff'],
    'concat':   ['merge', 'combine', 'join'],
    'listener': ['handler', 'observer', 'subscriber'],
    'watcher':  ['observer', 'monitor', 'tracker'],
}


def _split_identifier(ident: str) -> tuple[list[str], str]:
    """식별자를 단어 리스트와 케이스 스타일로 분리"""
    if '_' in ident:
        return [p for p in ident.split('_') if p], 'snake'
    parts = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', ident)
    parts = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', parts).split('_')
    parts = [p for p in parts if p]
    style = 'pascal' if ident[0].isupper() else 'camel'
    return parts, style


def _join_identifier(parts: list[str], style: str) -> str:
    """단어 리스트를 케이스 스타일로 조합"""
    if not parts:
        return ''
    # camel/pascal 조합 시 각 part 내부의 underscore 제거
    if style == 'snake':
        return '_'.join(p.lower() for p in parts)
    clean = [re.sub(r'[_\s]+', '', p) for p in parts]
    clean = [p for p in clean if p]
    if not clean:
        return ''
    if style == 'pascal':
        return ''.join(p.capitalize() for p in clean)
    # camel
    return clean[0].lower() + ''.join(p.capitalize() for p in clean[1:])


def _wordnet_synonyms(word: str) -> list[str]:
    """WordNet에서 코드에 쓸 만한 동의어 추출"""
    if not _HAS_WORDNET:
        return []
    word_lower = word.lower()
    synsets = _wn.synsets(word_lower, pos=[_wn.NOUN, _wn.VERB])
    candidates = set()
    for syn in synsets[:2]:              # 상위 2개 의미만 (너무 희귀한 의미 제외)
        for lemma in syn.lemmas()[:5]:   # synset당 상위 5개 lemma만
            name = lemma.name().replace('-', '_').lower()
            if (name != word_lower
                    and '_' not in name        # linguistic_context 같은 합성어 제외
                    and name.isalpha()         # 숫자/특수문자 없는 것만
                    and name.isidentifier()
                    and 2 <= len(name) <= len(word) + 3):
                candidates.add(name)
    return list(candidates)


def _rename_word(word: str) -> str:
    """단어 하나를 현실적으로 rename (약어사전 → WordNet → fallback)"""
    lower = word.lower()

    # 1. 약어 사전 우선
    if lower in _ABBREV:
        return random.choice(_ABBREV[lower])

    # 2. WordNet 동의어
    syns = _wordnet_synonyms(lower)
    if syns:
        chosen = random.choice(syns)
        # 원본이 snake_case 단어면 그대로, 아니면 스타일 유지
        if word[0].isupper():
            return chosen.capitalize()
        return chosen

    # 3. prefix/suffix 패턴 변환
    patterns = [
        ('get_', 'fetch_'), ('set_', 'update_'), ('is_', 'has_'),
        ('do_', 'perform_'), ('on_', 'handle_'), ('check_', 'verify_'),
        ('make_', 'create_'), ('build_', 'construct_'), ('parse_', 'decode_'),
    ]
    for src_p, dst_p in patterns:
        if lower.startswith(src_p):
            return dst_p + word[len(src_p):]

    camel_patterns = [
        ('get', 'fetch'), ('set', 'update'), ('is', 'has'),
        ('do', 'perform'), ('on', 'handle'), ('check', 'verify'),
        ('make', 'create'), ('build', 'construct'),
    ]
    for src_p, dst_p in camel_patterns:
        if lower.startswith(src_p) and len(word) > len(src_p):
            rest = word[len(src_p):]
            return dst_p + (rest[0].upper() + rest[1:] if word[0].islower() else rest)

    # 4. fallback: 접두사 추가 (짧으면 접미사)
    if len(word) <= 3:
        return word + random.choice(['_n', '_v', '_r'])
    return random.choice(['new_', 'cur_', 'my_']) + word


def realistic_rename_identifiers(code: str, language: str, ratio: float) -> str:
    """
    식별자를 ratio 비율만큼 현실적인 이름으로 치환.
    - 약어 사전 + WordNet 동의어 사용
    - 문자열/주석 내부 보호
    - camelCase/snake_case 스타일 유지
    """
    identifiers = extract_identifiers(code, language)
    if not identifiers:
        return code

    n_rename  = max(1, int(len(identifiers) * ratio))
    to_rename = random.sample(identifiers, min(n_rename, len(identifiers)))

    # 문자열/주석 마스킹
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

    # 식별자별 새 이름 생성
    rename_map: dict[str, str] = {}
    existing = set(identifiers)
    for i, ident in enumerate(to_rename):
        parts, style = _split_identifier(ident)

        if len(parts) == 1:
            new_word = _rename_word(parts[0])
            new_name = _join_identifier([new_word], style)
        else:
            # 복합 식별자: 첫 번째 또는 마지막 단어만 변경 (너무 많이 바꾸면 어색)
            idx_to_change = 0 if random.random() < 0.5 else -1
            new_parts = parts[:]
            new_parts[idx_to_change] = _rename_word(parts[idx_to_change])
            new_name = _join_identifier(new_parts, style)

        # 기존 식별자와 충돌하면 suffix 추가
        if new_name in existing:
            new_name += '_r' if style == 'snake' else 'R'

        rename_map[f'__TAG_{i}__'] = new_name
        masked = re.sub(r'\b' + re.escape(ident) + r'\b', f'__TAG_{i}__', masked)

    for tag, new_name in rename_map.items():
        masked = masked.replace(tag, new_name)

    for key, val in placeholders.items():
        masked = masked.replace(key, val)

    return masked


def evaluate_robustness(model, test_pairs: list) -> dict:
    """
    변수명 변경 비율별 탐지율 측정.
    - chunked_embedding으로 긴 함수 처리
    - 문자열/주석 보호된 rename 적용
    - 실패 샘플 통계 출력
    """
    ratios = [0.0, 0.25, 0.50, 0.75, 1.00]
    results = {}
    sample  = random.sample(test_pairs, min(500, len(test_pairs)))

    print(f"  (샘플 {len(sample)}개 기준)")
    for ratio in ratios:
        attacked_pairs = []
        for pair in sample:
            language = pair.get('language', 'python')
            positive_attacked = (
                rename_identifiers(pair['positive'], language, ratio)
                if ratio > 0 else pair['positive']
            )
            attacked_pairs.append({
                **pair,
                'positive': positive_attacked,
            })

        # 메인 평가와 동일한 경로
        path = save_tmp(attacked_pairs, f'robust_{ratio}')
        loader = DataLoader(
            PairDataset(path, tokenizer),
            batch_size=CONFIG['batch_size'], shuffle=False,
        )
        r = evaluate(model, loader)
        os.remove(path)
        
        results[ratio] = {'recall': r['recall'], 'total': r['total_pos']}
        print(f"  변수명 {int(ratio*100):3d}% 변경 | Recall={r['recall']:.4f}")

    return results


def evaluate_robustness_realistic(model, test_pairs: list) -> dict:
    """WordNet + 약어사전 기반 현실적 rename으로 강건성 평가"""
    ratios = [0.0, 0.25, 0.50, 0.75, 1.00]
    results = {}
    sample  = random.sample(test_pairs, min(500, len(test_pairs)))

    print(f"  (샘플 {len(sample)}개 기준, WordNet+약어사전 기반)")
    for ratio in ratios:
        attacked_pairs = []
        for pair in sample:
            language          = pair.get('language', 'python')
            positive_attacked = (
                realistic_rename_identifiers(pair['positive'], language, ratio)
                if ratio > 0 else pair['positive']
            )
            attacked_pairs.append({**pair, 'positive': positive_attacked})

        path   = save_tmp(attacked_pairs, f'realistic_{ratio}')
        loader = DataLoader(
            PairDataset(path, tokenizer),
            batch_size=CONFIG['batch_size'], shuffle=False,
        )
        r = evaluate(model, loader)
        os.remove(path)

        results[ratio] = r
        print(f"  변수명 {int(ratio*100):3d}% 변경 | "
              f"Recall={r['recall']:.4f} | {r['total_pos']}개")

    return results

def encode_single(model, code: str, language: str, use_dfg: bool = True):
    """단일 코드 스니펫을 임베딩으로 변환. use_dfg=False면 DFG 없이 순차 position만 사용."""
    try:
        if use_dfg:
            source_ids, position_idx, dfg_to_code, dfg_to_dfg = encode_with_dfg(
                code, language, tokenizer
            )
            attn_mask = build_attn_mask(position_idx, dfg_to_code, dfg_to_dfg)
        else:
            # DFG 없이 일반 BERT 방식으로 인코딩
            tokens = tokenizer.tokenize(code)[:TOTAL_LENGTH - 2]
            tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
            source_ids   = tokenizer.convert_tokens_to_ids(tokens)
            position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(tokens))]

            pad_len       = TOTAL_LENGTH - len(source_ids)
            source_ids   += [tokenizer.pad_token_id] * pad_len
            position_idx += [tokenizer.pad_token_id] * pad_len

            seq_len   = len(tokens)
            attn_mask = np.zeros((TOTAL_LENGTH, TOTAL_LENGTH), dtype=bool)
            attn_mask[:seq_len, :seq_len] = True

        with torch.no_grad():
            emb = model(
                torch.tensor([source_ids],   dtype=torch.long).to(DEVICE),
                torch.tensor([position_idx], dtype=torch.long).to(DEVICE),
                torch.tensor([attn_mask],    dtype=torch.bool).to(DEVICE),
            )
        return F.normalize(emb, dim=1).squeeze(0)

    except Exception:
        return None


def evaluate_robustness_ablation(model, test_pairs: list) -> None:
    """DFG 있을 때 vs 없을 때 강건성 비교"""
    sample = random.sample(test_pairs, min(200, len(test_pairs)))
    
    for use_dfg in [True, False]:
        label = "DFG O" if use_dfg else "DFG X"
        print(f"\n  [{label}] 강건성")
        
        for ratio in [0.0, 0.50, 1.00]:
            correct = total = 0
            for pair in sample:
                language = pair.get('language', 'python')
                positive_attacked = (
                    rename_identifiers(pair['positive'], language, ratio)
                    if ratio > 0 else pair['positive']
                )
                
                # DFG 없는 버전: position_ids를 순차 위치로만
                # (encode_with_dfg에 dfg_flag 파라미터 추가 필요)
                emb_a = encode_single(model, pair['anchor'], language,
                                       use_dfg)
                emb_p = encode_single(model, positive_attacked, language,
                                       use_dfg)
                
                if emb_a is None or emb_p is None:
                    continue
                
                total += 1
                if torch.dot(emb_a, emb_p).item() > 0.5:
                    correct += 1
            
            recall = correct / total if total > 0 else 0
            print(f"    변수명 {int(ratio*100):3d}% 변경 | Recall={recall:.4f}")


# ── 메인 평가 ──────────────────────────────────────────────────────────
def evaluate_all(model_path: str, test_path: str) -> None:

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


    print(f"Device: {DEVICE}")
    print("모델 로딩 중...")
    model = load_model(model_path)

    with open(test_path) as f:
        all_pairs = [json.loads(line) for line in f]
    print(f"테스트 데이터: {len(all_pairs)}개")

    # 1. 전체 평가
    print("\n=== 전체 평가 ===")
    test_loader = DataLoader(
        PairDataset(test_path, tokenizer),
        batch_size=CONFIG['batch_size'], shuffle=False,
    )
    r = evaluate(model, test_loader)
    print(f"  Positive 유사도: {r['avg_pos_sim']:.4f} (±{r['pos_sim_std']:.4f})")
    print(f"  Negative 유사도: {r['avg_neg_sim']:.4f} (±{r['neg_sim_std']:.4f})")
    print(f"  Recall (>0.5): {r['recall']:.4f}")
    print(f"  FPR       (>0.5): {r['fpr']:.4f}")

    # 2. 레벨별 평가
    print("\n=== 레벨별 평가 ===")
    for level in ['surface', 'structural', 'semantic']:
        pairs = [p for p in all_pairs if p.get('level') == level]
        if not pairs:
            continue
        r = load_and_eval(model, pairs, f'level_{level}')
        print(f"\n  [{level}] ({len(pairs)}개)")
        print(f"    Positive 유사도: {r['avg_pos_sim']:.4f}")
        print(f"    Negative 유사도: {r['avg_neg_sim']:.4f}")
        print(f"    Recall: {r['recall']:.4f}")
        print(f"    FPR:       {r['fpr']:.4f}")

    # 3. 언어별 평가
    print("\n=== 언어별 평가 ===")
    for lang in ['python', 'java', 'javascript', 'go', 'ruby', 'php']:
        pairs = [p for p in all_pairs if p.get('language') == lang]
        if not pairs:
            continue
        r = load_and_eval(model, pairs, f'lang_{lang}')
        print(f"\n  [{lang}] ({len(pairs)}개)")
        print(f"    Positive 유사도: {r['avg_pos_sim']:.4f}")
        print(f"    Negative 유사도: {r['avg_neg_sim']:.4f}")
        print(f"    Recall: {r['recall']:.4f}")
        print(f"    FPR:       {r['fpr']:.4f}")

    # 4. 라이선스별 평가
    print("\n=== 라이선스별 평가 ===")
    for lic in ['GPL-2.0', 'GPL-3.0', 'AGPL-3.0', 'LGPL-2.1', 'MIT', 'Apache-2.0']:
        pairs = [p for p in all_pairs if p.get('license') == lic]
        if not pairs:
            continue
        r = load_and_eval(model, pairs, f'lic_{lic.replace("/", "_")}')
        print(f"\n  [{lic}] ({len(pairs)}개)")
        print(f"    Positive 유사도: {r['avg_pos_sim']:.4f}")
        print(f"    Recall: {r['recall']:.4f}")
        print(f"    FPR:       {r['fpr']:.4f}")

    # 5. 강건성 평가
    print("\n=== 강건성 평가 (변수명 변경 공격) ===")
    evaluate_robustness(model, all_pairs)

    print("\n=== 강건성 DFG ===")
    evaluate_robustness_ablation(model, all_pairs)

    print("\n=== 강건성 평가 (현실적 rename: WordNet + 약어사전) ===")
    evaluate_robustness_realistic(model, all_pairs)



    print("\n평가 완료.")


if __name__ == '__main__':
    evaluate_all(CONFIG['model_path'], CONFIG['test_path'])
