import json
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

sys.path.append('/home/ngseokim/code-killr/parser')
from DFG import (DFG_python, DFG_java, DFG_javascript,
                 DFG_go, DFG_ruby, DFG_php, DFG_csharp)
from utils import (tree_to_token_index, index_to_code_token,
                   remove_comments_and_docstrings)
from tree_sitter import Language, Parser

# ── 파서 초기화 ──────────────────────────────────────
SO_PATH = '/home/ngseokim/code-killr/parser/my-languages.so'

PARSERS = {}
DFG_FUNC = {
    'python':     DFG_python,
    'java':       DFG_java,
    'javascript': DFG_javascript,
    'go':         DFG_go,
    'ruby':       DFG_ruby,
    'php':        DFG_php,
    'c_sharp':    DFG_csharp,
}

for lang in DFG_FUNC:
    parser = Parser()
    parser.set_language(Language(SO_PATH, lang))
    PARSERS[lang] = parser

CODE_LENGTH      = 256
DATA_FLOW_LENGTH = 64
TOTAL_LENGTH     = CODE_LENGTH + DATA_FLOW_LENGTH

LICENSE_WEIGHTS = {
    'GPL-2.0': 3.0, 'GPL-3.0': 3.0, 'AGPL-3.0': 3.0,
    'LGPL-2.1': 2.0, 'LGPL-3.0': 2.0,
}


# ── DFG 추출 ─────────────────────────────────────────
def extract_dataflow(code: str, language: str):
    parser  = PARSERS.get(language)
    dfg_fn  = DFG_FUNC.get(language)
    if parser is None or dfg_fn is None:
        return [], []

    try:
        if language == 'php':
            code = '<?php ' + code
        tree = parser.parse(bytes(code, 'utf-8'))
        root = tree.root_node

        tokens_index = tree_to_token_index(root)
        code_lines   = code.split('\n')
        index_to_code = {}
        for idx, (s, e) in enumerate(tokens_index):
            token = index_to_code_token((s, e), code_lines)
            index_to_code[(s, e)] = (idx, token)

        code_tokens = [index_to_code[x][1] for x in tokens_index if x in index_to_code]
        dfg, _      = dfg_fn(root, index_to_code, {})
        dfg         = sorted(dfg, key=lambda x: x[1])
        dfg         = [d for d in dfg if d[1] < len(code_tokens)]
        return code_tokens, dfg
    except Exception:
        return [], []


# ── 인코딩 ──────────────────────────────────────────
def encode_with_dfg(code: str, language: str, tokenizer):
    code_tokens, dfg = extract_dataflow(code, language)

    # 토큰화
    code_tokens = [
        tokenizer.tokenize('@ ' + x[:64])[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]

    # 원본→현재 위치 매핑
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens[i]))
    code_tokens = [t for sub in code_tokens for t in sub]

    # 트런케이션
    max_code = CODE_LENGTH + DATA_FLOW_LENGTH - 3 - min(len(dfg), DATA_FLOW_LENGTH)
    code_tokens = code_tokens[:max_code][:512-3]

    # [CLS] + code + [SEP]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids    = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx  = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    # DFG 노드 추가
    dfg = dfg[:TOTAL_LENGTH - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx  += [0 for _ in dfg]
    source_ids    += [tokenizer.unk_token_id for _ in dfg]

    # 패딩
    pad_len       = TOTAL_LENGTH - len(source_ids)
    position_idx  += [tokenizer.pad_token_id] * pad_len
    source_ids    += [tokenizer.pad_token_id] * pad_len

    # DFG 재인덱싱
    reverse_index = {x[1]: i for i, x in enumerate(dfg)}
    for i, x in enumerate(dfg):
        dfg[i] = x[:-1] + ([reverse_index[j] for j in x[-1] if j in reverse_index],)

    dfg_to_dfg  = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length      = 1  # [CLS] 길이
    dfg_to_code = [(s + length, e + length) for s, e in dfg_to_code]

    return source_ids, position_idx, dfg_to_code, dfg_to_dfg


def build_attn_mask(position_idx, dfg_to_code, dfg_to_dfg):
    attn_mask  = np.zeros((TOTAL_LENGTH, TOTAL_LENGTH), dtype=bool)
    node_index = sum(i > 1 for i in position_idx)
    max_length = sum(i != 1 for i in position_idx)

    # 코드 토큰끼리 attend
    attn_mask[:node_index, :node_index] = True

    # DFG → code attend
    for idx, (a, b) in enumerate(dfg_to_code):
        if idx + node_index < max_length:
            attn_mask[idx + node_index, :max_length] = True
            attn_mask[idx + node_index, a:b]         = True
            attn_mask[a:b, idx + node_index]         = True

    # DFG → DFG attend
    for idx, edges in enumerate(dfg_to_dfg):
        for a in edges:
            if a + node_index < max_length:
                attn_mask[idx + node_index, a + node_index] = True

    return attn_mask


# ── Dataset ──────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer):
        self.pairs     = []
        self.tokenizer = tokenizer
        self.cache     = {}

        with open(jsonl_path) as f:
            for line in f:
                self.pairs.append(json.loads(line))
        print(f"로드 완료: {len(self.pairs)}개 pair")

    def _encode(self, code: str, language: str):
        key = (code[:100], language)
        if key not in self.cache:
            self.cache[key] = encode_with_dfg(code, language, self.tokenizer)
        return self.cache[key]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        try:
            lang = pair.get('language', 'python')

            a_ids, a_pos, a_d2c, a_d2d = self._encode(pair['anchor'],   lang)
            p_ids, p_pos, p_d2c, p_d2d = self._encode(pair['positive'], lang)

            a_mask = build_attn_mask(a_pos, a_d2c, a_d2d)
            p_mask = build_attn_mask(p_pos, p_d2c, p_d2d)

            weight = LICENSE_WEIGHTS.get(pair.get('license', ''), 1.0)

            return {
                'anchor_input_ids':      torch.tensor(a_ids,  dtype=torch.long),
                'anchor_position_ids':   torch.tensor(a_pos,  dtype=torch.long),
                'anchor_attn_mask':      torch.tensor(a_mask, dtype=torch.bool),
                'positive_input_ids':    torch.tensor(p_ids,  dtype=torch.long),
                'positive_position_ids': torch.tensor(p_pos,  dtype=torch.long),
                'positive_attn_mask':    torch.tensor(p_mask, dtype=torch.bool),
                'weight':                torch.tensor(weight, dtype=torch.float),
            }
        except Exception:
            return self.__getitem__((idx + 1) % len(self.pairs))
