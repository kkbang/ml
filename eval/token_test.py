# 어느 tokenize 호출에서 664가 나오는지 찾기
import sys, json, warnings
sys.path.insert(0, '/home/ngseokim/code-killr/core')
sys.path.append('/home/ngseokim/code-killr/parser')
from transformers import AutoTokenizer
from dataset import extract_dataflow, TOTAL_LENGTH

tokenizer = AutoTokenizer.from_pretrained('microsoft/graphcodebert-base')

with open('/home/ngseokim/code-killr/data/test_v3.jsonl') as f:
    pairs = [json.loads(l) for l in f]

for i, pair in enumerate(pairs):
    lang = pair.get('language', 'python')
    for role in ['anchor', 'positive']:
        code = pair[role]
        code_tokens, dfg = extract_dataflow(code, lang)
        for idx, x in enumerate(code_tokens):
            raw = '@ ' + x if idx != 0 else x
            toks = tokenizer.tokenize(raw)
            if len(toks) > 100:  # 비정상적으로 긴 단일 토큰
                print(f"[{i}] {role} ({lang}) AST토큰→서브워드 {len(toks)}개")
                print(f"  원본 토큰: {repr(x[:80])}")
                break
