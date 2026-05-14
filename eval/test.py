# 디버깅용 - evaluate.py와 별도로 실행
import sys, json, random
sys.path.insert(0, '/home/ngseokim/code-killr/core')
sys.path.append('/home/ngseokim/code-killr/parser')
sys.path.append('/home/ngseokim/code-killr/eval')

# evaluate.py에서 필요한 함수들 import
from evaluate import (extract_identifiers, realistic_rename_identifiers,
                      rename_identifiers)

random.seed(42)

with open('/home/ngseokim/code-killr/data/test_v3.jsonl') as f:
    pairs = [json.loads(l) for l in f]

sample = random.sample(pairs, 3)

for pair in sample:
    lang = pair.get('language', 'python')
    code = pair['positive']
    idents = extract_identifiers(code, lang)

    print(f"\n{'='*60}")
    print(f"언어: {lang} | 레벨: {pair.get('level')} | 식별자 {len(idents)}개: {idents[:10]}")
    print(f"\n[원본]\n{code[:300]}")

    for ratio in [0.25, 0.50, 1.00]:
        renamed = realistic_rename_identifiers(code, lang, ratio)
        old_idents = set(extract_identifiers(code, lang))
        new_idents = set(extract_identifiers(renamed, lang))
        changed = old_idents - new_idents
        print(f"\n[{int(ratio*100)}% rename] 실제 변경된 식별자: {changed}")
        print(renamed[:300])

    print(f"\n--- 기존 __rnm__ 방식 100% ---")
    old_way = rename_identifiers(code, lang, 1.0)
    print(old_way[:300])
