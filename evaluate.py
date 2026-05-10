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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'test_path':  'data/test_v3.jsonl',
    'model_path': 'GCB_dfg_stage2.pt',
    'batch_size': 16,
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
        'precision':   correct_pos / total_pos if total_pos > 0 else 0,
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
    return list({t for t in tokens if t not in kw and len(t) > 1
                 and t not in ('__STR__', '__CMT__')})


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
        correct = total = skipped = 0

        for pair in sample:
            language  = pair.get('language', 'python')
            anchor    = pair['anchor']
            positive  = pair['positive']

            positive_attacked = (
                rename_identifiers(positive, language, ratio)
                if ratio > 0 else positive
            )

            # chunked_embedding으로 긴 함수도 처리
            emb_a = chunked_embedding(model, anchor,           language, tokenizer, DEVICE)
            emb_p = chunked_embedding(model, positive_attacked, language, tokenizer, DEVICE)

            if emb_a is None or emb_p is None:
                skipped += 1
                continue

            sim    = torch.dot(emb_a, emb_p).item()
            total += 1
            if sim > 0.5:
                correct += 1

        precision      = correct / total if total > 0 else 0
        results[ratio] = {'precision': precision, 'total': total, 'skipped': skipped}
        print(f"  변수명 {int(ratio*100):3d}% 변경 | "
              f"Precision={precision:.4f} | "
              f"유효={total}개 | 스킵={skipped}개")

    return results


# ── 메인 평가 ──────────────────────────────────────────────────────────
def evaluate_all(model_path: str, test_path: str) -> None:
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
    print(f"  Precision (>0.5): {r['precision']:.4f}")
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
        print(f"    Precision: {r['precision']:.4f}")
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
        print(f"    Precision: {r['precision']:.4f}")
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
        print(f"    Precision: {r['precision']:.4f}")
        print(f"    FPR:       {r['fpr']:.4f}")

    # 5. 강건성 평가
    print("\n=== 강건성 평가 (변수명 변경 공격) ===")
    evaluate_robustness(model, all_pairs)

    print("\n평가 완료.")


if __name__ == '__main__':
    evaluate_all(CONFIG['model_path'], CONFIG['test_path'])
