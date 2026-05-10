import torch
import json
import hashlib
import numpy as np
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from collections import Counter
import sys

sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import encode_with_dfg, build_attn_mask, TOTAL_LENGTH
from model import GraphCodeBERTEncoder  # 🔴 Fix 1: 올바른 모델 클래스 import

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'train_path':    'data/train_v3.jsonl',
    'anchor_path':   'data/opensearch_data_100k.json',
    'model_path':    'GCB_dfg_stage1.pt',
    'output_path':   'hard_negatives_gcb.jsonl',
    'model_name':    'microsoft/graphcodebert-base',
    'batch_size':    128,
    'sim_threshold': 0.6,
    'max_hard_neg':  15000,
}

tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])


def load_model():
    # 🔴 Fix 1: GraphCodeBERTEncoder로 로드 (state dict 키 일치)
    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model


def get_embeddings_batch(model, items: list) -> tuple[np.ndarray, list[int]]:
    """
    배치 단위 임베딩 생성
    Returns:
        embs:        (N, 768) numpy array
        valid_mask:  encode 성공한 인덱스 리스트 (실패한 샘플은 zero vector)
    """
    embs       = []
    valid_mask = []

    for i in range(0, len(items), CONFIG['batch_size']):
        batch      = items[i:i + CONFIG['batch_size']]
        batch_ids, batch_pos, batch_mask = [], [], []

        for code, lang in batch:
            try:
                ids, pos, d2c, d2d = encode_with_dfg(code, lang, tokenizer)
                mask = build_attn_mask(pos, d2c, d2d)
                batch_ids.append(ids)
                batch_pos.append(pos)
                batch_mask.append(mask)
                valid_mask.append(True)
            except Exception:
                batch_ids.append([tokenizer.pad_token_id] * TOTAL_LENGTH)
                batch_pos.append([1] * TOTAL_LENGTH)
                batch_mask.append(np.zeros((TOTAL_LENGTH, TOTAL_LENGTH), dtype=bool))
                valid_mask.append(False)

        input_ids    = torch.tensor(batch_ids,  dtype=torch.long).to(DEVICE)
        position_ids = torch.tensor(batch_pos,  dtype=torch.long).to(DEVICE)
        attn_mask    = torch.tensor(batch_mask, dtype=torch.bool).to(DEVICE)

        with torch.no_grad():
            # 🔴 Fix 2: GraphCodeBERTEncoder.forward() signature에 맞게 전달
            emb = model(input_ids, position_ids, attn_mask)   # (B, 768)
            emb = F.normalize(emb, dim=1)                     # model.py 미수정이므로 여기서 normalize
        embs.append(emb.cpu().numpy())

        done = min(i + CONFIG['batch_size'], len(items))
        print(f"  임베딩 진행: {done}/{len(items)}", end='\r')

    return np.vstack(embs), valid_mask


def mine_hard_negatives():
    print(f"Device: {DEVICE}")
    print("모델 로딩 중...")
    model = load_model()

    # ── pool 로딩 & 임베딩 ────────────────────────────────────────
    print(f"\n앵커 pool 로딩: {CONFIG['anchor_path']}")
    with open(CONFIG['anchor_path']) as f:
        anchor_data = json.load(f)
    print(f"총 {len(anchor_data)}개 로딩")

    print("pool 임베딩 계산 중...")
    pool_items = [(item['code'], item.get('language', 'python')) for item in anchor_data]
    pool_embs, pool_valid = get_embeddings_batch(model, pool_items)
    print(f"\npool 임베딩 완료: {pool_embs.shape}")

    # ── train anchor 임베딩 (🟡 Fix 3: 배치로 한 번에) ──────────
    print(f"\ntrain pair 로딩: {CONFIG['train_path']}")
    with open(CONFIG['train_path']) as f:
        train_pairs = [json.loads(line) for line in f]

    print("train anchor 임베딩 계산 중...")
    anchor_items = [
        (p['anchor'], p.get('language', 'python')) for p in train_pairs
    ]
    anchor_embs, anchor_valid = get_embeddings_batch(model, anchor_items)
    print(f"\nanchor 임베딩 완료: {anchor_embs.shape}")

    # ── Hard Negative Mining ──────────────────────────────────────
    print("\nHard Negative Mining 중...")
    hard_negatives = []
    seen = set()  # 🟡 Fix 4: MD5 해시 기반 중복 방지

    for idx, pair in enumerate(train_pairs):
        if len(hard_negatives) >= CONFIG['max_hard_neg']:
            break
        if not anchor_valid[idx]:
            continue

        anchor_emb  = anchor_embs[idx]          # (768,)
        anchor_repo = pair.get('repo', '')
        anchor_lang = pair.get('language', 'python')
        anchor_lic  = pair.get('license', '')
        anchor_code = pair['anchor']

        # 유사도 계산 (pool_embs: (100k, 768), anchor_emb: (768,))
        sims       = pool_embs @ anchor_emb      # (100k,)
        sorted_idx = np.argsort(-sims)

        for neg_idx in sorted_idx:
            sim = sims[neg_idx]
            if sim < CONFIG['sim_threshold']:
                break                            # 정렬된 상태이므로 이후는 전부 threshold 미만

            if not pool_valid[neg_idx]:
                continue

            neg_item = anchor_data[neg_idx]
            neg_code = neg_item['code']
            neg_repo = neg_item.get('repo', '')

            if neg_repo == anchor_repo:          # 같은 repo 제외
                continue
            if neg_code == anchor_code:          # 동일 코드 제외
                continue

            # 🟡 Fix 4: MD5 해시 기반 dedup
            key = (
                hashlib.md5(anchor_code.encode()).hexdigest(),
                hashlib.md5(neg_code.encode()).hexdigest(),
            )
            if key in seen:
                continue
            seen.add(key)

            hard_negatives.append({
                'anchor':           anchor_code,
                'negative':         neg_code,
                'similarity':       float(sim),
                'anchor_license':   anchor_lic,
                'negative_license': neg_item.get('license', ''),
                'anchor_repo':      anchor_repo,
                'negative_repo':    neg_repo,
                'language':         anchor_lang,
            })
            break  # anchor당 1개

        if (idx + 1) % 100 == 0:
            print(f"  {idx+1}/{len(train_pairs)} 처리 | Hard Negative {len(hard_negatives)}개")

    # ── 저장 & 통계 ───────────────────────────────────────────────
    with open(CONFIG['output_path'], 'w') as f:
        for hn in hard_negatives:
            f.write(json.dumps(hn, ensure_ascii=False) + '\n')

    print(f"\n완료: {len(hard_negatives)}개 저장 → {CONFIG['output_path']}")

    lic_dist = Counter(hn['anchor_license'] for hn in hard_negatives)
    print("라이선스 분포:")
    for lic, cnt in sorted(lic_dist.items(), key=lambda x: -x[1]):
        print(f"  {lic}: {cnt}개")

    avg_sim = np.mean([hn['similarity'] for hn in hard_negatives])
    print(f"평균 유사도: {avg_sim:.4f}")


if __name__ == '__main__':
    torch.cuda.empty_cache()
    mine_hard_negatives()
