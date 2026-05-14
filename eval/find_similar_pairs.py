# /home/ngseokim/code-killr/eval/find_similar_pairs.py
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / '.env')
ROOT = Path(os.getenv('PROJECT_ROOT'))

sys.path.insert(0, str(ROOT / 'core'))
sys.path.append(str(ROOT / 'parser'))

from transformers import AutoModel, AutoTokenizer
from model import GraphCodeBERTEncoder
from dataset import encode_with_dfg, build_attn_mask, TOTAL_LENGTH

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CONFIG = {
    'anchor_path': str(ROOT / 'data/opensearch_data_100k.json'),
    'model_path':  str(ROOT / 'model/GCB_dfg_stage2.pt'),
    'output_path': str(ROOT / 'data/top_similar_pairs.jsonl'),
    'model_name':  'microsoft/graphcodebert-base',
    'batch_size':  128,
    'top_k':       200,    # 최종 저장할 쌍 수
    'top_per_item': 5,     # 항목당 유사 후보 수
    'sim_threshold': 0.85, # 이 이상만 저장
}


def load_model():
    encoder = AutoModel.from_pretrained(CONFIG['model_name'])
    model   = GraphCodeBERTEncoder(encoder)
    ckpt    = torch.load(CONFIG['model_path'], map_location=DEVICE)
    model.load_state_dict(ckpt.get('model', ckpt))
    return model.to(DEVICE).eval()


def compute_embeddings(model, tokenizer, items: list) -> np.ndarray:
    """전체 임베딩 계산 → (N, 768) numpy"""
    all_embs = []
    for i in tqdm(range(0, len(items), CONFIG['batch_size']), desc='임베딩 계산'):
        batch = items[i:i + CONFIG['batch_size']]
        b_ids, b_pos, b_mask = [], [], []

        for code, lang in batch:
            try:
                ids, pos, d2c, d2d = encode_with_dfg(code, lang, tokenizer)
                mask = build_attn_mask(pos, d2c, d2d)
            except Exception:
                ids  = [tokenizer.pad_token_id] * TOTAL_LENGTH
                pos  = [1] * TOTAL_LENGTH
                mask = np.zeros((TOTAL_LENGTH, TOTAL_LENGTH), dtype=bool)
            b_ids.append(ids)
            b_pos.append(pos)
            b_mask.append(mask)

        with torch.no_grad():
            emb = model(
                torch.tensor(b_ids,  dtype=torch.long).to(DEVICE),
                torch.tensor(b_pos,  dtype=torch.long).to(DEVICE),
                torch.tensor(b_mask, dtype=torch.bool).to(DEVICE),
            )
            emb = F.normalize(emb, dim=1)
        all_embs.append(emb.cpu().numpy())

    return np.vstack(all_embs)


def find_top_pairs(embs: np.ndarray) -> list[tuple[int, int, float]]:
    """
    GPU matmul로 유사도 높은 쌍 탐색.
    같은 인덱스(자기 자신) 제외, 중복 쌍 제거.
    """
    N    = embs.shape[0]
    emb_t = torch.tensor(embs, dtype=torch.float32).to(DEVICE)  # (N, 768)

    pairs = {}  # (min_i, max_j) → sim
    batch = 512

    for i in tqdm(range(0, N, batch), desc='유사도 탐색'):
        q = emb_t[i:i + batch]                  # (batch, 768)
        sim = torch.matmul(q, emb_t.T)          # (batch, N)

        # 자기 자신 마스킹
        for bi in range(q.shape[0]):
            sim[bi, i + bi] = -1.0

        # 각 행에서 상위 top_per_item 인덱스 추출
        topk = torch.topk(sim, k=CONFIG['top_per_item'], dim=1)

        for bi in range(q.shape[0]):
            gi = i + bi
            for rank in range(CONFIG['top_per_item']):
                j    = topk.indices[bi, rank].item()
                s    = topk.values[bi, rank].item()
                if s < CONFIG['sim_threshold']:
                    continue
                key  = (min(gi, j), max(gi, j))
                if key not in pairs or pairs[key] < s:
                    pairs[key] = s

    # 유사도 내림차순 정렬 후 top_k 반환
    sorted_pairs = sorted(pairs.items(), key=lambda x: -x[1])
    return [(i, j, s) for (i, j), s in sorted_pairs[:CONFIG['top_k']]]


def main():
    print(f"Device: {DEVICE}")

    print(f"\n데이터 로딩: {CONFIG['anchor_path']}")
    with open(CONFIG['anchor_path']) as f:
        data = json.load(f)
    print(f"총 {len(data)}개")

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    model     = load_model()
    print("모델 로드 완료")

    items = [(d['code'], d.get('language', 'python')) for d in data]
    embs  = compute_embeddings(model, tokenizer, items)
    print(f"임베딩 완료: {embs.shape}")

    print("\n유사도 높은 쌍 탐색 중...")
    top_pairs = find_top_pairs(embs)
    print(f"발견된 쌍: {len(top_pairs)}개")

    # 결과 저장
    results = []
    for i, j, sim in top_pairs:
        results.append({
            'sim':      round(sim, 4),
            'code_a':   data[i]['code'],
            'code_b':   data[j]['code'],
            'lang_a':   data[i].get('language', ''),
            'lang_b':   data[j].get('language', ''),
            'repo_a':   data[i].get('repo', ''),
            'repo_b':   data[j].get('repo', ''),
            'license_a': data[i].get('license', ''),
            'license_b': data[j].get('license', ''),
        })

    with open(CONFIG['output_path'], 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n저장 완료: {CONFIG['output_path']}")
    print(f"상위 5개 유사도: {[r['sim'] for r in results[:5]]}")


if __name__ == '__main__':
    main()
