"""
chunker.py — SEA (Split-Encode-Aggregate)
512 토큰 초과 함수를 청크 단위로 분할 → 각각 임베딩 → mean pooling

사용:
    from chunker import chunked_embedding
    emb = chunked_embedding(model, code, language, tokenizer, device)
    # emb: (768,) normalized tensor or None
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import PreTrainedTokenizer

import sys
sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import encode_with_dfg, build_attn_mask, TOTAL_LENGTH, CODE_LENGTH

# ── 설정 ──────────────────────────────────────────────
CHUNK_TOKEN_LIMIT = 400    # 청크당 코드 토큰 수 (512 미만으로 여유 있게)
OVERLAP_LINES     = 3      # 청크 간 겹치는 줄 수 (문맥 연속성 유지)
MIN_CHUNK_LINES   = 5      # 이보다 짧은 청크는 버림


def _split_by_lines(code: str, tokenizer: PreTrainedTokenizer) -> list[str]:
    """
    코드를 줄 단위로 분할하여 CHUNK_TOKEN_LIMIT 이하 청크 생성.
    청크 간 OVERLAP_LINES 줄 겹침으로 문맥 연속성 유지.
    """
    lines = code.splitlines(keepends=True)
    if not lines:
        return [code]

    chunks       = []
    current_lines = []
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.tokenize(line))

        if current_tokens + line_tokens > CHUNK_TOKEN_LIMIT and current_lines:
            chunk = ''.join(current_lines)
            if len(current_lines) >= MIN_CHUNK_LINES:
                chunks.append(chunk)
            # overlap: 마지막 OVERLAP_LINES 줄을 다음 청크 시작으로
            current_lines  = current_lines[-OVERLAP_LINES:]
            current_tokens = sum(len(tokenizer.tokenize(l)) for l in current_lines)

        current_lines.append(line)
        current_tokens += line_tokens

    # 마지막 청크
    if current_lines and len(current_lines) >= MIN_CHUNK_LINES:
        chunks.append(''.join(current_lines))

    return chunks if chunks else [code]


def chunked_embedding(
    model,
    code: str,
    language: str,
    tokenizer: PreTrainedTokenizer,
    device: str = 'cuda',
) -> torch.Tensor | None:
    """
    SEA 방식 임베딩.
    - 512 토큰 이하: 그냥 encode_with_dfg 한 번
    - 512 토큰 초과: 청크 분할 → 각각 임베딩 → mean pooling

    Returns:
        (768,) normalized float32 tensor, or None (전체 실패 시)
    """
    # 토큰 수 빠르게 체크
    token_count = len(tokenizer.tokenize(code))

    if token_count <= CHUNK_TOKEN_LIMIT:
        # 단일 청크 — 기존 방식 그대로
        return _encode_single(model, code, language, tokenizer, device)

    # 청크 분할
    chunks = _split_by_lines(code, tokenizer)

    chunk_embs = []
    for chunk in chunks:
        emb = _encode_single(model, chunk, language, tokenizer, device)
        if emb is not None:
            chunk_embs.append(emb)

    if not chunk_embs:
        return None

    if len(chunk_embs) == 1:
        return chunk_embs[0]

    # Mean pooling → re-normalize
    stacked = torch.stack(chunk_embs, dim=0)          # (N, 768)
    pooled  = stacked.mean(dim=0)                     # (768,)
    return F.normalize(pooled, dim=0)


def _encode_single(
    model,
    code: str,
    language: str,
    tokenizer: PreTrainedTokenizer,
    device: str,
) -> torch.Tensor | None:
    """단일 코드 청크 → (768,) normalized embedding"""
    try:
        ids, pos, d2c, d2d = encode_with_dfg(code, language, tokenizer)
        mask = build_attn_mask(pos, d2c, d2d)

        input_ids    = torch.tensor([ids],  dtype=torch.long).to(device)
        position_ids = torch.tensor([pos],  dtype=torch.long).to(device)
        attn_mask    = torch.from_numpy(np.array([mask])).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                emb = model(input_ids, position_ids, attn_mask)   # (1, 768)
                emb = F.normalize(emb.float(), dim=1)
        return emb.squeeze(0)                                      # (768,)
    except Exception:
        return None


def chunked_embedding_batch(
    model,
    items: list[tuple[str, str]],   # [(code, language), ...]
    tokenizer: PreTrainedTokenizer,
    device: str = 'cuda',
    batch_size: int = 32,
) -> np.ndarray:
    """
    배치 단위 SEA 임베딩.
    실패한 샘플은 zero vector로 채움.

    Returns:
        (N, 768) numpy array
    """
    results = []
    for code, language in items:
        emb = chunked_embedding(model, code, language, tokenizer, device)
        if emb is not None:
            results.append(emb.cpu().numpy())
        else:
            results.append(np.zeros(768, dtype=np.float32))
    return np.vstack(results)
