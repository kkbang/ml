"""
embed_server.py — 500 QPS 목표 GraphCodeBERT 임베딩 서버 (Split 구조)

아키텍처:
  요청 → asyncio.Queue → Batcher(20ms or 16개)
       → ThreadPoolExecutor(DFG 전처리 병렬)
       → GPU: dfg_replace (PyTorch, word_embeddings)
       → TRT: RoBERTa Transformer
       → 응답

실행:
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  export KMP_AFFINITY=granularity=fine,compact,1,0
  uvicorn embed_server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import sys
import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import encode_with_dfg, build_attn_mask, TOTAL_LENGTH
from model import GraphCodeBERTEncoder

# ── 로깅 설정 ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("embed_server")

# ── 설정 ──────────────────────────────────────────────────────────────
TRT_PATH            = 'graphcodebert_encoder.trt'
MODEL_NAME          = 'microsoft/graphcodebert-base'
MODEL_PATH          = 'GCB_dfg_stage2.pt'
MAX_BATCH           = 64
BATCH_WAIT_MS       = 5         # 단건 요청 latency 최소화
BATCH_MAX_SIZE      = 8         # TRT sweet spot (1293 QPS)
NUM_PREPROC_WORKERS = 32        # CPU 64코어 중 절반 사용
L                   = TOTAL_LENGTH
D                   = 768
DEVICE              = 'cuda'

app_state: dict = {}


# ── TRT 추론 엔진 ──────────────────────────────────────────────────────
class TRTInferencer:
    """inputs_embeds (B,L,768) → cls_output (B,768)"""

    def __init__(self, trt_path: str):
        logger  = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        with open(trt_path, 'rb') as f:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        self.d_embeds = cuda.mem_alloc(MAX_BATCH * L * D * np.dtype(np.float32).itemsize)
        self.d_attn   = cuda.mem_alloc(MAX_BATCH * L * np.dtype(np.int64).itemsize)
        self.d_pos    = cuda.mem_alloc(MAX_BATCH * L * np.dtype(np.int64).itemsize)
        self.d_out    = cuda.mem_alloc(MAX_BATCH * D * np.dtype(np.float32).itemsize)
        print(f"  TRT 엔진 로드 완료: {trt_path}")

    def infer(self,
              inputs_embeds:  np.ndarray,   # (B, L, D) float32
              attention_mask: np.ndarray,   # (B, L)    int64
              position_ids:   np.ndarray,   # (B, L)    int64
              ) -> np.ndarray:              # (B, D)    float32 normalized
        B = inputs_embeds.shape[0]

        self.context.set_input_shape('inputs_embeds',  (B, L, D))
        self.context.set_input_shape('attention_mask', (B, L))
        self.context.set_input_shape('position_ids',   (B, L))
        self.context.set_tensor_address('inputs_embeds',  int(self.d_embeds))
        self.context.set_tensor_address('attention_mask', int(self.d_attn))
        self.context.set_tensor_address('position_ids',   int(self.d_pos))
        self.context.set_tensor_address('cls_output',     int(self.d_out))

        cuda.memcpy_htod_async(self.d_embeds, inputs_embeds.astype(np.float32), self.stream)
        cuda.memcpy_htod_async(self.d_attn,   attention_mask.astype(np.int64),  self.stream)
        cuda.memcpy_htod_async(self.d_pos,    position_ids.astype(np.int64),    self.stream)
        self.context.execute_async_v3(self.stream.handle)

        output = np.empty((B, D), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.d_out, self.stream)
        self.stream.synchronize()

        norms = np.linalg.norm(output, axis=1, keepdims=True) + 1e-8
        return output / norms

    def __del__(self):
        try:
            self.d_embeds.free(); self.d_attn.free()
            self.d_pos.free();    self.d_out.free()
        except Exception:
            pass


# ── DFG 전처리 (CPU, 병렬) ────────────────────────────────────────────
def preprocess_single(args) -> tuple:
    code, language, tokenizer = args
    try:
        ids, pos, d2c, d2d = encode_with_dfg(code, language, tokenizer)
        mask = build_attn_mask(pos, d2c, d2d)
        return ids, pos, mask, True
    except Exception:
        pad_id = tokenizer.pad_token_id
        return [pad_id]*L, [1]*L, np.zeros((L,L), dtype=bool), False


# ── DFG 노드 교체 (GPU, PyTorch) ──────────────────────────────────────
def dfg_replace_gpu(
    input_ids:    torch.Tensor,   # (B, L)
    position_idx: torch.Tensor,   # (B, L)
    attn_mask:    torch.Tensor,   # (B, L, L)
    word_embeddings,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns numpy arrays ready for TRT"""
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.float32):
        nodes_mask    = position_idx.eq(0)
        token_mask    = position_idx.ge(2)
        inputs_embeds = word_embeddings(input_ids)

        nodes_to_token_mask = (
            nodes_mask[:, :, None] & token_mask[:, None, :] & attn_mask
        ).float()
        nodes_to_token_mask = nodes_to_token_mask / (
            nodes_to_token_mask.sum(-1, keepdim=True) + 1e-10
        )
        avg_embeddings = torch.einsum(
            "abc,acd->abd", nodes_to_token_mask, inputs_embeds
        )
        inputs_embeds = (
            inputs_embeds * (~nodes_mask)[:, :, None]
            + avg_embeddings * nodes_mask[:, :, None]
        )
        attention_mask_1d = input_ids.ne(pad_token_id).long()

    return (
        inputs_embeds.cpu().numpy(),
        attention_mask_1d.cpu().numpy(),
        position_idx.cpu().numpy(),
    )


# ── 스키마 ────────────────────────────────────────────────────────────
class CodeItem(BaseModel):
    code:     str
    language: str = 'python'

class EmbedRequest(BaseModel):
    items: List[CodeItem]

class EmbedResponse(BaseModel):
    embeddings:     List[List[float]]
    failed_indices: List[int]
    elapsed_ms:     float


class PendingRequest:
    def __init__(self, items):
        self.items  = items
        self.future = asyncio.get_event_loop().create_future()


# ── Batcher 루프 ──────────────────────────────────────────────────────
async def batcher_loop():
    queue          = app_state['queue']
    inferencer     = app_state['inferencer']
    executor       = app_state['executor']
    tokenizer      = app_state['tokenizer']
    word_embeddings = app_state['word_embeddings']
    pad_token_id   = app_state['pad_token_id']
    loop           = asyncio.get_event_loop()

    while True:
        try:
            first = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        pending  = [first]
        deadline = time.perf_counter() + BATCH_WAIT_MS / 1000

        while len(pending) < BATCH_MAX_SIZE:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                req = await asyncio.wait_for(queue.get(), timeout=remaining)
                pending.append(req)
            except asyncio.TimeoutError:
                break

        # flatten + 길이순 정렬
        all_items, req_offsets = [], []
        offset = 0
        for req in pending:
            req_offsets.append((offset, offset + len(req.items)))
            all_items.extend(req.items)
            offset += len(req.items)

        indexed        = sorted(enumerate(all_items), key=lambda x: len(x[1].code))
        sorted_items   = [item for _, item in indexed]
        original_order = [i    for i, _    in indexed]

        # Step 1: CPU 병렬 DFG 전처리 (asyncio.gather로 각 샘플 독립 제출)
        t_pre   = time.perf_counter()
        args    = [(item.code, item.language, tokenizer) for item in sorted_items]
        futures = [loop.run_in_executor(executor, preprocess_single, arg) for arg in args]
        results = await asyncio.gather(*futures)
        ms_pre  = (time.perf_counter() - t_pre) * 1000

        ids_np   = np.array([r[0] for r in results], dtype=np.int64)
        pos_np   = np.array([r[1] for r in results], dtype=np.int64)
        mask_np  = np.array([r[2] for r in results], dtype=bool)
        valid    = [r[3] for r in results]

        # Step 2: GPU DFG 노드 교체 (PyTorch, word_embeddings)
        t_dfg  = time.perf_counter()
        ids_t  = torch.tensor(ids_np,  dtype=torch.long).to(DEVICE)
        pos_t  = torch.tensor(pos_np,  dtype=torch.long).to(DEVICE)
        mask_t = torch.tensor(mask_np, dtype=torch.bool).to(DEVICE)

        embeds_np, attn_np, posids_np = dfg_replace_gpu(
            ids_t, pos_t, mask_t, word_embeddings, pad_token_id
        )
        ms_dfg = (time.perf_counter() - t_dfg) * 1000

        # Step 3: TRT 추론
        t_trt       = time.perf_counter()
        embs_sorted = inferencer.infer(embeds_np, attn_np, posids_np)
        ms_trt      = (time.perf_counter() - t_trt) * 1000

        ms_total = ms_pre + ms_dfg + ms_trt
        logger.info(
            f"batch={len(all_items):2d} | "
            f"DFG전처리={ms_pre:.1f}ms | "
            f"DFG교체={ms_dfg:.1f}ms | "
            f"TRT={ms_trt:.1f}ms | "
            f"총={ms_total:.1f}ms"
        )

        # 원래 순서 복원
        embs = np.zeros_like(embs_sorted)
        embs[original_order] = embs_sorted
        failed_global = [original_order[i] for i, v in enumerate(valid) if not v]

        for req, (start, end) in zip(pending, req_offsets):
            req_failed = [i - start for i in failed_global if start <= i < end]
            req.future.set_result((embs[start:end].tolist(), req_failed))


# ── lifespan ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("서버 초기화 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 학습된 모델에서 word_embeddings 로드
    base_encoder    = AutoModel.from_pretrained(MODEL_NAME)
    model           = GraphCodeBERTEncoder(base_encoder)
    ckpt            = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt.get('model', ckpt))
    model           = model.to(DEVICE)
    model.eval()
    word_embeddings = model.encoder.embeddings.word_embeddings
    pad_token_id    = model.encoder.config.pad_token_id

    inferencer = TRTInferencer(TRT_PATH)
    executor   = ThreadPoolExecutor(max_workers=NUM_PREPROC_WORKERS)
    queue      = asyncio.Queue(maxsize=1000)

    app_state.update({
        'tokenizer':       tokenizer,
        'word_embeddings': word_embeddings,
        'pad_token_id':    pad_token_id,
        'inferencer':      inferencer,
        'executor':        executor,
        'queue':           queue,
    })

    batcher_task = asyncio.create_task(batcher_loop())
    print(f"준비 완료 | TRT={TRT_PATH} | preproc_workers={NUM_PREPROC_WORKERS}")
    yield
    batcher_task.cancel()
    executor.shutdown(wait=False)
    app_state.clear()


app = FastAPI(title='code-killr embedding API', lifespan=lifespan)


# ── 엔드포인트 ────────────────────────────────────────────────────────
@app.post('/embed', response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail='items가 비어 있습니다.')
    if len(request.items) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f'최대 {MAX_BATCH}개')

    t0  = time.perf_counter()
    req = PendingRequest(request.items)
    await app_state['queue'].put(req)
    embeddings, failed_indices = await req.future
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(f"/embed | items={len(request.items)} | {elapsed_ms}ms | failed={len(failed_indices)}")
    return EmbedResponse(
        embeddings=embeddings,
        failed_indices=failed_indices,
        elapsed_ms=elapsed_ms,
    )


@app.get('/health')
async def health():
    return {
        'status': 'ok', 'trt_engine': TRT_PATH,
        'max_batch': MAX_BATCH, 'preproc_workers': NUM_PREPROC_WORKERS,
    }


# ── OpenAI 호환 엔드포인트 (/v1/embeddings) ───────────────────────────
class OpenAIEmbedRequest(BaseModel):
    input: List[Union[CodeItem, str]]   # str이면 language="python" 기본값
    model: str = "code-killr"

class OpenAIEmbeddingData(BaseModel):
    object:    str = "embedding"
    embedding: List[float]
    index:     int

class OpenAIEmbedResponse(BaseModel):
    object: str = "list"
    data:   List[OpenAIEmbeddingData]
    model:  str
    usage:  dict


@app.post('/v1/embeddings', response_model=OpenAIEmbedResponse)
async def openai_embed(request: OpenAIEmbedRequest):
    if not request.input:
        raise HTTPException(status_code=400, detail='input이 비어 있습니다.')
    if len(request.input) > MAX_BATCH:
        raise HTTPException(status_code=400, detail=f'최대 {MAX_BATCH}개')

    # str 입력 → CodeItem 변환 (language 기본값: python)
    items = [
        CodeItem(code=item, language='python') if isinstance(item, str) else item
        for item in request.input
    ]

    t0  = time.perf_counter()
    req = PendingRequest(items)
    await app_state['queue'].put(req)
    embeddings, _ = await req.future
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    logger.info(f"/v1/embeddings | items={len(items)} | {elapsed_ms}ms")
    return OpenAIEmbedResponse(
        data=[
            OpenAIEmbeddingData(embedding=emb, index=i)
            for i, emb in enumerate(embeddings)
        ],
        model=request.model,
        usage={
            'prompt_tokens': sum(len(it.code.split()) for it in items),
            'total_tokens':  sum(len(it.code.split()) for it in items),
        },
    )
