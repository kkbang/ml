"""
embed_server.py — GraphCodeBERT 임베딩 서버 (ProcessPoolExecutor 적용)

핵심 변경: ThreadPoolExecutor → ProcessPoolExecutor
  - 각 Worker 프로세스가 독립 GIL 보유 → DFG 파싱 진짜 병렬화
  - Worker 시작 시 tokenizer 한 번만 초기화 (오버헤드 최소화)

아키텍처:
  요청 → asyncio.Queue → Batcher(5ms or 8개)
       → ProcessPoolExecutor (각 프로세스가 독립 GIL로 DFG 파싱)
       → GPU: dfg_replace (PyTorch)
       → TRT: RoBERTa Transformer
       → 응답

실행:
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  export KMP_AFFINITY=granularity=fine,compact,1,0
  uvicorn embed_server:app --host 0.0.0.0 --port 8000 --workers 1
"""

import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')
import asyncio
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor   # ← ThreadPool → ProcessPool
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

# ── 로깅 ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("embed_server")

# ── 설정 ──────────────────────────────────────────────────────────────
TRT_PATH            = 'graphcodebert_encoder.trt'
MODEL_NAME          = 'microsoft/graphcodebert-base'
MODEL_PATH          = 'model/GCB_dfg_stage2.pt'
MAX_BATCH           = 64     # TRT 엔진 빌드 시 설정값 (고정)
TRT_CHUNK_SIZE      = 64     # MAX_BATCH 초과 시 TRT를 이 크기로 나눠서 호출
BATCH_WAIT_MS       = 5     # 5 → 20ms: 더 많은 요청을 한 라운드에 묶기
BATCH_MAX_SIZE      = 8     # 8 → 16:  한 라운드 최대 PendingRequest 수
NUM_PREPROC_WORKERS = 32
L                   = TOTAL_LENGTH
D                   = 768
DEVICE              = 'cuda'

app_state: dict = {}


# ════════════════════════════════════════════════════════════════════
# ProcessPoolExecutor Worker 초기화
# ════════════════════════════════════════════════════════════════════

# 각 Worker 프로세스의 전역 tokenizer
# → 프로세스 시작 시 딱 한 번만 로드, 이후 모든 요청에서 재사용
_worker_tokenizer = None
_extract_rust      = None

def _init_worker(model_name: str) -> None:
    global _worker_tokenizer, _extract_rust
    sys.path.append('/home/ngseokim/code-killr/parser')
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_name)
    from dfg_rs import extract_dataflow_rust
    _extract_rust = extract_dataflow_rust

def _encode_with_dfg_rust(code: str, language: str, tokenizer) -> tuple:
    rs_tokens, rs_dfg = _extract_rust(code, language)
    dfg = list(rs_dfg)
    # 배치 tokenize (개별 호출 N번 → 1번으로 단축)
    prefixed        = [rs_tokens[0]] + ['@ ' + x for x in rs_tokens[1:]]
    batch_ids       = tokenizer(prefixed, add_special_tokens=False)['input_ids']
    code_tokens_sub = [tokenizer.convert_ids_to_tokens(batch_ids[0])] +                       [tokenizer.convert_ids_to_tokens(ids)[1:] for ids in batch_ids[1:]]
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens_sub)):
        ori2cur_pos[i] = (ori2cur_pos[i-1][1], ori2cur_pos[i-1][1] + len(code_tokens_sub[i]))
    code_tokens_flat = [t for sub in code_tokens_sub for t in sub]
    max_code = TOTAL_LENGTH - 3 - min(len(dfg), 64)
    code_tokens_flat = code_tokens_flat[:max_code][:512-3]
    source_tokens = [tokenizer.cls_token] + code_tokens_flat + [tokenizer.sep_token]
    source_ids    = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx  = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:TOTAL_LENGTH - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx  += [0] * len(dfg)
    source_ids    += [tokenizer.unk_token_id] * len(dfg)
    pad_len        = TOTAL_LENGTH - len(source_ids)
    position_idx  += [tokenizer.pad_token_id] * pad_len
    source_ids    += [tokenizer.pad_token_id] * pad_len
    reverse_index  = {x[1]: i for i, x in enumerate(dfg)}
    for i, (name, idx, sources) in enumerate(dfg):
        dfg[i] = (name, idx, [reverse_index[j] for j in sources if j in reverse_index])
    dfg_to_code = [(ori2cur_pos[x[1]][0] + 1, ori2cur_pos[x[1]][1] + 1) for x in dfg]
    dfg_to_dfg  = [x[2] for x in dfg]
    return source_ids, position_idx, dfg_to_code, dfg_to_dfg

def _preprocess_single_worker(args: tuple) -> tuple:
    global _worker_tokenizer
    code, language = args
    try:
        ids, pos, d2c, d2d = _encode_with_dfg_rust(code, language, _worker_tokenizer)
        return ids, pos, d2c, d2d, True
    except Exception:
        pad_id = _worker_tokenizer.pad_token_id
        return [pad_id] * L, [1] * L, [], [], False


# ════════════════════════════════════════════════════════════════════
# TRT 추론 엔진
# ════════════════════════════════════════════════════════════════════

class TRTInferencer:
    """inputs_embeds (B,L,768) → cls_output (B,768)"""

    def __init__(self, trt_path: str):
        logger_trt = trt.Logger(trt.Logger.WARNING)
        runtime    = trt.Runtime(logger_trt)
        with open(trt_path, 'rb') as f:
            self.engine  = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream  = cuda.Stream()

        # GPU 버퍼 사전 할당 (MAX_BATCH 기준으로 한 번만)
        self.d_embeds = cuda.mem_alloc(MAX_BATCH * L * D * np.dtype(np.float32).itemsize)
        self.d_attn   = cuda.mem_alloc(MAX_BATCH * L * np.dtype(np.int64).itemsize)
        self.d_pos    = cuda.mem_alloc(MAX_BATCH * L * np.dtype(np.int64).itemsize)
        self.d_out    = cuda.mem_alloc(MAX_BATCH * D * np.dtype(np.float32).itemsize)
        logger.info(f"TRT 엔진 로드 완료: {trt_path}")

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


# ════════════════════════════════════════════════════════════════════
# DFG 노드 교체 (GPU, PyTorch — 메인 프로세스에서 실행)
# ════════════════════════════════════════════════════════════════════

def dfg_replace_gpu(
    input_ids:    torch.Tensor,
    position_idx: torch.Tensor,
    attn_mask:    torch.Tensor,
    word_embeddings,
    pad_token_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    DFG 노드 임베딩 교체.
    CPU 파싱 결과를 받아 GPU에서 avg_embeddings 계산.
    메인 프로세스에서만 실행 (GPU는 메인 프로세스가 독점).
    """
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


# ════════════════════════════════════════════════════════════════════
# 요청/응답 스키마
# ════════════════════════════════════════════════════════════════════

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


# ════════════════════════════════════════════════════════════════════
# Batcher 루프
# ════════════════════════════════════════════════════════════════════

async def batcher_loop():
    """
    asyncio 이벤트 루프에서 돌아가는 배치 처리기.

    흐름:
    1. Queue에서 요청 수집 (BATCH_WAIT_MS 또는 BATCH_MAX_SIZE 도달 시)
    2. 각 코드 샘플을 ProcessPoolExecutor worker에 독립적으로 제출
       → asyncio.gather로 모든 worker 완료 대기
       → 각 worker는 독립 프로세스에서 진짜 병렬로 DFG 파싱
    3. GPU에서 DFG 노드 교체
    4. TRT 추론
    5. 결과를 각 요청 future에 분배
    """
    queue           = app_state['queue']
    inferencer      = app_state['inferencer']
    executor        = app_state['executor']
    word_embeddings = app_state['word_embeddings']
    pad_token_id    = app_state['pad_token_id']
    loop            = asyncio.get_event_loop()

    while True:
        # 첫 요청 대기
        try:
            first = await asyncio.wait_for(queue.get(), timeout=1.0)
        except asyncio.TimeoutError:
            continue

        # BATCH_WAIT_MS 동안 추가 요청 수집
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

        # 모든 PendingRequest의 아이템 flatten
        all_items, req_offsets = [], []
        offset = 0
        for req in pending:
            req_offsets.append((offset, offset + len(req.items)))
            all_items.extend(req.items)
            offset += len(req.items)

        # 길이순 정렬 (Dynamic Batching: padding 낭비 최소화)
        indexed        = sorted(enumerate(all_items), key=lambda x: len(x[1].code))
        sorted_items   = [item for _, item in indexed]
        original_order = [i    for i, _    in indexed]

        # ── Step 1: ProcessPoolExecutor로 DFG 파싱 (진짜 병렬) ──────
        t_pre = time.perf_counter()

        args    = [(item.code, item.language) for item in sorted_items]
        futures = [
            loop.run_in_executor(executor, _preprocess_single_worker, arg)
            for arg in args
        ]
        results = await asyncio.gather(*futures)

        # mask는 메인 프로세스에서 생성 (100KB×N → 2.5KB×N pickle 절감)
        ids_np  = np.array([r[0] for r in results], dtype=np.int64)
        pos_np  = np.array([r[1] for r in results], dtype=np.int64)
        mask_np = np.array(
            [build_attn_mask(r[1], r[2], r[3]) for r in results],
            dtype=bool
        )
        valid   = [r[4] for r in results]

        ms_pre  = (time.perf_counter() - t_pre) * 1000

        # ── Step 2: GPU DFG 노드 교체 ────────────────────────────────
        t_dfg  = time.perf_counter()
        ids_t  = torch.tensor(ids_np,  dtype=torch.long).to(DEVICE)
        pos_t  = torch.tensor(pos_np,  dtype=torch.long).to(DEVICE)
        mask_t = torch.tensor(mask_np, dtype=torch.bool).to(DEVICE)

        embeds_np, attn_np, posids_np = dfg_replace_gpu(
            ids_t, pos_t, mask_t, word_embeddings, pad_token_id
        )
        ms_dfg = (time.perf_counter() - t_dfg) * 1000

        # ── Step 3: TRT 추론 (MAX_BATCH 초과 시 청크 분할) ─────────
        t_trt = time.perf_counter()
        N = len(all_items)
        if N <= TRT_CHUNK_SIZE:
            # 일반 케이스: 한 번에 처리
            embs_sorted = inferencer.infer(embeds_np, attn_np, posids_np)
        else:
            # MAX_BATCH 초과 시: TRT_CHUNK_SIZE(64)씩 나눠서 처리 후 합치기
            chunks = []
            for c_start in range(0, N, TRT_CHUNK_SIZE):
                c_end = min(c_start + TRT_CHUNK_SIZE, N)
                chunk = inferencer.infer(
                    embeds_np[c_start:c_end],
                    attn_np[c_start:c_end],
                    posids_np[c_start:c_end],
                )
                chunks.append(chunk)
            embs_sorted = np.vstack(chunks)
        ms_trt = (time.perf_counter() - t_trt) * 1000

        ms_total = ms_pre + ms_dfg + ms_trt
        logger.info(
            f"batch={len(all_items):2d} | "
            f"DFG전처리={ms_pre:.1f}ms | "
            f"DFG교체={ms_dfg:.1f}ms | "
            f"TRT={ms_trt:.1f}ms | "
            f"총={ms_total:.1f}ms"
        )

        # 원래 순서 복원 후 각 요청에 결과 분배
        embs = np.zeros_like(embs_sorted)
        embs[original_order] = embs_sorted
        failed_global = [original_order[i] for i, v in enumerate(valid) if not v]

        for req, (start, end) in zip(pending, req_offsets):
            req_failed = [i - start for i in failed_global if start <= i < end]
            req.future.set_result((embs[start:end].tolist(), req_failed))


# ════════════════════════════════════════════════════════════════════
# 서버 lifespan
# ════════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("서버 초기화 중...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 학습된 모델에서 word_embeddings만 로드 (TRT 추론에 필요)
    base_encoder    = AutoModel.from_pretrained(MODEL_NAME, attn_implementation="eager")
    model           = GraphCodeBERTEncoder(base_encoder)
    ckpt            = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt.get('model', ckpt))
    model           = model.to(DEVICE)
    model.eval()
    word_embeddings = model.encoder.embeddings.word_embeddings
    pad_token_id    = model.encoder.config.pad_token_id

    inferencer = TRTInferencer(TRT_PATH)

    # ── ProcessPoolExecutor 초기화 ────────────────────────────────
    # initializer: 각 worker 프로세스 시작 시 _init_worker(MODEL_NAME) 실행
    # → 32개 프로세스 각각이 tokenizer를 독립적으로 로드
    # → 이후 요청 시 tokenizer 전달 오버헤드 없음
    executor = ProcessPoolExecutor(
        max_workers=NUM_PREPROC_WORKERS,
        initializer=_init_worker,         # worker 시작 시 실행할 함수
        initargs=(MODEL_NAME,),            # initializer에 전달할 인자
    )
    logger.info(f"ProcessPoolExecutor 초기화 완료 ({NUM_PREPROC_WORKERS} workers)")

    queue = asyncio.Queue(maxsize=1000)

    app_state.update({
        'tokenizer':       tokenizer,
        'word_embeddings': word_embeddings,
        'pad_token_id':    pad_token_id,
        'inferencer':      inferencer,
        'executor':        executor,
        'queue':           queue,
    })

    batcher_task = asyncio.create_task(batcher_loop())
    logger.info(f"서버 준비 완료 | TRT={TRT_PATH} | workers={NUM_PREPROC_WORKERS}")
    yield

    # 종료 시 정리
    batcher_task.cancel()
    executor.shutdown(wait=False)
    app_state.clear()


app = FastAPI(title='code-killr embedding API', lifespan=lifespan)


# ════════════════════════════════════════════════════════════════════
# 엔드포인트
# ════════════════════════════════════════════════════════════════════

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
        'executor_type': 'ProcessPoolExecutor',
    }


# ── OpenAI 호환 엔드포인트 (/v1/embeddings) ───────────────────────────
class OpenAIEmbedRequest(BaseModel):
    input: List[Union[CodeItem, str]]
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
