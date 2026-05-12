# code-killr 🔍

> **라이선스 위반 코드 표절 탐지 시스템**  
> GraphCodeBERT + DFG + Contrastive Learning 기반 코드 임베딩 모델

---

## 개요

`code-killr`는 오픈소스 라이선스(GPL, LGPL, MIT 등)가 부여된 코드가 리팩토링을 거쳐 무단 복제되었는지 탐지하는 ML 시스템입니다.  
변수명 변경, 구조 변환, 의미 보존 리팩토링 등 다양한 난이도의 표절 유형을 탐지할 수 있습니다.

### 핵심 특징

- **GraphCodeBERT + DFG**: Data Flow Graph를 입력으로 받아 변수명 변경에 강건한 임베딩 생성
- **Weighted NT-Xent Loss**: GPL/LGPL 라이선스에 높은 가중치를 부여한 대조 학습
- **2단계 학습**: Stage1 (기본 대조 학습) → Hard Negative Mining → Stage2 (강화 학습)
- **Split Architecture 서빙**: DFG 전처리(ProcessPoolExecutor) + Transformer(TensorRT FP16)
- **OpenAI 호환 API**: `/v1/embeddings` 엔드포인트 제공

---

## 성능

### 모델 성능

| 지표 | 값 |
|---|---|
| Positive 유사도 | 0.9605 (±0.0640) |
| Negative 유사도 | 0.0328 (±0.0830) |
| Precision (>0.5) | 0.9986 |
| FPR (>0.5) | 0.0005 |

### 리팩토링 레벨별 성능

| 레벨 | Precision | FPR |
|---|---|---|
| Surface (변수명 변경 등) | 0.9991 | 0.0004 |
| Structural (구조 변환) | 0.9994 | 0.0002 |
| Semantic (의미 보존) | 0.9961 | 0.0007 |

### 변수명 변경 강건성

| 변경 비율 | Precision |
|---|---|
| 0% | 1.0000 |
| 25% | 0.9700 |
| 50% | 0.8920 |
| 75% | 0.6140 |
| 100% | 0.1660 |

### 서빙 성능 (RTX 3090, TensorRT FP16 + ProcessPoolExecutor)

| 동시 요청 수 | Throughput | p50 Latency | p99 Latency |
|---|---|---|---|
| 1 | ~80 items/sec | ~64ms | - |
| 8 | 322 items/sec | 195ms | 220ms |
| 16 | **365 items/sec** | 328ms | 415ms |
| 32 | 349 items/sec | 663ms | 969ms |

> 실제 프로덕션 코드(복잡한 함수) 기준: 단건 요청 ~100~200ms

---

## 아키텍처

### 모델 구조

```
코드 입력
   ↓
DFG 추출 (tree-sitter + GraphCodeBERT parser)
   ↓
GraphCodeBERTEncoder
   ├── DFG 노드 임베딩 교체 (avg_embeddings)
   └── RoBERTa Transformer (12 layers)
   ↓
[CLS] 임베딩 (768-dim, L2 normalized)
   ↓
Weighted NT-Xent Loss
```

### 서빙 구조 (Split Architecture)

```
클라이언트 요청
      ↓
asyncio.Queue
      ↓ BATCH_WAIT_MS(5ms) or BATCH_MAX_SIZE(8개) 도달 시
   Batcher
      ↓ ProcessPoolExecutor(32 workers) — 각 프로세스 독립 GIL
        DFG 파싱 (encode_with_dfg) 진짜 병렬 실행       ~15~60ms
      ↓ 메인 프로세스: build_attn_mask + GPU DFG 교체     ~20ms
      ↓ TensorRT FP16 Transformer 추론 (batch=8)          ~10ms
      ↓ L2 정규화 → 결과 분배
   응답
```

> **Split Architecture 이유**: GraphCodeBERT의 커스텀 DFG forward는 ONNX/TRT export 시
> Transformers 버전 호환 문제가 있어, DFG 전처리(Python/PyTorch)와
> Transformer(TRT)를 분리하여 각각 최적화합니다.

### 최적화 적용 내역

| 순위 | 기법 | 적용 여부 | 효과 |
|---|---|---|---|
| 1 | TensorRT FP16 | ✅ | GPU 추론 1,293 QPS (batch=8) |
| 2 | ProcessPoolExecutor | ✅ | DFG 파싱 진짜 병렬화 (GIL 우회) |
| 3 | Dynamic Batching | ✅ | 길이순 정렬로 padding 최소화 |
| 4 | Thread Affinity | ✅ | KMP_AFFINITY 환경변수 |
| 5 | Pickle 최적화 | ✅ | mask 직렬화 제거 (12.8MB → 320KB) |
| 6 | CUDA Streams | ❌ | 파이프라인 구조 변경 필요 |

### 모델 스펙

- Base: `microsoft/graphcodebert-base` (125M params)
- 입력: `input_ids` + `position_idx` (DFG) + `attn_mask` (DFG edge)
- 출력: 768-dim L2 normalized embedding
- 지원 언어: Python, Java, JavaScript, Go, Ruby, PHP, C#

---

## 프로젝트 구조

```
code-killr/
├── parser/                         # GraphCodeBERT 공식 파서
│   ├── DFG.py                      # 언어별 DFG 추출
│   ├── utils.py                    # 토큰 유틸리티
│   └── my-languages.so             # tree-sitter 컴파일 바이너리
│
├── model.py                        # GraphCodeBERTEncoder 정의
├── dataset.py                      # PairDataset, encode_with_dfg, build_attn_mask
├── loss.py                         # Weighted NT-Xent Loss
├── chunker.py                      # SEA (Split-Encode-Aggregate) 512 토큰 초과 처리
│
├── train_graphcodebert.py          # Stage1 학습
├── hard_negative_mining_gcb.py     # Hard Negative Mining
├── train_stage2_gcb.py             # Stage2 학습 (Hard Negative 혼합)
│
├── evaluate.py                     # 성능 평가 (레벨별/언어별/라이선스별/강건성)
├── benchmark.py                    # 서버 Latency/Throughput/QPS 측정
│
├── export_onnx.py                  # RoBERTa encoder → ONNX (Split 구조)
├── convert_trt.py                  # ONNX → TensorRT FP16 변환
├── embed_server.py                 # FastAPI 임베딩 서버 (OpenAI 호환)
│
├── data/
│   ├── train_v3.jsonl
│   ├── val_v3.jsonl
│   ├── test_v3.jsonl
│   └── opensearch_data_100k.json
│
├── hard_negatives_gcb.jsonl        # Mining 결과 (10,253개, avg sim: 0.6925)
├── GCB_dfg_stage1.pt               # Stage1 체크포인트 (val loss: 0.0179)
├── GCB_dfg_stage2.pt               # Stage2 체크포인트 (val loss: 0.0078)
├── graphcodebert_encoder.onnx      # ONNX 모델 (RoBERTa encoder)
└── graphcodebert_encoder.trt       # TensorRT FP16 엔진 (MAX_BATCH=64)
```

---

## 환경 설정

```bash
# 학습 환경 (Transformers 5.x)
conda create -n code-killr python=3.12 -y
conda activate code-killr
pip install torch transformers tree-sitter==0.20.4 fastapi uvicorn

# 서빙 환경 (Transformers 4.40.0 — TRT export 호환)
conda create -n code-killr-serve python=3.12 -y
conda activate code-killr-serve
pip install torch transformers==4.40.0 fastapi uvicorn \
            onnx onnxruntime tensorrt pycuda \
            tree-sitter==0.20.4 aiohttp numpy
```

---

## 학습 파이프라인

```bash
# Stage1
python train_graphcodebert.py
# → best val loss: 0.0179 (epoch 6, early stop epoch 11)

# Hard Negative Mining
python hard_negative_mining_gcb.py
# → 10,253개 저장, 평균 유사도 0.6925

# Stage2
python train_stage2_gcb.py
# → best val loss: 0.0078

# 성능 평가
python evaluate.py
```

---

## 서빙 파이프라인

```bash
# Step 1: ONNX export (서빙 환경 conda activate code-killr-serve)
python export_onnx.py

# Step 2: TensorRT 변환 (~40초 소요)
python convert_trt.py

# Step 3: 서버 실행
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
uvicorn embed_server:app --host 0.0.0.0 --port 8000 --workers 1
```

> 서버 시작 시 ProcessPoolExecutor 32개 프로세스가 초기화됩니다 (30~60초 소요).  
> `서버 준비 완료` 로그 확인 후 사용하세요.

---

## API 사용법

### OpenAI 호환 방식 (권장)

```python
from openai import OpenAI

client = OpenAI(api_key="dummy", base_url="http://<서버IP>:8000")

# 문자열 입력 (language 기본값: python)
response = client.embeddings.create(
    model="code-killr",
    input=["def add(x, y): return x + y"]
)

# 언어 명시
response = client.embeddings.create(
    model="code-killr",
    input=[
        {"code": "def add(x, y): return x + y", "language": "python"},
        {"code": "public int add(int a, int b) { return a + b; }", "language": "java"},
    ]
)

vectors = [d.embedding for d in response.data]  # (N, 768)
```

### 응답 형식

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [...768차원...], "index": 0}
  ],
  "model": "code-killr",
  "usage": {"prompt_tokens": 8, "total_tokens": 8}
}
```

### 배치 권장 패턴 (OpenSearch 연동)

```python
BATCH_SIZE = 8   # TRT sweet spot

for i in range(0, len(docs), BATCH_SIZE):
    batch = docs[i:i + BATCH_SIZE]
    response = client.embeddings.create(
        model="code-killr",
        input=[
            {"code": d["code"], "language": d["language"]}
            for d in batch
        ]
    )
    embeddings = [d.embedding for d in response.data]
```

### 제약사항

- 한 번에 최대 64개
- 지원 언어: `python`, `java`, `javascript`, `go`, `ruby`, `php`
- 언어 미지정 시 `python` 기본값

---

## 벤치마크

```bash
# 기본
python benchmark.py

# 파라미터 조정
python benchmark.py --concurrency 16 --batch-size 8 --total 400 --url http://localhost:8000
```

### 서버 로그 예시

```
16:23:11 [INFO] batch= 8 | DFG전처리=18.3ms | DFG교체=4.2ms | TRT=6.1ms | 총=28.6ms
16:23:11 [INFO] /v1/embeddings | items=8 | 34.2ms
```

---

## 라이선스 가중치

| 라이선스 | 가중치 | 이유 |
|---|---|---|
| GPL-2.0 / GPL-3.0 / AGPL-3.0 | 3.0 | 강한 카피레프트 |
| LGPL-2.1 / LGPL-3.0 | 2.0 | 약한 카피레프트 |
| MIT / Apache-2.0 / 기타 | 1.0 | 허용적 라이선스 |

---

## 인프라

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 3090 (24GB, Ampere CC 8.6) |
| CPU | 64코어 |
| OS | Ubuntu (Remote SSH) |
| 학습 환경 | conda `code-killr`, Python 3.12, Transformers 5.8.0 |
| 서빙 환경 | conda `code-killr-serve`, Python 3.12, Transformers 4.40.0 |
| 파서 | tree-sitter v0.20.4 |
| 추론 | TensorRT FP16, PyTorch 2.x |

---

## 팀

- **모델 파이프라인 / 서빙**: ShinMK
- **데이터 크롤링 / OpenSearch**: 팀원
