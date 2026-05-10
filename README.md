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
- **1,293 QPS 서빙**: TensorRT FP16 + 비동기 Dynamic Batching + OpenAI 호환 API

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

### 서빙 성능 (RTX 3090, TensorRT FP16)

| batch | latency | QPS |
|---|---|---|
| 1 | 1.9ms | 528 |
| 8 | 6.2ms | **1,293** |
| 16 | 12.9ms | 1,244 |
| 32 | 26.7ms | 1,198 |
| 64 | 53.0ms | 1,208 |

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
      ↓ BATCH_WAIT_MS(5ms) or BATCH_MAX_SIZE(8개) 대기
   Batcher
      ↓ CPU 병렬 DFG 파싱 (ThreadPoolExecutor × 8)  ~2ms
      ↓ GPU DFG 노드 임베딩 교체 (PyTorch)           ~6ms
      ↓ TensorRT FP16 Transformer 추론               ~3ms
      ↓ L2 정규화 → 결과 분배
   응답 (총 ~11ms)
```

> **Split Architecture 이유**: GraphCodeBERT의 커스텀 DFG forward는 TRT export 시 호환 문제가 있어,  
> DFG 전처리(Python/PyTorch)와 Transformer(TRT)를 분리하여 각각 최적화합니다.

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
│
├── export_onnx.py                  # RoBERTa encoder → ONNX 변환 (Split)
├── convert_trt.py                  # ONNX → TensorRT FP16 변환
├── embed_server.py                 # FastAPI 임베딩 서버 (OpenAI 호환)
│
├── data/
│   ├── train_v3.jsonl              # 학습 데이터 (positive pair)
│   ├── val_v3.jsonl                # 검증 데이터
│   ├── test_v3.jsonl               # 테스트 데이터
│   └── opensearch_data_100k.json   # Hard Negative Mining 풀 (98,006개)
│
├── hard_negatives_gcb.jsonl        # Mining 결과 (10,253개, avg sim: 0.6925)
├── GCB_dfg_stage1.pt               # Stage1 체크포인트 (val loss: 0.0179)
├── GCB_dfg_stage2.pt               # Stage2 체크포인트 (val loss: 0.0078)
├── graphcodebert_encoder.onnx      # ONNX 모델 (RoBERTa encoder)
└── graphcodebert_encoder.trt       # TensorRT FP16 엔진
```

---

## 데이터 형식

### train/val/test (JSONL)
```json
{
  "anchor":   "def foo(x): return x + 1",
  "positive": "def bar(y): return y + 1",
  "language": "python",
  "license":  "GPL-3.0",
  "level":    "surface",
  "repo":     "https://github.com/..."
}
```

### hard_negatives_gcb (JSONL)
```json
{
  "anchor":           "...",
  "negative":         "...",
  "similarity":       0.6832,
  "anchor_license":   "GPL-3.0",
  "negative_license": "MIT",
  "language":         "java"
}
```

---

## 실행 방법

### 환경 설정

```bash
# 학습 환경
conda create -n code-killr python=3.12 -y
conda activate code-killr
pip install torch transformers==5.8.0 tree-sitter==0.20.4 fastapi uvicorn

# 서빙 환경 (TRT 호환)
conda create -n code-killr-serve python=3.12 -y
conda activate code-killr-serve
pip install torch transformers==4.40.0 fastapi uvicorn onnx onnxruntime tensorrt pycuda tree-sitter==0.20.4
```

### Stage1 학습

```bash
python train_graphcodebert.py
# best val loss: 0.0179 (epoch 6, early stop at epoch 11)
```

### Hard Negative Mining

```bash
python hard_negative_mining_gcb.py
# 결과: 10,253개 저장, 평균 유사도 0.6925
```

### Stage2 학습

```bash
python train_stage2_gcb.py
# best val loss: 0.0078
```

### 성능 평가

```bash
python evaluate.py
```

### 서빙 파이프라인

```bash
# Step 1: ONNX export (서빙 환경에서)
conda activate code-killr-serve
python export_onnx.py

# Step 2: TensorRT 변환 (~40초 소요)
python convert_trt.py

# Step 3: 서버 실행
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
uvicorn embed_server:app --host 0.0.0.0 --port 8000 --workers 1
```

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

vectors = [d.embedding for d in response.data]  # List[List[float]], shape: (N, 768)
```

### curl

```bash
# 문자열 입력
curl -X POST http://<서버IP>:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "code-killr", "input": ["def add(x, y): return x + y"]}'

# 언어 명시
curl -X POST http://<서버IP>:8000/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "code-killr",
    "input": [
      {"code": "def add(x, y): return x + y", "language": "python"},
      {"code": "public int add(int a, int b) {}", "language": "java"}
    ]
  }'
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
    # OpenSearch에 저장
```

### 제약사항

- 한 번에 최대 64개
- 지원 언어: `python`, `java`, `javascript`, `go`, `ruby`, `php`
- 언어 미지정 시 `python` 기본값 적용

---

## 서버 로그 예시

```
03:50:39 [INFO] batch= 1 | DFG전처리=2.1ms | DFG교체=5.7ms | TRT=3.3ms | 총=11.2ms
03:50:39 [INFO] /v1/embeddings | items=1 | 31.87ms
```

---

## 라이선스 가중치

| 라이선스 | 가중치 | 이유 |
|---|---|---|
| GPL-2.0 / GPL-3.0 / AGPL-3.0 | 3.0 | 강한 카피레프트, 위반 시 법적 영향 큼 |
| LGPL-2.1 / LGPL-3.0 | 2.0 | 약한 카피레프트 |
| MIT / Apache-2.0 / 기타 | 1.0 | 허용적 라이선스 |

---

## 인프라

| 항목 | 사양 |
|---|---|
| GPU | NVIDIA RTX 3090 (24GB, Ampere CC 8.6) |
| OS | Ubuntu (Remote SSH) |
| 학습 환경 | conda `code-killr`, Python 3.12, Transformers 5.8.0 |
| 서빙 환경 | conda `code-killr-serve`, Python 3.12, Transformers 4.40.0 |
| 파서 | tree-sitter v0.20.4 |
| 추론 | TensorRT FP16, PyTorch 2.x |

---

## 팀

- **모델 파이프라인 / 서빙**: ShinMK
- **데이터 크롤링 / OpenSearch**: 팀원
