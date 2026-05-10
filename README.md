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
- **500 QPS 목표 서빙**: TensorRT FP16 + 비동기 Dynamic Batching

---

## 성능

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

---

## 아키텍처

```
코드 입력
   ↓
DFG 추출 (tree-sitter + GraphCodeBERT parser)
   ↓
GraphCodeBERTEncoder
   ├── DFG 노드 임베딩 교체 (avg_embeddings)
   └── RoBERTa Transformer
   ↓
[CLS] 임베딩 (768-dim)
   ↓
Weighted NT-Xent Loss
```

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
├── export_onnx.py                  # GraphCodeBERTEncoder → ONNX 변환
├── convert_trt.py                  # ONNX → TensorRT 변환
├── embed_server.py                 # FastAPI 임베딩 서버 (500 QPS 목표)
│
├── data/
│   ├── train_v3.jsonl              # 학습 데이터 (positive pair)
│   ├── val_v3.jsonl                # 검증 데이터
│   ├── test_v3.jsonl               # 테스트 데이터
│   └── opensearch_data_100k.json   # Hard Negative Mining 풀
│
├── hard_negatives_gcb.jsonl        # Mining 결과 (10,253개)
├── GCB_dfg_stage1.pt               # Stage1 체크포인트
├── GCB_dfg_stage2.pt               # Stage2 체크포인트 (최종)
├── graphcodebert.onnx              # ONNX 모델
└── graphcodebert.trt               # TensorRT 엔진
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
conda create -n code-killr python=3.12
conda activate code-killr
pip install torch transformers tree-sitter==0.20.4 fastapi uvicorn onnx onnxruntime tensorrt pycuda
```

### Stage1 학습

```bash
python train_graphcodebert.py
```

### Hard Negative Mining

```bash
python hard_negative_mining_gcb.py
```

### Stage2 학습

```bash
python train_stage2_gcb.py
```

### 성능 평가

```bash
python evaluate.py
```

### ONNX Export → TensorRT 변환 → 서버 실행

```bash
# 1. ONNX export
python export_onnx.py

# 2. TensorRT 변환 (수 분 소요)
python convert_trt.py

# 3. 서버 실행
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export KMP_AFFINITY=granularity=fine,compact,1,0
uvicorn embed_server:app --host 0.0.0.0 --port 8000 --workers 1
```

### API 호출

```python
import requests

response = requests.post("http://<서버IP>:8000/embed", json={
    "items": [
        {"code": "def foo(x): return x + 1", "language": "python"},
        {"code": "public int bar(int y) { return y + 1; }", "language": "java"},
    ]
})

data = response.json()
# data["embeddings"]      → List[List[float]]  shape: (N, 768)
# data["failed_indices"]  → List[int]          DFG 파싱 실패 인덱스
# data["elapsed_ms"]      → float              처리 시간 (ms)
```

---

## 서빙 아키텍처

```
클라이언트 요청들
      ↓
asyncio.Queue
      ↓ 20ms or 32개 대기
   Batcher
      ↓ 병렬 DFG 파싱 (ThreadPoolExecutor × 8)
  ids / pos / mask 배치
      ↓ 길이순 정렬 (Dynamic Batching)
  TensorRT FP16 추론
      ↓
  각 요청에 결과 분배 → 응답
```

| 최적화 | 적용 |
|---|---|
| TensorRT FP16 | GPU 추론 |
| Dynamic Batching | 길이순 정렬로 padding 최소화 |
| 비동기 Request Queue | asyncio 기반 배치 수집 |
| 병렬 DFG 전처리 | ThreadPoolExecutor × 8 |
| Thread Affinity | KMP_AFFINITY 환경변수 |

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
| GPU | NVIDIA RTX 3090 (24GB) |
| OS | Ubuntu (Remote SSH) |
| 환경 | conda `code-killr`, Python 3.12 |
| 프레임워크 | PyTorch, HuggingFace Transformers |
| 파서 | tree-sitter v0.20.4 |

---

## 팀

- **모델 파이프라인**: ShinMK
- **데이터 크롤링**: 팀원
