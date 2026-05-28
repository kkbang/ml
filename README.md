# 프로젝트 이름

> License Violation Code Plagiarism Detection System
> 오픈소스 라이선스(GPL, LGPL, MIT 등) 코드의 무단 복제·리팩터링 탐지

---

## Overview

오픈소스 라이선스 코드가 변수명 변경, 구조 변형, 의미 보존 리팩터링 등을 거쳐 무단 사용되는 경우를 탐지하는 ML 시스템이다.

**Bi-encoder 단독 모델**과 **OpenSearch 기반 hybrid retrieval + license_review heuristic reranker**로 이뤄진 production 파이프라인을 함께 운영한다.

- **Base model**: `microsoft/graphcodebert-base` (125M params)
- **Embedding**: 768-dim, L2-normalized
- **지원 언어**: Python, Java, JavaScript, Go, Ruby, PHP, C#

---

## Architecture

### Model

```
Source Code
    ↓
[DFG Parser]  ← Rust(dfg_rs) + Python parser
    ↓
Code Tokens + Data Flow Graph
    ↓
[GraphCodeBERT Encoder]
    ↓
768-dim Embedding (L2-normalized)
```

- **GraphCodeBERTEncoder** — DFG를 함께 입력해 변수명 변형에 강건한 표현 학습
- **Loss** — Weighted NT-Xent (copyleft 계열에 더 높은 weight)

### Repo Structure

| 디렉토리 | 역할 |
|---------|------|
| `core/` | 모델 컴포넌트 (encoder, loss 등) |
| `parser/` | GraphCodeBERT용 DFG 추출 |
| `dfg_rs/` | DFG Rust 구현 (속도 최적화) |
| `train/` | 학습 파이프라인 (stage1 ~ stage6) |
| `eval/` | 평가 스크립트 |
| `serve/` | FastAPI 임베딩 서버 |
| `bench/` | 벤치마크 유틸 |
| `test-suite/judged/` | Production 평가 테스트셋 (4 카테고리) |

---

## Training Pipeline

다단계 점진 학습 구조 (`train/` 디렉토리 기준).

| 단계 | 파일 | 역할 |
|------|------|------|
| Stage 1 | `train_stage1.py` | 기본 contrastive learning |
| Mining | `hard_negative_mining.py` | Hard negative 채굴 |
| Stage 2 | `train_stage2.py` | Hard negative 반영 강화 학습 |
| Stage 3 | `train_stage3.py` | 추가 정제 |
| Stage 3.5 | `train_stage3_5.py`, `train_stage3_5_clean.py` | 중간 정제 |
| Stage 4 | `train_stage4.py` | **현재 production 모델 (bi-encoder)** |
| Stage 5 | `train_stage5.py` | 추가 실험 |
| Stage 6 | `train_stage6.py` | 최신 실험 |

### 핵심 기법
- Contrastive learning (anchor / positive / negative)
- Hard negative mining
- License-weighted loss (copyleft 강화)

---

## Serving

### Split Architecture (GIL 우회)

```
[Client]
   ↓ /v1/embeddings  (OpenAI 호환, batch ≤ 64)
[FastAPI]
   ├── DFG Preprocessing  ← ProcessPoolExecutor (32 workers)
   └── Transformer        ← TensorRT FP16
   ↓
768-dim L2-normalized vector
```

### Benchmark (RTX 3090, TensorRT FP16)

| 메트릭 | 값 |
|--------|-----|
| Peak throughput | 365 items/sec (16 concurrent) |
| Median latency (p50) | 328ms @ 16 concurrent |
| Single request | 100~200ms |

> ※ 위 수치는 RTX 3090 벤치 환경 기준이며 실서비스 하드웨어에 따라 달라질 수 있음.

---

## Model Performance (Bi-encoder Standalone)

| 메트릭 | 값 |
|--------|-----|
| Positive 평균 유사도 | 0.9605 (±0.0640) |
| Negative 평균 유사도 | 0.0328 (±0.0830) |
| Precision (>0.5 threshold) | 0.9986 |
| False Positive Rate | 0.0005 |

### 한계
변수명 100% 치환 시 precision은 **0.166**까지 하락. 표면 변형엔 강하나 극단적 변형엔 한계가 있음.

---

## Production Pipeline Evaluation

### 평가 목적
기존엔 bi-encoder 단독 성능만 측정했으나, 실서비스 경로는 **Retrieval(hybrid) → license_review reranker → 사용자 노출**로 이뤄진다. End-to-end production 파이프라인을 처음으로 정량 평가했다.

### 테스트셋 (`test-suite/judged/`)

| 파일 | 카테고리 | 사용 개수 | 의미 |
|------|---------|---------|------|
| `TP_final100.jsonl` | True Positive | 64 | 표면 변형 (rename 등) |
| `FN_final100.jsonl` | False Negative 후보 | 100 | 의미 동등 + 구조 다름 (어려운 양성) |
| `FP_final100.jsonl` | False Positive 후보 | 100 | structural twin (보이는 건 비슷, 실제 무관) |
| `TN_final100.jsonl` | True Negative | 100 | 완전 무관 |

**총 364 페어** (TP는 품질 기준 적용 후 64개).

**큐레이션 워크플로 (3단계)**
1. 휴리스틱 필터로 후보 풀 생성
2. LLM Judge(Claude)로 카테고리 적합성 검증
3. 휴먼 체크

### 평가 파이프라인 (`production_eval_pipeline.py`)

6 phase, 중간 상태 저장(resumable):

| Phase | 역할 |
|-------|------|
| 0 | OpenSearch에서 anchor를 **raw_code content 기반**으로 매칭 → manifest 생성 |
| 1 | Sanity check (positive control, 응답 구조 검증) |
| 2 | Eval repo 빌드 (code_b 파일화) |
| 3 | GitHub push (또는 수동) |
| 4 | Production API 호출 (`POST /retrieve/hybrid/by-repo-url`) |
| 5 | 메트릭 산출 (Loose/Strict accuracy, F1, 카테고리 분해) |

### 신뢰도 검증 (`verify_eval_pipeline.py`)
7가지 sanity check로 평가 자체의 신뢰성 입증.

---

## Iteration & Results

| 버전 | 변경 사항 | Strict Acc | F1 |
|------|----------|-----------|-----|
| v1 | rule_based retrieval only | 0.866 | 0.809 |
| v2 | + kNN(embedding) 활성화 | 0.913 | 0.888 |
| v3 | + license_review cap 3종 (medium ceiling) | 0.909 | 0.882 |
| **v4** | **+ cap 엄격화 (low ceiling, 조건 강화)** | **0.926** | **0.896** |

### v4 최종 Cap
- **Cap #1**: kNN-only + `embedding_knn_match` + `call_overlap=0` + `identifier_overlap<3` → `low`
- **Cap #2**: rule+knn + `anonymized_code_match` + (`call_overlap<3` OR `domain_conflict`) → `low`
- **Cap #3**: structural twin + no call_overlap + weak domain → `low`

### v4 결과

**종합**
```
Strict Accuracy: 0.9264
F1 Score:        0.8957
```

**카테고리별**
| 카테고리 | 정확도 |
|---------|-------|
| TP | 100% |
| TN | 100% |
| FP | 94.7% (남은 false alarm 3건) |
| FN | 77.4% (의미 동등 변형 14건 누락) |

**Group별**
| Group | 개수 | 정확도 |
|-------|-----|-------|
| Ready (embedding 정상) | 213 | 0.944 |
| Rule-only (embedding 누락) | 18 | 0.722 |

**Retrieval Source 분포**
- kNN + rule_based: 55건
- rule_based only: 17건
- kNN only: 4건

---

## Insights

1. **Production end-to-end = bi-encoder 단독의 ~96% 수준** (gap 4.1pp).
2. **FP 11 → 3건**으로 false alarm 위험 대폭 축소.
3. **Heuristic tuning은 plateau** 도달 — 추가 개선은 모델/reranker 레이어 영역.
4. 평가 신뢰성을 별도 verify 스크립트로 입증.

---

## Next Steps

| 옵션 | 효과 | 비용 | 권장도 |
|-----|------|-----|-------|
| **A. v4 production 배포 + 운영 로그 수집** | 즉시 가치, 실분포 파악 | 낮음 | ★★★ |
| B. Cross-encoder reranker 도입 | FP borderline 해결 | Latency↑ | ★★ |
| C. 모델 재학습 (hard negative + 도메인 다양성) | FN 근본 개선 | 사이클 김 | ★★ |

권장: **A 선행** → 운영 데이터로 B/C 중 ROI 높은 쪽 결정.

---

## API

OpenAI 호환 `/v1/embeddings` 엔드포인트.

- Input: code string (또는 batch ≤ 64)
- Output: 768-dim L2-normalized vector

Production retrieval API:
```
POST /retrieve/hybrid/by-repo-url
{
  "repo_url": "https://github.com/owner/repo",
  "include_same_repo": false,
  "merged_top_k": 100
}
```

---