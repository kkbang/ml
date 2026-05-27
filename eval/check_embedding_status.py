"""
check_embedding_status.py — 매칭된 chunk가 embedding index에도 있는지 확인

입력: manifest_match.jsonl  (check_anchor_match.py 결과)
출력: manifest_final.jsonl  (embedding_indexed 추가)

흐름:
  1. manifest 로드
  2. status가 matched 또는 code_only_repo_diff인 케이스만 (anchor_chunk_id 존재)
  3. 각 chunk_id가 embedding_index에 있는지 확인
  4. embedding_indexed 필드 추가하고 저장

사용:
  python check_embedding_status.py \\
    --input  /home/shinmk/code-killr/test-suite/eval_results/manifest_match.jsonl \\
    --output /home/shinmk/code-killr/test-suite/eval_results/manifest_final.jsonl \\
    --os-host http://localhost:9200
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm


def check_embedding(host: str, embedding_index: str, chunk_id: str) -> bool:
    if not chunk_id:
        return False
    body = {
        "size": 1,
        "_source": False,
        "query": {"term": {"_id": chunk_id}},
    }
    try:
        r = requests.post(
            f"{host}/{embedding_index}/_search",
            json=body,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        if r.status_code == 404:
            return False
        r.raise_for_status()
        total = r.json().get("hits", {}).get("total", {})
        if isinstance(total, dict):
            return total.get("value", 0) > 0
        return total > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="manifest_match.jsonl")
    ap.add_argument("--output", required=True, help="manifest_final.jsonl")
    ap.add_argument("--os-host", default="http://localhost:9200")
    ap.add_argument("--embedding-index", default="code_chunk_embedding_index_v1")
    args = ap.parse_args()

    in_path = Path(args.input)
    records = [json.loads(l) for l in in_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"전체 케이스: {len(records)}")

    # 매칭된 것만 (chunk_id 있는 것)
    has_chunk = [r for r in records if r.get("anchor_chunk_id")]
    print(f"chunk index 매칭된 케이스: {len(has_chunk)}")
    print(f"embedding 확인 시작...\n")

    # 각 케이스 embedding 체크
    for r in tqdm(records, desc="embedding 확인"):
        chunk_id = r.get("anchor_chunk_id")
        if not chunk_id:
            r["embedding_indexed"] = False
            continue
        r["embedding_indexed"] = check_embedding(
            args.os_host, args.embedding_index, chunk_id
        )

    # status 업데이트
    for r in records:
        prev = r.get("status")
        if prev in ("matched", "code_only_repo_diff"):
            if r["embedding_indexed"]:
                r["eval_status"] = "ready"           # chunk + embedding 둘 다 OK
            else:
                r["eval_status"] = "rule_based_only"  # chunk만, embedding 없음
        else:
            r["eval_status"] = prev  # not_in_os, search_error 등 그대로

    # 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 요약 ──
    print("\n" + "=" * 70)
    print("최종 평가 가능성 분포")
    print("=" * 70)
    eval_counts = Counter(r["eval_status"] for r in records)
    print(f"전체: {len(records)}\n")
    for status, count in eval_counts.most_common():
        pct = count / len(records) * 100
        bar = "█" * int(pct / 2)
        print(f"  {status:<25} {count:>4} ({pct:>5.1f}%) {bar}")

    # 카테고리별
    print("\n" + "-" * 70)
    print("카테고리별 평가 가능성")
    print("-" * 70)
    print(f"  {'Cat':<5} {'Total':>6} {'Ready':>6} {'+rule_only':>11} {'합계':>6} {'%':>6}")
    by_cat = defaultdict(lambda: defaultdict(int))
    for r in records:
        by_cat[r["category"]]["total"] += 1
        by_cat[r["category"]][r["eval_status"]] += 1

    for cat in ["TP", "FN", "FP", "TN"]:
        v = by_cat.get(cat, {})
        if not v:
            continue
        total = v["total"]
        ready = v.get("ready", 0)
        rule_only = v.get("rule_based_only", 0)
        usable = ready + rule_only
        pct = usable / total * 100 if total else 0
        print(f"  {cat:<5} {total:>6} {ready:>6} {rule_only:>11} {usable:>6} {pct:>5.1f}%")

    # 평가 가능 케이스 = ready + rule_based_only
    usable = eval_counts.get("ready", 0) + eval_counts.get("rule_based_only", 0)
    print("\n" + "=" * 70)
    print(f"→ Manifest 저장: {out_path}")
    print(f"\n  Ready (chunk + embedding):     {eval_counts.get('ready', 0)}/{len(records)}")
    print(f"  Rule-based only (no embedding): {eval_counts.get('rule_based_only', 0)}/{len(records)}")
    print(f"  ──")
    print(f"  평가 가능 (합계):                {usable}/{len(records)} = {usable/len(records)*100:.1f}%")
    print(f"  평가 불가 (not_in_os 등):       {len(records) - usable}/{len(records)}")

    print("\n다음 단계:")
    print("  - Ready 케이스: bi-encoder + rule-based 둘 다 평가 가능")
    print("  - Rule-based only 케이스: kNN 안 함, rule_based만 사용")
    print("  - Phase 1 (sanity checks) 진행")


if __name__ == "__main__":
    main()
