"""
check_anchor_match.py — anchor가 OS chunk index에 있는지 확인

매칭 조건 (둘 다 만족):
  1. raw_code.strip() byte-for-byte 일치
  2. normalize(repo_id) 일치
     - `https://github.com/owner/repo` ↔ `github:owner/repo` 동일 처리
     - 대소문자 무시 (`BettaFish` ↔ `bettafish` 동일 처리)

레포 변화만 normalize, 코드는 변형 없음 (수집 스크립트와 동일하게 .strip()만).

사용:
  python check_anchor_match.py \\
    --testdir /home/shinmk/code-killr/test-suite/judged \\
    --os-host http://localhost:9200 \\
    --output /home/shinmk/code-killr/test-suite/eval_results/manifest_match.jsonl
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import requests
from tqdm import tqdm


def normalize_repo(s: str) -> str:
    """레포 ID 정규화: prefix 통일 + 소문자화."""
    if not s:
        return ""
    s = s.strip()
    if s.startswith("https://github.com/"):
        s = "github:" + s[len("https://github.com/"):]
    elif s.startswith("http://github.com/"):
        s = "github:" + s[len("http://github.com/"):]
    return s.lower()


def query_os(host: str, index: str, body: dict, timeout: int = 30):
    r = requests.post(
        f"{host}/{index}/_search",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def search_by_content(host: str, chunk_index: str, anchor_code: str,
                      language: str, use_filters: bool = True) -> list:
    """anchor의 raw_code로 match_phrase 검색."""
    lines = anchor_code.strip().split("\n")
    if len(lines) >= 3:
        sample = "\n".join(lines[:3])
    elif lines:
        sample = lines[0]
    else:
        return []

    must = [{"match_phrase": {"raw_code": sample}}]
    filters = []
    if use_filters:
        if language:
            filters.append({"term": {"language": language}})
        filters.append({"term": {"chunk_type": "function"}})

    body = {
        "size": 30,
        "_source": ["chunk_id", "repo_id", "symbol_name", "file_path",
                    "language", "chunk_type", "raw_code"],
        "query": {"bool": {"must": must, "filter": filters}},
    }
    return query_os(host, chunk_index, body).get("hits", {}).get("hits", [])


def find_match(hits: list, anchor_code: str, anchor_repo_norm: str) -> tuple:
    """hits에서 (code match + repo match) 둘 다 만족하는 hit 찾기.

    Returns:
        (hit, match_kind)
        match_kind:
          - "code+repo": 둘 다 일치
          - "code_only": 코드만 일치, repo 다름
          - None: 아무것도 없음
    """
    a = anchor_code.strip()
    code_only_hit = None

    for hit in hits:
        src = hit.get("_source", {})
        found_code = src.get("raw_code", "")
        found_repo = src.get("repo_id", "")

        if found_code.strip() != a:
            continue  # 코드부터 안 맞으면 skip

        # 코드는 맞음. repo도 보자
        if normalize_repo(found_repo) == anchor_repo_norm:
            return hit, "code+repo"
        if code_only_hit is None:
            code_only_hit = hit

    if code_only_hit is not None:
        return code_only_hit, "code_only"
    return None, None


def load_all_pairs(testdir: Path) -> list[dict]:
    pairs = []
    for cat in ["TP", "FN", "FP", "TN"]:
        p = testdir / f"{cat}_final100.jsonl"
        if not p.exists():
            print(f"  ⚠ {p.name} 없음", file=sys.stderr)
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            d = json.loads(line)
            d["__case_id"] = f"{cat}-{i:03d}"
            d["__category"] = cat
            d["__seq"] = i
            pairs.append(d)
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testdir", required=True)
    ap.add_argument("--os-host", default="http://localhost:9200")
    ap.add_argument("--chunk-index", default="code_chunk_index")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pairs = load_all_pairs(Path(args.testdir))
    print(f"테스트 페어: {len(pairs)}개")
    print(f"OS host: {args.os_host}")
    print(f"매칭: raw_code byte-match + repo_id normalize match\n")

    manifest = []
    status_counts = Counter()

    for p in tqdm(pairs, desc="OS 매칭 확인"):
        case_id = p["__case_id"]
        anchor_code = p.get("anchor") or p.get("code", "") or ""
        anchor_repo = p.get("repo", "")
        anchor_function = p.get("function", "")
        language = p.get("language", "")
        pair_code = p.get("positive") or p.get("negative", "")

        anchor_repo_norm = normalize_repo(anchor_repo)

        record = {
            "case_id": case_id,
            "category": p["__category"],
            "language": language,
            "anchor_repo_meta": anchor_repo,
            "anchor_repo_norm": anchor_repo_norm,
            "anchor_function_meta": anchor_function,
            "chunk_indexed": False,
            "anchor_chunk_id": None,
            "found_repo_id": None,
            "found_symbol_name": None,
            "found_file_path": None,
            "fallback_used": False,
            "match_kind": None,
            "pair_code": pair_code,
            "status": "unknown",
        }

        if not anchor_code:
            record["status"] = "missing_anchor_code"
            status_counts["missing_anchor_code"] += 1
            manifest.append(record)
            continue

        # 1) 기본 검색 (filter 적용)
        try:
            hits = search_by_content(args.os_host, args.chunk_index,
                                     anchor_code, language, use_filters=True)
        except Exception as e:
            record["status"] = f"search_error: {type(e).__name__}"
            status_counts["search_error"] += 1
            manifest.append(record)
            continue

        hit, kind = find_match(hits, anchor_code, anchor_repo_norm)

        # 2) 못 찾으면 fallback (filter 없이)
        if hit is None:
            try:
                hits_fb = search_by_content(args.os_host, args.chunk_index,
                                            anchor_code, language, use_filters=False)
                hit, kind = find_match(hits_fb, anchor_code, anchor_repo_norm)
                if hit is not None:
                    record["fallback_used"] = True
            except Exception:
                pass

        if hit is None:
            record["status"] = "not_in_os"
            status_counts["not_in_os"] += 1
            manifest.append(record)
            continue

        # 3) 매칭 성공 (code+repo or code_only)
        src = hit.get("_source", {})
        chunk_id = hit.get("_id") or src.get("chunk_id")
        record["chunk_indexed"] = True
        record["anchor_chunk_id"] = chunk_id
        record["found_repo_id"] = src.get("repo_id")
        record["found_symbol_name"] = src.get("symbol_name")
        record["found_file_path"] = src.get("file_path")
        record["match_kind"] = kind
        record["status"] = "matched" if kind == "code+repo" else "code_only_repo_diff"
        status_counts[record["status"]] += 1

        manifest.append(record)

    # 저장
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # ── 요약 ──
    print("\n" + "=" * 70)
    print("OS chunk index 매칭 결과 (raw_code byte-match + repo_id normalize)")
    print("=" * 70)
    print(f"전체: {len(manifest)}\n")
    for status, count in status_counts.most_common():
        pct = count / len(manifest) * 100
        bar = "█" * int(pct / 2)
        print(f"  {status:<25} {count:>4} ({pct:>5.1f}%) {bar}")

    # 카테고리별
    print("\n" + "-" * 70)
    print("카테고리별 매칭률 (code+repo 둘 다 일치 기준)")
    print("-" * 70)
    by_cat = defaultdict(lambda: defaultdict(int))
    for m in manifest:
        by_cat[m["category"]]["total"] += 1
        by_cat[m["category"]][m["status"]] += 1

    for cat in ["TP", "FN", "FP", "TN"]:
        v = by_cat.get(cat, {})
        if not v:
            continue
        total = v["total"]
        matched = v.get("matched", 0)
        code_only = v.get("code_only_repo_diff", 0)
        not_in = v.get("not_in_os", 0)
        pct = matched / total * 100 if total else 0
        print(f"  {cat:<5} {matched:>3}/{total:<3} = {pct:>5.1f}%  "
              f"(code_only_repo_diff: {code_only}, not_in_os: {not_in})")

    # Fallback 사용 케이스
    fallback_count = sum(1 for m in manifest if m["fallback_used"])
    if fallback_count:
        print(f"\n  Fallback (filter 제거) 사용: {fallback_count}건")

    # 메타데이터 변화 (matched만)
    matched_recs = [m for m in manifest if m["status"] == "matched"]
    if matched_recs:
        # repo가 다른 표기지만 같은 경우
        repo_exact = sum(1 for m in matched_recs
                         if m["anchor_repo_meta"] == m["found_repo_id"])
        repo_normalized = sum(1 for m in matched_recs
                              if m["anchor_repo_meta"] != m["found_repo_id"])
        print("\n" + "-" * 70)
        print("Matched 케이스 중 repo 표기 차이")
        print("-" * 70)
        print(f"  완전 동일:        {repo_exact}/{len(matched_recs)}")
        print(f"  normalize로 매칭: {repo_normalized}/{len(matched_recs)}")

    # code_only sample (코드는 같은데 repo가 다른 케이스)
    code_only_recs = [m for m in manifest if m["status"] == "code_only_repo_diff"]
    if code_only_recs:
        print("\n" + "-" * 70)
        print(f"code_only_repo_diff sample ({len(code_only_recs)}건 중)")
        print("-" * 70)
        for m in code_only_recs[:5]:
            print(f"  [{m['case_id']}] {m['anchor_repo_meta']}")
            print(f"           → 코드는 같은데 OS의 다른 repo에서 발견: {m['found_repo_id']}")

    total = len(manifest)
    matched = status_counts.get("matched", 0)
    print("\n" + "=" * 70)
    print(f"→ Manifest 저장: {out_path}")
    print(f"\n진짜 매칭 (code+repo): {matched}/{total} = {matched/total*100:.1f}%")
    print(f"코드만 일치 (다른 repo): {status_counts.get('code_only_repo_diff', 0)}/{total}")
    print(f"OS에 없음: {status_counts.get('not_in_os', 0)}/{total}")


if __name__ == "__main__":
    main()
