"""
verify_eval_pipeline.py — Production eval 결과의 신뢰성 검증

검증 항목:
  1. Coverage: 보낸 290 → 261 차이가 어떻게 생긴 건지
  2. case_id 매핑: 모든 finding이 정확한 case_id로 매핑됐는지
  3. Expected anchor 매칭 로직: 정말 같은 chunk를 찾은 건지
  4. CASE_MARKER 보존: chunker가 주석을 제거했는지
  5. Spot check: 랜덤 5 케이스의 수동 검증 데이터 출력
  6. False positive 검증: FP라고 판정한 케이스가 진짜 FP인지
  7. False negative 검증: FN이라고 판정한 케이스가 진짜 FN인지
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def normalize_repo(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.startswith("https://github.com/"):
        s = "github:" + s[len("https://github.com/"):]
    return s.lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    manifest = load_jsonl(workdir / "manifest_final.jsonl")
    eval_manifest = load_jsonl(workdir / "eval_repo" / "manifest.jsonl")
    api_response = json.loads((workdir / "api_response.json").read_text(encoding="utf-8"))
    predictions = load_jsonl(workdir / "predictions.jsonl")
    findings = api_response.get("findings", [])

    print("=" * 70)
    print("Eval Pipeline 신뢰성 검증")
    print("=" * 70)

    # ─── 검증 1: Coverage ───────────────────────────────────────────
    print("\n[1] Coverage (input → output)")
    input_count = len(eval_manifest)
    analyzed = api_response.get("analyzed_source_count", 0)
    findings_count = len(findings)
    pred_count = len(predictions)
    print(f"  eval_repo manifest 입력:         {input_count}")
    print(f"  API analyzed_source_count:       {analyzed}")
    print(f"  API findings 개수:               {findings_count}")
    print(f"  predictions.jsonl 매칭된 페어:   {pred_count}")

    lost_to_analyze = input_count - analyzed
    lost_to_findings = analyzed - findings_count
    lost_to_pred = findings_count - pred_count
    print(f"\n  손실:")
    print(f"    eval_manifest → analyzed:  {lost_to_analyze}  (chunker skip 가능성: 너무 짧음 등)")
    print(f"    analyzed → findings:       {lost_to_findings}  (candidates 0개인 chunk)")
    print(f"    findings → predictions:    {lost_to_pred}  (case_id 매핑 실패)")

    # 어떤 case_id가 input에 있는데 predictions에 없나
    input_case_ids = {m["case_id"] for m in eval_manifest}
    pred_case_ids = {p["case_id"] for p in predictions}
    missing_in_pred = input_case_ids - pred_case_ids
    if missing_in_pred:
        print(f"\n  predictions에 없는 case_id ({len(missing_in_pred)}개):")
        for cid in list(missing_in_pred)[:15]:
            print(f"    {cid}")
        if len(missing_in_pred) > 15:
            print(f"    ... 외 {len(missing_in_pred) - 15}개")

        # 카테고리별 분포
        cat_missing = Counter(c.split("-")[0] for c in missing_in_pred)
        print(f"\n  누락 카테고리별: {dict(cat_missing)}")

    # ─── 검증 2: case_id 매핑 정확성 ────────────────────────────────
    print("\n[2] case_id 매핑 검증")
    by_file = {m["file_path"]: m for m in eval_manifest}

    # findings 직접 파싱해서 우리 로직대로 case_id 추출 검증
    case_id_extraction = {
        "marker_extracted": 0,
        "file_path_fallback": 0,
        "no_case_id": 0,
        "marker_mismatch": 0,    # marker 있는데 file_path와 다름
    }
    findings_with_marker = []
    findings_no_marker = []

    for f in findings:
        src = f.get("source", {})
        src_path = src.get("file_path", "")
        src_code = src.get("raw_code", "")
        m = re.search(r"CASE_MARKER:\s*([A-Z]+-\d+)", src_code)
        marker_cid = m.group(1) if m else None

        # file_path 기반 추출 (queries/TP/001.py → TP-001)
        # 실제 file_path는 eval_repo 내 path임 (e.g., TP/001.py 또는 pairs/TP/001.py)
        path_cid = None
        # 마지막 두 segment에서 TP/001 같은 패턴 찾기
        path_match = re.search(r"([A-Z]+)/(\d+)\.[a-z]+$", src_path)
        if path_match:
            path_cid = f"{path_match.group(1)}-{int(path_match.group(2)):03d}"

        if marker_cid:
            findings_with_marker.append(f)
            case_id_extraction["marker_extracted"] += 1
            if path_cid and path_cid != marker_cid:
                case_id_extraction["marker_mismatch"] += 1
                print(f"  ⚠ marker/path 불일치: marker={marker_cid}, path={path_cid}, file={src_path}")
        elif path_cid:
            findings_no_marker.append(f)
            case_id_extraction["file_path_fallback"] += 1
        else:
            findings_no_marker.append(f)
            case_id_extraction["no_case_id"] += 1

    for k, v in case_id_extraction.items():
        print(f"  {k:<25} {v}")

    if case_id_extraction["no_case_id"] > 0:
        print(f"\n  ⚠ case_id 추출 실패 sample (file_path):")
        for f in findings_no_marker[:5]:
            print(f"    {f.get('source', {}).get('file_path')!r}")

    # CASE_MARKER가 보존됐는지 = chunker가 주석 살렸는지
    marker_preserved = case_id_extraction["marker_extracted"] / max(1, len(findings))
    print(f"\n  CASE_MARKER 보존율: {marker_preserved*100:.1f}%")
    if marker_preserved < 0.9:
        print(f"  ⚠ chunker가 CASE_MARKER 주석을 자주 제거함 → file_path 매칭에 의존 중")
    else:
        print(f"  ✓ CASE_MARKER 잘 보존됨")

    # ─── 검증 3: Expected anchor 매칭 로직 ──────────────────────────
    print("\n[3] Expected anchor 매칭 검증")
    print("  expected_found=True 케이스에서 진짜 같은 chunk를 찾았나?")

    # predictions에서 expected_found=True인 케이스
    found_preds = [p for p in predictions if p["expected_found"]]
    print(f"  expected_found=True: {len(found_preds)}건")

    # 매칭 방식별 비율
    repo_path_match = 0
    for p in found_preds[:30]:  # 검증용 30개만
        case_id = p["case_id"]
        em = next((m for m in eval_manifest if m["case_id"] == case_id), None)
        if not em:
            continue
        # finding 찾기
        f_match = None
        for f in findings:
            src_code = f.get("source", {}).get("raw_code", "")
            m = re.search(r"CASE_MARKER:\s*([A-Z]+-\d+)", src_code)
            if m and m.group(1) == case_id:
                f_match = f
                break
        if f_match is None:
            continue
        # candidates에서 expected_repo + expected_path 매칭
        expected_repo_norm = normalize_repo(em.get("expected_repo_id", ""))
        for cand in f_match.get("candidates", []):
            cand_repo_norm = normalize_repo((cand.get("repository") or {}).get("repo_url", ""))
            cand_path = (cand.get("location") or {}).get("file_path", "")
            if cand_repo_norm == expected_repo_norm and cand_path == em.get("expected_file_path"):
                repo_path_match += 1
                break

    print(f"  Sample 30개 중 (repo + file_path) 정확 매칭: {repo_path_match}/30")

    # ─── 검증 4: Spot check 5개 ─────────────────────────────────────
    print("\n[4] Spot check: 랜덤 5개 수동 검증 데이터")

    # 1개 TP, 1개 FN, 1개 FP, 1개 TN, 1개 worst FN
    samples = []
    pred_by_cat = defaultdict(list)
    for p in predictions:
        pred_by_cat[p["category"]].append(p)
    for cat in ["TP", "FN", "FP", "TN"]:
        if pred_by_cat[cat]:
            samples.append(pred_by_cat[cat][0])
    # worst FN: label=1인데 expected_found=False 중 top_risk가 high인 것
    worst_fn = [p for p in predictions
                if p["label"] == 1 and not p["expected_found"] and p["top_risk_level"] == "high"]
    if worst_fn:
        samples.append(worst_fn[0])

    for sample in samples:
        case_id = sample["case_id"]
        em = next((m for m in eval_manifest if m["case_id"] == case_id), None)
        if not em:
            continue
        print(f"\n  ─── {case_id} (label={sample['label']}, cat={sample['category']}) ───")
        print(f"    eval_manifest:")
        print(f"      expected_repo_id:    {em.get('expected_repo_id')}")
        print(f"      expected_symbol:     {em.get('expected_symbol_name')}")
        print(f"      expected_file_path:  {em.get('expected_file_path')}")
        print(f"    predictions:")
        print(f"      top_risk_level:      {sample['top_risk_level']}")
        print(f"      expected_found:      {sample['expected_found']}")
        print(f"      expected_risk_level: {sample.get('expected_risk_level')}")
        print(f"      candidate_count:     {sample['candidate_count']}")
        print(f"      matched_by:          {sample.get('expected_matched_by')}")

        # finding 찾아서 후보 상위 3개 보여주기
        for f in findings:
            src_code = f.get("source", {}).get("raw_code", "")
            m = re.search(r"CASE_MARKER:\s*([A-Z]+-\d+)", src_code)
            if not m or m.group(1) != case_id:
                continue
            cands = f.get("candidates", [])[:3]
            print(f"    상위 3 candidates:")
            for i, c in enumerate(cands, 1):
                rl = (c.get("review_priority") or {}).get("level")
                rs = (c.get("review_priority") or {}).get("score")
                cr = (c.get("repository") or {}).get("repo_url", "")[:50]
                ch = c.get("matched_chunk", {}).get("raw_code", "")[:60].replace("\n", " ")
                mb = c.get("matched_by")
                print(f"      [{i}] {rl} ({rs}) {mb} repo={cr}")
                print(f"          chunk: {ch}...")
            break

    # ─── 검증 5: matched_by 분포 sanity ──────────────────────────────
    print("\n[5] matched_by 분포 sanity")
    print("  bi-encoder 평가에서 96.7% 정확도였는데 production에서 kNN이 0건이라는 게 정상인가?")

    # 모든 findings의 모든 candidates 중 matched_by 분포
    all_matched_by = Counter()
    for f in findings:
        for c in f.get("candidates", []):
            mb = tuple(sorted(c.get("matched_by", [])))
            all_matched_by[mb] += 1

    print(f"\n  전체 candidates의 matched_by 분포:")
    for mb, count in all_matched_by.most_common():
        print(f"    {','.join(mb) if mb else '(empty)':<30} {count}")

    knn_total = sum(c for mb, c in all_matched_by.items() if "knn" in mb)
    print(f"\n  kNN으로 잡힌 candidate 총: {knn_total}건")
    if knn_total == 0:
        print(f"  ⚠ kNN이 전혀 작동 안 함 → API 또는 OS embedding_index 문제 의심")
    elif knn_total < 10:
        print(f"  ⚠ kNN이 거의 작동 안 함")

    # ─── 검증 6: Risk level 분포 ─────────────────────────────────────
    print("\n[6] Risk level 분포 sanity")
    rl_dist = Counter()
    for f in findings:
        rl = (f.get("top_review_priority") or {}).get("level", "low")
        rl_dist[rl] += 1
    print(f"  top_review_priority 분포 (전체 findings):")
    for k, v in rl_dist.most_common():
        print(f"    {k:<10} {v} ({v/max(1,len(findings))*100:.1f}%)")

    # ─── 검증 7: Worst FN/FP 깊이 분석 ──────────────────────────────
    print("\n[7] Worst FN/FP 패턴 검증")
    # Worst FN: label=1인데 expected_found=False
    worst_fns = [p for p in predictions if p["label"] == 1 and not p["expected_found"]]
    print(f"  Worst FN: {len(worst_fns)}건")
    # top_risk가 high면서 expected_found=False인 케이스의 의미:
    #   "production이 뭔가 high로 찾았는데 그게 우리 expected anchor는 아님"
    #   → 같은 코드가 OS의 다른 chunk에 있고 그게 더 가깝게 잡힌 것일 수도
    top_high_no_expected = [p for p in worst_fns if p["top_risk_level"] == "high"]
    print(f"  그 중 top_risk=high이지만 expected 미발견: {len(top_high_no_expected)}")
    print(f"  → 'production이 다른 거 잡음' 시나리오")

    # ─── 종합 ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("검증 종합")
    print("=" * 70)

    issues = []
    if lost_to_pred > 5:
        issues.append(f"predictions 손실 {lost_to_pred}건 (case_id 매핑 실패 가능)")
    if case_id_extraction["no_case_id"] > 0:
        issues.append(f"case_id 추출 실패 {case_id_extraction['no_case_id']}건")
    if marker_preserved < 0.9:
        issues.append(f"CASE_MARKER 보존 {marker_preserved*100:.1f}% (낮음)")
    if knn_total == 0:
        issues.append("kNN 0건 (의심)")
    if repo_path_match < 25:
        issues.append(f"sample expected 매칭 sanity: {repo_path_match}/30")

    if not issues:
        print("✅ 모든 검증 항목 정상 — 평가 신뢰 가능")
    else:
        print("⚠ 발견된 의심 항목:")
        for iss in issues:
            print(f"   - {iss}")


if __name__ == "__main__":
    main()
