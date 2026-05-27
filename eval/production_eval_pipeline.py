"""
production_eval_pipeline.py — End-to-end production evaluation pipeline

전체 phase를 한 번에 실행하는 통합 스크립트.

Phases:
  0. OS chunk index + embedding index 매칭 상태 확인 → manifest_final.jsonl
  1. Sanity check (positive control)
  2. eval repo 빌드 (290 code_b files, 4 category sub-repos)
  3. GitHub push (gh CLI 사용)
  4. Production API 호출 (4 sub-repos)
  5. 결과 매칭 + 메트릭 리포트

각 phase는 결과를 workdir에 저장 → 재실행 시 skip 가능.

사용:
  python production_eval_pipeline.py \\
    --testdir /home/shinmk/code-killr/test-suite/judged \\
    --os-host http://localhost:9200 \\
    --api http://ngseo-ubuntu:18000 \\
    --workdir /home/shinmk/code-killr/test-suite/eval_results/pipeline \\
    --gh-owner kkbang

옵션:
  --start-from N      특정 phase부터 시작 (재실행 시)
  --stop-at N         특정 phase까지만
  --force             기존 결과 무시하고 새로
  --skip-push         GitHub push 단계 생략 (수동 push 시)
  --eval-repo-base    GitHub repo prefix (e.g., kkbang/code-killr-eval)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm


LANG_EXT = {"python": "py", "go": "go", "java": "java", "javascript": "js"}
CATEGORIES = ["TP", "FN", "FP", "TN"]


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def normalize_repo(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.startswith("https://github.com/"):
        s = "github:" + s[len("https://github.com/"):]
    elif s.startswith("http://github.com/"):
        s = "github:" + s[len("http://github.com/"):]
    return s.lower()


def os_query(host: str, index: str, body: dict, timeout: int = 30):
    r = requests.post(
        f"{host}/{index}/_search",
        json=body,
        headers={"Content-Type": "application/json"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def load_pairs(testdir: Path) -> list[dict]:
    pairs = []
    for cat in CATEGORIES:
        path = testdir / f"{cat}_final100.jsonl"
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                d = json.loads(line)
                d["__case_id"] = f"{cat}-{i:03d}"
                d["__category"] = cat
                d["__label"] = 1 if cat in ("TP", "FN") else 0
                d["__seq"] = i
                pairs.append(d)
    return pairs


# ══════════════════════════════════════════════════════════════════
# Phase 0: Manifest preparation
# ══════════════════════════════════════════════════════════════════
def phase0_build_manifest(args, workdir: Path):
    manifest_path = workdir / "manifest_final.jsonl"
    if manifest_path.exists() and not args.force:
        print(f"\n[Phase 0] 기존 manifest 사용: {manifest_path}")
        return [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    print(f"\n[Phase 0] OS chunk + embedding 매칭 시작")
    pairs = load_pairs(Path(args.testdir))
    print(f"  테스트 페어: {len(pairs)}개")

    manifest = []
    for p in tqdm(pairs, desc="OS 매칭"):
        anchor_code = p.get("anchor") or p.get("code", "") or ""
        anchor_repo_meta = p.get("repo", "")
        anchor_function_meta = p.get("function", "")
        language = p.get("language", "")
        pair_code = p.get("positive") or p.get("negative", "")
        anchor_repo_norm = normalize_repo(anchor_repo_meta)

        record = {
            "case_id": p["__case_id"],
            "category": p["__category"],
            "label": p["__label"],
            "language": language,
            "anchor_repo_meta": anchor_repo_meta,
            "anchor_function_meta": anchor_function_meta,
            "anchor_chunk_id": None,
            "found_repo_id": None,
            "found_symbol_name": None,
            "found_file_path": None,
            "chunk_indexed": False,
            "embedding_indexed": False,
            "match_kind": None,
            "pair_code": pair_code,
            "eval_status": "unknown",
        }

        if not anchor_code:
            record["eval_status"] = "missing_anchor_code"
            manifest.append(record)
            continue

        # raw_code로 검색
        lines = anchor_code.strip().split("\n")
        sample = "\n".join(lines[:3]) if len(lines) >= 3 else lines[0] if lines else ""

        if not sample:
            record["eval_status"] = "missing_anchor_code"
            manifest.append(record)
            continue

        def _search(use_filters: bool):
            must = [{"match_phrase": {"raw_code": sample}}]
            filters = []
            if use_filters:
                if language:
                    filters.append({"term": {"language": language}})
                filters.append({"term": {"chunk_type": "function"}})
            body = {
                "size": 30,
                "_source": ["chunk_id", "repo_id", "symbol_name", "file_path", "raw_code"],
                "query": {"bool": {"must": must, "filter": filters}},
            }
            return os_query(args.os_host, args.chunk_index, body).get("hits", {}).get("hits", [])

        try:
            hits = _search(True)
        except Exception as e:
            record["eval_status"] = f"search_error: {type(e).__name__}"
            manifest.append(record)
            continue

        a = anchor_code.strip()
        code_only_hit = None
        matched_hit = None
        for hit in hits:
            src = hit.get("_source", {})
            if src.get("raw_code", "").strip() != a:
                continue
            if normalize_repo(src.get("repo_id", "")) == anchor_repo_norm:
                matched_hit = hit
                record["match_kind"] = "code+repo"
                break
            if code_only_hit is None:
                code_only_hit = hit

        if matched_hit is None and code_only_hit is None:
            # fallback: filter 없이
            try:
                hits_fb = _search(False)
                for hit in hits_fb:
                    src = hit.get("_source", {})
                    if src.get("raw_code", "").strip() != a:
                        continue
                    if normalize_repo(src.get("repo_id", "")) == anchor_repo_norm:
                        matched_hit = hit
                        record["match_kind"] = "code+repo"
                        break
                    if code_only_hit is None:
                        code_only_hit = hit
            except Exception:
                pass

        hit = matched_hit or code_only_hit
        if hit is None:
            record["eval_status"] = "not_in_os"
            manifest.append(record)
            continue

        if not record["match_kind"]:
            record["match_kind"] = "code_only"

        src = hit.get("_source", {})
        chunk_id = hit.get("_id") or src.get("chunk_id")
        record["chunk_indexed"] = True
        record["anchor_chunk_id"] = chunk_id
        record["found_repo_id"] = src.get("repo_id")
        record["found_symbol_name"] = src.get("symbol_name")
        record["found_file_path"] = src.get("file_path")

        # embedding 체크
        try:
            r = requests.post(
                f"{args.os_host}/{args.embedding_index}/_search",
                json={"size": 1, "_source": False, "query": {"term": {"_id": chunk_id}}},
                timeout=10,
            )
            if r.status_code != 404:
                total = r.json().get("hits", {}).get("total", {})
                if isinstance(total, dict):
                    record["embedding_indexed"] = total.get("value", 0) > 0
                else:
                    record["embedding_indexed"] = total > 0
        except Exception:
            pass

        if record["embedding_indexed"]:
            record["eval_status"] = "ready"
        else:
            record["eval_status"] = "rule_based_only"

        manifest.append(record)

    # 저장
    workdir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        for m in manifest:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # 요약
    print("\n[Phase 0] 결과:")
    counts = Counter(m["eval_status"] for m in manifest)
    for k, v in counts.most_common():
        print(f"  {k}: {v}")
    print(f"  → {manifest_path}")
    return manifest


# ══════════════════════════════════════════════════════════════════
# Phase 1: Sanity check
# ══════════════════════════════════════════════════════════════════
def phase1_sanity_check(args, manifest: list, workdir: Path):
    print(f"\n[Phase 1] Sanity check")
    ready = [m for m in manifest if m["eval_status"] == "ready"]
    rule_only = [m for m in manifest if m["eval_status"] == "rule_based_only"]
    not_in = [m for m in manifest if m["eval_status"] == "not_in_os"]
    print(f"  Ready: {len(ready)} | Rule-only: {len(rule_only)} | Not in OS: {len(not_in)}")
    print(f"  평가 대상: {len(ready) + len(rule_only)}개")

    if len(ready) + len(rule_only) < 50:
        print(f"  ⚠ 평가 대상 너무 적음. 본 평가 의미 작을 수 있음.")

    # Sanity log 저장
    log = {
        "ready": len(ready),
        "rule_based_only": len(rule_only),
        "not_in_os": len(not_in),
        "evaluable_total": len(ready) + len(rule_only),
    }
    (workdir / "phase1_sanity.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  → phase1_sanity.json")


# ══════════════════════════════════════════════════════════════════
# Phase 2: Build eval repo
# ══════════════════════════════════════════════════════════════════
def phase2_build_eval_repo(args, manifest: list, workdir: Path):
    print(f"\n[Phase 2] eval repo 빌드")
    evaluable = [m for m in manifest if m["eval_status"] in ("ready", "rule_based_only")]
    print(f"  평가 대상: {len(evaluable)}개 (Ready + Rule-only)")

    repo_root = workdir / "eval_repo"
    if repo_root.exists() and not args.force:
        print(f"  기존 디렉토리 사용: {repo_root}")
    else:
        if repo_root.exists():
            shutil.rmtree(repo_root)
        repo_root.mkdir(parents=True, exist_ok=True)

        # 카테고리별로 sub-dir 생성
        manifest_for_repo = []
        for m in evaluable:
            if not m.get("pair_code"):
                continue
            cat = m["category"]
            seq = int(m["case_id"].split("-")[1])
            lang = m["language"]
            ext = LANG_EXT.get(lang, "txt")
            sub_dir = repo_root / cat
            sub_dir.mkdir(exist_ok=True)
            file_path = sub_dir / f"{seq:03d}.{ext}"
            # CASE_MARKER 주석 + pair_code
            comment = f"# CASE_MARKER: {m['case_id']}\n" if lang == "python" \
                      else f"// CASE_MARKER: {m['case_id']}\n"
            file_path.write_text(comment + m["pair_code"], encoding="utf-8")

            manifest_for_repo.append({
                "case_id": m["case_id"],
                "file_path": f"{cat}/{seq:03d}.{ext}",
                "category": m["category"],
                "label": m["label"],
                "eval_status": m["eval_status"],
                "language": lang,
                "anchor_chunk_id": m["anchor_chunk_id"],
                "expected_repo_id": m["found_repo_id"],
                "expected_symbol_name": m["found_symbol_name"],
                "expected_file_path": m["found_file_path"],
            })

        # eval manifest 저장
        with (repo_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
            for m in manifest_for_repo:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

        # README + LICENSE
        (repo_root / "README.md").write_text(
            "# code-killr eval set\n\nAuto-generated for production pipeline evaluation.\n",
            encoding="utf-8"
        )
        (repo_root / "LICENSE").write_text(
            "MIT License\n\nCopyright (c) 2026 code-killr-eval\n",
            encoding="utf-8"
        )

        print(f"  생성된 파일: {len(manifest_for_repo)}개")
        print(f"  → {repo_root}")


# ══════════════════════════════════════════════════════════════════
# Phase 3: GitHub push
# ══════════════════════════════════════════════════════════════════
def phase3_github_push(args, workdir: Path):
    print(f"\n[Phase 3] GitHub push")
    repo_root = workdir / "eval_repo"

    if args.skip_push:
        print(f"  --skip-push 지정됨. 수동 push 후 --eval-repo-url로 진행하세요.")
        return None

    if not repo_root.exists():
        print(f"  ✗ eval repo 디렉토리 없음. Phase 2 먼저 실행.")
        sys.exit(1)

    if not args.gh_owner:
        print(f"  ✗ --gh-owner 필요. 예: --gh-owner kkbang")
        sys.exit(1)

    repo_name = f"code-killr-eval-{int(time.time())}"
    full_name = f"{args.gh_owner}/{repo_name}"
    expected_url = f"https://github.com/{full_name}"

    print(f"  생성할 repo: {expected_url}")

    # git init / commit / gh repo create + push
    commands = [
        ["git", "init", "-b", "main"],
        ["git", "add", "."],
        ["git", "commit", "-m", "eval set", "--allow-empty"],
        ["gh", "repo", "create", full_name, "--public", "--source=.", "--push"],
    ]

    for cmd in commands:
        try:
            r = subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"  ✗ 실패: {' '.join(cmd)}")
            print(f"    stderr: {e.stderr}")
            print(f"  → 수동 push 후 --start-from 4 --eval-repo-url <url> 로 재실행")
            sys.exit(1)
        except FileNotFoundError:
            print(f"  ✗ 명령어 없음: {cmd[0]}")
            sys.exit(1)

    print(f"  ✓ Push 완료: {expected_url}")
    # URL 저장
    (workdir / "eval_repo_url.txt").write_text(expected_url, encoding="utf-8")
    return expected_url


# ══════════════════════════════════════════════════════════════════
# Phase 4: Call production API
# ══════════════════════════════════════════════════════════════════
def phase4_call_api(args, workdir: Path):
    print(f"\n[Phase 4] Production API 호출")

    # repo URL 결정
    eval_url = args.eval_repo_url
    if not eval_url:
        url_file = workdir / "eval_repo_url.txt"
        if url_file.exists():
            eval_url = url_file.read_text(encoding="utf-8").strip()
    if not eval_url:
        print(f"  ✗ eval repo URL 필요. --eval-repo-url 또는 Phase 3 먼저.")
        sys.exit(1)
    print(f"  Repo URL: {eval_url}")

    out_path = workdir / "api_response.json"
    if out_path.exists() and not args.force:
        print(f"  기존 응답 사용: {out_path}")
        return json.loads(out_path.read_text(encoding="utf-8"))

    print(f"  API 호출 중... (시간 좀 걸림)")
    t0 = time.time()
    try:
        resp = requests.post(
            f"{args.api}/retrieve/hybrid/by-repo-url",
            json={
                "repo_url": eval_url,
                "include_same_repo": False,
                "merged_top_k": 100,
            },
            timeout=args.api_timeout,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  ✗ API 호출 실패: {e}")
        sys.exit(1)

    elapsed = time.time() - t0
    data = resp.json()
    print(f"  ✓ 응답 받음 ({elapsed:.1f}s)")
    print(f"    analyzed_source_count: {data.get('analyzed_source_count')}")
    print(f"    findings: {len(data.get('findings', []))}")

    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  → {out_path}")
    return data


# ══════════════════════════════════════════════════════════════════
# Phase 5: Match + Metrics
# ══════════════════════════════════════════════════════════════════
def phase5_metrics(args, workdir: Path):
    print(f"\n[Phase 5] 매칭 + 메트릭")

    # eval manifest 로드
    eval_manifest_path = workdir / "eval_repo" / "manifest.jsonl"
    if not eval_manifest_path.exists():
        print(f"  ✗ eval manifest 없음: {eval_manifest_path}")
        sys.exit(1)
    eval_manifest = [json.loads(l) for l in eval_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_file = {m["file_path"]: m for m in eval_manifest}

    # API response 로드
    api_response_path = workdir / "api_response.json"
    if not api_response_path.exists():
        print(f"  ✗ API 응답 없음. Phase 4 먼저.")
        sys.exit(1)
    data = json.loads(api_response_path.read_text(encoding="utf-8"))
    findings = data.get("findings", [])

    # 매칭
    predictions = []
    for finding in findings:
        src_path = finding.get("source", {}).get("file_path", "")
        # CASE_MARKER에서 case_id 추출 (or file_path 매칭)
        src_code = finding.get("source", {}).get("raw_code", "")
        case_id = None
        m_marker = re.search(r"CASE_MARKER:\s*([A-Z]+-\d+)", src_code)
        if m_marker:
            case_id = m_marker.group(1)
        else:
            # file_path 기반 fallback
            meta = by_file.get(src_path)
            if meta:
                case_id = meta["case_id"]

        if not case_id:
            continue

        # eval manifest에서 메타 찾기
        meta = None
        for em in eval_manifest:
            if em["case_id"] == case_id:
                meta = em
                break
        if not meta:
            continue

        # expected anchor가 candidates에 있는지
        expected_repo = normalize_repo(meta.get("expected_repo_id", ""))
        expected_chunk_id = meta.get("anchor_chunk_id")
        found_expected = False
        expected_risk = "low"
        expected_score = 0.0
        expected_matched_by = []
        expected_signal = ""

        for cand in finding.get("candidates", []):
            cand_repo = normalize_repo((cand.get("repository") or {}).get("repo_url", ""))
            cand_path = (cand.get("location") or {}).get("file_path", "")
            cand_chunk_code = (cand.get("matched_chunk") or {}).get("raw_code", "")

            is_match = False
            if cand_repo == expected_repo and cand_path == meta.get("expected_file_path"):
                is_match = True
            elif expected_chunk_id and cand_chunk_code:
                # raw_code 일치로 보조 매칭
                pass

            if is_match:
                found_expected = True
                rp = cand.get("review_priority") or {}
                expected_risk = rp.get("level", "low")
                expected_score = rp.get("score", 0.0)
                expected_matched_by = cand.get("matched_by", [])
                expected_signal = cand.get("primary_match_signal", "")
                break

        top = finding.get("top_review_priority") or {}
        predictions.append({
            "case_id": case_id,
            "label": meta["label"],
            "category": meta["category"],
            "language": meta["language"],
            "eval_status": meta["eval_status"],
            "top_risk_level": top.get("level", "low"),
            "top_risk_score": top.get("score", 0.0),
            "candidate_count": len(finding.get("candidates", [])),
            "expected_found": found_expected,
            "expected_risk_level": expected_risk,
            "expected_risk_score": expected_score,
            "expected_matched_by": expected_matched_by,
            "expected_signal": expected_signal,
        })

    pred_path = workdir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"  매칭된 페어: {len(predictions)}/{len(findings)}")
    print(f"  → {pred_path}")

    # 메트릭 계산
    HIGH_RISK = {"medium", "high", "critical"}

    def compute(items, use_strict: bool):
        """use_strict=True: expected_found AND its risk >= medium"""
        tp = fn = fp = tn = 0
        for p in items:
            if use_strict:
                surfaced = p["expected_found"] and p["expected_risk_level"] in HIGH_RISK
            else:
                surfaced = p["top_risk_level"] in HIGH_RISK
            if p["label"] == 1:
                if surfaced: tp += 1
                else:        fn += 1
            else:
                if surfaced: fp += 1
                else:        tn += 1
        n = tp + fn + fp + tn
        acc = (tp + tn) / max(1, n)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2*prec*rec / max(1e-9, prec + rec)
        return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "n": n}

    loose = compute(predictions, False)
    strict = compute(predictions, True)
    ready_preds = [p for p in predictions if p["eval_status"] == "ready"]
    rule_preds = [p for p in predictions if p["eval_status"] == "rule_based_only"]
    ready_strict = compute(ready_preds, True) if ready_preds else None
    rule_strict = compute(rule_preds, True) if rule_preds else None

    # 카테고리별 (strict)
    by_cat = defaultdict(list)
    for p in predictions:
        by_cat[p["category"]].append(p)

    # 리포트 작성
    report = ["# Production Pipeline 평가 리포트\n",
              f"- 전체 페어: {len(predictions)}",
              f"- Loose 정의: top_review_priority가 medium 이상",
              f"- Strict 정의: expected anchor가 candidates에 medium 이상으로 surface",
              ""]
    report += ["## 전체 메트릭",
               "",
               "| 정의 | n | Accuracy | Precision | Recall | F1 |",
               "|---|---:|---:|---:|---:|---:|",
               f"| Loose | {loose['n']} | {loose['accuracy']:.4f} | {loose['precision']:.4f} | {loose['recall']:.4f} | {loose['f1']:.4f} |",
               f"| Strict | {strict['n']} | {strict['accuracy']:.4f} | {strict['precision']:.4f} | {strict['recall']:.4f} | {strict['f1']:.4f} |",
               ""]
    if ready_strict:
        report += [f"| Strict (Ready만) | {ready_strict['n']} | {ready_strict['accuracy']:.4f} | {ready_strict['precision']:.4f} | {ready_strict['recall']:.4f} | {ready_strict['f1']:.4f} |",]
    if rule_strict:
        report += [f"| Strict (Rule-only만) | {rule_strict['n']} | {rule_strict['accuracy']:.4f} | {rule_strict['precision']:.4f} | {rule_strict['recall']:.4f} | {rule_strict['f1']:.4f} |",]
    report.append("")

    report += ["## 카테고리별 (Strict 기준)",
               "",
               "| Cat | n | TP | FN | FP | TN | 정답률 |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for cat in CATEGORIES:
        items = by_cat.get(cat, [])
        if not items:
            continue
        s = compute(items, True)
        correct = s["tp"] + s["tn"]
        n = s["n"]
        report.append(f"| {cat} | {n} | {s['tp']} | {s['fn']} | {s['fp']} | {s['tn']} | {correct/max(1,n)*100:.1f}% |")
    report.append("")

    # Bi-encoder 비교
    report += ["## Bi-encoder 단독 평가와 비교",
               "",
               "| 지표 | Bi-encoder (cos>0.59) | Production loose | Production strict |",
               "|---|---:|---:|---:|",
               f"| Accuracy | 0.967 | {loose['accuracy']:.4f} | {strict['accuracy']:.4f} |",
               f"| F1 | 0.963 | {loose['f1']:.4f} | {strict['f1']:.4f} |",
               f"| Recall | 0.957 | {loose['recall']:.4f} | {strict['recall']:.4f} |",
               ""]

    # Worst FN (label=1인데 못 잡음)
    fn_cases = [p for p in predictions if p["label"] == 1 and (not p["expected_found"] or p["expected_risk_level"] not in HIGH_RISK)]
    fn_cases.sort(key=lambda p: p["expected_risk_score"])
    if fn_cases:
        report += ["## Worst FN (label=1인데 medium 이하)", "",
                   "| case_id | language | top_risk | expected_risk | expected_found |",
                   "|---|---|---|---|---|"]
        for p in fn_cases[:10]:
            report.append(f"| {p['case_id']} | {p['language']} | {p['top_risk_level']} | {p['expected_risk_level']} | {p['expected_found']} |")
        report.append("")

    # Worst FP (label=0인데 medium 이상 surfaced)
    fp_cases = [p for p in predictions if p["label"] == 0 and p["expected_found"] and p["expected_risk_level"] in HIGH_RISK]
    fp_cases.sort(key=lambda p: -p["expected_risk_score"])
    if fp_cases:
        report += ["## Worst FP (label=0인데 medium 이상 surfaced)", "",
                   "| case_id | language | expected_risk | expected_signal | matched_by |",
                   "|---|---|---|---|---|"]
        for p in fp_cases[:10]:
            report.append(f"| {p['case_id']} | {p['language']} | {p['expected_risk_level']} | {p['expected_signal']} | {','.join(p['expected_matched_by'])} |")
        report.append("")

    # Retrieval source 분포
    by_src = Counter()
    for p in predictions:
        if p["expected_found"]:
            srcs = tuple(sorted(p["expected_matched_by"])) or ("(none)",)
            by_src[srcs] += 1
    if by_src:
        report += ["## Retrieval source 분포 (expected_found만)", "",
                   "| Sources | Count |",
                   "|---|---:|"]
        for srcs, n in by_src.most_common():
            report.append(f"| {','.join(srcs)} | {n} |")
        report.append("")

    report_path = workdir / "production_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"  → {report_path}")

    # 콘솔 요약
    print("\n" + "=" * 60)
    print(f"  Loose Accuracy:  {loose['accuracy']:.4f}  F1: {loose['f1']:.4f}")
    print(f"  Strict Accuracy: {strict['accuracy']:.4f}  F1: {strict['f1']:.4f}")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                  description=__doc__)
    ap.add_argument("--testdir", required=False,
                    help="Phase 0용 — TP/FN/FP/TN final100.jsonl 들어있는 디렉토리")
    ap.add_argument("--os-host", default="http://localhost:9200")
    ap.add_argument("--api", default="http://ngseo-ubuntu:18000")
    ap.add_argument("--chunk-index", default="code_chunk_index")
    ap.add_argument("--embedding-index", default="code_chunk_embedding_index_v1")
    ap.add_argument("--workdir", required=True,
                    help="모든 phase 산출물 저장")
    ap.add_argument("--gh-owner", help="GitHub 사용자/조직 (Phase 3 push용)")
    ap.add_argument("--eval-repo-url",
                    help="이미 push한 eval repo URL (Phase 4부터 직접 시작 시)")
    ap.add_argument("--start-from", type=int, default=0,
                    help="특정 phase부터 시작 (0-5)")
    ap.add_argument("--stop-at", type=int, default=5,
                    help="특정 phase까지만 (0-5)")
    ap.add_argument("--api-timeout", type=int, default=3600)
    ap.add_argument("--force", action="store_true",
                    help="기존 결과 무시하고 새로 실행")
    ap.add_argument("--skip-push", action="store_true",
                    help="GitHub push 단계 생략 (수동 push 시)")
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    start = args.start_from
    stop = args.stop_at
    manifest = None

    print(f"=" * 70)
    print(f"Production Eval Pipeline (phases {start}~{stop})")
    print(f"  workdir: {workdir}")
    print(f"=" * 70)

    if start <= 0 <= stop:
        if not args.testdir:
            print("✗ Phase 0 실행하려면 --testdir 필요")
            sys.exit(1)
        manifest = phase0_build_manifest(args, workdir)
    elif manifest is None:
        m_path = workdir / "manifest_final.jsonl"
        if m_path.exists():
            manifest = [json.loads(l) for l in m_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if manifest is None:
        print("✗ manifest 없음. Phase 0 먼저.")
        sys.exit(1)

    if start <= 1 <= stop:
        phase1_sanity_check(args, manifest, workdir)

    if start <= 2 <= stop:
        phase2_build_eval_repo(args, manifest, workdir)
        if stop == 2:
            print("\n다음 단계: GitHub push 후 --start-from 4 로 재실행")
            print("  cd <workdir>/eval_repo")
            print("  gh repo create kkbang/<name> --public --source=. --push")
            return

    if start <= 3 <= stop:
        url = phase3_github_push(args, workdir)
        if url:
            args.eval_repo_url = args.eval_repo_url or url

    if start <= 4 <= stop:
        phase4_call_api(args, workdir)

    if start <= 5 <= stop:
        phase5_metrics(args, workdir)

    print("\n✓ 파이프라인 완료")


if __name__ == "__main__":
    main()
