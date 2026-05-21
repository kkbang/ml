"""
select_best_100.py — 각 카테고리 approved 풀에서 품질 100개 선별 (v2: 강화 필터)

선별 기준:
  - 언어 균형 (Python/Java/Go/JavaScript 골고루)
  - Repo 다양성 (단일 repo 과대표집 제한)
  - 카테고리별 우선순위 시그널

v2 강화 필터 (1차 LLM judge 결과 반영):
  - TP에서 minify-only 페어 완전 배제 (토큰 시퀀스 동일)
  - `true→not false`, `false→not true` 트릭만 있는 페어 배제
  - 변수명만 다르고 skeleton 토큰 시퀀스 100% 동일한 페어 배제
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

JUDGED = Path("/home/shinmk/code-killr/test-suite/judged")
TARGET = 100
SEED = 2024


# ── trivial 페어 감지 유틸 ──────────────────────────────────────────

def _normalize_whitespace(code: str) -> str:
    """공백/주석/세미콜론/콤마 제거 후 정규화 (aggressive)."""
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    code = re.sub(r"\s+", "", code)
    # ★ trailing comma, 잉여 세미콜론, 문자열 concat `+`, `!` 제거 — minify trick 대응
    code = code.replace(";", "").replace(",", "")
    return code


_KEYWORDS = {
    "if","else","elif","for","while","return","def","class","import","from","as",
    "try","except","finally","raise","with","lambda","yield","pass","break","continue",
    "func","var","const","let","switch","case","default","new","this","null","true","false",
    "public","private","protected","static","void","int","string","bool","float","double",
    "package","interface","struct","type","go","chan","range","map","make",
    "in","is","not","and","or","None","True","False","throw","throws","extends","implements",
    "self","cls",
}

def _tokenize_skeleton(code: str) -> list:
    """식별자를 'ID', 문자열을 'STR', 숫자를 'NUM'으로 치환한 토큰 시퀀스."""
    # 주석 제거
    code = re.sub(r"#.*", "", code)
    code = re.sub(r"//.*", "", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    # 문자열 리터럴 → STR
    code = re.sub(r'"(?:[^"\\]|\\.)*"', '"STR"', code)
    code = re.sub(r"'(?:[^'\\]|\\.)*'", "'STR'", code)
    code = re.sub(r"`(?:[^`\\]|\\.)*`", '`STR`', code)
    # 숫자 → NUM
    code = re.sub(r"\b\d+(?:\.\d+)?\b", "NUM", code)
    toks = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[^\sA-Za-z_0-9]", code)
    out = []
    for t in toks:
        if re.match(r"^[A-Za-z_]", t):
            out.append(t if t in _KEYWORDS else "ID")
        else:
            out.append(t)
    return out


# `true → not false` 류 trick 패턴 (B에서만 등장하면 의심)
_TRICK_PATTERNS = [
    r"\bnot\s+False\b",
    r"\bnot\s+True\b",
    r"\bnot\s+true\b",
    r"\bnot\s+false\b",
    r"!\s*true\b",
    r"!\s*false\b",
    r"!\s*True\b",
    r"!\s*False\b",
    r"\(1\s*==\s*0\)",
    r"\(0\s*==\s*1\)",
    r"\(NUM\s*==\s*NUM\)",  # after skeleton — too generic, skip
]
_TRICK_RE = re.compile("|".join(_TRICK_PATTERNS[:-1]))


def _is_trivial_TP(a: str, b: str) -> tuple[bool, str]:
    """A→B 변형이 trivial인지 판단. (is_trivial, reason)"""
    # 1) whitespace/주석만 다름
    if _normalize_whitespace(a) == _normalize_whitespace(b):
        return True, "공백/주석만 다름"

    sk_a = _tokenize_skeleton(a)
    sk_b = _tokenize_skeleton(b)

    # 2) skeleton(식별자/문자열/숫자 무시) 완전 동일 → minify 또는 변수 리네임만
    if sk_a == sk_b:
        return True, "skeleton 토큰 시퀀스 완전 동일"

    # 3) skeleton이 거의 동일 (97%+) + 토큰 수 거의 같음
    if len(sk_a) > 5 and len(sk_b) > 5:
        m = min(len(sk_a), len(sk_b))
        same = sum(1 for x, y in zip(sk_a, sk_b) if x == y)
        ratio = same / max(len(sk_a), len(sk_b))
        if ratio > 0.97 and abs(len(sk_a) - len(sk_b)) < 3:
            return True, f"skeleton ~동일 ({ratio:.0%})"

    # 4) `not true / not false / !true / !false / (1==0)` 트릭이 B에만 존재
    tricks_a = len(_TRICK_RE.findall(a))
    tricks_b = len(_TRICK_RE.findall(b))
    trick_diff = tricks_b - tricks_a
    # B에서만 트릭이 새로 생긴 게 2개 이상 + 길이 변화 거의 없음 → 트릭만 적용된 변형
    if trick_diff >= 2 and abs(len(sk_a) - len(sk_b)) <= max(8, trick_diff * 4):
        return True, f"`not true/false` 트릭만 ({trick_diff}개)"

    return False, ""


# ── 점수 함수 ──────────────────────────────────────────────────────

def score_TP(p):
    sig = p.get("signal") or p.get("_filter_signal") or {}
    j = sig.get("jaccard", 0)
    lr = sig.get("len_ratio", 1)
    a, b = p.get("anchor", ""), p.get("positive", "")
    a_len = len(a.splitlines())

    # ★ jaccard ≥ 0.98이면 거의 항상 trivial (실제 1차 LLM judge에서 100% reject)
    if j >= 0.98:
        return float("-inf")

    # ★ trivial 페어 완전 제외
    trivial, _ = _is_trivial_TP(a, b)
    if trivial:
        return float("-inf")

    short_pen = -2 if a_len < 6 else 0
    jac_bonus = 1 if 0.35 <= j <= 0.75 else 0
    trivial_pen = -3 if j > 0.9 else 0
    lr_pen = -1 if lr > 2.5 else 0
    return short_pen + jac_bonus + trivial_pen + lr_pen + a_len * 0.1


def score_FN(p):
    sig = p.get("signal") or p.get("_filter_signal") or {}
    j = sig.get("jaccard", 1.0)
    a_len = len(p.get("anchor", "").splitlines())
    hard_bonus = (1.0 - min(j, 1.0)) * 3
    short_pen = -3 if a_len < 6 else 0
    level = p.get("level", "")
    level_bonus = {
        "semantic_hard": 2,
        "mixed_hard": 1.5,
        "structural_hard": 1,
        "surface_hard": 0.5,
    }.get(level, 0)
    return hard_bonus + short_pen + level_bonus + a_len * 0.05


def score_FP(p):
    sig = p.get("signal") or p.get("_filter_signal") or {}
    io = sig.get("id_overlap", 1.0)
    a_len = len(p.get("anchor", "").splitlines())
    hard = (1.0 - min(io, 1.0)) * 2
    short_pen = -3 if a_len < 6 else 0
    return hard + short_pen + a_len * 0.05


def score_TN(p):
    sig = p.get("signal") or p.get("_filter_signal") or {}
    j = sig.get("jaccard", 1.0)
    a_len = len(p.get("anchor", "").splitlines())
    return (1.0 - min(j, 1.0)) * 2 + min(a_len, 30) * 0.05


SCORERS = {"TP": score_TP, "FN": score_FN, "FP": score_FP, "TN": score_TN}

MAX_PER_REPO = 8
TARGET_LANG_DIST = {
    "python": 30, "go": 25, "java": 25, "javascript": 20,
}


def select_for_category(cat: str):
    src = JUDGED / f"{cat}_approved.jsonl"
    out = JUDGED / f"{cat}_final100.jsonl"
    if not src.exists():
        print(f"  {src.name} 없음 — 건너뜀")
        return

    pool = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n[{cat}] approved pool: {len(pool)}개")

    scorer = SCORERS[cat]

    # 사전 필터 통계 (TP만)
    if cat == "TP":
        trivial_count = 0
        trivial_reasons = Counter()
        for p in pool:
            tr, reason = _is_trivial_TP(p.get("anchor", ""), p.get("positive", ""))
            if tr:
                trivial_count += 1
                # 사유 키워드만 추출
                key = reason.split("(")[0].strip()
                trivial_reasons[key] += 1
        print(f"  ⚠ trivial 페어 자동 배제: {trivial_count}개")
        for reason, cnt in trivial_reasons.most_common():
            print(f"     • {reason}: {cnt}개")
        usable = len(pool) - trivial_count
        print(f"  유효 풀: {usable}개")

    random.seed(SEED)
    scored = sorted(pool, key=lambda p: (-scorer(p), random.random()))

    selected = []
    repo_count = Counter()
    lang_count = Counter()
    aws_count = 0
    AWS_CAP = 12 if cat == "FP" else 999

    for p in scored:
        if len(selected) >= TARGET:
            break
        # -inf 점수는 trivial → 절대 선택 안 함
        if scorer(p) == float("-inf"):
            continue

        repo = p.get("repo", "") or p.get("anchor_repo", "unknown")
        lang = p.get("language", "unknown").lower()
        func = p.get("function", "") or p.get("anchor_function", "")

        is_aws_consumer = (
            cat == "FP"
            and "awslabs/aws-sdk-python" in repo
            and func == "_consumer"
        )
        if is_aws_consumer and aws_count >= AWS_CAP:
            continue

        if repo_count[repo] >= MAX_PER_REPO:
            continue

        target = TARGET_LANG_DIST.get(lang, 10)
        if lang_count[lang] >= target + 5:
            continue

        selected.append(p)
        repo_count[repo] += 1
        lang_count[lang] += 1
        if is_aws_consumer:
            aws_count += 1

    # 100개 못 채웠으면 캡 풀고 채움 (단, trivial은 끝까지 배제)
    if len(selected) < TARGET:
        already = {id(p) for p in selected}
        for p in scored:
            if len(selected) >= TARGET:
                break
            if id(p) in already:
                continue
            if scorer(p) == float("-inf"):
                continue
            selected.append(p)

    with open(out, "w", encoding="utf-8") as f:
        for p in selected:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  → 선별: {len(selected)}개 → {out.name}")
    print(f"  언어 분포: {dict(lang_count)}")
    print(f"  상위 repo: {repo_count.most_common(5)}")
    if cat == "FP":
        print(f"  AWS _consumer: {aws_count}개")


if __name__ == "__main__":
    print("=" * 60)
    print("각 카테고리에서 품질 100개 선별 (v2 강화 필터)")
    print("=" * 60)
    for cat in ["TP", "FN", "FP", "TN"]:
        select_for_category(cat)
    print(f"\n출력 디렉토리: {JUDGED}")
