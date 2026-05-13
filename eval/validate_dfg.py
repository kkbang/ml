"""
validate_dfg.py — Python extract_dataflow() vs Rust extract_dataflow_rust() 비교
"""

import sys
sys.path.insert(0, '/home/ngseokim/code-killr/core')
import time
from transformers import AutoTokenizer

sys.path.append('/home/ngseokim/code-killr/parser')
from dataset import encode_with_dfg, extract_dataflow, TOTAL_LENGTH

try:
    from dfg_rs import extract_dataflow_rust
    print("✅ dfg_rs 로드 성공")
except ImportError as e:
    print(f"❌ dfg_rs 로드 실패: {e}")
    sys.exit(1)

MODEL_NAME = 'microsoft/graphcodebert-base'
tokenizer  = AutoTokenizer.from_pretrained(MODEL_NAME)

SAMPLES = [
    ("python",     "def add(x, y):\n    result = x + y\n    return result\n"),
    ("python",     "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n"),
    ("python",     "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n"),
    ("java",       "public int binarySearch(int[] arr, int target) {\n    int left = 0, right = arr.length - 1;\n    while (left <= right) {\n        int mid = (left + right) / 2;\n        if (arr[mid] == target) return mid;\n        if (arr[mid] < target) left = mid + 1;\n        else right = mid - 1;\n    }\n    return -1;\n}\n"),
    ("javascript", "function debounce(func, wait) {\n    let timeout;\n    return function(...args) {\n        clearTimeout(timeout);\n        timeout = setTimeout(() => func(...args), wait);\n    };\n}\n"),
    ("ruby", "def fibonacci(n)\n  return n if n <= 1\n  fibonacci(n-1) + fibonacci(n-2)\nend\n"),
    ("php",  "<?php\nfunction add($x, $y) {\n  $result = $x + $y;\n  return $result;\n}\n"),
    ("c_sharp", "public int BinarySearch(int[] arr, int target) {\n  int left = 0, right = arr.Length - 1;\n  while (left <= right) {\n    int mid = (left + right) / 2;\n    if (arr[mid] == target) return mid;\n    if (arr[mid] < target) left = mid + 1;\n    else right = mid - 1;\n  }\n  return -1;\n}\n"),
    ("go",         "func mergeSort(arr []int) []int {\n    if len(arr) <= 1 {\n        return arr\n    }\n    mid := len(arr) / 2\n    left := mergeSort(arr[:mid])\n    right := mergeSort(arr[mid:])\n    return merge(left, right)\n}\n"),
]


# ── [1] extract_dataflow 레벨 비교 ────────────────────────────────────
def compare_extract(code, language, verbose=False):
    result = {'language': language, 'match': False, 'errors': []}
    try:
        py_tokens, py_dfg = extract_dataflow(code, language)
    except Exception as e:
        result['errors'].append(f"Python 실패: {e}")
        return result
    try:
        rs_tokens, rs_dfg = extract_dataflow_rust(code, language)
    except Exception as e:
        result['errors'].append(f"Rust 실패: {e}")
        return result

    result['py_tokens'] = len(py_tokens)
    result['rs_tokens'] = len(rs_tokens)
    result['py_dfg']    = len(py_dfg)
    result['rs_dfg']    = len(rs_dfg)

    if len(py_tokens) != len(rs_tokens):
        result['errors'].append(f"토큰 수 불일치: Python={len(py_tokens)} Rust={len(rs_tokens)}")
        if verbose:
            print(f"    Python: {py_tokens[:8]}")
            print(f"    Rust:   {rs_tokens[:8]}")

    if len(py_dfg) != len(rs_dfg):
        result['errors'].append(f"DFG 노드 수 불일치: Python={len(py_dfg)} Rust={len(rs_dfg)}")

    py_names = [x[0] for x in py_dfg]
    rs_names = [x[0] for x in rs_dfg]
    if py_names != rs_names:
        result['errors'].append("DFG 변수명 불일치")
        if verbose:
            print(f"    Python names: {py_names[:8]}")
            print(f"    Rust   names: {rs_names[:8]}")

    py_idxs = [x[1] for x in py_dfg]
    rs_idxs = [x[1] for x in rs_dfg]
    if py_idxs != rs_idxs:
        result['errors'].append("DFG token_idx 불일치")
        if verbose:
            print(f"    Python idxs: {py_idxs[:8]}")
            print(f"    Rust   idxs: {rs_idxs[:8]}")

    result['match'] = len(result['errors']) == 0
    return result


# ── [2] encode_with_dfg 하이브리드 비교 ──────────────────────────────
def encode_hybrid(code, language):
    """Rust DFG + Python 토크나이징 → encode_with_dfg와 동일한 결과"""
    rs_tokens, rs_dfg = extract_dataflow_rust(code, language)
    code_tokens = rs_tokens
    dfg         = list(rs_dfg)

    code_tokens_sub = [
        tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x)
        for idx, x in enumerate(code_tokens)
    ]
    ori2cur_pos = {-1: (0, 0)}
    for i in range(len(code_tokens_sub)):
        ori2cur_pos[i] = (
            ori2cur_pos[i-1][1],
            ori2cur_pos[i-1][1] + len(code_tokens_sub[i])
        )
    code_tokens_flat = [t for sub in code_tokens_sub for t in sub]
    max_code = TOTAL_LENGTH - 3 - min(len(dfg), 64)
    code_tokens_flat = code_tokens_flat[:max_code][:512-3]

    source_tokens = [tokenizer.cls_token] + code_tokens_flat + [tokenizer.sep_token]
    source_ids    = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx  = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]

    dfg = dfg[:TOTAL_LENGTH - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx  += [0 for _ in dfg]
    source_ids    += [tokenizer.unk_token_id for _ in dfg]

    pad_len = TOTAL_LENGTH - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * pad_len
    source_ids   += [tokenizer.pad_token_id] * pad_len

    reverse_index = {x[1]: i for i, x in enumerate(dfg)}
    for i, x in enumerate(dfg):
        name, idx, sources = x
        dfg[i] = (name, idx, [reverse_index[j] for j in sources if j in reverse_index])

    dfg_to_dfg  = [x[2] for x in dfg]
    dfg_to_code = [(ori2cur_pos[x[1]][0] + 1, ori2cur_pos[x[1]][1] + 1) for x in dfg]
    return source_ids, position_idx, dfg_to_code, dfg_to_dfg


def compare_encode(code, language):
    result = {'language': language, 'match': False, 'errors': []}
    try:
        py = encode_with_dfg(code, language, tokenizer)
    except Exception as e:
        result['errors'].append(f"Python 실패: {e}")
        return result
    try:
        rs = encode_hybrid(code, language)
    except Exception as e:
        result['errors'].append(f"Hybrid 실패: {e}")
        return result

    if py[0] != rs[0]: result['errors'].append("input_ids 불일치")
    if py[1] != rs[1]: result['errors'].append("position_ids 불일치")
    if py[2] != rs[2]: result['errors'].append(f"dfg_to_code 불일치 Py={py[2][:2]} Rs={rs[2][:2]}")
    result['match'] = len(result['errors']) == 0
    return result


# ── [3] 속도 비교 ─────────────────────────────────────────────────────
def benchmark(n=200):
    codes = [s[1] for s in SAMPLES * (n // len(SAMPLES) + 1)][:n]
    langs = [s[0] for s in SAMPLES * (n // len(SAMPLES) + 1)][:n]

    t0 = time.perf_counter()
    for c, l in zip(codes, langs):
        try: extract_dataflow(c, l)
        except: pass
    py_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for c, l in zip(codes, langs):
        try: extract_dataflow_rust(c, l)
        except: pass
    rs_ms = (time.perf_counter() - t0) * 1000

    return {
        'py_per': round(py_ms/n, 2), 'rs_per': round(rs_ms/n, 2),
        'speedup': round(py_ms/rs_ms, 2) if rs_ms > 0 else 0,
    }


def main():
    print("=" * 60)
    print("  DFG 검증: Python vs Rust")
    print("=" * 60)

    print("\n[1] extract_dataflow 레벨 (raw token)")
    p1 = t1 = 0
    for lang, code in SAMPLES:
        r = compare_extract(code, lang, verbose=True)
        t1 += 1
        ok = "✅" if r['match'] else "❌"
        print(f"  {ok} {lang:12s} | tokens Py={r.get('py_tokens','?')} Rs={r.get('rs_tokens','?')} | DFG Py={r.get('py_dfg','?')} Rs={r.get('rs_dfg','?')}")
        for e in r.get('errors', []): print(f"     → {e}")
        if r['match']: p1 += 1
    print(f"  결과: {p1}/{t1} 통과")

    print("\n[2] encode_with_dfg 하이브리드 (임베딩 입력 수준)")
    p2 = t2 = 0
    for lang, code in SAMPLES:
        r = compare_encode(code, lang)
        t2 += 1
        ok = "✅" if r['match'] else "❌"
        print(f"  {ok} {lang:12s}")
        for e in r.get('errors', []): print(f"     → {e}")
        if r['match']: p2 += 1
    print(f"  결과: {p2}/{t2} 통과")

    print("\n[3] 속도 비교 (200 샘플)")
    b = benchmark(200)
    print(f"  Python: {b['py_per']}ms/sample")
    print(f"  Rust:   {b['rs_per']}ms/sample")
    print(f"  속도 향상: {b['speedup']}×")

    print("\n" + ("✅ 검증 완료. 서버에 적용 가능." if p2 == t2 else f"⚠️  {t2-p2}개 불일치. 수정 필요."))
    print("=" * 60)


if __name__ == '__main__':
    main()
