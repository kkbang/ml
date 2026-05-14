import argparse, json, random, time
import numpy as np
import requests

def get_embeddings_batch(endpoint, items):
    payload = {"items": [{"code": c, "language": l} for c, l in items]}
    resp = requests.post(endpoint, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return [np.array(e, dtype=np.float32) for e in data["embeddings"]], data.get("failed_indices", [])

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def check_collapse(embeddings):
    sample = random.sample(embeddings, min(100, len(embeddings)))
    sims = [cosine_sim(sample[i], sample[j]) for i in range(len(sample)) for j in range(i+1, len(sample))]
    mean_sim = float(np.mean(sims))
    return {"mean_inter_sim": round(mean_sim, 4), "collapsed": mean_sim > 0.95}

def main(endpoint, test_path, n_samples, batch_size=64):
    pairs = [json.loads(l) for l in open(test_path, encoding="utf-8") if l.strip()]
    sampled = random.sample(pairs, min(n_samples, len(pairs)))
    print(f"샘플 {len(sampled)}개 | 배치 {batch_size} | {endpoint}\n")

    all_codes = []
    for p in sampled:
        lang = p.get("language", "python")
        all_codes.append((p["anchor"], lang))
        all_codes.append((p["positive"], lang))

    embeddings = [None] * len(all_codes)
    failed_global = set()
    latencies_ms = []

    for start in range(0, len(all_codes), batch_size):
        chunk = all_codes[start:start+batch_size]
        t0 = time.perf_counter()
        embs, failed = get_embeddings_batch(endpoint, chunk)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed / len(chunk))
        for i, emb in enumerate(embs):
            embeddings[start+i] = emb
        for fi in failed:
            failed_global.add(start+fi)
        print(f"  {min(start+batch_size, len(all_codes))}/{len(all_codes)} 완료  ({elapsed:.0f}ms)", end="\r")
    print()

    pos_sims, neg_sims, norms, anchor_embs = [], [], [], []
    for i, pair in enumerate(sampled):
        ai, pi = i*2, i*2+1
        ni = ((i+1) % len(sampled)) * 2
        if ai in failed_global or pi in failed_global or ni in failed_global:
            continue
        ea, ep, en = embeddings[ai], embeddings[pi], embeddings[ni]
        if ea is None or ep is None or en is None:
            continue
        pos_sims.append(cosine_sim(ea, ep))
        neg_sims.append(cosine_sim(ea, en))
        norms.append(float(np.linalg.norm(ea)))
        anchor_embs.append(ea)

    pos, neg = np.array(pos_sims), np.array(neg_sims)
    recall = float((pos > 0.5).mean())
    fpr    = float((neg > 0.5).mean())
    gap    = float(pos.mean() - neg.mean())
    collapse = check_collapse(anchor_embs)

    print("\n========== 임베딩 품질 체크 결과 ==========")
    print(f"\n[유사도 분포]")
    print(f"  Positive 평균 : {pos.mean():.4f}  (±{pos.std():.4f})")
    print(f"  Negative 평균 : {neg.mean():.4f}  (±{neg.std():.4f})")
    print(f"  Separation gap: {gap:.4f}")
    print(f"\n[임계값 0.5 기준]")
    print(f"  Recall : {recall:.4f}  (기대 ≥ 0.99)")
    print(f"  FPR    : {fpr:.4f}  (기대 ≤ 0.001)")
    print(f"\n[정규화 체크]")
    print(f"  norm 평균: {np.mean(norms):.4f}  (±{np.std(norms):.4f})  ← 1.0이면 정상")
    print(f"\n[임베딩 붕괴 체크]")
    print(f"  Inter-embedding 평균 유사도: {collapse['mean_inter_sim']}")
    print(f"  붕괴 여부: {'⚠  붕괴 의심' if collapse['collapsed'] else '정상'}")
    print(f"\n[응답 속도]")
    print(f"  평균 레이턴시: {np.mean(latencies_ms):.1f}ms / 건")
    print(f"\n[실패] {len(failed_global)} / {len(all_codes)}")

    print("\n[종합 판정]")
    issues = []
    if recall < 0.99: issues.append(f"Recall {recall:.4f} < 0.99")
    if fpr > 0.005:   issues.append(f"FPR {fpr:.4f} > 0.005")
    if gap < 0.80:    issues.append(f"Separation gap {gap:.4f} 낮음")
    if collapse["collapsed"]: issues.append("임베딩 붕괴 감지")
    if abs(np.mean(norms) - 1.0) > 0.05: issues.append(f"norm {np.mean(norms):.4f} 정규화 이상")
    for iss in issues: print(f"  ⚠  {iss}")
    if not issues: print("  ✓ 모든 지표 정상")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint",   default="http://localhost:8000/embed")
    parser.add_argument("--test_path",  default="/home/ngseokim/code-killr/data/test_v3_anon.jsonl")
    parser.add_argument("--n_samples",  type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args.endpoint, args.test_path, args.n_samples, args.batch_size)
