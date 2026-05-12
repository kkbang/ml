"""
benchmark.py — 임베딩 서버 Latency / Throughput / QPS 측정

실행:
  # 기본 (동시 요청 8개, 배치 크기 8, 총 200개)
  python benchmark.py

  # 파라미터 조정
  python benchmark.py --concurrency 16 --batch-size 8 --total 500 --url http://localhost:8000
"""

import asyncio
import argparse
import time
import json
import numpy as np
import aiohttp
from dataclasses import dataclass, field
from typing import List

# ── 테스트용 샘플 코드 (언어별 다양하게) ──────────────────────────────
SAMPLE_CODES = [
    ("python", """
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i]); i += 1
        else:
            result.append(right[j]); j += 1
    return result + left[i:] + right[j:]
"""),
    ("java", """
public List<Integer> twoSum(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement)) {
            return Arrays.asList(map.get(complement), i);
        }
        map.put(nums[i], i);
    }
    return new ArrayList<>();
}
"""),
    ("javascript", """
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}
"""),
    ("go", """
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := (left + right) / 2
        if arr[mid] == target {
            return mid
        } else if arr[mid] < target {
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return -1
}
"""),
    ("python", """
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.order = []

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.order.append(key)
"""),
    ("java", """
public TreeNode buildTree(int[] preorder, int[] inorder) {
    if (preorder.length == 0) return null;
    TreeNode root = new TreeNode(preorder[0]);
    int mid = 0;
    for (int i = 0; i < inorder.length; i++) {
        if (inorder[i] == preorder[0]) { mid = i; break; }
    }
    root.left  = buildTree(Arrays.copyOfRange(preorder, 1, mid + 1),
                           Arrays.copyOfRange(inorder, 0, mid));
    root.right = buildTree(Arrays.copyOfRange(preorder, mid + 1, preorder.length),
                           Arrays.copyOfRange(inorder, mid + 1, inorder.length));
    return root;
}
"""),
    ("python", """
def longest_common_subsequence(text1: str, text2: str) -> int:
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]
"""),
    ("go", """
func maxProfit(prices []int) int {
    if len(prices) == 0 { return 0 }
    minPrice := prices[0]
    maxProfit := 0
    for _, price := range prices {
        if price < minPrice {
            minPrice = price
        } else if price - minPrice > maxProfit {
            maxProfit = price - minPrice
        }
    }
    return maxProfit
}
"""),
]


@dataclass
class BenchmarkResult:
    total_requests:   int = 0
    total_items:      int = 0
    success_count:    int = 0
    error_count:      int = 0
    latencies_ms:     List[float] = field(default_factory=list)
    elapsed_total_s:  float = 0.0

    def items_per_sec(self) -> float:
        if self.elapsed_total_s == 0:
            return 0.0
        return self.total_items / self.elapsed_total_s

    def qps(self) -> float:
        if self.elapsed_total_s == 0:
            return 0.0
        return self.success_count / self.elapsed_total_s

    def percentile(self, p: float) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, p))

    def print_report(self) -> None:
        print("\n" + "="*55)
        print("  벤치마크 결과")
        print("="*55)
        print(f"  총 요청 수:       {self.total_requests}개")
        print(f"  총 아이템 수:     {self.total_items}개")
        print(f"  성공 / 실패:      {self.success_count} / {self.error_count}")
        print(f"  총 소요 시간:     {self.elapsed_total_s:.2f}s")
        print()
        print(f"  Throughput:       {self.items_per_sec():.1f} items/sec")
        print(f"  QPS:              {self.qps():.1f} req/sec")
        print()
        print(f"  Latency p50:      {self.percentile(50):.1f}ms")
        print(f"  Latency p75:      {self.percentile(75):.1f}ms")
        print(f"  Latency p95:      {self.percentile(95):.1f}ms")
        print(f"  Latency p99:      {self.percentile(99):.1f}ms")
        print(f"  Latency max:      {max(self.latencies_ms):.1f}ms")
        print(f"  Latency min:      {min(self.latencies_ms):.1f}ms")
        print(f"  Latency avg:      {np.mean(self.latencies_ms):.1f}ms")
        print("="*55)


async def send_batch(
    session: aiohttp.ClientSession,
    url: str,
    batch: list,
    result: BenchmarkResult,
    semaphore: asyncio.Semaphore,
) -> None:
    payload = {
        "model": "code-killr",
        "input": [{"code": code, "language": lang} for lang, code in batch],
    }
    async with semaphore:
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{url}/v1/embeddings",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                data = await resp.json()
                elapsed_ms = (time.perf_counter() - t0) * 1000

                if resp.status == 200 and "data" in data:
                    result.success_count += 1
                    result.total_items   += len(data["data"])
                    result.latencies_ms.append(elapsed_ms)
                else:
                    result.error_count += 1
                    print(f"  ⚠️  오류 응답: status={resp.status}")
        except Exception as e:
            result.error_count += 1
            print(f"  ❌ 요청 실패: {e}")

        result.total_requests += 1
        done = result.total_requests
        total = result.total_requests + (0)  # progress hint
        if done % 10 == 0:
            avg = np.mean(result.latencies_ms) if result.latencies_ms else 0
            print(f"  진행: {done}req | avg={avg:.0f}ms | "
                  f"items/s={result.items_per_sec():.0f}", end='\r')


async def run_benchmark(
    url: str,
    concurrency: int,
    batch_size: int,
    total_items: int,
) -> BenchmarkResult:
    # 총 아이템 수로 배치 구성
    batches = []
    items_added = 0
    while items_added < total_items:
        batch = []
        for _ in range(min(batch_size, total_items - items_added)):
            sample = SAMPLE_CODES[items_added % len(SAMPLE_CODES)]
            batch.append(sample)
            items_added += 1
        batches.append(batch)

    print(f"\n{'='*55}")
    print(f"  벤치마크 시작")
    print(f"{'='*55}")
    print(f"  서버 URL:         {url}")
    print(f"  동시 요청 수:     {concurrency}")
    print(f"  배치 크기:        {batch_size}")
    print(f"  총 아이템:        {total_items}")
    print(f"  총 요청 수:       {len(batches)}")
    print(f"{'='*55}\n")

    result    = BenchmarkResult()
    semaphore = asyncio.Semaphore(concurrency)

    # warm-up
    print("  Warm-up 중...")
    async with aiohttp.ClientSession() as session:
        warmup_batch = batches[0][:2]
        await send_batch(session, url, warmup_batch, BenchmarkResult(), semaphore)

    result = BenchmarkResult()
    t_start = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        tasks = [
            send_batch(session, url, batch, result, semaphore)
            for batch in batches
        ]
        await asyncio.gather(*tasks)

    result.elapsed_total_s = time.perf_counter() - t_start
    print()
    return result


def main():
    parser = argparse.ArgumentParser(description="임베딩 서버 벤치마크")
    parser.add_argument("--url",         default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, default=8,   help="동시 요청 수")
    parser.add_argument("--batch-size",  type=int, default=8,   help="요청당 아이템 수")
    parser.add_argument("--total",       type=int, default=200, help="총 아이템 수")
    args = parser.parse_args()

    result = asyncio.run(run_benchmark(
        url=args.url,
        concurrency=args.concurrency,
        batch_size=args.batch_size,
        total_items=args.total,
    ))
    result.print_report()

    # JSON 저장
    output = {
        "config": {
            "url":         args.url,
            "concurrency": args.concurrency,
            "batch_size":  args.batch_size,
            "total_items": args.total,
        },
        "results": {
            "qps":             round(result.qps(), 2),
            "items_per_sec":   round(result.items_per_sec(), 2),
            "latency_p50_ms":  round(result.percentile(50), 2),
            "latency_p75_ms":  round(result.percentile(75), 2),
            "latency_p95_ms":  round(result.percentile(95), 2),
            "latency_p99_ms":  round(result.percentile(99), 2),
            "latency_avg_ms":  round(float(np.mean(result.latencies_ms)), 2),
            "latency_max_ms":  round(max(result.latencies_ms), 2),
            "success_count":   result.success_count,
            "error_count":     result.error_count,
            "elapsed_s":       round(result.elapsed_total_s, 2),
        }
    }
    with open("benchmark_result.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  결과 저장: benchmark_result.json")


if __name__ == "__main__":
    main()
