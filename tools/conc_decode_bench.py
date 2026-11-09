#!/usr/bin/env python3
"""并发解码测试：对 Qwenox 服务（OpenAI 兼容端点）发 N 路并发请求，
报告每路 decode tok/s 与聚合吞吐。greedy（temperature=0）走投机解码。"""
import json, sys, time, threading, urllib.request

HOST = "http://192.168.5.12:8731"
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
    "Write a Python function that finds the longest palindromic substring in a string.",
    "Write a Python class implementing a LRU cache with get and put in O(1).",
    "Write a Python function that serializes a binary tree to a string and deserializes it back.",
    "用 Python 写一个函数：统计一段文本中每个汉字出现的次数，按频率降序返回。",
    "用 Python 写一个带重试和指数退避的 HTTP 下载函数，要求有类型注解。",
    "Write a Python function that validates and formats a list of phone numbers.",
    "写一个 Python 函数：解析 Markdown 表格并返回字典列表，处理转义字符。",
]

def one_request(idx, prompt, max_tokens, results):
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            resp = json.loads(r.read())
        dt = time.perf_counter() - t0
        usage = resp.get("usage", {})
        ct = usage.get("completion_tokens", 0)
        pt = usage.get("prompt_tokens", 0)
        results[idx] = (ct, pt, dt, None)
    except Exception as e:
        results[idx] = (0, 0, time.perf_counter() - t0, str(e))

def run_level(nconc, max_tokens=256):
    results = [None] * nconc
    threads = []
    t0 = time.perf_counter()
    for i in range(nconc):
        th = threading.Thread(target=one_request,
                              args=(i, PROMPTS[i % len(PROMPTS)] + f" (case {i})", max_tokens, results))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0
    total_ct = sum(r[0] for r in results if r and not r[3])
    errs = [r[3] for r in results if r and r[3]]
    per = [(r[0], r[2], r[0] / r[2] if r[2] > 0 else 0) for r in results if r and not r[3]]
    print(f"\n=== concurrency {nconc} ===")
    for i, (ct, dt, rate) in enumerate(per):
        print(f"  stream {i}: {ct} tokens in {dt:.1f}s = {rate:.1f} tok/s")
    print(f"  wall {wall:.1f}s, aggregate {total_ct / wall:.1f} tok/s, "
          f"per-stream mean {sum(p[2] for p in per) / max(1, len(per)):.1f} tok/s")
    if errs:
        print("  ERRORS:", errs)

def health():
    try:
        with urllib.request.urlopen(HOST + "/health", timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "health":
        print(json.dumps(health(), indent=1)[:1500])
        sys.exit(0)
    levels = [int(x) for x in sys.argv[1:]] or [1, 2, 4]
    for n in levels:
        run_level(n)
