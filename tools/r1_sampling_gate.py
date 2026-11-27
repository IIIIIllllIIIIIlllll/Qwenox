#!/usr/bin/env python3
"""Short-ctx sampling decode bench + bit-exact gate for gfx1151-engine R1.
Issues seeded sampling requests to the serve on 192.168.5.12:8731, prints
per-request tok/s (server-side usage / wall) and writes outputs for SHA256
comparison across builds/env toggles."""
import hashlib
import json
import sys
import time
import urllib.request

HOST = "http://192.168.5.12:8731"
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
    "用 Python 写一个函数：统计一段文本中每个汉字出现的次数，按频率降序返回。",
    "Write a Python class implementing a LRU cache with get and put in O(1).",
    "Write a short story about a robot learning to paint, about 150 words.",
]


def one(prompt, max_tokens, seed):
    body = json.dumps({
        "model": "qwen",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "top_k": 40,
        "seed": seed,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    return text, usage, dt


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "run"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    max_tokens = 192
    out = {}
    rates = []
    for i in range(n):
        text, usage, dt = one(PROMPTS[i % len(PROMPTS)], max_tokens, 42)
        ct = usage.get("completion_tokens", 0)
        sha = hashlib.sha256(text.encode()).hexdigest()[:16]
        rate = ct / dt if dt > 0 else 0
        if i > 0:
            rates.append(rate)
        print(f"req {i}: {ct} tok in {dt:.1f}s = {rate:.1f} tok/s (client) sha={sha}")
        out[f"req{i}"] = {"sha256": hashlib.sha256(text.encode()).hexdigest(),
                          "tokens": ct, "text": text}
    if rates:
        print(f"mean (excl warmup): {sum(rates)/len(rates):.2f} tok/s client-side")
    with open(f"outputs/r1gate_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"saved outputs/r1gate_{tag}.json")


if __name__ == "__main__":
    main()
