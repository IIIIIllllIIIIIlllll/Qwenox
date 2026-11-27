#!/usr/bin/env python3
import json, sys, time, urllib.request

HOST = "http://192.168.5.12:8731"
PROMPTS = [
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring.",
    "用 Python 写一个函数：统计一段文本中每个汉字出现的次数，按频率降序返回。",
    "Write a Python class implementing a LRU cache with get and put in O(1).",
    "Write a short story about a robot learning to paint, about 150 words.",
]

def one(prompt, max_tokens, seed):
    body = json.dumps({"model": "qwen", "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0.7, "top_k": 40, "seed": seed,
                       "max_tokens": max_tokens, "stream": False}).encode()
    req = urllib.request.Request(HOST + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    return resp.get("usage", {}).get("completion_tokens", 0), dt

for i in range(4):
    ct, dt = one(PROMPTS[i % len(PROMPTS)], 192, 42)
    print(f"req {i}: {ct} tok in {dt:.2f}s = {ct/dt:.1f} tok/s", flush=True)
