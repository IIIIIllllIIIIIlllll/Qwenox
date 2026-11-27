#!/usr/bin/env python3
"""Long-context bit-exact gate: big synthetic prompt (10K or 43.5K ctx) +
seeded sampling decode; SHA of the completion compared across engine configs."""
import hashlib
import json
import sys
import time
import urllib.request

HOST = "http://192.168.5.12:8731"


def make_prompt(target_tokens):
    base = ("The committee reviewed the quarterly report on renewable energy "
            "grid integration, noting that transformer substation 17 required "
            "a firmware audit before the winter peak. ")
    # ~14 tokens per sentence; pad by repeating with an index prefix
    n = int(target_tokens / 14) + 2
    parts = []
    for i in range(n):
        parts.append(f"[entry {i:06d}] " + base)
    return "".join(parts) + "\nSummarize entry 000000 in one sentence."


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
    with urllib.request.urlopen(req, timeout=900) as r:
        resp = json.loads(r.read())
    dt = time.perf_counter() - t0
    text = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage", {})
    return text, usage, dt


def main():
    tag = sys.argv[1]
    ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    text, usage, dt = one(make_prompt(ctx), 128, 42)
    sha = hashlib.sha256(text.encode()).hexdigest()
    pt = usage.get("prompt_tokens", 0)
    ct = usage.get("completion_tokens", 0)
    print(f"ctx={ctx} prompt_tokens={pt} completion={ct} dt={dt:.1f}s sha={sha[:16]}")
    with open(f"outputs/longctx_{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"prompt_tokens": pt, "completion_tokens": ct,
                   "sha256": sha, "text": text}, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
