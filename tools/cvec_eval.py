#!/usr/bin/env python3
"""cvec_eval.py — 控制向量（refusal projection）开/关对比的轻量抓取工具。

对已经在监听的 qwenox API（默认 http://127.0.0.1:8731）发固定数量的中性 prompt，
贪心生成固定长度，记录客户端 wall 时间、usage 里的 completion_tokens、decode tok/s
和输出前 80 字，写成 JSON。开/关向量各跑一次，用同一份 JSON 对比即可。

用法：
  python3 tools/cvec_eval.py --tag stock
  python3 tools/cvec_eval.py --tag project --runs 3
  python3 tools/cvec_eval.py --base http://127.0.0.1:8731 --out logs/cvec/stock.json

不包含任何有害 prompt；它只验证加载后模型仍能正常连贯生成，以及开销量级。
"""
import argparse
import json
import time
import urllib.request

PROMPTS = [
    "写一段 80 字左右的短文，介绍秋天的清晨。",
    "用三句话解释什么是梯度下降。",
    "写一个 Python 函数，判断一个字符串是不是回文。",
]


def chat(base, prompt, max_tokens, timeout):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    wall = time.time() - t0
    data = json.loads(raw)
    usage = data.get("usage") or {}
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return {
        "prompt": prompt,
        "wall_s": round(wall, 3),
        "completion_tokens": usage.get("completion_tokens", 0),
        "text80": text.replace("\n", " ")[:80],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8731")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--runs", type=int, default=1, help="每个 prompt 重复次数")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="", help="默认 logs/cvec/<tag>.json")
    args = ap.parse_args()

    rows = []
    for _ in range(args.runs):
        for p in PROMPTS:
            rows.append(chat(args.base, p, args.max_tokens, args.timeout))

    # decode 速度：只统计有足够 completion token 的样本（wall 含 prefill/TTFT）
    speeds = [r["completion_tokens"] / r["wall_s"] for r in rows
              if r["wall_s"] > 0 and r["completion_tokens"] >= 32]
    summary = {
        "tag": args.tag,
        "base": args.base,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "wall_mean_s": round(sum(r["wall_s"] for r in rows) / len(rows), 3),
        "client_tps_min": round(min(speeds), 2) if speeds else None,
        "client_tps_mean": round(sum(speeds) / len(speeds), 2) if speeds else None,
        "empty_outputs": sum(1 for r in rows if not r["text80"]),
    }
    out = args.out or f"logs/cvec/{args.tag}.json"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))
    for r in rows:
        print(f"  {r['completion_tokens']:4d} tok {r['wall_s']:7.2f}s | {r['text80']}")


if __name__ == "__main__":
    main()
