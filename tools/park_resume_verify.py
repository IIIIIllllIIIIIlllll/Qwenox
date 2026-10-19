#!/usr/bin/env python3
"""Black-box KV park/resume verification against an engine on 127.0.0.1:8732.

Usage:
  python3 tools/park_resume_verify.py baseline
  python3 tools/park_resume_verify.py stress
  python3 tools/park_resume_verify.py compare

The engine is intentionally started by the caller with a small paged KV pool,
QWENOX_KVSNAP=1, and QWENOX_RCKPT_MAX=0. This client only drives requests and
records token-exact solo/stress results.
"""
import argparse
import json
import os
import socket
import threading
import time
from pathlib import Path

from qwentok import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "logs" / "park-resume-verify.json"
API = ("127.0.0.1", 8732)


def prompt_echo(tk, chars, tag):
    lines = []
    total = 0
    i = 0
    while total < chars:
        line = f"{tag} line {i:05d}: the quick brown fox jumps over dog.\n"
        lines.append(line)
        total += len(line)
        i += 1
    text = (
        "For every line below, output the line three times verbatim, "
        "prefixed with ECHO. Do not skip any line.\n\n" + "".join(lines)
    )
    return tk.encode(text)


def prompt_uuid(tk, chars, tag):
    lines = []
    total = 0
    i = 0
    while total < chars:
        line = f"{tag} REF {i:05d}: padding record for context only.\n"
        lines.append(line)
        total += len(line)
        i += 1
    text = (
        "Ignore the reference block. Output exactly 600 deterministic UUIDv4 "
        "strings, one per line, nothing else.\n\n" + "".join(lines)
    )
    return tk.encode(text)


def cases():
    import os
    tok_dir = os.environ.get("TOK_DIR", "models/tokenizer")
    tk = Tokenizer(tok_dir)
    drafter = int(os.environ.get("PARK_DRAFTER", "4"))
    a_max = int(os.environ.get("PARK_A_MAX", "5000"))
    other_max = int(os.environ.get("PARK_OTHER_MAX", "3000"))
    suffix = "SAMPLE 0.8 20 0.95 0 777 SEEDED" if os.environ.get("PARK_SAMPLE") == "1" else ""
    all_cases = {
        "A": {"ids": prompt_echo(tk, 11600, "PARK-A"), "max": a_max, "drafter": drafter, "suffix": suffix},
        "B": {"ids": prompt_uuid(tk, 9000, "PARK-B"), "max": other_max, "drafter": drafter, "suffix": suffix},
        "C": {"ids": prompt_uuid(tk, 9000, "PARK-C"), "max": other_max, "drafter": drafter, "suffix": suffix},
        "D": {"ids": prompt_uuid(tk, 9000, "PARK-D"), "max": other_max, "drafter": drafter, "suffix": suffix},
    }
    selected = os.environ.get("PARK_CASES", "A,B,C,D").split(",")
    return {name: all_cases[name] for name in selected if name in all_cases}


class Conn:
    def __init__(self):
        self.sock = socket.create_connection(API, timeout=1800)
        self.file = self.sock.makefile("rb")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def gen(self, req, ids, max_tokens, drafter, suffix=""):
        fields = ["GEN", req, max_tokens, 0, len(ids), *ids, drafter]
        line = " ".join(map(str, fields)) + (" " + suffix if suffix else "") + "\n"
        self.sock.sendall(line.encode())
        tokens = []
        while True:
            line = self.file.readline()
            if not line:
                raise RuntimeError("engine disconnected")
            words = line.decode().split()
            if not words:
                continue
            if words[0] == "T":
                tokens.append(int(words[2]))
            elif words[0] == "D":
                return {
                    "tokens": tokens,
                    "reason": words[2] if len(words) > 2 else "",
                    "done": " ".join(words),
                    "prompt": len(ids),
                }


def run_one(name, case, req=1):
    c = Conn()
    t0 = time.time()
    try:
        result = c.gen(req, case["ids"], case["max"], case["drafter"], case.get("suffix", ""))
        result["elapsed"] = time.time() - t0
        result["request"] = req
        result["case"] = name
        result["prompt"] = len(case["ids"])
        result["max_tokens"] = case["max"]
        print(
            f"{name}: {result['reason']} elapsed={result['elapsed']:.1f}s "
            f"prompt={result['prompt']} completion={len(result['tokens'])}",
            flush=True,
        )
        return result
    finally:
        c.close()


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def baseline():
    all_cases = cases()
    state = {"cases": {k: {"prompt": len(v["ids"]), "max": v["max"]} for k, v in all_cases.items()}, "baseline": {}}
    for i, (name, case) in enumerate(all_cases.items(), 1):
        state["baseline"][name] = run_one(name, case, i)
        save_state(state)
    print("BASELINE saved", flush=True)


def stress():
    all_cases = cases()
    result = {}
    errors = []
    # A owns the earliest sequence; later requests enter while A is decoding.
    # PARK_DELAYS can shorten/reshape the queue without changing the cases.
    delay_text = os.environ.get("PARK_DELAYS", "0,8,16,24").split(",")
    delays = {name: float(delay_text[i]) if i < len(delay_text) else 0.0
              for i, name in enumerate(all_cases)}

    def worker(i, name):
        try:
            time.sleep(delays[name])
            result[name] = run_one(name, all_cases[name], i + 1)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")

    threads = [threading.Thread(target=worker, args=(i, name)) for i, name in enumerate(all_cases)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        print("ERROR", "; ".join(errors), flush=True)
    state = load_state()
    state["stress"] = result
    save_state(state)
    if errors or len(result) != len(all_cases):
        raise SystemExit(1)
    compare_results(state["baseline"], result, "stress")
    print("STRESS saved", flush=True)


def compare():
    state = load_state()
    if "baseline" not in state or "stress" not in state:
        raise SystemExit("run baseline and stress first")
    compare_results(state["baseline"], state["stress"], "stress")


def compare_results(base, actual, label):
    bad = 0
    for name in sorted(base):
        a = base.get(name)
        b = actual.get(name)
        if not b:
            print(f"IDENTITY {name}: MISSING")
            bad += 1
            continue
        d = first_diff(a["tokens"], b["tokens"])
        ok = d is None and a["reason"] == b["reason"]
        if ok:
            print(f"IDENTITY {name}: MATCH")
        else:
            print(
                f"IDENTITY {name}: MISMATCH first_token={d} "
                f"base=({a['reason']},{len(a['tokens'])}) "
                f"{label}=({b['reason']},{len(b['tokens'])})"
            )
            bad += 1
    if bad:
        raise SystemExit(1)


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("baseline", "stress", "compare"))
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--cases", dest="cases", help="comma-separated case names")
    args = ap.parse_args()
    if args.cases:
        os.environ["PARK_CASES"] = args.cases
    STATE = Path(args.state)
    if args.mode == "baseline":
        baseline()
    elif args.mode == "stress":
        stress()
    else:
        compare()


if __name__ == "__main__":
    main()
