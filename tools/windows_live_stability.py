"""Bounded live smoke/soak tests against Qwenox. Does not change server config.

Uses synthetic prompts only. Results may contain server configuration/address;
keep --output outside version control. An idle server is required at startup.
"""

import argparse
import concurrent.futures
import json
import pathlib
import time
import urllib.request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base", required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--soak-seconds", type=int, default=0)
parser.add_argument("--pause-seconds", type=float, default=2)
args = parser.parse_args()
if not 0 <= args.soak_seconds <= 3600:
    parser.error("--soak-seconds must be 0..3600")
if not 0.5 <= args.pause_seconds <= 60:
    parser.error("--pause-seconds must be 0.5..60")
BASE = args.base.rstrip("/")
OUT = args.output
OUT.mkdir(parents=True, exist_ok=True)
rows = []


def call(path, body=None, timeout=45):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=timeout)


def health():
    with call("/health", timeout=5) as r:
        return json.load(r)


def idle():
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        h = health()
        if h["engine"] != "running":
            raise RuntimeError("engine stopped")
        if not h["in_flight"] and not h["queued"]:
            return h
        time.sleep(0.5)
    raise RuntimeError("engine did not become idle")


def run(name, prompt, stream=False):
    body = {
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 64,
        "stream": stream,
    }
    start = time.monotonic()
    with call("/v1/chat/completions", body) as r:
        if stream:
            done = False
            text = ""
            first = None
            for line in r:
                if not line.startswith(b"data: "):
                    continue
                data = line[6:].strip()
                if data == b"[DONE]":
                    done = True
                    break
                event = json.loads(data)
                for choice in event.get("choices", []):
                    part = choice.get("delta", {}).get("content", "")
                    if part and first is None:
                        first = time.monotonic() - start
                    text += part
            if not done or not text:
                raise RuntimeError("incomplete stream")
            row = {
                "case": name,
                "seconds": time.monotonic() - start,
                "first_content_s": first,
                "characters": len(text),
            }
        else:
            data = json.load(r)
            choice = data["choices"][0]
            if not choice["message"]["content"]:
                raise RuntimeError("empty content")
            row = {
                "case": name,
                "seconds": time.monotonic() - start,
                "usage": data["usage"],
                "finish_reason": choice["finish_reason"],
            }
    return row


def save(row):
    rows.append(row)
    print(json.dumps(row), flush=True)
    (OUT / "smoke-results.json").write_text(json.dumps(rows, indent=2))


try:
    initial = health()
    if initial.get("in_flight") or initial.get("queued"):
        raise RuntimeError("server is busy; retry when idle")
    save({"case": "initial_health", "health": idle()})
    for i in range(10):
        save(
            run(
                "serial-" + str(i),
                "Write one short sentence about testing software. Case " + str(i),
            )
        )
        idle()
    save(run("stream", "List three benefits of software tests.", True))
    idle()
    for words in (256, 1024, 4096):
        prompt = "Read the notes and reply with exactly OK.\n" + (
            "The system is being tested. " * words
        )
        save(run("prefill-" + str(words) + "-sentences", prompt))
        idle()
    if health().get("slots", 1) >= 2:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for row in pool.map(
                lambda n: run(
                    "parallel-" + str(n), "Count from one to twenty. Test " + str(n)
                ),
                range(2),
            ):
                save(row)
        idle()
    body = {
        "messages": [
            {"role": "user", "content": "Count from 1 to 1000, one number per line."}
        ],
        "enable_thinking": False,
        "temperature": 0,
        "max_tokens": 512,
        "stream": True,
    }
    got = False
    start = time.monotonic()
    with call("/v1/chat/completions", body) as r:
        for line in r:
            if line.startswith(b"data: ") and line[6:].strip() != b"[DONE]":
                event = json.loads(line[6:])
                if any(
                    c.get("delta", {}).get("content") for c in event.get("choices", [])
                ):
                    got = True
                    break
    if not got:
        raise RuntimeError("cancel test had no content")
    idle()
    save({"case": "disconnect-to-idle", "seconds": time.monotonic() - start})
    save(run("after-cancel", "Reply with exactly OK."))
    save({"case": "final_health", "health": idle()})
    deadline = time.monotonic() + args.soak_seconds
    round_no = 0
    while time.monotonic() < deadline:
        idle()
        save(
            run(
                "soak-" + str(round_no),
                "Write a short explanation of why software testing is useful. Case "
                + str(round_no),
                round_no % 2 == 0,
            )
        )
        idle()
        round_no += 1
        time.sleep(args.pause_seconds)
    if args.soak_seconds:
        save({"case": "post-soak-health", "health": idle()})
    print("ALL PASS", flush=True)
except Exception as e:
    save({"case": "failure", "error": repr(e)})
    raise
