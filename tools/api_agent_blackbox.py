#!/usr/bin/env python3
"""Agent-like blackbox test against an OpenAI-compatible endpoint.

A self-contained client: it needs no external agent binary, only this repository's
API and the Python standard library. Each "agent" is a multi-step conversation
that grows its context the way a coding agent does -- a long shared system block,
a task, then a few follow-up steps that append synthetic tool output.

The identity check is the point: every agent must produce the same turn-by-turn
transcript when it runs alone (baseline) and when the agents run concurrently
(stress), even though concurrency changes KV admission and speculation batching.

Run against an already-started service, from a terminal that is not itself using
the service. Cases run sequentially in the baseline phase, so any other traffic
to the same API invalidates the comparison.

  python3 tools/api_agent_blackbox.py run --base http://127.0.0.1:8731 --output /tmp/agent-run
  python3 tools/api_agent_blackbox.py baseline --base ... --output ...   # phase 1
  python3 tools/api_agent_blackbox.py stress   --base ... --output ...   # phase 2
  python3 tools/api_agent_blackbox.py compare  --base ... --output ...   # phase 3
"""

import argparse
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
import uuid
from urllib.parse import urlsplit

_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_MAX_LOG_BYTES = 4 * 1024 * 1024
_MAX_PROMPT_CHARS = 2_000_000
_MAX_STATE_BYTES = 64 * 1024 * 1024
_MAX_CASES = 16
_CASE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")


class ApiError(RuntimeError):
    pass


class AgentFailure(RuntimeError):
    pass


# ---------------------------------------------------------------- HTTP plumbing

def _parse_base(base):
    parsed = urlsplit(base)
    if parsed.scheme != "http" or not parsed.hostname or parsed.query or parsed.fragment:
        raise ValueError("base must be an http:// URL with a host and no query/fragment")
    return parsed


def _request(base, method, path, body, timeout):
    parsed = _parse_base(base)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=timeout)
    try:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        conn.request(method, parsed.path.rstrip("/") + path,
                     body=None if body is None else json.dumps(body).encode("utf-8"),
                     headers=headers)
        response = conn.getresponse()
        data = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(data) > _MAX_RESPONSE_BYTES:
            raise ApiError("response exceeds safety limit")
        return response.status, data
    except (OSError, http.client.HTTPException) as exc:
        raise ApiError(f"{method} {path}: {exc}") from exc
    finally:
        conn.close()


def wait_ready(base, timeout):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            status, data = _request(base, "GET", "/health", None,
                                    min(5.0, max(0.1, deadline - time.monotonic())))
            last = json.loads(data)
            if (status == 200 and isinstance(last, dict) and last.get("status") == "ok"
                    and last.get("tokenizer") is True and last.get("engine") == "running"):
                return
        except (ApiError, ValueError, json.JSONDecodeError) as exc:
            last = str(exc)
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
    raise ApiError(f"API did not become ready: {last}")


def chat(base, body, timeout):
    started = time.monotonic()
    status, data = _request(base, "POST", "/v1/chat/completions", body, timeout)
    elapsed = time.monotonic() - started
    if status != 200:
        raise ApiError(f"chat returned {status}: {data[:2000]!r}")
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ApiError(f"chat returned invalid JSON: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("choices"), list) \
            or not value["choices"]:
        raise ApiError(f"chat response has no choices: {str(value)[:500]}")
    choice = value["choices"][0]
    message = choice.get("message")
    if not isinstance(message, dict):
        raise ApiError("chat choice has no message object")
    return {"content": message.get("content") or "",
            "finish": choice.get("finish_reason"),
            "usage": value.get("usage") if isinstance(value.get("usage"), dict) else {},
            "elapsed": round(elapsed, 3)}


# ---------------------------------------------------------------- agent pattern

def shared_context(chars, cases):
    """One prefix shared by every agent, then a per-case tail.

    A shared prefix is what real agents have (the same repo/system prompt), and it
    is what makes the engine share KV pages between concurrent requests.
    """
    if not isinstance(chars, int) or chars <= 0 or chars > _MAX_PROMPT_CHARS:
        raise ValueError(f"context chars must be in 1..{_MAX_PROMPT_CHARS}")
    lines, total, index = [], 0, 0
    while total < chars:
        line = f"reference {index:05d}: the service keeps paged KV state shared.\n"
        lines.append(line)
        total += len(line)
        index += 1
    return "".join(lines)


SYSTEM = ("You are a terse engineering assistant working in a repository. "
          "Answer each step in one short paragraph. Never repeat the reference block.")

TASK = ("Step 1: state one rule for admitting a request into a shared KV pool, "
        "and one rule for choosing which request to suspend under pressure.")


def step_message(step):
    return (f"Step {step}: a tool printed the following fixture output.\n"
            f"fixture[{step}] = {{admitted: true, pages: {16 + step}, owner: worker-{step}}}\n"
            f"Continue with the next operational check for step {step}. "
            "Answer in one short paragraph.")


def transcript_digest(turns):
    payload = json.dumps(turns, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_agent(args, name, context, delay=0.0, deadline=None):
    if delay:
        if deadline is not None and time.monotonic() + delay >= deadline:
            raise AgentFailure(f"{name}: global deadline reached before start")
        time.sleep(delay)
    if deadline is not None and time.monotonic() >= deadline:
        raise AgentFailure(f"{name}: global deadline reached before start")
    messages = [
        {"role": "system", "content": SYSTEM + "\n\n" + context + f"\n\nagent={name}"},
        {"role": "user", "content": TASK},
    ]
    turns, started = [], time.monotonic()
    for step in range(1, args.turns + 1):
        remaining = args.request_timeout
        if deadline is not None:
            remaining = min(remaining, max(0.01, deadline - time.monotonic()))
        body = {"model": args.model, "messages": messages, "max_tokens": args.max_tokens,
                "temperature": args.temperature}
        if args.seed is not None:
            body["seed"] = args.seed
        reply = chat(args.base, body, remaining)
        turns.append({"step": step, "content": reply["content"], "finish": reply["finish"]})
        messages.append({"role": "assistant", "content": reply["content"]})
        if step < args.turns:
            messages.append({"role": "user", "content": step_message(step)})
    return {"name": name, "turns": turns, "digest": transcript_digest(turns),
            "context_chars": len(context), "elapsed": round(time.monotonic() - started, 3)}


# ---------------------------------------------------------------- phases

def case_names(value):
    names = [item.strip() for item in value.split(",")]
    if not names or any(name in (".", "..") or not _CASE_RE.fullmatch(name) for name in names):
        raise ValueError("cases must be safe names [A-Za-z0-9_.-]")
    if len(set(names)) != len(names):
        raise ValueError("cases must be unique")
    if len(names) > _MAX_CASES:
        raise ValueError(f"too many cases; maximum is {_MAX_CASES}")
    return names


def parse_delays(value, count):
    try:
        values = [float(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("delays must be finite non-negative numbers") from exc
    if len(values) != count or any(not math.isfinite(item) or item < 0 for item in values):
        raise ValueError("delays count must equal cases count and values must be non-negative")
    return values


def save(path, value):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_STATE_BYTES:
        raise AgentFailure("state exceeds safety limit")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load(path):
    try:
        path = path.resolve()
        if path.stat().st_size > _MAX_STATE_BYTES:
            raise AgentFailure("state exceeds safety limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AgentFailure(f"cannot load state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AgentFailure(f"state must contain an object: {path}")
    return value


def config_of(args, context):
    return {"base": args.base, "model": args.model, "cases": args.case_names,
            "delays": args.delays_value, "context_chars": len(context),
            "turns": args.turns, "max_tokens": args.max_tokens,
            "temperature": args.temperature, "seed": args.seed}


def baseline(args, context):
    state = {"config": config_of(args, context), "baseline": {}, "baseline_errors": []}
    try:
        for name in args.case_names:
            state["baseline"][name] = run_agent(args, name, context)
            print(f"BASELINE {name}: {len(state['baseline'][name]['turns'])} turns "
                  f"{state['baseline'][name]['digest'][:16]}", flush=True)
    except Exception as exc:  # noqa: BLE001
        state["baseline_errors"].append(str(exc))
        raise
    finally:
        save(args.state, state)
    print(f"BASELINE saved {args.state}", flush=True)


def stress(args, context):
    state = load(args.state)
    if state.get("config") != config_of(args, context):
        raise AgentFailure("stress config does not exactly match baseline config")
    if state.get("baseline_errors") or set(state.get("baseline", {})) != set(args.case_names):
        raise AgentFailure("baseline is incomplete or failed")
    result, errors, lock = {}, [], threading.Lock()
    deadline = time.monotonic() + args.worker_timeout

    def worker(index, name):
        try:
            value = run_agent(args, name, context, args.delays_value[index], deadline)
            with lock:
                result[name] = value
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{name}: {exc}")

    threads = [threading.Thread(target=worker, args=(index, name), name=f"stress-{name}")
               for index, name in enumerate(args.case_names)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    alive = [thread.name for thread in threads if thread.is_alive()]
    state.update({"stress": result, "stress_errors": errors, "stress_threads_alive": alive})
    try:
        if errors or len(result) != len(args.case_names) or alive:
            raise AgentFailure("stress failed: " + "; ".join(errors or alive or ["missing result"]))
        print(f"STRESS saved {args.state}", flush=True)
    finally:
        save(args.state, state)


def compare(args, context):
    state = load(args.state)
    failures = []
    if state.get("config") != config_of(args, context):
        failures.append("config mismatch")
    if state.get("baseline_errors") or state.get("stress_errors") or state.get("stress_threads_alive"):
        failures.append("recorded run errors")
    base, actual = state.get("baseline", {}), state.get("stress", {})
    if set(base) != set(actual) or set(base) != set(args.case_names):
        failures.append("case set mismatch")
    for name in args.case_names:
        before, after = base.get(name), actual.get(name)
        same = (isinstance(before, dict) and isinstance(after, dict)
                and isinstance(before.get("turns"), list) and before.get("turns")
                and before.get("turns") == after.get("turns"))
        print(f"IDENTITY {name}: {'MATCH' if same else 'MISMATCH'}", flush=True)
        if not same:
            first = None
            if isinstance(before, dict) and isinstance(after, dict):
                a, b = before.get("turns") or [], after.get("turns") or []
                for index, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        first = index
                        break
                if first is None and len(a) != len(b):
                    first = min(len(a), len(b))
            print(f"  first differing turn: {first}", flush=True)
            failures.append(f"case {name} mismatch")
    if failures:
        raise SystemExit("RESULT FAIL: " + "; ".join(failures))
    print("RESULT PASS", flush=True)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("run", "baseline", "stress", "compare"))
    parser.add_argument("--base", default="http://127.0.0.1:8731")
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory; created by baseline/run, reused by later phases")
    parser.add_argument("--cases", default="A,B")
    parser.add_argument("--delays", default="0,2",
                        help="per-case start delay in the stress phase")
    parser.add_argument("--context-chars", type=int, default=6000,
                        help="length of the shared system prefix in characters")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy, the only setting where transcripts must match")
    parser.add_argument("--seed", type=int, default=None,
                        help="optional sampling seed; leave unset for greedy")
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--request-timeout", type=positive, default=300)
    parser.add_argument("--worker-timeout", type=positive, default=900)
    parser.add_argument("--ready-timeout", type=positive, default=1800)
    args = parser.parse_args()
    try:
        args.output = args.output.expanduser().resolve()
        args.state = args.output / "results.json"
        args.case_names = case_names(args.cases)
        args.delays_value = parse_delays(args.delays, len(args.case_names))
        if not 1 <= args.turns <= 16:
            parser.error("turns must be in 1..16")
        if not 1 <= args.max_tokens <= 8192:
            parser.error("max-tokens must be in 1..8192")
        if args.temperature != 0 and args.seed is None:
            parser.error("temperature != 0 needs --seed, otherwise transcripts are random")
        context = shared_context(args.context_chars, args.case_names)
        if args.mode in ("run", "baseline"):
            args.output.mkdir(parents=True, exist_ok=False)
        wait_ready(args.base, args.ready_timeout)
        if args.mode == "run":
            baseline(args, context)
            stress(args, context)
            compare(args, context)
        elif args.mode == "baseline":
            baseline(args, context)
        elif args.mode == "stress":
            stress(args, context)
        else:
            compare(args, context)
    except (ApiError, OSError, ValueError, AgentFailure) as exc:
        print(f"RESULT FAIL: {exc}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
