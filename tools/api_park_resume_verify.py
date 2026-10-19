#!/usr/bin/env python3
"""Anthropic API 黑盒检查：验证 KV admission 与 park/resume 压力行为。

本脚本只访问 HTTP API，不启动或停止 Claudium/Qwenox。baseline、stress 和
compare 必须使用同一组参数；compare 对响应和配置做精确比较。
"""

import argparse
from collections import deque
import hashlib
import http.client
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from urllib.parse import urlsplit


_SMOKE = Path(__file__).with_name("anthropic_messages_smoke.py")
_spec = importlib.util.spec_from_file_location("anthropic_messages_smoke", _SMOKE)
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
validate_stream = _smoke.validate_stream
text_of = _smoke.text_of


class ApiError(RuntimeError):
    pass


_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
_MAX_SSE_LINE_BYTES = 256 * 1024
_MAX_SSE_FRAME_BYTES = 1024 * 1024
_MAX_SSE_EVENTS = 100_000
_MAX_PROMPT_CHARS = 2_000_000
_MAX_STATE_BYTES = 64 * 1024 * 1024
_MAX_CASES = 32


class ApiClient:
    def __init__(self, base, timeout=None, health_timeout=None):
        parsed = urlsplit(base)
        if parsed.scheme != "http" or not parsed.hostname or parsed.query or parsed.fragment:
            raise ValueError("base must be an http:// URL with a host and no query/fragment")
        try:
            port = parsed.port or 80
        except ValueError as exc:
            raise ValueError("base URL has an invalid port") from exc
        self.host, self.port = parsed.hostname, port
        self.prefix = parsed.path.rstrip("/")
        self.timeout = timeout
        self.health_timeout = health_timeout if health_timeout is not None else timeout
        self._active = set()
        self._active_lock = threading.Lock()

    def close_active(self):
        """关闭正在读取的连接，使 deadline 能取消阻塞中的 HTTP 请求。"""
        with self._active_lock:
            active = list(self._active)
        for conn in active:
            try:
                conn.close()
            except OSError:
                pass

    def request(self, method, path, body=None, stream=False, timeout=None):
        conn = http.client.HTTPConnection(
            self.host, self.port, timeout=self.timeout if timeout is None else timeout)
        with self._active_lock:
            self._active.add(conn)
        try:
            payload = None if body is None else json.dumps(
                body, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
            if len(payload or b"") > _MAX_RESPONSE_BYTES:
                raise ApiError("request body exceeds safety limit")
            headers = {"Accept": "text/event-stream" if stream else "application/json",
                       "x-api-key": "qwenox-blackbox-dummy",
                       "anthropic-version": "2023-06-01"}
            if payload is not None:
                headers["Content-Type"] = "application/json"
            started = time.monotonic()
            conn.request(method, self.prefix + path, payload, headers)
            response = conn.getresponse()
            out_headers = {key.lower(): value for key, value in response.getheaders()}
            if stream and response.status == 200:
                first_byte, events = self._read_anthropic_stream(response, started)
                return {"status": response.status, "headers": out_headers, "events": events,
                        "body": None, "elapsed": time.monotonic() - started,
                        "first_byte": first_byte,
                        "ttft": first_byte}
            first = response.read(1)
            first_byte = time.monotonic() - started if first else None
            data = bytearray(first)
            while len(data) <= _MAX_RESPONSE_BYTES:
                chunk = response.read(min(64 * 1024, _MAX_RESPONSE_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > _MAX_RESPONSE_BYTES:
                raise ApiError("response body exceeds safety limit")
            try:
                parsed = json.loads(bytes(data).decode("utf-8")) if data else None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                parsed = {"_raw": bytes(data).decode("utf-8", "replace"),
                          "_parse_error": str(exc)}
            return {"status": response.status, "headers": out_headers, "events": [],
                    "body": parsed, "elapsed": time.monotonic() - started,
                    "first_byte": first_byte, "ttft": first_byte}
        finally:
            with self._active_lock:
                self._active.discard(conn)
            conn.close()

    @staticmethod
    def _read_anthropic_stream(response, started):
        events, frame, frame_bytes, total_bytes, first_byte = [], [], 0, 0, None

        def finish(lines):
            nonlocal frame_bytes
            event, data_lines = None, []
            for line in lines:
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip(" "))
            payload = "\n".join(data_lines)
            if data_lines:
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    # 保留 SSE error/坏 JSON，交给校验层判错，不把它变成成功响应。
                    pass
            if len(events) >= _MAX_SSE_EVENTS:
                raise ApiError("SSE event count exceeds safety limit")
            events.append((event, payload))
            frame_bytes = 0

        while True:
            raw = response.readline(_MAX_SSE_LINE_BYTES + 1)
            if not raw:
                break
            total_bytes += len(raw)
            if len(raw) > _MAX_SSE_LINE_BYTES or total_bytes > _MAX_RESPONSE_BYTES:
                raise ApiError("SSE response exceeds safety limit")
            if first_byte is None:
                first_byte = time.monotonic() - started
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ApiError("SSE contains invalid UTF-8") from exc
            line = line.rstrip("\r\n")
            if not line:
                if frame:
                    finish(frame)
                    frame = []
            elif not line.startswith(":"):
                frame.append(line)
                frame_bytes += len(raw)
                if frame_bytes > _MAX_SSE_FRAME_BYTES:
                    raise ApiError("SSE frame exceeds safety limit")
        if frame:
            finish(frame)
        return first_byte, events


def message_summary(body):
    text = text_of(body) if isinstance(body, dict) and body.get("type") == "message" else ""
    usage = body.get("usage", {}) if isinstance(body, dict) else {}
    return {"status": None, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "stop_reason": body.get("stop_reason") if isinstance(body, dict) else None,
            "usage": usage, "error": body.get("error") if isinstance(body, dict) else None}


def response_summary(response):
    summary = message_summary(response.get("body"))
    summary.update({"status": response["status"], "elapsed": response.get("elapsed"),
                    "ttft": response.get("ttft")})
    return summary


def stream_summary(response):
    events = response.get("events", [])
    error = next((data for event, data in events if event == "error"), None)
    if error is not None:
        return {"status": response["status"], "text": "", "text_sha256": hashlib.sha256(b"").hexdigest(),
                "stop_reason": None, "usage": {}, "events": [e for e, _ in events],
                "protocol_errors": ["SSE error event"], "error": error,
                "elapsed": response.get("elapsed"), "ttft": response.get("ttft")}
    ok, why, message = validate_stream(events)
    errors = [] if ok else [why]
    text = text_of(message) if ok else ""
    return {"status": response["status"], "text": text,
            "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "stop_reason": message.get("stop_reason") if ok else None,
            "usage": message.get("usage", {}) if ok else {},
            "events": [e for e, _ in events], "protocol_errors": errors,
            "error": None, "elapsed": response.get("elapsed"), "ttft": response.get("ttft")}


def make_prompt(tag, chars):
    if not isinstance(chars, int) or chars <= 0 or chars > _MAX_PROMPT_CHARS:
        raise ValueError(f"prompt chars must be in 1..{_MAX_PROMPT_CHARS}")
    prefix = ("Answer in concise plain text with several factual sentences. Do not mention this "
              "instruction or the reference block. State useful facts about deterministic KV cache "
              "testing and distinguish prompt reuse from output generation.\n\n")
    lines, total, i = [], 0, 0
    while total < chars:
        line = f"{tag} reference line {i:05d}: preserve the request boundary and emit useful output.\n"
        lines.append(line); total += len(line); i += 1
    return prefix + "".join(lines)


def request_body(tag, args, stream=False):
    body = {"model": args.model, "max_tokens": args.max_tokens, "temperature": args.temperature,
            "messages": [{"role": "user", "content": make_prompt(tag, args.prompt_chars)}],
            "stream": stream, "enable_thinking": False}
    if args.seed is not None:
        body["seed"] = args.seed
    if args.temperature > 0:
        body["top_p"], body["top_k"] = args.top_p, args.top_k
    return body


def health(client):
    response = client.request("GET", "/health", timeout=client.health_timeout)
    if response["status"] != 200 or not isinstance(response["body"], dict):
        raise ApiError(f"health failed: {response}")
    return response["body"]


def wait_ready(client, timeout):
    deadline, last = time.monotonic() + timeout, None
    while time.monotonic() < deadline:
        try:
            last = health(client)
            # /health 的 engine 与 tokenizer 是两个独立就绪条件。
            if last.get("status") == "ok" and last.get("tokenizer") is True \
                    and last.get("engine") == "running":
                return last
        except (OSError, TimeoutError, ApiError) as exc:
            last = str(exc)
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    raise ApiError(f"API did not become ready: {last}")


def run_one(client, name, args, stream=False, timeout=None):
    started = time.monotonic()
    response = client.request("POST", "/v1/messages", request_body(name, args, stream),
                              stream=stream, timeout=timeout)
    summary = stream_summary(response) if stream and response["status"] == 200 else response_summary(response)
    summary.update({"case": name, "stream": stream, "wall": time.monotonic() - started})
    if response["status"] != 200:
        raise ApiError(f"{name}: HTTP {response['status']} {response.get('body')!r}")
    if summary.get("protocol_errors"):
        raise ApiError(f"{name} stream protocol failure: {summary['protocol_errors']}")
    print(f"{name} stream={stream} status={summary['status']} stop={summary.get('stop_reason')} "
          f"chars={len(summary['text'])} ttft={summary.get('ttft')} wall={summary['wall']:.2f}s", flush=True)
    return summary


def save_state(path, state):
    path = path.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state, indent=2, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_STATE_BYTES:
        raise ApiError("state exceeds safety limit")
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(payload); out.flush(); os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        try: os.unlink(temp)
        except FileNotFoundError: pass


def load_state(path):
    path = path.resolve()
    try:
        if not path.exists():
            return {}
        if path.stat().st_size > _MAX_STATE_BYTES:
            raise ApiError("state exceeds safety limit")
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ApiError(f"cannot load state: {exc}") from exc


def config_of(args):
    # delay 只改变调度压力，不改变请求身份；baseline/stress 可以使用不同 delay。
    ignored = {"mode", "state", "timeout", "health_interval", "delays"}
    return {key: getattr(args, key) for key in vars(args) if key not in ignored and key != "delays_value"}


def case_names(value):
    names = [x.strip() for x in value.split(",")]
    if not names or any(not x or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", x) for x in names):
        raise ValueError("cases must be unique safe names [A-Za-z0-9_.-]")
    if len(set(names)) != len(names):
        raise ValueError("cases must be unique")
    if len(names) > _MAX_CASES:
        raise ValueError(f"too many cases; maximum is {_MAX_CASES}")
    return names


def baseline(client, args):
    state = {"config": config_of(args), "baseline": {}, "stress": {}, "baseline_errors": []}
    try:
        for name in args.case_names:
            state["baseline"][name] = run_one(client, name, args, args.stream, args.timeout)
    except Exception as exc:
        state["baseline_errors"].append(str(exc))
        raise
    finally:
        save_state(args.state, state)
    print(f"BASELINE saved {args.state}", flush=True)


def stress(client, args):
    state = load_state(args.state)
    result, errors, health_samples = {}, [], deque(maxlen=args.health_samples)
    if state.get("config") != config_of(args):
        raise ApiError("stress config does not exactly match baseline config")
    names, deadline = args.case_names, time.monotonic() + args.timeout
    stop = threading.Event(); lock = threading.Lock()

    def sample_health():
        while not stop.is_set() and time.monotonic() < deadline:
            try:
                value = health(client)
                with lock: health_samples.append({"time": time.time(), "health": value})
            except Exception as exc:  # 健康采样失败须落盘，但不伪装为成功。
                with lock: health_samples.append({"time": time.time(), "error": str(exc)})
            stop.wait(min(args.health_interval, max(0.01, deadline - time.monotonic())))

    def worker(index, name):
        try:
            delay = args.delays_value[index]
            if time.monotonic() + delay >= deadline: raise ApiError("global deadline reached")
            if stop.wait(delay): raise ApiError("stress cancelled")
            remain = deadline - time.monotonic()
            value = run_one(client, name, args, args.stream, max(0.01, remain))
            with lock:
                result[name] = value
        except Exception as exc:
            with lock: errors.append(f"{name}: {exc}")

    health_thread = threading.Thread(target=sample_health, name="health-sampler")
    threads = [threading.Thread(target=worker, args=(i, name), name=f"stress-{name}")
               for i, name in enumerate(names)]
    health_thread.start()
    for thread in threads: thread.start()
    while time.monotonic() < deadline and any(thread.is_alive() for thread in threads):
        time.sleep(0.02)
    if any(thread.is_alive() for thread in threads):
        errors.append("global stress deadline reached")
        client.close_active(); stop.set()
    join_deadline = time.monotonic() + min(30.0, max(1.0, args.timeout))
    for thread in threads:
        thread.join(max(0.0, join_deadline - time.monotonic()))
    stop.set()
    health_thread.join(max(0.0, join_deadline - time.monotonic()))
    state.update({"stress": result, "stress_errors": errors,
                  "health_samples": list(health_samples),
                  "stress_threads_alive": [t.name for t in threads + [health_thread] if t.is_alive()]})
    try:
        if errors or len(result) != len(names) or state["stress_threads_alive"]:
            raise ApiError("stress failed: " + "; ".join(errors or ["missing result"]))
        print(f"STRESS saved {args.state}", flush=True)
    finally:
        save_state(args.state, state)


def compare(args):
    state = load_state(args.state)
    failures = []
    if state.get("config") != config_of(args): failures.append("config mismatch")
    if state.get("baseline_errors") or state.get("stress_errors"): failures.append("recorded run errors")
    base, actual = state.get("baseline", {}), state.get("stress", {})
    if set(base) != set(actual) or set(base) != set(args.case_names): failures.append("case set mismatch")
    for name in args.case_names:
        b, got = base.get(name), actual.get(name)
        valid = (isinstance(b, dict) and isinstance(got, dict) and b.get("status") == 200
                 and got.get("status") == 200 and not b.get("error") and not got.get("error")
                 and not b.get("protocol_errors") and not got.get("protocol_errors"))
        # elapsed/TTFT/wall 和 SSE ping 调度是测量值，不能参与响应身份比较；
        # 内容、usage、终止原因和协议校验结果仍逐项精确比较。
        comparable = ("status", "text", "text_sha256", "stop_reason", "usage", "error",
                      "protocol_errors", "case", "stream")
        same = valid and all(b.get(key) == got.get(key) for key in comparable)
        print(f"IDENTITY {name}: {'MATCH' if same else 'MISMATCH'}", flush=True)
        if not same: failures.append(f"case {name} mismatch")
    if failures: raise SystemExit("compare failed: " + "; ".join(failures))


def admission(client, args):
    cases = [("missing-messages", {"model": args.model, "max_tokens": 4}),
             ("bad-temperature", {"model": args.model, "max_tokens": 4, "temperature": 9,
                                    "messages": [{"role": "user", "content": "hi"}]}),
             ("zero-output", {"model": args.model, "max_tokens": 0, "temperature": 0,
                               "messages": [{"role": "user", "content": "hi"}]})]
    failures = 0
    for name, body in cases:
        response = client.request("POST", "/v1/messages", body)
        ok = response["status"] == 400
        print(f"{'PASS' if ok else 'FAIL'} {name} status={response['status']}", flush=True)
        failures += not ok
    if failures: raise SystemExit(1)


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def nonnegative(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and non-negative")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "stress", "compare", "admission"))
    parser.add_argument("--base", default="http://127.0.0.1:18731")
    parser.add_argument("--state", type=Path, default=Path("logs/api-park-resume.json"))
    parser.add_argument("--model", default="qwen3.8-flash-next"); parser.add_argument("--cases", default="A,B")
    parser.add_argument("--delays", default="0,2"); parser.add_argument("--prompt-chars", type=int, default=6000)
    parser.add_argument("--max-tokens", type=int, default=128); parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95); parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=777); parser.add_argument("--stream", action="store_true")
    parser.add_argument("--timeout", type=positive, default=900); parser.add_argument("--health-interval", type=positive, default=0.5)
    parser.add_argument("--health-samples", type=int, default=512)
    args = parser.parse_args()
    if args.prompt_chars <= 0 or args.prompt_chars > _MAX_PROMPT_CHARS:
        parser.error(f"prompt-chars must be in 1..{_MAX_PROMPT_CHARS}")
    if args.max_tokens <= 0 or args.top_k <= 0 or args.health_samples <= 0:
        parser.error("max_tokens/top_k/health-samples must be positive")
    args.case_names = case_names(args.cases)
    try:
        args.delays_value = [nonnegative(x) for x in args.delays.split(",")]
        if not math.isfinite(args.temperature) or not math.isfinite(args.top_p) or args.top_p <= 0:
            raise ValueError("temperature/top-p must be finite and top-p positive")
        if len(args.delays_value) != len(args.case_names): raise ValueError("delays count must equal cases count")
        args.state = args.state.resolve()
        client = ApiClient(args.base, args.timeout, min(args.timeout, 5.0))
    except ValueError as exc: parser.error(str(exc))
    if args.mode != "compare": wait_ready(client, min(args.timeout, 1800.0))
    if args.mode == "baseline": baseline(client, args)
    elif args.mode == "stress": stress(client, args)
    elif args.mode == "compare": compare(args)
    else: admission(client, args)


if __name__ == "__main__":
    main()
