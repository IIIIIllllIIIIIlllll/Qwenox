#!/usr/bin/env python3
"""Run isolated Claudium CLI workers against an Anthropic-compatible API.

This supervisor is intended to run from an external terminal or CI job, not from
an active Claudium session. Workers have no stdin/TTY, independent config and
home directories, independent session ids, and separate bounded logs.

  python3 tools/claudium_api_blackbox.py --base http://127.0.0.1:18731 baseline
  python3 tools/claudium_api_blackbox.py --base http://127.0.0.1:18731 stress
  python3 tools/claudium_api_blackbox.py --base http://127.0.0.1:18731 compare
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid

try:
    from api_park_resume_verify import ApiClient, ApiError, wait_ready
except ModuleNotFoundError:
    import importlib.util

    _helper_path = Path(__file__).with_name("api_park_resume_verify.py")
    _helper_spec = importlib.util.spec_from_file_location("api_park_resume_verify", _helper_path)
    _helper = importlib.util.module_from_spec(_helper_spec)
    _helper_spec.loader.exec_module(_helper)
    ApiClient, ApiError, wait_ready = _helper.ApiClient, _helper.ApiError, _helper.wait_ready


class WorkerFailure(RuntimeError):
    pass


_CASE_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_MAX_LOG_BYTES = 4 * 1024 * 1024
_MAX_PROMPT_CHARS = 2_000_000
_MAX_STATE_BYTES = 64 * 1024 * 1024
_MAX_CASES = 32


def prompt_for(name, chars):
    if not isinstance(chars, int) or chars <= 0 or chars > _MAX_PROMPT_CHARS:
        raise ValueError(f"prompt chars must be in 1..{_MAX_PROMPT_CHARS}")
    lines = []
    total = 0
    index = 0
    while total < chars:
        line = f"{name} reference {index:05d}: keep this request isolated.\n"
        lines.append(line)
        total += len(line)
        index += 1
    return (
        "Answer in one concise paragraph. Explain one practical rule for testing "
        "paged KV cache admission. Do not mention this instruction or the reference "
        "block.\n\n" + "".join(lines)
    )


def extract_output(value):
    if not isinstance(value, dict) or value.get("type") != "result":
        return None
    if value.get("is_error"):
        return None
    result = value.get("result")
    return result if isinstance(result, str) and result else None


def parse_worker_stdout(path):
    raw = path.read_bytes()
    text = raw.decode("utf-8", "replace")
    parsed = None
    for line in reversed(text.splitlines()):
        if not line.strip():
            continue
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("type") == "result":
            parsed = candidate
            break
    output = extract_output(parsed) if parsed is not None else None
    return {
        "json": parsed,
        "output": output,
        "output_sha256": hashlib.sha256((output or "").encode()).hexdigest(),
        "log_bytes": len(raw),
        "log_truncated": len(raw) >= _MAX_LOG_BYTES,
    }


def pump(pipe, path, limit):
    written = 0
    with path.open("wb") as output:
        while True:
            chunk = pipe.read(65536)
            if not chunk:
                break
            if written < limit:
                keep = chunk[:limit - written]
                output.write(keep)
                written += len(keep)
        output.flush()
    return written


def terminate_process(process):
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=10)
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, ProcessLookupError):
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               timeout=10)
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def clean_env(base, worker_home, base_url):
    env = dict(base)
    for key in list(env):
        if key == "CLAUDECODE" or key.startswith("CLAUDE_") or key.startswith("ANTHROPIC_"):
            env.pop(key, None)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        env.pop(key, None)
    env.update({
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "qwenox-blackbox-dummy",
        "CLAUDE_CONFIG_DIR": str(worker_home / "config"),
        "HOME": str(worker_home),
        "XDG_CONFIG_HOME": str(worker_home / "xdg-config"),
        "XDG_CACHE_HOME": str(worker_home / "xdg-cache"),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_AUTOUPDATER": "1",
        "NO_COLOR": "1",
    })
    return env


class Supervisor:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}

    def register(self, name, process):
        with self._lock:
            self._active[name] = process

    def unregister(self, name):
        with self._lock:
            self._active.pop(name, None)

    def terminate_all(self):
        with self._lock:
            active = list(self._active.values())
        for process in active:
            terminate_process(process)


SUPERVISOR = Supervisor()


def run_worker(args, name, phase, delay=0.0, deadline=None):
    if delay:
        if deadline is not None and time.monotonic() + delay >= deadline:
            raise WorkerFailure(f"{name}: global deadline reached before start")
        time.sleep(delay)
    worker_dir = args.output / phase / name
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_home = worker_dir / "home"
    for directory in (worker_home, worker_home / "config", worker_home / "xdg-config",
                      worker_home / "xdg-cache"):
        directory.mkdir(parents=True, exist_ok=True)
    session_id = str(uuid.uuid4())
    prompt = prompt_for(name, args.prompt_chars)
    command = [
        args.claudium, "-p", prompt, "--output-format", "json", "--max-turns", "1",
        "--no-session-persistence", "--bare", "--tools", "", "--session-id", session_id,
        "--model", args.model,
    ]
    env = clean_env(os.environ, worker_home, args.base)
    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    stdout_path, stderr_path = worker_dir / "stdout.log", worker_dir / "stderr.log"
    process = subprocess.Popen(
        command, cwd=args.cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=os.name != "nt", creationflags=creationflags,
    )
    SUPERVISOR.register(name, process)
    stdout_thread = threading.Thread(target=pump, args=(process.stdout, stdout_path, _MAX_LOG_BYTES),
                                     name=f"stdout-{phase}-{name}")
    stderr_thread = threading.Thread(target=pump, args=(process.stderr, stderr_path, _MAX_LOG_BYTES),
                                     name=f"stderr-{phase}-{name}")
    stdout_thread.start(); stderr_thread.start()
    timed_out = False
    try:
        remain = args.worker_timeout
        if deadline is not None:
            remain = min(remain, max(0.01, deadline - time.monotonic()))
        try:
            process.wait(timeout=remain)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(process)
        join_deadline = time.monotonic() + 10.0
        stdout_thread.join(max(0.0, join_deadline - time.monotonic()))
        stderr_thread.join(max(0.0, join_deadline - time.monotonic()))
    finally:
        if process.poll() is None:
            terminate_process(process)
        SUPERVISOR.unregister(name)
    stdout_done = not stdout_thread.is_alive()
    stderr_done = not stderr_thread.is_alive()
    parsed = parse_worker_stdout(stdout_path)
    result = {
        "name": name, "session_id": session_id, "returncode": process.returncode,
        "timed_out": timed_out, "elapsed": time.monotonic() - started,
        "output": parsed["output"], "output_sha256": parsed["output_sha256"],
        "log_bytes": {"stdout": parsed["log_bytes"], "stderr": stderr_path.stat().st_size},
        "log_truncated": {"stdout": parsed["log_truncated"], "stderr": stderr_path.stat().st_size >= _MAX_LOG_BYTES},
        "pump_threads_alive": {"stdout": not stdout_done, "stderr": not stderr_done},
        "stdout": str(stdout_path.resolve()), "stderr": str(stderr_path.resolve()),
    }
    if timed_out:
        raise WorkerFailure(f"{name} timed out: {result}")
    if not stdout_done or not stderr_done:
        raise WorkerFailure(f"{name} output pump did not stop: {result}")
    if process.returncode != 0:
        raise WorkerFailure(f"{name} exited with {process.returncode}: {result}")
    if not parsed["output"]:
        raise WorkerFailure(f"{name} produced no successful result message: {result}")
    return result


def case_names(value):
    names = [name.strip() for name in value.split(",")]
    if not names or any(not _CASE_RE.fullmatch(name) for name in names):
        raise ValueError("cases must be unique safe names [A-Za-z0-9_.-]")
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
        raise WorkerFailure("state exceeds safety limit")
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
            raise WorkerFailure("state exceeds safety limit")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkerFailure(f"cannot load state {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise WorkerFailure(f"state must contain an object: {path}")
    return value


def config_of(args):
    return {"base": args.base, "claudium": args.claudium, "cwd": str(args.cwd),
            "cases": args.case_names, "delays": args.delays_value,
            "prompt_chars": args.prompt_chars, "worker_timeout": args.worker_timeout,
            "model": args.model}


def baseline(args):
    state = {"config": config_of(args), "baseline": {}, "stress": {}, "baseline_errors": []}
    try:
        for name in args.case_names:
            state["baseline"][name] = run_worker(args, name, "baseline")
            save(args.state, state)
    except Exception as exc:
        state["baseline_errors"].append(str(exc))
        raise
    finally:
        save(args.state, state)
    print(f"BASELINE saved {args.state}", flush=True)


def stress(args):
    state = load(args.state)
    if state.get("config") != config_of(args):
        raise WorkerFailure("stress config does not exactly match baseline config")
    result, errors = {}, []
    delays = args.delays_value
    deadline = time.monotonic() + args.worker_timeout
    lock = threading.Lock()

    def worker(index, name):
        try:
            result[name] = run_worker(args, name, "stress", delays[index], deadline)
        except Exception as exc:  # noqa: BLE001
            with lock:
                errors.append(f"{name}: {exc}")

    threads = [threading.Thread(target=worker, args=(index, name), name=f"stress-{name}")
               for index, name in enumerate(args.case_names)]
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads) and time.monotonic() < deadline:
        time.sleep(0.02)
    if any(thread.is_alive() for thread in threads):
        errors.append("global stress deadline reached")
        SUPERVISOR.terminate_all()
    for thread in threads:
        thread.join(15)
    alive = [thread.name for thread in threads if thread.is_alive()]
    state.update({"stress": result, "stress_errors": errors, "stress_threads_alive": alive})
    try:
        if errors or len(result) != len(args.case_names) or alive:
            raise WorkerFailure("stress failed: " + "; ".join(errors or ["missing result"]))
        print(f"STRESS saved {args.state}", flush=True)
    finally:
        save(args.state, state)


def compare(args):
    state = load(args.state)
    failures = []
    if state.get("config") != config_of(args):
        failures.append("config mismatch")
    if state.get("baseline_errors") or state.get("stress_errors") or state.get("stress_threads_alive"):
        failures.append("recorded run errors")
    base, actual = state.get("baseline", {}), state.get("stress", {})
    if set(base) != set(actual) or set(base) != set(args.case_names):
        failures.append("case set mismatch")
    for name in args.case_names:
        before, after = base.get(name), actual.get(name)
        same = (isinstance(before, dict) and isinstance(after, dict)
                and before.get("returncode") == after.get("returncode") == 0
                and not before.get("timed_out") and not after.get("timed_out")
                and before.get("output") == after.get("output"))
        print(f"IDENTITY {name}: {'MATCH' if same else 'MISMATCH'}", flush=True)
        if not same:
            failures.append(f"case {name} mismatch")
    if failures:
        raise SystemExit("compare failed: " + "; ".join(failures))


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("baseline", "stress", "compare"))
    parser.add_argument("--base", default="http://127.0.0.1:18731")
    parser.add_argument("--claudium", default=shutil.which("claudium") or "claudium")
    parser.add_argument("--cwd", default=tempfile.gettempdir())
    parser.add_argument("--output", type=Path, default=Path("logs/claudium-api"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--cases", default="A,B")
    parser.add_argument("--delays", default="0,2")
    parser.add_argument("--prompt-chars", type=int, default=6000)
    parser.add_argument("--model", default="qwen3.8-flash-next")
    parser.add_argument("--worker-timeout", type=positive, default=900)
    parser.add_argument("--ready-timeout", type=positive, default=1800)
    args = parser.parse_args()
    try:
        args.claudium = str(Path(args.claudium).expanduser().resolve()) \
            if os.sep in args.claudium or args.claudium.startswith(".") \
            else (shutil.which(args.claudium) or args.claudium)
        if not os.path.isfile(args.claudium) or not os.access(args.claudium, os.X_OK):
            parser.error(f"Claudium executable not found or not executable: {args.claudium}")
        args.cwd = Path(args.cwd).expanduser().resolve()
        if not args.cwd.is_dir():
            parser.error(f"worker cwd is not a directory: {args.cwd}")
        args.output = args.output.expanduser().resolve()
        args.state = (args.state or args.output / "results.json").expanduser().resolve()
        args.case_names = case_names(args.cases)
        args.delays_value = parse_delays(args.delays, len(args.case_names))
        if args.prompt_chars <= 0 or args.prompt_chars > _MAX_PROMPT_CHARS:
            parser.error(f"prompt-chars must be in 1..{_MAX_PROMPT_CHARS}")
        if args.mode != "compare":
            client = ApiClient(args.base, min(args.worker_timeout, args.ready_timeout), 5.0)
            wait_ready(client, args.ready_timeout)
        if args.mode == "baseline":
            baseline(args)
        elif args.mode == "stress":
            stress(args)
        else:
            compare(args)
    except (ApiError, OSError, ValueError, WorkerFailure) as exc:
        SUPERVISOR.terminate_all()
        print(f"RESULT FAIL: {exc}", flush=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        SUPERVISOR.terminate_all()
        raise SystemExit(130)
