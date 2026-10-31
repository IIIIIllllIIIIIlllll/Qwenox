#!/usr/bin/env python3
"""CPU-only disconnect regression using a fake engine and synthetic tokenizer.

Run: python tools/api_disconnect_test.py --api build/qwenox-win.exe
     python tools/api_disconnect_test.py --api build/qwenox-api --slots 2
     python tools/api_disconnect_test.py --engine-client  # local protocol tests
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time

from api_regression_test import check, get, request


# Compile this small driver only for --engine-client, so the tests can set
# short deadlines and inspect callbacks without starting an API server.
ENGINE_CLIENT_DRIVER = r"""
#include "engine_client.h"
#include <chrono>
#include <iostream>
#include <nlohmann/json.hpp>
#ifdef _WIN32
#include "../gpu/os_win32.h"
#endif

int main(int argc, char** argv) {
#ifdef _WIN32
    if (!wsa_init()) return 2;
#endif
    if (argc != 5) return 2;
    qwenox::EngineClient client;
    std::string error;
    if (!client.connect(argv[1], &error, 2.0)) {
        std::cerr << error << '\n';
        return 2;
    }
    client.set_timeouts(std::stod(argv[2]), std::stod(argv[3]));
    const std::string mode = argv[4];
    qwenox::GenParams params;
    params.req = 42;
    params.max_tokens = 5;
    params.ids = {1};
    std::vector<int> callbacks;
    int heartbeats = 0;
    qwenox::WaitFn wait = [&] { ++heartbeats; return mode != "cancel-wait"; };
    if (mode == "no-wait") wait = {};
    const auto start = std::chrono::steady_clock::now();
    const auto result = client.generate(params, [&](int token, float) {
        callbacks.push_back(token);
        return mode != "cancel-token";
    }, wait);
    const double elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    nlohmann::json output = {
        {"transport_ok", result.transport_ok}, {"timed_out", result.timed_out},
        {"connected", client.connected()}, {"reason", result.reason},
        {"tokens", result.tokens}, {"logprobs", result.logprobs},
        {"callbacks", callbacks}, {"heartbeats", heartbeats}, {"elapsed", elapsed},
        {"n_prompt", result.n_prompt}, {"n_gen", result.n_gen},
        {"prefill_ms", result.prefill_ms}, {"decode_ms", result.decode_ms},
        {"drafter", result.drafter}, {"rounds", result.rounds},
        {"commit", result.commit}, {"n_cached", result.n_cached},
        {"proposed", result.proposed},
    };
    std::cout << output.dump() << '\n';
}
"""


def engine_client_case(driver, first, between, mode, serve):
    """Run one client against a loopback peer, returning result and commands."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)
        address = f"127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [str(driver), address, str(first), str(between), mode],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            conn, _ = listener.accept()
            with conn:
                conn.settimeout(3)
                request_line = b""
                while not request_line.endswith(b"\n"):
                    chunk = conn.recv(1)
                    if not chunk:
                        raise AssertionError("client closed before GEN")
                    request_line += chunk
                if not request_line.startswith(b"GEN 42 "):
                    raise AssertionError(f"unexpected request: {request_line!r}")
                observation = serve(conn, process) or {}
                stdout, stderr = process.communicate(timeout=3)
                if process.returncode:
                    raise AssertionError(f"client exited {process.returncode}: {stderr}")
                received = observation.pop("received", b"")
                while True:
                    try:
                        chunk = conn.recv(4096)
                    except ConnectionResetError:
                        break
                    if not chunk:
                        break
                    received += chunk
                observation["finished_at"] = time.monotonic()
                return json.loads(stdout), received, observation
        finally:
            if process.poll() is None:
                process.kill()
                process.communicate()


def scheduled_lines(events):
    def serve(conn, unused_process):
        start = time.monotonic()
        for delay, line in events:
            remaining = start + delay - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            conn.sendall((line + "\n").encode("ascii"))
    return serve


def flood_lines(conn, process, lines, duration, interval=0.03):
    """Keep sending while also collecting X, until the client closes or duration ends."""
    end = time.monotonic() + duration
    received = b""
    cancel_at = None
    index = 0
    while time.monotonic() < end and process.poll() is None:
        try:
            conn.sendall((lines[index % len(lines)] + "\n").encode("ascii"))
        except (BrokenPipeError, ConnectionResetError):
            break
        index += 1
        if select.select([conn], [], [], interval)[0]:
            try:
                chunk = conn.recv(4096)
            except ConnectionResetError:
                break
            if not chunk:
                break
            received += chunk
            if cancel_at is None and b"X 42\n" in received:
                cancel_at = time.monotonic()
    return {"received": received, "cancel_at": cancel_at}


def run_engine_client_checks(driver):
    # Before the first T, W must refresh first_timeout, not between_timeout.
    # After T, both W phases refresh the between-token timeout instead.
    result, commands, _ = engine_client_case(
        driver, 1.5, 0.55, "normal", scheduled_lines([
            (0.75, "W 42 wait"), (1.55, "W 42 resume"),
            (2.0, "T 42 101 -0.125"), (2.3, "W 42 wait"),
            (2.6, "W 42 resume"), (2.9, "T 42 102"),
            (3.02, "D 42 done 7 2 1.25 2.5 3 4 5 6 7"),
        ]))
    check(result["transport_ok"] and not result["timed_out"] and
          result["reason"] == "done" and result["tokens"] == [101, 102] and
          result["callbacks"] == [101, 102] and result["logprobs"] == [-0.125, 0.0] and
          result["heartbeats"] >= 2 and result["elapsed"] > 2.8 and not commands and
          [result[key] for key in ("n_prompt", "n_gen", "prefill_ms", "decode_ms",
                                   "drafter", "rounds", "commit", "n_cached", "proposed")]
          == [7, 2, 1.25, 2.5, 3, 4, 5, 6, 7], "engine-wait-progress", result)

    result, commands, _ = engine_client_case(
        driver, 0.8, 0.3, "no-wait", scheduled_lines([
            (0.4, "W 42 wait"), (0.85, "W 42 resume"),
            (1.15, "D 42 done 4 0 1 2"),
        ]))
    check(result["transport_ok"] and result["tokens"] == [] and
          result["callbacks"] == [] and result["n_prompt"] == 4 and
          result["n_gen"] == 0 and not commands, "engine-wait-no-token", result)

    # W after a T must not revert to the longer first-token timeout.
    result, commands, _ = engine_client_case(
        driver, 1.5, 0.55, "no-wait", scheduled_lines([
            (0.05, "T 42 101"), (0.35, "W 42 resume"),
        ]))
    check(result["timed_out"] and not result["transport_ok"] and
          result["tokens"] == [101] and 0.75 < result["elapsed"] < 1.3 and
          commands == b"X 42\n", "engine-wait-between-timeout", result)

    # Legitimate T lines still refresh the interval and allow heartbeat calls.
    result, commands, _ = engine_client_case(
        driver, 0.9, 0.55, "normal", scheduled_lines([
            (0.05, "T 42 11"), (0.37, "T 42 12 -inf"),
            (0.69, "T 42 13 -0.25"), (1.01, "T 42 14"),
            (1.2, "D 42 length 8 4 1 2"),
        ]))
    check(result["transport_ok"] and result["reason"] == "length" and
          result["tokens"] == [11, 12, 13, 14] and
          result["callbacks"] == [11, 12, 13, 14] and result["heartbeats"] >= 1 and
          not commands, "engine-token-progress-and-heartbeat", result)

    invalid = [
        "W", "W 42", "W 42wait", "W 42wait resume", "W 43 wait",
        "W 42 pause", "W 42 wait extra", "W 42 resume extra", "W 42 waitx",
        "W 42.0 wait", "W 9223372036854775808 wait", "W 42 wait\x00",
        "T 43 91", "T 42junk 91", "T 42 91junk", "T 42 91 bad",
        "T 42 91 -0.5 extra", "D 43 done 0 0 0 0", "PONG", "unknown",
    ]
    for after_token in (False, True):
        def send_invalid(conn, process):
            if after_token:
                conn.sendall(b"T 42 101\n")
            return flood_lines(conn, process, invalid, 2.4)

        result, commands, _ = engine_client_case(
            driver, 1.35, 0.5, "no-wait" if after_token else "normal", send_invalid)
        minimum, maximum = (0.3, 1.2) if after_token else (1.1, 1.9)
        check(result["timed_out"] and not result["transport_ok"] and
              result["tokens"] == ([101] if after_token else []) and
              result["callbacks"] == ([101] if after_token else []) and
              result["heartbeats"] == (0 if after_token else 1) and
              minimum < result["elapsed"] < maximum and commands == b"X 42\n",
              f"engine-invalid-lines-timeout after_token={after_token}", result)

    def cancel_ack(conn, process, mode):
        if mode == "cancel-token":
            conn.sendall(b"T 42 101 -0.125\n")
        observed = b""
        end = time.monotonic() + 3
        while b"X 42\n" not in observed and time.monotonic() < end:
            conn.sendall(b"W 42 wait\n")
            if select.select([conn], [], [], 0.2)[0]:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                observed += chunk
        if observed != b"X 42\n":
            raise AssertionError(f"cancel not received: {observed!r}")
        # Continue W long enough to catch a mistakenly re-enabled heartbeat.
        for unused in range(6):
            conn.sendall(b"W 42 resume\n")
            time.sleep(0.2)
        conn.sendall(b"T 42 102\nD 42 cancel 0 0 0 0\n")
        return {"received": observed}

    for mode in ("cancel-wait", "cancel-token"):
        result, commands, _ = engine_client_case(
            driver, 8, 8, mode, lambda conn, process: cancel_ack(conn, process, mode))
        expected_tokens = [102] if mode == "cancel-wait" else [101, 102]
        check(result["transport_ok"] and result["reason"] == "cancel" and
              result["connected"] and result["tokens"] == expected_tokens and
              result["callbacks"] == ([] if mode == "cancel-wait" else [101]) and
              result["heartbeats"] == (1 if mode == "cancel-wait" else 0) and
              commands == b"X 42\n", f"engine-wait-cancel-ack {mode}", result)

    def cancel_deadline_case(mode):
        def serve(conn, process):
            if mode == "cancel-token":
                conn.sendall(b"T 42 101\n")
            return flood_lines(conn, process, ["W 42 wait", "W 42 resume"], 14, 0.15)
        return engine_client_case(driver, 30, 30, mode, serve)

    # Test both cancellation paths concurrently; W must not push either cap out.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {mode: executor.submit(cancel_deadline_case, mode)
                   for mode in ("cancel-wait", "cancel-token")}
        for mode, future in futures.items():
            result, commands, observation = future.result()
            cancel_at = observation["cancel_at"]
            duration = (observation["finished_at"] - cancel_at
                        if cancel_at is not None else -1)
            check(result["timed_out"] and not result["transport_ok"] and
                  not result["connected"] and commands == b"X 42\n" and
                  9.6 < duration < 11.5 and
                  result["heartbeats"] == (1 if mode == "cancel-wait" else 0),
                  f"engine-wait-cancel-deadline {mode}",
                  f"{result}, cancel-to-close={duration:.3f}s")
    print("RESULT PASS")


def wait_for(predicate, name, timeout=3.5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(f"{name}: timed out after {timeout}s")


def free_port():
    with socket.socket() as conn:
        conn.bind(("127.0.0.1", 0))
        return conn.getsockname()[1]


def make_tokenizer(directory):
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    alphabet = {value: value for value in byte_values}
    extra = 256
    for value in range(256):
        if value not in alphabet:
            alphabet[value] = extra
            extra += 1
    vocab = {chr(alphabet[value]): 1024 + value for value in range(256)}
    vocab.update({"Hi": 12675, chr(alphabet[32]) + "Par": 4130, "is": 284})
    special = [("<|endoftext|>", 248043), ("<|im_start|>", 248045),
               ("<|im_end|>", 248046), ("<think>", 248068), ("</think>", 248069)]
    tokenizer = {"model": {"vocab": vocab, "merges": []}, "added_tokens": [
        {"content": content, "id": token_id, "special": True}
        for content, token_id in special
    ]}
    directory.mkdir()
    (directory / "tokenizer.json").write_text(json.dumps(tokenizer), encoding="utf-8")


def run_checks(base, api_port, engine_port, slots):
    def state():
        with socket.create_connection(("127.0.0.1", engine_port), timeout=2) as conn:
            conn.sendall(b"TESTSTATE\n")
            with conn.makefile("rb") as reader:
                return json.loads(reader.readline())

    def health():
        return json.loads(get(base, "/health")[2])

    def idle():
        status = health()
        return status["in_flight"] == 0 and status["queued"] == 0

    def body_for(path, stream=False, seed=616161):
        body = {"stream": stream, "temperature": 1, "seed": seed,
                "max_tokens": 65536, "enable_thinking": False}
        if path == "/v1/completions":
            body["prompt"] = "hi"
        elif path == "/v1/chat/completions":
            body["messages"] = [{"role": "user", "content": "hi"}]
        else:
            body["input"] = "hi"
        return body

    def open_request(path, body):
        payload = json.dumps(body).encode("utf-8")
        conn = socket.create_connection(("127.0.0.1", api_port), timeout=2)
        header = (f"POST {path} HTTP/1.1\r\nHost: localhost\r\n"
                  f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
                  "Connection: close\r\n\r\n").encode("ascii")
        conn.sendall(header + payload)
        return conn

    def next_request():
        status, content_type, raw = request(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hi"}],
            "enable_thinking": False, "temperature": 0, "max_tokens": 4,
        })
        check(status == 200 and "application/json" in content_type and
              json.loads(raw)["choices"][0]["message"]["content"] == "Hi",
              "next-request-after-draining", raw[:300])

    paths = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
    for path in paths:
        status, content_type, raw = request(base, path, body_for(path, seed=616163))
        check(status == 200 and "application/json" in content_type and raw,
              f"nonstream-normal-completion {path}")
        check(state()[-1]["cancel"] == 0, f"live-client-not-cancelled {path}")

    for seed in (616161, 616162):
        for stream in (False, True):
            for path in paths:
                before = len(state())
                with open_request(path, body_for(path, stream, seed)) as conn:
                    wait_for(lambda: len(state()) == before + 1, "generation-started")
                    if seed == 616161:
                        wait_for(lambda: state()[-1]["tokens"] > 0, "continuous-tokens")
                    if stream:
                        check(b"text/event-stream" in conn.recv(4096), "sse-started")
                    else:
                        conn.settimeout(0.1)
                        try:
                            data = conn.recv(1)
                        except socket.timeout:
                            data = None
                        check(data is None, "nonstream-no-premature-response")
                wait_for(idle, f"disconnect seed={seed} stream={stream} {path}")
                entry = state()[-1]
                check(entry["cancel"] == 1 and entry["done"],
                      f"cancel-once-and-drain seed={seed} stream={stream} {path}", entry)
                next_request()

    before = len(state())
    with open_request(paths[0], body_for(paths[0])) as conn:
        wait_for(lambda: len(state()) == before + 1, "reset-generation-started")
        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                       struct.pack("hh" if os.name == "nt" else "ii", 1, 0))
    wait_for(idle, "reset-client-disconnect")
    check(state()[-1]["cancel"] == 1 and state()[-1]["done"], "reset-cancel-and-drain")
    next_request()

    owners = []
    before = len(state())
    try:
        for unused in range(slots):
            owners.append(open_request(paths[0], body_for(paths[0])))
        wait_for(lambda: len(state()) == before + slots, "all-slots-occupied")
        with open_request(paths[0], body_for(paths[0], seed=616162)):
            wait_for(lambda: health()["queued"] == 1, "disconnected-request-queued")
        wait_for(lambda: health()["queued"] == 0, "disconnected-request-left-queue")
        check(len(state()) == before + slots, "queued-disconnect-never-sends-gen")
    finally:
        for conn in owners:
            conn.close()
    wait_for(idle, "queue-owners-cancelled")
    next_request()

    status, _, raw = request(base, paths[1], dict(body_for(paths[1]), stop="Hi"))
    check(status == 200 and json.loads(raw)["choices"][0]["text"] == "",
          "stop-sequence-cancels", raw[:300])
    check(state()[-1]["cancel"] == 1 and state()[-1]["done"],
          "no-heartbeat-after-token-cancellation")
    next_request()
    print("RESULT PASS")


def main():
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--api", help="path to the compiled qwenox-api binary")
    target.add_argument("--engine-client", action="store_true",
                        help="compile and test the client against a local fake engine")
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"),
                        help="C++ compiler for --engine-client")
    parser.add_argument("--slots", type=int, default=1)
    args = parser.parse_args()
    if args.engine_client:
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory(prefix="qwenox-engine-client-test-") as temporary:
            driver = Path(temporary) / ("engine-client-test.exe" if os.name == "nt"
                                        else "engine-client-test")
            command = [args.cxx, "-std=c++17", "-O2", "-pthread",
                       "-Wall", "-Wextra", "-Wpedantic", "-Werror",
                       "-I", str(root / "src/api"), "-I", str(root / "third_party"),
                       str(root / "src/api/engine_client.cpp"), "-x", "c++", "-",
                       "-o", str(driver)]
            if os.name == "nt":
                command.append("-lws2_32")
            subprocess.run(command, input=ENGINE_CLIENT_DRIVER, text=True, check=True)
            run_engine_client_checks(driver)
        return
    api_binary = str(Path(args.api).resolve())
    fake_engine = str(Path(__file__).with_name("api_fake_engine.py").resolve())
    engine_port, api_port = free_port(), free_port()
    while engine_port == api_port:
        api_port = free_port()
    base = f"http://127.0.0.1:{api_port}"
    children = []
    with tempfile.TemporaryDirectory(prefix="qwenox-api-disconnect-") as temporary:
        root = Path(temporary)
        make_tokenizer(root / "tokenizer")
        env = dict(os.environ, QWENOX_API_TOKCACHE_FILE="", QWENOX_API_ADMIN_KEY="",
                   QWENOX_API_TOKCACHE="0", QWENOX_REQSTAT="0", ROPE_FACTOR="1")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (root / "test.log").open("w+") as log:
            try:
                children.append(subprocess.Popen(
                    [sys.executable, fake_engine, "--port", str(engine_port),
                     "--slots", str(args.slots)], cwd=root, stdout=log, stderr=log,
                    creationflags=creationflags))

                def engine_ready():
                    try:
                        with socket.create_connection(("127.0.0.1", engine_port), timeout=0.1):
                            return True
                    except OSError:
                        return False

                wait_for(engine_ready, "fake-engine-ready")
                children.append(subprocess.Popen(
                    [api_binary, "--tokenizer", str(root / "tokenizer"),
                     "--engine", f"127.0.0.1:{engine_port}", "--host", "127.0.0.1",
                     "--port", str(api_port), "--overrides", ""],
                    cwd=root, env=env, stdout=log, stderr=log, creationflags=creationflags))

                def api_ready():
                    try:
                        return get(base, "/health")[0] == 200
                    except OSError:
                        return False

                wait_for(api_ready, "api-ready", timeout=10)
                run_checks(base, api_port, engine_port, args.slots)
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read(), file=sys.stderr)
                raise
            finally:
                for child in reversed(children):
                    child.terminate()
                    child.wait(timeout=5)


if __name__ == "__main__":
    main()
