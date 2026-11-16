#!/usr/bin/env python3
"""<tool_call> ban for requests without callable tools.

Starts api_fake_engine.py and the given qwenox-api, sends chat / responses /
messages requests and reads back the BIAS the API put on each GEN (fake
engine BIASLOG). A sampled request with no tools, or with tool_choice "none",
must carry <tool_call> (248058) and </tool_call> (248059) at -100; one with
tools or a greedy one must not; a client's own bias on either id is kept as
sent; with QWENOX_NOTOOLS_BAN=0 nothing is added. Ends with RESULT PASS or a traceback.

    python3 tools/api_notools_ban_test.py --api build/qwenox-api
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from api_regression_test import check, get, request
from api_disconnect_test import free_port, wait_for

# The Windows api build is a tray app that serves only with --console; the
# Linux console build has no such flag.
CONSOLE_FLAG = ["--console"] if os.name == "nt" else []

TOOL_CALL, TOOL_CALL_END = 248058, 248059
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
RESPONSE_TOOLS = [{"type": "function", "name": "get_weather",
                   "parameters": TOOLS[0]["function"]["parameters"]}]
ANTHROPIC_TOOLS = [{"name": "get_weather",
                    "input_schema": TOOLS[0]["function"]["parameters"]}]
USER = [{"role": "user", "content": "Generate a title for: merge this patch"}]
SAMPLED = {"temperature": 0.7, "seed": 3}


def bias_log(engine_port):
    with socket.create_connection(("127.0.0.1", engine_port), timeout=2) as conn:
        conn.sendall(b"BIASLOG\n")
        data = b""
        while not data.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            data += chunk
    return json.loads(data)


def send(base, engine_port, path, body, name):
    bias_log(engine_port)  # drop earlier entries
    status, _, raw = request(base, path, body)
    check(status == 200, f"{name}-status", raw[:600])
    log = bias_log(engine_port)
    check(len(log) >= 1, f"{name}-gen", log)
    return {b[0]: b[1] for b in log[0]["bias"]}


def banned(b):
    return b.get(TOOL_CALL) == -100 and b.get(TOOL_CALL_END) == -100


def clean(b):
    return TOOL_CALL not in b and TOOL_CALL_END not in b


def run_checks(base, ep):
    chat = "/v1/chat/completions"
    b = send(base, ep, chat, dict(SAMPLED, messages=USER, max_tokens=64), "chat-no-tools")
    check(banned(b), "chat-no-tools-banned", b)
    b = send(base, ep, chat, dict(SAMPLED, messages=USER, max_tokens=64, tools=TOOLS),
             "chat-tools")
    check(clean(b), "chat-tools-not-banned", b)
    b = send(base, ep, chat, dict(SAMPLED, messages=USER, max_tokens=64, tools=TOOLS,
                                  tool_choice="none"), "chat-tool-choice-none")
    check(banned(b), "chat-tool-choice-none-banned", b)
    b = send(base, ep, chat, dict(messages=USER, max_tokens=64, temperature=0), "chat-greedy")
    check(clean(b), "chat-greedy-not-banned", b)
    b = send(base, ep, chat, dict(SAMPLED, messages=USER, max_tokens=64,
                                  logit_bias={str(TOOL_CALL): 5, "13": -2}), "chat-client-bias")
    check(b == {TOOL_CALL: 5, 13: -2, TOOL_CALL_END: -100}, "chat-client-bias-kept", b)

    resp = "/v1/responses"
    b = send(base, ep, resp, dict(SAMPLED, input="merge this patch", max_output_tokens=64),
             "responses-no-tools")
    check(banned(b), "responses-no-tools-banned", b)
    b = send(base, ep, resp, dict(SAMPLED, input="merge this patch", max_output_tokens=64,
                                  tools=RESPONSE_TOOLS), "responses-tools")
    check(clean(b), "responses-tools-not-banned", b)

    msgs = "/v1/messages"
    b = send(base, ep, msgs, dict(SAMPLED, messages=USER, max_tokens=64), "messages-no-tools")
    check(banned(b), "messages-no-tools-banned", b)
    b = send(base, ep, msgs, dict(SAMPLED, messages=USER, max_tokens=64, tools=ANTHROPIC_TOOLS),
             "messages-tools")
    check(clean(b), "messages-tools-not-banned", b)
    b = send(base, ep, msgs, dict(SAMPLED, messages=USER, max_tokens=64, tools=ANTHROPIC_TOOLS,
                                  tool_choice={"type": "none"}), "messages-tool-choice-none")
    check(banned(b), "messages-tool-choice-none-banned", b)


def run_kill_switch(base, ep):
    b = send(base, ep, "/v1/chat/completions", dict(SAMPLED, messages=USER, max_tokens=64),
             "off-chat-no-tools")
    check(clean(b), "off-not-banned", b)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="path to the compiled qwenox-api binary")
    parser.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent.parent /
                                                   "data" / "tokenizer"))
    args = parser.parse_args()
    api_binary = str(Path(args.api).resolve())
    fake_engine = str(Path(__file__).with_name("api_fake_engine.py").resolve())
    with tempfile.TemporaryDirectory(prefix="qwenox-notools-ban-") as temporary:
        root = Path(temporary)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (root / "test.log").open("w+", encoding="utf-8") as log:
            children = []
            try:
                for ban_env in (None, "0"):
                    engine_port, api_port = free_port(), free_port()
                    while engine_port == api_port:
                        api_port = free_port()
                    base = f"http://127.0.0.1:{api_port}"
                    children.append(subprocess.Popen(
                        [sys.executable, fake_engine, "--port", str(engine_port)], cwd=root,
                        stdout=log, stderr=log, creationflags=creationflags))

                    def engine_ready():
                        try:
                            with socket.create_connection(("127.0.0.1", engine_port), timeout=0.1):
                                return True
                        except OSError:
                            return False

                    wait_for(engine_ready, "fake-engine-ready")
                    child_env = dict(os.environ, QWENOX_API_TOKCACHE_FILE="",
                                     QWENOX_API_ADMIN_KEY="", QWENOX_API_TOKCACHE="0",
                                     QWENOX_REQSTAT="0", ROPE_FACTOR="1")
                    child_env.pop("QWENOX_NOTOOLS_BAN", None)
                    if ban_env is not None:
                        child_env["QWENOX_NOTOOLS_BAN"] = ban_env
                    children.append(subprocess.Popen(
                        [api_binary, *CONSOLE_FLAG, "--tokenizer", args.tokenizer,
                         "--engine", f"127.0.0.1:{engine_port}", "--host", "127.0.0.1",
                         "--port", str(api_port), "--overrides", ""],
                        cwd=root, env=child_env, stdout=log, stderr=log,
                        creationflags=creationflags))

                    def api_ready():
                        try:
                            return get(base, "/health")[0] == 200
                        except OSError:
                            return False

                    wait_for(api_ready, "api-ready", timeout=15)
                    if ban_env is None:
                        run_checks(base, engine_port)
                    else:
                        run_kill_switch(base, engine_port)
                    for child in reversed(children):
                        child.terminate()
                        child.wait(timeout=5)
                    children.clear()
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-4000:], file=sys.stderr)
                raise
            finally:
                for child in children:
                    child.kill()
    print("RESULT PASS")


if __name__ == "__main__":
    main()
