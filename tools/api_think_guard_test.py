#!/usr/bin/env python3
"""Think-guard checks: replies that end with reasoning only.

Starts api_fake_engine.py and the given qwenox-api, then drives the fake
engine's think-guard scenarios (max_tokens 7100..7900, see
api_fake_engine.guard_scenario) through /v1/chat/completions, /v1/responses
and /v1/messages, streaming and not. Ends with RESULT PASS or a traceback.

    python3 tools/api_think_guard_test.py --api build/qwenox-api
"""
import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

from api_regression_test import check, get, parse_sse, request
from api_disconnect_test import free_port, wait_for

SENTENCE = ("Considering the limited time by the user, I have to give the solution "
            "based on the thinking directly now.")
THINK = "Let me check the weather."
TOOLS = [{"type": "function", "function": {
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
RESPONSE_TOOLS = [{"type": "function", "name": "get_weather",
                   "parameters": TOOLS[0]["function"]["parameters"]}]
ANTHROPIC_TOOLS = [{"name": "get_weather",
                    "input_schema": TOOLS[0]["function"]["parameters"]}]
# "\n\n" + SENTENCE + "\n</think>\n\n" and SENTENCE + "\n</think>\n\n"
CLOSER_IDS = [271, 79939, 279, 6973, 854, 539, 279, 1156, 11, 353, 599, 310, 2873, 279,
              6093, 3018, 383, 279, 7047, 5774, 1381, 13, 198, 248069, 271]
THINK_IDS = [9764, 728, 1716, 279, 8831, 13]


class Env:
    def __init__(self, base, engine_port):
        self.base = base
        self.engine_port = engine_port

    def guard_log(self):
        with socket.create_connection(("127.0.0.1", self.engine_port), timeout=2) as conn:
            conn.sendall(b"GUARDLOG\n")
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
        return json.loads(data)


# ------------------------------------------------------------ route drivers --

def chat(env, body, stream):
    body = dict(body, stream=stream)
    if stream:
        body["stream_options"] = {"include_usage": True}
    status, _, raw = request(env.base, "/v1/chat/completions", body)
    check(status == 200, "chat-status", raw[:800])
    if not stream:
        j = json.loads(raw)
        choice = j["choices"][0]
        msg = choice["message"]
        calls = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                 for c in msg.get("tool_calls") or []]
        return {"reasoning": msg.get("reasoning_content") or "", "content": msg["content"] or "",
                "calls": calls, "finish": choice["finish_reason"], "usage": j["usage"],
                "raw": raw}
    out = {"reasoning": "", "content": "", "calls": [], "finish": None, "usage": None,
           "raw": raw}
    names, args = {}, {}
    for frame in parse_sse(raw):
        if frame == "[DONE]":
            continue
        if frame.get("usage"):
            out["usage"] = frame["usage"]
        for item in frame.get("choices") or []:
            out["finish"] = item.get("finish_reason") or out["finish"]
            delta = item.get("delta", {})
            out["reasoning"] += delta.get("reasoning_content", "") or ""
            out["content"] += delta.get("content", "") or ""
            for tc in delta.get("tool_calls", []):
                names[tc["index"]] = names.get(tc["index"], "") + tc.get("function", {}).get("name", "")
                args[tc["index"]] = args.get(tc["index"], "") + tc.get("function", {}).get("arguments", "")
    out["calls"] = [(names[i], json.loads(args[i])) for i in sorted(names)]
    return out


def responses(env, body, stream):
    body = dict(body, stream=stream)
    status, _, raw = request(env.base, "/v1/responses", body)
    check(status == 200, "responses-status", raw[:800])
    if stream:
        final = None
        for frame in parse_sse(raw):
            if isinstance(frame, dict) and frame.get("type") in (
                    "response.completed", "response.incomplete"):
                final = frame["response"]
        check(final is not None, "responses-stream-final", raw[-800:])
        response = final
    else:
        response = json.loads(raw)
    text = "".join(part["text"] for item in response["output"] if item["type"] == "message"
                   for part in item["content"])
    calls = [(item["name"], json.loads(item["arguments"])) for item in response["output"]
             if item["type"] == "function_call"]
    return {"content": text, "calls": calls, "status": response["status"],
            "usage": response["usage"], "raw": raw}


def messages(env, body, stream):
    body = dict(body, stream=stream)
    status, _, raw = request(env.base, "/v1/messages", body)
    check(status == 200, "messages-status", raw[:800])
    out = {"reasoning": "", "content": "", "calls": [], "finish": None, "usage": None,
           "raw": raw}
    if not stream:
        m = json.loads(raw)
        for block in m["content"]:
            if block["type"] == "thinking":
                out["reasoning"] += block["thinking"]
            elif block["type"] == "text":
                out["content"] += block["text"]
            elif block["type"] == "tool_use":
                out["calls"].append((block["name"], block["input"]))
        out["finish"], out["usage"] = m["stop_reason"], m["usage"]
        return out
    blocks = {}
    for frame in raw.decode("utf-8").split("\n\n"):
        data = [line[6:] for line in frame.split("\n") if line.startswith("data: ")]
        if not data:
            continue
        e = json.loads(data[0])
        if e["type"] == "content_block_start":
            blocks[e["index"]] = dict(e["content_block"], partial="")
        elif e["type"] == "content_block_delta":
            b, d = blocks[e["index"]], e["delta"]
            if d["type"] == "thinking_delta":
                out["reasoning"] += d["thinking"]
            elif d["type"] == "text_delta":
                out["content"] += d["text"]
            elif d["type"] == "input_json_delta":
                b["partial"] += d["partial_json"]
        elif e["type"] == "message_delta":
            out["finish"], out["usage"] = e["delta"]["stop_reason"], e["usage"]
    for i in sorted(blocks):
        if blocks[i]["type"] == "tool_use":
            out["calls"].append((blocks[i]["name"], json.loads(blocks[i]["partial"] or "{}")))
    return out


def guard_of(result):
    return (result["usage"] or {}).get("think_guard")


# ------------------------------------------------------------------- checks --

def run_checks(env):
    user = [{"role": "user", "content": "weather?"}]
    sampled = {"temperature": 1, "seed": 5}
    greedy = {"temperature": 0}

    # A: ends inside <think>. The sentence and </think> are streamed into the
    # reasoning; the resume answers.
    want_reasoning = THINK + "\n\n" + SENTENCE
    for stream in (False, True):
        tag = "stream" if stream else "nonstream"
        env.guard_log()
        r = chat(env, dict(greedy, messages=user, max_tokens=7100), stream)
        check(r["content"] == "Hi" and r["reasoning"] == want_reasoning and
              r["finish"] == "stop" and guard_of(r) == "resumed_unclosed",
              f"A-chat-{tag}", r["raw"][:1500])
        check(r["usage"]["completion_tokens"] == len(THINK_IDS) + 1 and
              r["usage"]["prompt_tokens"] < 200, f"A-chat-usage-{tag}", r["usage"])
        log = env.guard_log()
        check(len(log) == 2 and not log[0]["resumed"] and log[1]["resumed"] and
              log[1]["tail"][-len(CLOSER_IDS) - len(THINK_IDS):] == THINK_IDS + CLOSER_IDS and
              log[1]["max_tokens"] == 7100 - len(THINK_IDS),
              f"A-chat-resume-context-{tag}", log)

        r = responses(env, dict(greedy, input="weather?", max_output_tokens=7100), stream)
        check(r["content"] == "Hi" and r["status"] == "completed" and
              guard_of(r) == "resumed_unclosed", f"A-responses-{tag}", r["raw"][:1500])

        r = messages(env, dict(greedy, messages=user, max_tokens=7100), stream)
        check(r["content"] == "Hi" and r["reasoning"] == want_reasoning and
              r["finish"] == "end_turn" and guard_of(r) == "resumed_unclosed",
              f"A-messages-{tag}", r["raw"][:1500])

    # B: </think> then nothing (or only whitespace). Nothing extra is
    # streamed; the resume context swaps the </think> for sentence + </think>.
    for key, name in ((7200, "B"), (7300, "B-space")):
        for stream in (False, True):
            tag = "stream" if stream else "nonstream"
            env.guard_log()
            r = chat(env, dict(greedy, messages=user, max_tokens=key), stream)
            check(r["content"].strip() == "Hi" and r["reasoning"] == "" and
                  SENTENCE not in r["raw"].decode("utf-8") and r["finish"] == "stop" and
                  guard_of(r) == "resumed_empty_answer", f"{name}-chat-{tag}", r["raw"][:1500])
            log = env.guard_log()
            tail = log[-1]["tail"]
            check(len(log) == 2 and log[1]["resumed"] and
                  tail[-len(CLOSER_IDS) + 1:] == CLOSER_IDS[1:] and tail.count(248069) == 1,
                  f"{name}-chat-resume-context-{tag}", log)
        r = messages(env, dict(greedy, messages=user, max_tokens=key), True)
        check(r["content"].strip() == "Hi" and guard_of(r) == "resumed_empty_answer",
              f"{name}-messages-stream", r["raw"][:1500])

    # The first resume ends at once: greedy does not retry (it would replay),
    # sampling retries with a fresh seed.
    env.guard_log()
    r = chat(env, dict(greedy, messages=user, max_tokens=7400), False)
    log = env.guard_log()
    check(r["content"] == "" and r["reasoning"] == want_reasoning and r["finish"] == "stop" and
          guard_of(r) == "resume_failed" and len(log) == 2,
          "resume-empty-greedy-no-retry", (r["raw"][:1200], log))
    for stream in (False, True):
        tag = "stream" if stream else "nonstream"
        env.guard_log()
        r = chat(env, dict(sampled, messages=user, max_tokens=7400), stream)
        log = env.guard_log()
        resumes = [entry for entry in log if entry["resumed"]]
        check(r["content"] == "Hi" and guard_of(r) == "resumed_unclosed" and
              len(resumes) == 2 and resumes[0]["seed"] != resumes[1]["seed"],
              f"resume-empty-sampled-retry-{tag}", (r["raw"][:1200], log))

    # C: a finished tool call inside the reasoning is made as a call.
    for stream in (False, True):
        tag = "stream" if stream else "nonstream"
        env.guard_log()
        r = chat(env, dict(greedy, messages=user, tools=TOOLS, max_tokens=7500), stream)
        log = env.guard_log()
        check(r["finish"] == "tool_calls" and r["calls"] == [("get_weather", {"city": "北京"})] and
              "<tool_call>" in r["reasoning"] and r["content"].strip() == "" and
              guard_of(r) == "tool_call_from_reasoning" and len(log) == 1,
              f"C-chat-{tag}", (r["raw"][:1500], log))
        r = responses(env, dict(greedy, input="weather?", tools=RESPONSE_TOOLS,
                                max_output_tokens=7500), stream)
        check(r["calls"] == [("get_weather", {"city": "北京"})] and r["status"] == "completed" and
              guard_of(r) == "tool_call_from_reasoning", f"C-responses-{tag}", r["raw"][:1500])
        r = messages(env, dict(greedy, messages=user, tools=ANTHROPIC_TOOLS, max_tokens=7500),
                     stream)
        check(r["calls"] == [("get_weather", {"city": "北京"})] and r["finish"] == "tool_use" and
              guard_of(r) == "tool_call_from_reasoning", f"C-messages-{tag}", r["raw"][:1500])
    # Without tools the same text is only reasoning: shape A instead.
    r = chat(env, dict(greedy, messages=user, max_tokens=7500), False)
    check(r["calls"] == [] and r["content"] == "Hi" and guard_of(r) == "resumed_unclosed",
          "C-without-tools-resumes", r["raw"][:1500])

    # D: EOS right after a bare <tool_call> opener. The resume continues on the
    # same slot with EOS banned and a stop at </tool_call>, so the call body is
    # written and the call closes. The first run is think + "</think>\n\n" +
    # the opener = 9 tokens; the resume prompt ends with the opener.
    for stream in (False, True):
        tag = "stream" if stream else "nonstream"
        env.guard_log()
        r = chat(env, dict(sampled, messages=user, tools=TOOLS, max_tokens=8000), stream)
        log = env.guard_log()
        check(r["finish"] == "tool_calls" and
              r["calls"] == [("get_weather", {"city": "北京"})] and
              r["content"] == "" and guard_of(r) == "resumed_open_tool_call" and
              len(log) == 2 and not log[0]["resumed"] and log[1]["resumed"] and
              log[1]["tail"][-1] == 248058 and log[1]["max_tokens"] == 8000 - 9 and
              any(value == -100.0 for _, value in log[1]["bias"]),
              f"D-chat-{tag}", (r["raw"][:1500], log))
    r = responses(env, dict(sampled, input="weather?", tools=RESPONSE_TOOLS,
                            max_output_tokens=8000), False)
    check(r["calls"] == [("get_weather", {"city": "北京"})] and
          r["status"] == "completed" and guard_of(r) == "resumed_open_tool_call",
          "D-responses-nonstream", r["raw"][:1500])
    r = messages(env, dict(sampled, messages=user, tools=ANTHROPIC_TOOLS, max_tokens=8000),
                 False)
    check(r["calls"] == [("get_weather", {"city": "北京"})] and r["finish"] == "tool_use" and
          guard_of(r) == "resumed_open_tool_call", "D-messages-nonstream", r["raw"][:1500])
    # Greedy cannot ban the EOS (engine BIAS is sampler-only): the single
    # resume attempt replays the early end of turn, the reply stays as it was.
    # finish_reason still reports the real stop reason, not "length".
    env.guard_log()
    r = chat(env, dict(greedy, messages=user, tools=TOOLS, max_tokens=8000), False)
    log = env.guard_log()
    check(r["finish"] == "stop" and r["calls"] == [] and r["content"] == "" and
          guard_of(r) == "resume_failed" and len(log) == 2 and log[1]["resumed"],
          "D-greedy-resume-failed", (r["raw"][:1200], log))
    # The resume never closes the call. Running into our own 8192-token resume
    # ceiling is not the client's budget: finish stays "stop". Running into
    # the client's max_tokens (8150 - 9 left, below the ceiling) is a real
    # "length". Neither retries: the attempt's text is already out.
    for max_tokens, want, tag in ((8300, "stop", "ceiling"), (8150, "length", "budget")):
        env.guard_log()
        r = chat(env, dict(sampled, messages=user, tools=TOOLS, max_tokens=max_tokens), False)
        log = env.guard_log()
        check(r["finish"] == want and r["calls"] == [] and r["content"] == "" and
              guard_of(r) == "resume_failed" and len(log) == 2 and log[1]["resumed"] and
              log[1]["max_tokens"] == min(8192, max_tokens - 9),
              f"D-resume-{tag}-{want}", (r["raw"][:600], log))
    # Without tools the opener is literal text and the guard stays out of it.
    env.guard_log()
    r = chat(env, dict(greedy, messages=user, max_tokens=8000), False)
    check(r["calls"] == [] and guard_of(r) is None and len(env.guard_log()) == 1,
          "D-without-tools-no-guard", r["raw"][:800])

    # Not the guard's cases: cut by max_tokens, no think block, forced tool
    # choice, a client stop string.
    env.guard_log()
    r = chat(env, dict(greedy, messages=user, max_tokens=7600), False)
    check(r["finish"] == "length" and guard_of(r) is None and len(env.guard_log()) == 1,
          "skip-length", r["raw"][:800])
    r = chat(env, dict(greedy, messages=user, max_tokens=7700, enable_thinking=False), False)
    check(r["content"] == THINK and guard_of(r) is None and len(env.guard_log()) == 1,
          "skip-thinking-off", r["raw"][:800])
    r = chat(env, dict(greedy, messages=user, tools=TOOLS, tool_choice="required",
                       max_tokens=7700), False)
    check(guard_of(r) is None and len(env.guard_log()) == 1, "skip-forced-tool", r["raw"][:800])
    r = chat(env, dict(greedy, messages=user, max_tokens=7900, stop=["weather"]), False)
    check(guard_of(r) is None and len(env.guard_log()) == 1, "skip-client-stop", r["raw"][:800])
    r = messages(env, dict(greedy, max_tokens=7700, messages=user + [
        {"role": "assistant", "content": "Sure"}]), False)
    check(guard_of(r) is None and len(env.guard_log()) == 1, "skip-messages-prefill",
          r["raw"][:800])

    # Test hook: think_guard_test_cut ends the first run early, as the model
    # would. (The resume's max_tokens, 7800 - 7704, falls outside the
    # scenario keys; the fake engine's default reply after </think> is "Hi".)
    r = chat(env, dict(greedy, messages=user, max_tokens=7800, think_guard_test_cut=7704), True)
    log = env.guard_log()
    check(r["content"] == "Hi" and r["reasoning"] == THINK * 1284 + "\n\n" + SENTENCE and
          guard_of(r) == "resumed_unclosed" and len(log) == 1 and log[0]["max_tokens"] == 7704,
          "test-cut-hook", (r["raw"][-1500:], log))
    r = chat(env, dict(greedy, messages=user, max_tokens=7800), False)
    check(r["finish"] == "length" and guard_of(r) is None and len(env.guard_log()) == 1,
          "test-cut-absent-no-guard", r["raw"][-800:])


def run_kill_switch(env):
    r = chat(env, dict(temperature=0, messages=[{"role": "user", "content": "weather?"}],
                       max_tokens=7100), False)
    check(r["content"] == "" and r["reasoning"] == THINK and guard_of(r) is None and
          len(env.guard_log()) == 1, "kill-switch-QWENOX_THINK_GUARD=0", r["raw"][:800])


# ------------------------------------------------------------------ harness --

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", required=True, help="path to the compiled qwenox-api binary")
    parser.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent.parent /
                                                   "data" / "tokenizer"))
    args = parser.parse_args()
    api_binary = str(Path(args.api).resolve())
    fake_engine = str(Path(__file__).with_name("api_fake_engine.py").resolve())
    with tempfile.TemporaryDirectory(prefix="qwenox-think-guard-") as temporary:
        root = Path(temporary)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with (root / "test.log").open("w+", encoding="utf-8") as log:
            children = []
            try:
                for guard_env in (None, "0"):
                    engine_port, api_port = free_port(), free_port()
                    while engine_port == api_port:
                        api_port = free_port()
                    env = Env(f"http://127.0.0.1:{api_port}", engine_port)
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
                    child_env.pop("QWENOX_THINK_GUARD", None)
                    if guard_env is not None:
                        child_env["QWENOX_THINK_GUARD"] = guard_env
                    children.append(subprocess.Popen(
                        [api_binary, "--tokenizer", args.tokenizer,
                         "--engine", f"127.0.0.1:{engine_port}", "--host", "127.0.0.1",
                         "--port", str(api_port), "--overrides", ""],
                        cwd=root, env=child_env, stdout=log, stderr=log,
                        creationflags=creationflags))

                    def api_ready():
                        try:
                            return get(env.base, "/health")[0] == 200
                        except OSError:
                            return False

                    wait_for(api_ready, "api-ready", timeout=15)
                    if guard_env is None:
                        run_checks(env)
                    else:
                        run_kill_switch(env)
                    for child in reversed(children):
                        child.terminate()
                        child.wait(timeout=5)
                    children.clear()
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:], file=sys.stderr)
                print("RESULT FAIL")
                raise
            finally:
                for child in reversed(children):
                    child.terminate()
                    child.wait(timeout=5)
    print("RESULT PASS")


if __name__ == "__main__":
    main()
