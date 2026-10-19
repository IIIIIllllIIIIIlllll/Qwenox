#!/usr/bin/env python3
"""在真实 HTTP 服务上验证日常并发、排队取消、多轮工具和图片请求。

不启动服务，不修改配置，不注入 snapshot/文件系统故障。建议在隔离实例运行，
使用默认 decode 预留与缓存配置；只在实际观察到时才报告 park 覆盖。
--timeout 发起主动关闭；无人值守运行时另用 timeout 等 supervisor 设置硬截止。
"""

import argparse
import base64
import http.client
import json
import re
from pathlib import Path
import socket
import threading
import time

from api_park_resume_verify import (
    ApiClient, health, positive, response_summary, save_state, stream_summary,
    validate_stream, wait_ready,
)


class LoadClient(ApiClient):
    def __init__(self, base):
        super().__init__(base, timeout=90, health_timeout=3)
        self.stream_sockets = set()
        self.expired = threading.Event()

    def request(self, *args, **kwargs):
        if self.expired.is_set():
            raise TimeoutError("测试总时限已到")
        return super().request(*args, **kwargs)

    def _read_anthropic_stream(self, response, started):
        sock = response.fp.raw._sock
        with self._active_lock:
            self.stream_sockets.add(sock)
        try:
            return super()._read_anthropic_stream(response, started)
        finally:
            with self._active_lock:
                self.stream_sockets.discard(sock)

    def close_active(self):
        with self._active_lock:
            sockets = list(self.stream_sockets) + [conn.sock for conn in self._active if conn.sock]
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        super().close_active()

    def expire(self):
        self.expired.set()
        self.close_active()


ROOT = Path(__file__).resolve().parent.parent


def body(prompt, maximum=128, stream=True, temperature=0):
    return {"model": "qwen3.8-flash-next", "max_tokens": maximum,
            "temperature": temperature, "seed": 777, "stream": stream,
            "enable_thinking": False,
            "messages": [{"role": "user", "content": prompt}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18731")
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=ROOT / "src/launch_win_icon.png")
    parser.add_argument("--skip-image", action="store_true")
    parser.add_argument("--timeout", type=positive, default=480)
    args = parser.parse_args()
    client = LoadClient(args.base)
    state = {"started_at": time.time(), "base": args.base, "checks": [], "phases": {}}
    initial_log = args.log.stat().st_size if args.log else 0
    source = (ROOT / "src/kv_admission.h").read_text()
    guide = (ROOT / "docs/CONCURRENCY.md").read_text()
    context = "<code>\n" + source + "\n</code>\n<documentation>\n" + guide + "\n</documentation>"
    cases = {
        "code_review": body("Review the actual C++ admission code and accompanying documentation. "
                            "Explain its arithmetic and invariants, then discuss practical deployment "
                            "tradeoffs and normal user-facing failure modes. Give a detailed engineering "
                            "review with examples; do not invent missing functions.\n" + context, 1024),
        "deployment_guide": body("Write a detailed operator runbook based on this documentation. "
                                 "Cover normal concurrency, sensible KV sizing, cancellations, monitoring, "
                                 "and a rollback checklist. Distinguish what is documented from inference.\n"
                                 + context, 1024, stream=False),
        "translation": body("将下面的发布说明翻译成自然中文，只输出译文："
                            "Queued requests can be cancelled without interrupting active users.", 96),
        "release_notes": body("Write five concise release-note bullets for a local coding assistant "
                              "server that supports shared KV memory, fair queuing, resumable requests, "
                              "streaming replies and client cancellation. Do not add unsupported claims.",
                              256, temperature=0.8),
    }

    def checkpoint():
        save_state(args.state, state)

    def check(name, condition, detail=None, fatal=True):
        state["checks"].append({"name": name, "passed": bool(condition), "detail": detail})
        checkpoint()
        print(("PASS " if condition else "FAIL ") + name, flush=True)
        if not condition and fatal:
            raise AssertionError(f"{name}: {detail}")

    def wait_for(predicate, name, timeout=15):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = health(client)
            if predicate(last):
                return last
            time.sleep(0.1)
        raise AssertionError(f"{name}: {last}")

    def idle():
        return wait_for(lambda h: h["in_flight"] == 0 and h["queued"] == 0, "资源归零")

    def request(request_body, path="/v1/messages"):
        response = client.request("POST", path, request_body,
                                  stream=request_body.get("stream", False))
        summary = stream_summary(response) if request_body.get("stream") else response_summary(response)
        if summary["status"] != 200 or summary.get("error") or summary.get("protocol_errors") \
                or not summary.get("stop_reason") or not summary.get("text"):
            raise AssertionError(f"无效生成响应：{summary}")
        usage = summary["usage"]
        fields = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        if any(type(usage.get(key)) is not int or usage[key] < 0 for key in fields) \
                or usage["output_tokens"] == 0 or usage["output_tokens"] > request_body["max_tokens"]:
            raise AssertionError(f"无效 usage：{usage}")
        summary["input_total"] = sum(usage[key] for key in fields if key != "output_tokens")
        return summary

    def group(name, requests, during=None):
        record = {"results": {}, "errors": [], "health": []}
        state["phases"][name] = record
        stop = threading.Event()
        lock = threading.Lock()

        def worker(key, request_body):
            try:
                value = request(request_body)
                with lock:
                    record["results"][key] = value
            except Exception as exc:
                with lock:
                    record["errors"].append({"case": key, "error": repr(exc)})

        def sample():
            while not stop.is_set():
                try:
                    value = health(client)
                    with lock:
                        if len(record["health"]) < 1500:
                            record["health"].append({"time": time.time(), "in_flight": value["in_flight"],
                                                     "queued": value["queued"]})
                except Exception as exc:
                    with lock:
                        record["errors"].append({"case": "health", "error": repr(exc)})
                    break
                stop.wait(0.1)

        sampler = threading.Thread(target=sample, daemon=True)
        workers = [threading.Thread(target=worker, args=(key, value), daemon=True)
                   for key, value in requests.items()]
        deadline = time.monotonic() + 180
        sampler.start()
        for thread in workers:
            thread.start()
        try:
            if during:
                during(record)
            for thread in workers:
                thread.join(max(0, deadline - time.monotonic()))
            check(name + "：全部请求完成", not any(t.is_alive() for t in workers)
                  and not record["errors"] and set(record["results"]) == set(requests), record["errors"])
        finally:
            stop.set()
            sampler.join(timeout=4)
            if any(t.is_alive() for t in workers):
                client.close_active()
                cleanup_deadline = time.monotonic() + 10
                for thread in workers:
                    thread.join(max(0, cleanup_deadline - time.monotonic()))
            record["threads_alive"] = [thread.name for thread in workers if thread.is_alive()]
            checkpoint()
        record["final_health"] = idle()
        return record

    def identical(name, baseline, actual):
        keys = ("text", "stop_reason", "input_total")
        mismatch = [key for key in keys if baseline.get(key) != actual.get(key)]
        if baseline["usage"].get("output_tokens") != actual["usage"].get("output_tokens"):
            mismatch.append("output_tokens")
        check(name, not mismatch, mismatch, fatal=False)

    def open_request(request_body, path="/v1/messages"):
        conn = http.client.HTTPConnection(client.host, client.port, timeout=30)
        conn.request("POST", client.prefix + path, json.dumps(request_body).encode(),
                     {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
        return conn

    def log_since(offset):
        with args.log.open("rb") as reader:
            reader.seek(offset)
            return reader.read().decode("utf-8", "replace").splitlines()

    def disconnect(conn, response=None):
        sock = conn.sock
        if sock is None and response is not None:
            sock = response.fp.raw._sock
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if response is not None:
            response.close()
        conn.close()

    def follow_up(name):
        value = request(body("What is 17 multiplied by 23? Reply with only the number.", 32))
        state["phases"][name] = value
        check(name, "391" in value["text"], value["text"])
        idle()

    timer = threading.Timer(args.timeout, client.expire)
    timer.daemon = True
    timer.start()
    try:
        state["initial_health"] = wait_ready(client, min(args.timeout, 120))
        check("独立双槽服务处于空闲", state["initial_health"]["slots"] == 2
              and state["initial_health"]["in_flight"] == 0 and state["initial_health"]["queued"] == 0)
        state["requests"] = cases
        state["baseline"] = {}
        for key, value in cases.items():
            state["baseline"][key] = request(value)
            idle()
            checkpoint()
        check("现实任务串行基线", len(state["baseline"]) == 4)
        state["serial_repeat"] = request(cases["code_review"])
        idle()
        identical("代码审查串行重复与初始基线一致", state["baseline"]["code_review"], state["serial_repeat"])

        for wave in range(2):
            result = group(f"mixed_wave_{wave + 1}", cases)
            for key in cases:
                identical(f"混合并发第{wave + 1}轮 {key} 与串行一致",
                          state["baseline"][key], result["results"][key])
            check(f"混合并发第{wave + 1}轮实际发生排队", any(h["queued"] > 0 for h in result["health"]))

        def cancel_queued(record):
            record["before_cancel_request"] = wait_for(lambda h: h["in_flight"] == 2, "两路工作中")
            conn = open_request(body("Summarize the benefits of request queuing in one sentence.", 64))
            try:
                record["queued_observed"] = wait_for(lambda h: h["queued"] >= 1 and h["in_flight"] == 2,
                                                     "第三路实际排队")
            finally:
                disconnect(conn)
            record["queue_drained"] = wait_for(lambda h: h["queued"] == 0 and h["in_flight"] == 2,
                                               "取消排队请求，不影响两路生成")

        queued_log_start = args.log.stat().st_size
        queued = group("queued_disconnect", {key: cases[key] for key in ("code_review", "deployment_guide")},
                       during=cancel_queued)
        queued["admissions"] = [line for line in log_since(queued_log_start)
                                if re.search(r"req \d+ -> slot", line)]
        check("排队取消请求未进入 engine", len(queued["admissions"]) == 2, queued["admissions"])
        for key in queued["results"]:
            identical("排队取消不改变 " + key, state["baseline"][key], queued["results"][key])
        follow_up("after_queued_cancel")

        for streaming in (True, False):
            idle()
            record = {"stream": streaming}
            state["phases"][f"active_cancel_{streaming}"] = record
            cancel_log_start = args.log.stat().st_size
            conn = open_request(dict(cases["code_review"], stream=streaming, max_tokens=1536))
            response = None
            try:
                record["active_observed"] = wait_for(lambda h: h["in_flight"] == 1, "取消前实际运行")
                if streaming:
                    response = conn.getresponse()
                    check("运行中 SSE 请求返回200", response.status == 200)
                    deadline = time.monotonic() + 30
                    found = False
                    while time.monotonic() < deadline:
                        line = response.readline(262145)
                        if not line:
                            break
                        if line.startswith(b"data:"):
                            event = json.loads(line[5:])
                            if event.get("delta", {}).get("type") == "text_delta" \
                                    and event["delta"].get("text"):
                                found = True
                                break
                    check("用户停止前已经收到正文 token", found)
                else:
                    time.sleep(0.5)
                record["before_disconnect"] = health(client)
                check(f"取消时请求仍在运行 stream={streaming}", record["before_disconnect"]["in_flight"] == 1)
            finally:
                disconnected_at = time.monotonic()
                disconnect(conn, response)
            record["after_disconnect"] = idle()
            record["cancel_elapsed"] = time.monotonic() - disconnected_at
            record["cancel_evidence"] = [line for line in log_since(cancel_log_start)
                                         if re.search(r"req \d+ \| decode eval = .* \| cancel$", line)
                                         or re.search(r"serve: req \d+ prefill aborted: client gone$", line)]
            check(f"运行中取消而非自然完成 stream={streaming}",
                  len(record["cancel_evidence"]) == 1 and record["cancel_elapsed"] < 10,
                  record["cancel_evidence"])
            check(f"运行中取消释放资源 stream={streaming}", True)
            follow_up(f"after_active_cancel_{streaming}")

        dialogs = {}
        dialog_reference = context + "\n<configuration_parser>\n" + "\n".join(
            (ROOT / "src/service_conf.h").read_text().splitlines()[:180]) + "\n</configuration_parser>"
        for label, port, limit in (("Atlas", 9201, 64), ("Borealis", 9302, 128)):
            dialogs[label] = body(f"We are reviewing deployment {label}. Its HTTP port is {port} "
                                  f"and queue limit is {limit}. Briefly acknowledge these values.\n" + dialog_reference, 96)
        turn1 = group("conversation_turn_1", dialogs)
        for label in dialogs:
            dialogs[label] = dict(dialogs[label], messages=dialogs[label]["messages"] + [
                {"role": "assistant", "content": turn1["results"][label]["text"]},
                {"role": "user", "content": "What HTTP port and queue limit did I give for this deployment? "
                 "Use only those values, not values from the reference documentation."}])
        turn2 = group("conversation_turn_2", dialogs)
        check("正常多轮请求实际复用了长上下文缓存", all(
            value["usage"]["cache_read_input_tokens"] >= 4096 for value in turn2["results"].values()),
            {key: value["usage"] for key, value in turn2["results"].items()}, fatal=False)
        for label, port, limit, other in (("Atlas", "9201", "64", "9302"),
                                          ("Borealis", "9302", "128", "9201")):
            text = turn2["results"][label]["text"]
            numbers = re.findall(r"\b\d+\b", text)
            check("多轮上下文隔离 " + label, port in numbers and limit in numbers and other not in numbers, text)

        tool = {"name": "get_service_limits", "description": "Read the current local service limits.",
                "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}}
        tool_body = body("Read the current local service limits using the tool, then tell me "
                         "the number of parallel slots and reserved decode tokens.", 256)
        tool_body.update(tools=[tool], tool_choice={"type": "tool", "name": tool["name"]})
        raw = client.request("POST", "/v1/messages", tool_body, stream=True)
        ok, why, message = validate_stream(raw["events"])
        check("真实工具调用 SSE 协议", raw["status"] == 200 and ok, why)
        calls = [block for block in message["content"] if block["type"] == "tool_use"]
        check("工具调用名称与参数", message["stop_reason"] == "tool_use" and len(calls) == 1
              and calls[0]["name"] == tool["name"] and calls[0]["input"] == {}, message)
        tool_result = {"parallel": state["initial_health"]["slots"], "reserve_decode": 4096,
                       "context": state["initial_health"]["context"]}
        tool_body["tool_choice"] = {"type": "auto"}
        tool_body["messages"] += [{"role": "assistant", "content": message["content"]},
                                  {"role": "user", "content": [{"type": "tool_result",
                                   "tool_use_id": calls[0]["id"], "content": json.dumps(tool_result)}]}]
        tool_answer = request(tool_body)
        state["phases"]["tool_round_trip"] = {"call": message, "result": tool_result, "answer": tool_answer}
        numbers = re.findall(r"\b\d+\b", tool_answer["text"])
        check("工具结果回传后正常回答", "4096" in numbers and "2" in numbers
              and tool_answer["stop_reason"] == "end_turn", tool_answer["text"])
        idle()

        if not args.skip_image:
            image_body = body("", 128)
            image_body["messages"][0]["content"] = [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                 "data": base64.b64encode(args.image.read_bytes()).decode()}},
                {"type": "text", "text": "Describe this application icon in one concise sentence."}]
            image = group("image_and_text", {"image": image_body, "text": cases["translation"]})
            identical("图片并发不改变纯文本输出", state["baseline"]["translation"], image["results"]["text"])
            follow_up("after_image")
        state["final_health"] = idle()
        state["passed"] = all(item["passed"] for item in state["checks"])
        if not state["passed"]:
            raise AssertionError("失败检查：" + "; ".join(item["name"] for item in state["checks"] if not item["passed"]))
    except Exception as exc:
        state["passed"] = False
        state["failure"] = repr(exc)
        raise
    finally:
        timer.cancel()
        state["deadline_exceeded"] = client.expired.is_set()
        state["finished_at"] = time.time()
        if args.log:
            with args.log.open("rb") as reader:
                reader.seek(initial_log)
                lines = reader.read().decode("utf-8", "replace").splitlines()
            patterns = {"park": "parked to kvsnap", "restore": "kvsnap: restored",
                        "fallback": "falling back to abort", "failure": " failed:",
                        "ram_restore": "rckpt", "admission": "rejected at admission"}
            state["log_evidence"] = {key: [line for line in lines if pattern in line]
                                     for key, pattern in patterns.items()}
        checkpoint()
        print("RESULT " + ("PASS" if state.get("passed") else "FAIL"), flush=True)


if __name__ == "__main__":
    main()
