#!/usr/bin/env python3
"""Exercise cancellation isolation against an already-running two-slot engine.

Fixtures are JSON token arrays named a, b (long multi-chunk prompts), recovery
(short prompt), and optional stop_ids. Alternatively, --tokenizer generates
fixed synthetic audit prompts with the model's tokenizer, retaining their IDs.
For the explicit alternate-per-chunk reproducer, launch the isolated engine
with QWENOX_PREFILL_SRPT=0. The test records SNAP observations and requires
observing A -> B -> A before sending X to B. It never starts/stops a service.
"""
import argparse
import datetime
import json
from pathlib import Path
import socket
import threading
import time


def make_fixtures(tokenizer_path):
    from qwentok import Tokenizer
    tk = Tokenizer(str(tokenizer_path))
    corpus = "\n".join(f"记录{i:05d}：服务{chr(65+i%8)}，成功请求{1000+i%53}，失败请求{i%7}，耗时{20+i%29}毫秒，版本r{i%17}，负责人团队{i%5}，变更经过测试但需审计追踪。" for i in range(9500))
    content = tk.encode(corpus)

    def prompt(rep, size):
        prefix = tk.encode("<|im_start|>system\n你是软件工程审计员。仅根据提供的合成日志撰写详细报告。<|im_end|>\n<|im_start|>user\n" + f"资料批次{rep}。以下是待审计记录：\n")
        suffix = tk.encode("\n请写一份至少2000字的详细工程审计报告，包括现状、风险分析、具体整改步骤、验收办法和运行建议。连续完整撰写，不要只给提纲，不要调用工具。<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n")
        ids = prefix + content[:size - len(prefix) - len(suffix)] + suffix
        if len(ids) != size:
            raise ValueError("tokenizer did not produce enough fixture tokens")
        return ids

    return dict(a=prompt(0, 32768), b=prompt(1, 32768), recovery=prompt(2, 1024),
                stop_ids=[248046, 248044])


class Connection:
    def __init__(self, endpoint):
        self.sock = socket.create_connection(endpoint, timeout=10)
        self.sock.settimeout(180)
        self.file = self.sock.makefile("rb")
        self.rows = []

    def send(self, line):
        self.rows.append(dict(t=time.monotonic(), direction="send", line=line))
        self.sock.sendall((line + "\n").encode("ascii"))

    def read(self):
        line = self.file.readline()
        if not line:
            raise EOFError("engine disconnected")
        line = line.decode("ascii").strip()
        self.rows.append(dict(t=time.monotonic(), direction="recv", line=line))
        return line.split()

    def start(self, req, ids, stop_ids, drafter=0, budget=16):
        self.send(" ".join(map(str, ["GEN", req, budget, len(stop_ids), *stop_ids,
                                     len(ids), *ids, drafter])))
        self.tokens, self.done, self.error = [], None, None

        def receive():
            try:
                while True:
                    parts = self.read()
                    if parts[0] == "T":
                        self.tokens.append(int(parts[2]))
                    if parts[0] == "D":
                        if int(parts[1]) != req:
                            raise ValueError("unexpected request ID")
                        self.done = parts
                        break
            except Exception as exc:
                self.error = repr(exc)

        self.worker = threading.Thread(target=receive, daemon=True)
        self.worker.start()

    def finish(self):
        self.worker.join(180)
        if self.worker.is_alive() or self.error or not self.done:
            raise RuntimeError(self.error or "completion timeout")
        return dict(tokens=self.tokens, done=self.done)

    def close(self):
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        if hasattr(self, "worker"):
            self.worker.join(5)
        self.file.close()
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="127.0.0.1:18960")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--fixtures", type=Path)
    inputs.add_argument("--tokenizer", type=Path, help="Flash-Next tokenizer directory for synthetic fixtures")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--id-base", type=int, default=910000)
    args = parser.parse_args()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started_monotonic = time.monotonic()
    host, port = args.endpoint.rsplit(":", 1)
    endpoint = host, int(port)
    fixtures = (json.loads(args.fixtures.read_text(encoding="utf-8")) if args.fixtures
                else make_fixtures(args.tokenizer))
    stop_ids = fixtures.get("stop_ids", [])
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "fixtures.json").write_text(json.dumps(fixtures), encoding="utf-8")
    conns, records, checks = [], [], []

    def conn():
        c = Connection(endpoint)
        conns.append(c)
        return c

    def serial(key, req, drafter=0):
        c = conn()
        c.start(req, fixtures[key], stop_ids, drafter)
        result = c.finish()
        records.append(dict(case=key, req=req, drafter=drafter, **result))
        checks.append(dict(case=f"{key}-{req}-completed", passed=result["done"][2] in ["done", "length"]
                           and len(result["tokens"]) > 0))
        if drafter == 1:
            checks.append(dict(case=f"mtp-{req}-active", passed=len(result["done"]) > 11
                               and int(result["done"][7]) == 1 and int(result["done"][11]) > 0))
        return c, result

    req = args.id_base
    try:
        _, reference = serial("a", req)
        req += 1
        control, a, b = conn(), conn(), conn()
        a_req, b_req = req, req + 1
        a.start(a_req, fixtures["a"], stop_ids)

        def wait_snap(predicate, label):
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                control.send("SNAP")
                parts = control.read()
                if parts[0] != "P":
                    raise ValueError("unexpected SNAP response")
                snap = dict(state=int(parts[2]), req=int(parts[3]), done=int(parts[5]), total=int(parts[6]))
                if predicate(snap):
                    records.append(dict(case=label, t=time.monotonic(), **snap))
                    return
                if a.done or b.done:
                    raise RuntimeError("request ended before required overlapping handoff: " + label)
                time.sleep(0.05)
            raise TimeoutError(label)

        # B has not been started; initialize the observation used by wait_snap.
        b.done = None
        wait_snap(lambda s: s["state"] == 1 and s["req"] == a_req and s["done"] > 0, "a-active")
        b.start(b_req, fixtures["b"], stop_ids)
        wait_snap(lambda s: s["state"] == 1 and s["req"] == b_req and s["done"] > 0, "b-active")
        wait_snap(lambda s: s["state"] == 1 and s["req"] == a_req and s["done"] > 0, "a-resumed-after-b")
        b.send(f"X {b_req}")
        records.append(dict(case="cancel-b-while-a-active", t=time.monotonic()))
        a_result, b_result = a.finish(), b.finish()
        records.append(dict(case="uncancelled-peer", req=a_req, **a_result))
        records.append(dict(case="cancelled-peer", req=b_req, **b_result))
        checks.append(dict(case="uncancelled-peer-finish", passed=a_result["done"][2] in ["done", "length"]))
        checks.append(dict(case="uncancelled-peer-token-parity", passed=a_result["tokens"] == reference["tokens"]))
        checks.append(dict(case="requested-cancel", passed=b_result["done"][2] == "cancel"))
        # Reuse the cancelled connection, then verify fresh serial and MTP work.
        b.start(req + 2, fixtures["recovery"], stop_ids)
        recovery = b.finish()
        records.append(dict(case="cancelled-connection-recovery", **recovery))
        checks.append(dict(case="cancelled-connection-recovery", passed=recovery["done"][2] in ["done", "length"]
                           and len(recovery["tokens"]) > 0))
        _, clean = serial("recovery", req + 3)
        checks.append(dict(case="recovery-token-parity", passed=recovery["tokens"] == clean["tokens"]))
        serial("recovery", req + 4, drafter=1)
    except Exception as exc:
        records.append(dict(error=repr(exc)))
        checks.append(dict(case="test-completed", passed=False))
    finally:
        for c in conns:
            c.close()
        result = dict(passed=bool(checks) and all(x["passed"] for x in checks),
                      started_utc=started_utc, started_monotonic=started_monotonic,
                      checks=checks, records=records, connections=[c.rows for c in conns])
        (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for check in checks:
        print(("PASS" if check["passed"] else "FAIL") + ": " + check["case"], flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
