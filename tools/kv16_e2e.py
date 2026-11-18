#!/usr/bin/env python3
"""Real-model BF16/FP16 snapshot, format-isolation and paged concurrency tests.

Run on the machine with the GPU and model files. Each invocation requires a
new --output directory; production caches and existing processes are untouched.
Example (Windows):
  python tools/kv16_e2e.py --engine build/qwenox-engine.exe \
    --model C:/Models/base.hgn --model C:/Models/overlay.hgn \
    --model C:/Models/mtp.hgn --output build/kv16-results

Only continuations within one precision are compared. FP16 and BF16 are not
expected to produce identical tokens. The line protocol uses greedy decoding.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import subprocess
import threading
import time


ROOT = Path(__file__).resolve().parent.parent


class Suite:
    def __init__(self, args):
        self.args = args
        self.out = Path(args.output).resolve()
        self.out.mkdir(parents=True, exist_ok=False)
        self.report_lock = threading.RLock()
        self.report = {"engine": str(Path(args.engine).resolve()),
                       "engine_sha256": hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),
                       "platform": platform.platform(),
                       "models": args.model, "started": time.time(),
                       "checks": [], "runs": {}}
        self.save()

    def save(self):
        with self.report_lock:
            (self.out / "results.json").write_text(
                json.dumps(self.report, indent=2), encoding="utf-8")

    def check(self, ok, label, **detail):
        self.report["checks"].append(dict(ok=bool(ok), label=label, **detail))
        self.save()
        print(("PASS " if ok else "FAIL ") + label +
              (" " + json.dumps(detail) if detail else ""), flush=True)
        if not ok:
            raise AssertionError(label)

    def run(self, name, precision, cache, paged=0, parallel=1, maxctx=12288,
            extra=None):
        return Engine(self, name, precision, cache, paged, parallel, maxctx, extra)

    def prompt(self, n):
        source = ROOT / "data" / "qsa-oracle" / (str(n) + ".json")
        return json.loads(source.read_text(encoding="utf-8"))["prompt_ids"]

    def snapshot(self):
        for precision in ("bf16", "fp16"):
            for n in self.args.sizes:
                for drafter in ([0, 1] if n == min(self.args.sizes) else [1]):
                    label = f"{precision}-{n}-d{drafter}"
                    cache = self.out / (label + "-cache")
                    p = self.prompt(n)
                    maxctx = max(12288, ((len(p) + 1024 + 4095) // 4096) * 4096)
                    with self.run(label + "-save", precision, cache,
                                  maxctx=maxctx) as engine:
                        first = engine.gen(p, 24, drafter=drafter,
                                           snaps=[min(1024, len(p) // 2)])
                        engine.sync()
                        q = p + first["tokens"] + [198, 198, 1000, 13, 198]
                        gold = engine.gen(q, 64, drafter=drafter)
                        engine.sync()
                        self.check(gold["cached"] >= len(p) + len(first["tokens"]) - 1,
                                   label + " uses live continuation state")
                    headers = inventory(cache)
                    self.check(bool(headers) and all(v["kvb"] == tag(precision)
                               for v in headers.values()), label + " headers")
                    self.check(any(v.get("mtp") for v in headers.values())
                               if drafter else True, label + " MTP saved")
                    for paged in (2, 1):
                        with self.run(label + f"-restore-p{paged}", precision,
                                      cache, paged=paged, maxctx=maxctx) as engine:
                            restored = engine.gen(q, 64, drafter=drafter)
                            engine.sync()
                            self.check(restored["tokens"] == gold["tokens"],
                                       label + f" restore p{paged} token identity")
                            self.check(restored["cached"] >= len(p),
                                       label + f" restore p{paged} cache hit",
                                       cached=restored["cached"])
                            self.check("kvsnap: restored" in engine.logtext(),
                                       label + f" restore p{paged} SSD log")
                            if drafter:
                                self.check("(mtp live)" in engine.logtext(),
                                           label + f" restore p{paged} MTP live")

    def isolation(self):
        # The SAME directory is reused through BF16 -> FP16 -> BF16 -> FP16.
        # Each direction also gets an independent cold-directory control.
        p = self.prompt(8192)
        for source, target in (("bf16", "fp16"), ("fp16", "bf16")):
            label = source + "-to-" + target
            cache = self.out / (label + "-shared")
            with self.run(label + "-source", source, cache, paged=2) as engine:
                first = engine.gen(p, 24)
                q = p + first["tokens"] + [198, 1000, 13, 198]
                source_gold = engine.gen(q, 64)
                engine.sync()
            source_files = inventory(cache)
            with self.run(label + "-target", target, cache, paged=2) as engine:
                cold_shared = engine.gen(q, 64)
                engine.sync()
                self.check(cold_shared["cached"] == 0 and
                           "kvsnap: restored" not in engine.logtext(),
                           label + " foreign cache never restored")
            shared_files = inventory(cache)
            self.check(all(shared_files.get(k) == v for k, v in source_files.items()),
                       label + " original files preserved")
            other_files = {k: v for k, v in shared_files.items() if k not in source_files}
            self.check(bool(other_files) and
                       all(v["kvb"] == tag(target) for v in other_files.values()),
                       label + " disjoint page/checkpoint names and tags")
            source_seeds = {v["seed"] for v in source_files.values() if "seed" in v}
            target_seeds = {v["seed"] for v in other_files.values() if "seed" in v}
            self.check(source_seeds.isdisjoint(target_seeds),
                       label + " distinct snapshot fingerprints")
            with self.run(label + "-cold", target,
                          self.out / (label + "-clean"), paged=2) as engine:
                cold_clean = engine.gen(q, 64)
                engine.sync()
            self.check(cold_shared["tokens"] == cold_clean["tokens"],
                       label + " shared-directory cold equals clean cold")
            # The target saved the end of q's generation. Continue beyond that
            # exact state, first live and then after restarting into mixed files.
            target_q = q + cold_shared["tokens"] + [198, 1000, 198]
            with self.run(label + "-target-live", target, cache, paged=1) as engine:
                target_gold = engine.gen(target_q, 64)
                engine.sync()
                self.check(target_gold["cached"] > len(q),
                           label + " own target snapshot available")
            with self.run(label + "-switchback", source, cache, paged=1) as engine:
                again = engine.gen(q, 64)
                engine.sync()
                self.check(again["tokens"] == source_gold["tokens"] and
                           again["cached"] >= len(p),
                           label + " switch back restores original precision")
            with self.run(label + "-target-again", target, cache, paged=2) as engine:
                again = engine.gen(target_q, 64)
                engine.sync()
                self.check(again["tokens"] == target_gold["tokens"] and
                           again["cached"] > len(q),
                           label + " target remains restorable in mixed directory")

    def parallel(self):
        p = self.prompt(8192)
        for precision in ("bf16", "fp16"):
            label = precision + "-parallel"
            cache = self.out / (label + "-cache")
            with self.run(label + "-gold", precision, cache, paged=1) as engine:
                first = engine.gen(p, 24)
                engine.sync()
                queries = [p + first["tokens"] + tail
                           for tail in ([198, 1000, 13, 198], [198, 1001, 13, 198])]
                gold = [engine.gen(q, 96) for q in queries]
                engine.sync()
            with self.run(label + "-two-slots", precision, cache, paged=1,
                          parallel=2) as engine:
                with concurrent.futures.ThreadPoolExecutor(2) as pool:
                    actual = list(pool.map(lambda q: engine.gen(q, 96), queries))
                engine.sync()
                for i, (a, b) in enumerate(zip(actual, gold)):
                    self.check(a["tokens"] == b["tokens"] and a["cached"] >= len(p),
                               label + f" shared-prefix/COW slot {i} token identity")
                self.check("[vr]" in engine.logtext(), label + " verify pool exercised")

    def guards(self):
        # Preserve the valid fingerprint and filename, then deliberately change
        # only a format tag in this disposable directory. This bypasses matching
        # isolation and exercises the restore-time checkpoint/page defenses.
        p = self.prompt(2051)
        for precision in ("bf16", "fp16"):
            label = precision + "-guards"
            cache = self.out / (label + "-valid")
            with self.run(label + "-save", precision, cache, paged=2) as engine:
                first = engine.gen(p, 24)
                engine.sync()
            q = p + first["tokens"] + [198, 1000, 198]
            with self.run(label + "-cold", precision,
                          self.out / (label + "-clean"), paged=2) as engine:
                gold = engine.gen(q, 64)
                engine.sync()
            for kind, suffix, offset in (("checkpoint", ".kck", 52),
                                          ("page", ".kvc", 36)):
                damaged = self.out / (label + "-wrong-" + kind)
                shutil.copytree(cache, damaged)
                paths = list(damaged.glob("*" + suffix))
                self.check(bool(paths), label + " has " + kind + " files")
                for path in paths:
                    with path.open("r+b") as stream:
                        stream.seek(offset)
                        stream.write(struct.pack("<i", 2 if precision == "fp16" else 16))
                with self.run(label + "-wrong-" + kind, precision, damaged,
                              paged=2) as engine:
                    actual = engine.gen(q, 64)
                    engine.sync()
                    self.check(actual["cached"] == 0 and
                               "kvsnap: restored" not in engine.logtext(),
                               label + " rejects wrong " + kind + " encoding")
                    self.check("failed (" in engine.logtext(),
                               label + " reaches " + kind + " restore defense")
                    self.check(actual["tokens"] == gold["tokens"],
                               label + " wrong " + kind + " fallback equals cold")


def tag(precision):
    return 16 if precision == "fp16" else 2


def inventory(directory):
    result = {}
    for path in sorted(directory.iterdir()):
        if path.suffix not in (".kck", ".kvc", ".kvm"):
            continue
        with path.open("rb") as stream:
            header = stream.read(128)
            digest = hashlib.sha256(header)
            for block in iter(lambda: stream.read(4 << 20), b""):
                digest.update(block)
        entry = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
        if path.suffix == ".kck":
            assert header[:8] == b"GDKCK1\0\0", path
            entry.update(seed=struct.unpack_from("<Q", header, 8)[0],
                         n=struct.unpack_from("<i", header, 40)[0],
                         kvb=struct.unpack_from("<i", header, 52)[0],
                         mtp=struct.unpack_from("<i", header, 60)[0])
        else:
            assert header[:8] in (b"GDKVC1\0\0", b"GDKVM1\0\0"), path
            entry["kvb"] = struct.unpack_from("<i", header, 36)[0]
        result[path.name] = entry
    return result


class Engine:
    def __init__(self, suite, name, precision, cache, paged, parallel, maxctx, extra):
        self.suite, self.name = suite, name
        self.cache, self.paged = cache, paged
        self.parallel, self.maxctx, self.precision = parallel, maxctx, precision
        self.extra = extra or {}
        self.log = suite.out / (name + ".log")
        self.process = None
        self.conn = self.reader = None

    def __enter__(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("QWENOX_")}
        env.update({"QWENOX_KV_TYPE": self.precision,
                    "QWENOX_KV_PAGED": str(self.paged),
                    "QWENOX_PARALLEL": str(self.parallel),
                    "QWENOX_KV_POOL_TOKENS": str(self.maxctx * (self.parallel + 1)),
                    "QWENOX_KVSNAP": "1", "QWENOX_KVSNAP_DIR": str(self.cache),
                    "QWENOX_KVSNAP_MIN": "512", "QWENOX_KVSNAP_MAX_GB": "12",
                    "QWENOX_PREFILL_CHUNK": "8192", "QWENOX_CONC_PREFILL_CHUNK": "8192",
                    "QWENOX_SPEC_GAMMA": "4", "QWENOX_SPEC_ADAPT": "0"})
        for key in ("QSA_KV_BF16", "QSA_WMMA", "QSA_WMMA_BTV", "GEMM_WMMA",
                    "GDN_FUSED", "MOE_LT", "MOE_LT_BF16", "GR_BF16", "GDN_STREAM",
                    "GDN_WAVE", "NOWARMUP", "INDEX_FUSED2", "PP_MOE_OUT",
                    "INDEX_STREAM_SELECT"):
            env["QWENOX_" + key] = "1"
        env.update(self.extra)
        # Reserve a free local port, then start our child only. GPU tests run
        # sequentially; each concurrent case uses one shared engine process.
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            self.port = listener.getsockname()[1]
        command = [self.suite.report["engine"], *self.suite.args.model, "--serve",
                   "--host", "127.0.0.1", "--port", str(self.port),
                   "--maxctx", str(self.maxctx)]
        self.stream = self.log.open("wb")
        self.process = subprocess.Popen(command, env=env, cwd=ROOT,
                                        stdout=self.stream, stderr=subprocess.STDOUT)
        self.suite.report["runs"][self.name] = dict(precision=self.precision,
            paged=self.paged, parallel=self.parallel, pid=self.process.pid,
            cache=str(self.cache), env={k: v for k, v in env.items()
                                       if k.startswith("QWENOX_")}, requests=[])
        self.suite.save()
        print("START " + self.name, flush=True)
        deadline = time.monotonic() + 600
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"engine exited {self.process.returncode}: {self.log}")
                try:
                    with socket.create_connection(("127.0.0.1", self.port), timeout=1):
                        pass
                    if self.parallel == 1:
                        # A single-slot engine deliberately clears live history
                        # when the connection changes. Keep the same connection
                        # for the save/live oracle and KVSYNC barrier.
                        self.conn = socket.create_connection(("127.0.0.1", self.port),
                                                             timeout=1800)
                        self.reader = self.conn.makefile("rb")
                    return self
                except OSError:
                    time.sleep(.5)
            raise TimeoutError("engine startup: " + str(self.log))
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.reader:
            self.reader.close()
        if self.conn:
            self.conn.close()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.stream.close()

    def logtext(self):
        return self.log.read_text(encoding="utf-8", errors="replace")

    def sync(self):
        context = (contextlib.nullcontext(self.conn) if self.conn else
                   socket.create_connection(("127.0.0.1", self.port), timeout=600))
        with context as conn:
            conn.sendall(b"KVSYNC\n")
            stream = self.reader if self.conn else conn.makefile("rb")
            line = stream.readline().decode().strip()
            if not line.startswith("K "):
                raise RuntimeError("KVSYNC failed: " + line)
        return line

    def gen(self, ids, count, drafter=1, snaps=None):
        # Independent connections can use the same request id.
        line = f"GEN 1 {count} 0 {len(ids)} " + " ".join(map(str, ids)) + f" {drafter}"
        if snaps:
            line += " SNAPS " + str(len(snaps)) + " " + " ".join(map(str, snaps))
        start, tokens, done = time.monotonic(), [], None
        context = (contextlib.nullcontext(self.conn) if self.conn else
                   socket.create_connection(("127.0.0.1", self.port), timeout=1800))
        with context as conn:
            conn.sendall((line + "\n").encode())
            stream = self.reader if self.conn else conn.makefile("rb")
            while True:
                raw = stream.readline()
                if not raw:
                    raise RuntimeError("engine disconnected: " + str(self.log))
                fields = raw.decode().split()
                if fields[0] == "T":
                    tokens.append(int(fields[2]))
                elif fields[0] == "D":
                    done = fields
                    break
        if done[2] != "length" or len(tokens) != count:
            raise RuntimeError(f"unexpected generation: {done}, {len(tokens)} tokens")
        result = dict(tokens=tokens, done=done, cached=int(done[10]),
                      seconds=round(time.monotonic() - start, 3),
                      input_count=len(ids), drafter=drafter)
        self.suite.report["runs"][self.name]["requests"].append(result)
        self.suite.save()
        print(f"GEN {self.name} prompt={len(ids)} cached={result['cached']} "
              f"seconds={result['seconds']}", flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[2051, 8192, 32768])
    parser.add_argument("--only", nargs="+", choices=["snapshot", "isolation", "parallel", "guards"],
                        default=["snapshot", "isolation", "parallel", "guards"])
    args = parser.parse_args()
    suite = Suite(args)
    try:
        for phase in args.only:
            getattr(suite, phase)()
        suite.report["passed"] = True
        print("ALL PASS", flush=True)
    finally:
        suite.report["finished"] = time.time()
        suite.save()


if __name__ == "__main__":
    main()
