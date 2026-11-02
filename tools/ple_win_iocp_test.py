#!/usr/bin/env python3
"""Windows CPU-only regression and fault injection for the production PleWin reader.

No ROCm runtime, GPU, or model is required. Each fatal-path test runs in its own
child process. --source can select an unmodified source file for before/after
reproduction; --output retains source identity, compile command, and raw results.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("this regression requires Windows IOCP")
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")
    root = Path(__file__).resolve().parent.parent
    source = args.source or root / "src/gpu/parts/10_ple_io.inc"
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    start = text.index("struct PleWin {")
    end = text.index("\n};", start) + len("\n};")
    temporary = None
    if args.output:
        work = args.output.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="ple-win-iocp-")
        work = Path(temporary.name)
    try:
        (work / "ple_win_under_test.inc").write_text(text[start:end], encoding="utf-8")
        binary = work / "ple_win_iocp_test.exe"
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-D_CRT_SECURE_NO_WARNINGS", "-I", str(work),
                   str(root / "tools/ple_win_iocp_test.cpp"), "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (work / "compile.stdout").write_text(compiled.stdout, encoding="utf-8")
        (work / "compile.stderr").write_text(compiled.stderr, encoding="utf-8")
        compiled.check_returncode()
        rows = []
        for mode in ["positive", "run-null", "run-foreign", "run-unaligned",
                     "split-null", "split-foreign", "split-unaligned", "run-short", "split-short"]:
            fixture = work / (mode + ".bin")
            start_time = time.perf_counter()
            try:
                result = subprocess.run([str(binary), mode, str(fixture)], capture_output=True,
                                        text=True, timeout=45)
                diagnostic = "read completion failed" if "short" in mode else "completion outside active batch"
                passed = result.returncode == 0 if mode == "positive" else (
                    result.returncode != 0 and diagnostic in result.stderr)
                row = dict(mode=mode, passed=passed, exit_code=result.returncode,
                           stdout=result.stdout, stderr=result.stderr)
            except subprocess.TimeoutExpired:
                row = dict(mode=mode, passed=False, error="timeout")
            finally:
                fixture.unlink(missing_ok=True)
            row["seconds"] = time.perf_counter() - start_time
            rows.append(row)
            print(("PASS" if row["passed"] else "FAIL") + ": " + mode, flush=True)
        summary = dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                       command=command, passed=all(r["passed"] for r in rows), cases=rows)
        (work / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0 if summary["passed"] else 1
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
