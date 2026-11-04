#!/usr/bin/env python3
"""CPU regression of production prefill_batch cancellation across slot handoffs.

GPU/chunk operations are host stubs; the production loop and cancellation checks
are extracted unchanged. This verifies control flow, not numerical inference.
--source supports reproducing the old shared-flag behavior before the fix.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")
    root = Path(__file__).resolve().parent.parent
    source = args.source or root / "src/gpu/parts/40_model.inc"
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    start = text.index("  int prefill_batch(")
    end = text.index("  int prefill_chunk(", start)
    method = text[start:end]
    call_local = "const std::atomic<bool>* cancel" in method
    temporary = None
    if args.output:
        work = args.output.resolve()
        work.mkdir(parents=True, exist_ok=False)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="prefill-cancel-")
        work = Path(temporary.name)
    try:
        (work / "prefill_batch_under_test.inc").write_text(method, encoding="utf-8")
        binary = work / ("prefill_cancel_test.exe" if os.name == "nt" else "prefill_cancel_test")
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-D_CRT_SECURE_NO_WARNINGS",
                   "-DCALL_LOCAL_CANCEL=" + str(int(call_local)), "-I", str(work),
                   str(root / "tools/prefill_cancel_test.cpp"), "-o", str(binary)]
        compiled = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (work / "compile.stdout").write_text(compiled.stdout, encoding="utf-8")
        (work / "compile.stderr").write_text(compiled.stderr, encoding="utf-8")
        compiled.check_returncode()
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        print(result.stdout, end="", flush=True)
        summary = dict(source=str(source), source_sha256=hashlib.sha256(raw).hexdigest(),
                       command=command, passed=result.returncode == 0, exit_code=result.returncode,
                       stdout=result.stdout, stderr=result.stderr)
        (work / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return result.returncode
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
