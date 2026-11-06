#!/usr/bin/env python3
"""Compile and exercise the production host-side KV page accounting."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    args = parser.parse_args()
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler and os.name == "nt":
        candidate = Path("C:/Program Files/LLVM/bin/clang++.exe")
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")

    root = Path(__file__).resolve().parent.parent
    source = (root / "src/gpu/parts/10_ple_io.inc").read_text(encoding="utf-8")
    start = source.index("struct KvPagePool {")
    end = source.index("// A3: physical pool size in pages.", start)
    build = root / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kv-page-", dir=build) as directory:
        work = Path(directory)
        (work / "kv_page_alloc.inc").write_text(source[start:end], encoding="utf-8")
        binary = work / ("kv_page_test.exe" if os.name == "nt" else "kv_page_test")
        subprocess.run([compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-D_CRT_SECURE_NO_WARNINGS", "-I", str(work),
                        str(root / "tools/kv_page_test.cpp"), "-o", str(binary)],
                       check=True, timeout=120)
        result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=30)
        print(result.stdout, end="", flush=True)
        if result.returncode:
            print(result.stderr, end="", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
