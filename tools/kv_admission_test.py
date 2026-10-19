#!/usr/bin/env python3
"""Compile and exercise the standalone KV admission arithmetic."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--tsan", action="store_true",
                        help="kept for compatibility; no threading is tested")
    args = parser.parse_args()
    if args.tsan:
        parser.error("--tsan is not applicable to the standalone arithmetic test")
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler and os.name == "nt":
        candidate = Path("C:/Program Files/LLVM/bin/clang++.exe")
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")

    root = Path(__file__).resolve().parent.parent
    build = root / "build"
    build.mkdir(exist_ok=True)
    binary = build / ("kv_admission_test.exe" if os.name == "nt" else "kv_admission_test")
    command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
               "-D_CRT_SECURE_NO_WARNINGS", str(root / "tools/kv_admission_test.cpp"),
               "-o", str(binary)]
    if os.name != "nt":
        command.append("-pthread")
    subprocess.run(command, check=True, timeout=120)
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=60)
    print(result.stdout, end="", flush=True)
    if result.returncode:
        print(result.stderr, end="", flush=True)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
