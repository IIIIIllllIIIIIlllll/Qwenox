#!/usr/bin/env python3
"""Compile and exercise the production admission code with a host-only model."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", default=os.environ.get("CXX"))
    parser.add_argument("--tsan", action="store_true")
    args = parser.parse_args()
    compiler = args.cxx or shutil.which("clang++") or shutil.which("g++")
    if not compiler and os.name == "nt":
        candidate = Path("C:/Program Files/LLVM/bin/clang++.exe")
        if candidate.is_file():
            compiler = str(candidate)
    if not compiler:
        parser.error("a C++17 compiler is required; pass --cxx PATH")
    if args.tsan and os.name == "nt":
        parser.error("--tsan requires a compiler with ThreadSanitizer on Linux")

    root = Path(__file__).resolve().parent.parent
    source = (root / "src/gpu/parts/51_host_cfg.inc").read_text(encoding="utf-8")
    start = source.index("struct Turnstile {")
    end = source.index("// 10-13 prefill scheduling knobs", start)
    build = root / "build"
    build.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kv-admission-", dir=build) as directory:
        work = Path(directory).resolve()
        work.relative_to(build.resolve())
        starve = source.index("static bool starve(GpuModel& m)", end)
        conn = source.index("struct Conn {", starve)
        serve = source[start:end] + source[starve:conn]
        # Deterministically schedule a peer in the real turn-release window,
        # before self-suspension reacquires the turn and revalidates its slot.
        marker = "  g_turn.release();\n  auto reacquire_original = [&] {"
        if serve.count(marker) != 1:
            raise RuntimeError("self-suspension turn-release hook changed")
        serve = serve.replace(marker, "  g_turn.release();\n"
                              "  if (self && on_self_turn_released) on_self_turn_released();\n"
                              "  auto reacquire_original = [&] {")
        (work / "kv_admission_serve.inc").write_text(serve, encoding="utf-8")
        tier_start = source.index("// Drop a match() that is not restored")
        tier_end = source.index("// ---- restore", tier_start)
        (work / "kv_admission_tier.inc").write_text(source[tier_start:tier_end], encoding="utf-8")
        binary = work / ("kv_admission_test.exe" if os.name == "nt" else "kv_admission_test")
        command = [compiler, "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                   "-D_CRT_SECURE_NO_WARNINGS", "-I", str(work),
                   str(root / "tools/kv_admission_test.cpp"), "-o", str(binary)]
        if os.name != "nt":
            command.append("-pthread")
        if args.tsan:
            command.extend(["-fsanitize=thread", "-g", "-O1"])
        subprocess.run(command, check=True, timeout=120)
        env = os.environ.copy()
        env["QWENOX_KVSNAP_OVER_RAM"] = "1"
        result = subprocess.run([str(binary)], timeout=60, env=env,
                                stderr=subprocess.PIPE, text=True)
        sys.stderr.write(result.stderr)
        if result.returncode:
            return result.returncode
        expected = [
            "req 876: KV-pressure recall candidate saved=4 SSD_prefix=0 SSD_mtp=0 boundary=0",
            "req 877: KV-pressure recall candidate saved=4 SSD_prefix=3 SSD_mtp=1 boundary=0",
            "req 878: KV-pressure recall candidate saved=4 SSD_prefix=4 SSD_mtp=1 boundary=1",
            "[kvpage] host shared slot=1 mapped=3 reclaimable=1 free_before=3 free_after=4",
            "[kvpage] host last reference slot=2 mapped=2 reclaimable=2 free_before=4 free_after=6",
            "[kvpage] host shared reference slot=1 mapped=1 reclaimable=0 free_before=5 free_after=5",
            "[kvpage] host final reference slot=2 mapped=1 reclaimable=1 free_before=5 free_after=6",
            "[kvpage] host prediction failure slot=0 mapped=4 reclaimable=-1 free_before=6 free_after=10",
        ]
        for line in expected:
            if result.stderr.count(line) != 1:
                raise RuntimeError(f"missing or repeated telemetry: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
