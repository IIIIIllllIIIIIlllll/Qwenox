#!/usr/bin/env python3
"""Exercise recoverable VisionTower allocation failures without a main model."""

import argparse
import array
import os
from pathlib import Path
import subprocess
import tempfile


def write_patches(path):
    rows, cols = 4, 1536
    values = array.array("f", (((i * 17) % 101 - 50) / 50.0 for i in range(rows * cols)))
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 1536), }"
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    with path.open("wb") as file:
        file.write(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode())
        values.tofile(file)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="build/qwenox-engine-win.exe")
    parser.add_argument("--vision-tower", default="models/qwen38-flash-next-vision.hgn")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    engine = Path(args.engine)
    tower = Path(args.vision_tower)
    if not engine.is_absolute():
        engine = root / engine
    if not tower.is_absolute():
        tower = root / tower
    build = root / "build"
    build.mkdir(exist_ok=True)
    failures = []
    with tempfile.TemporaryDirectory(prefix="vision-oom-", dir=build) as directory:
        work = Path(directory)
        patches = work / "patches.npy"
        write_patches(patches)

        def run(index):
            env = os.environ.copy()
            env.pop("QWENOX_VISION_EMBED_OOM", None)
            env.pop("QWENOX_VISION_OOM_AT", None)
            if index:
                env["QWENOX_VISION_OOM_AT"] = str(index)
            command = [
                str(engine), "-", "--vision-test", str(work / f"case-{index}"),
                "--vision-tower", str(tower), "--patches", str(patches),
                "--mrope-grid", "1,2,2",
            ]
            result = subprocess.run(command, cwd=root, env=env, capture_output=True,
                                    text=True, timeout=180)
            return result.returncode, result.stdout + result.stderr

        rc, log = run(0)
        if rc != 0 or "vision-test: P=4 M=1" not in log:
            failures.append(f"baseline failed (rc={rc})\n{log[-2000:]}")
        else:
            print("PASS baseline forward")
        for index in range(1, 17):
            rc, log = run(index)
            expected = rc == 1 and "injected allocation failure" in log and \
                       "vision allocation d_" in log and "CRASH fatal" not in log
            if not expected:
                failures.append(f"allocation {index} was not recoverable (rc={rc})\n{log[-2000:]}")
            else:
                print(f"PASS injected allocation {index}/16")
    if failures:
        print("\n\n".join(failures))
        return 1
    print("ALL PASS (VisionTower allocation recovery)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
