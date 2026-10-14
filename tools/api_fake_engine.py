#!/usr/bin/env python3
"""Deterministic line-protocol engine used by api_regression_test.py.

Run this on port 18730, start qwenox-api with --engine 127.0.0.1:18730 and
--port 18731, then run api_regression_test.py against that API port.
"""
import argparse
import json
import select
import socket
import struct
import threading
import time


def send_line(conn, line):
    conn.sendall((line + "\n").encode("ascii"))


SLOTS = 1
POOL = 0
TRACE = []
TRACE_LOCK = threading.Lock()


def handle(conn):
    pending = b""

    def record_cancel(req):
        with TRACE_LOCK:
            for entry in TRACE:
                if entry["req"] == req:
                    entry["cancel"] += 1

    def cancellable(req, seed):
        nonlocal pending
        entry = {"req": req, "seed": seed, "cancel": 0, "tokens": 0, "done": False}
        with TRACE_LOCK:
            TRACE.append(entry)
        deadline = time.monotonic() + (2.2 if seed == 616163 else 6.0)
        cancelled = False
        while time.monotonic() < deadline:
            if seed != 616162:
                conn.sendall((f"T {req} 12675 -0.125\n" * 32).encode("ascii"))
                with TRACE_LOCK:
                    entry["tokens"] += 32
            if select.select([conn], [], [], 0.01)[0]:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                pending += chunk
            while b"\n" in pending:
                command, pending = pending.split(b"\n", 1)
                if command == f"X {req}".encode("ascii"):
                    record_cancel(req)
                    if not cancelled:
                        cancelled = True
                        deadline = time.monotonic() + 1.2
        send_line(conn, f"T {req} 12675 -0.125")
        reason = "cancel" if cancelled else "done"
        send_line(conn, f"D {req} {reason} 0 0 0.0 0.0 0 0 0 0 0")
        with TRACE_LOCK:
            entry["done"] = True

    def read_exact(size):
        nonlocal pending
        while len(pending) < size:
            chunk = conn.recv(min(65536, size - len(pending)))
            if not chunk:
                return None
            pending += chunk
        result, pending = pending[:size], pending[size:]
        return result

    with conn:
        while True:
            while b"\n" not in pending:
                chunk = conn.recv(65536)
                if not chunk:
                    return
                pending += chunk
            raw, pending = pending.split(b"\n", 1)
            line = raw.decode("ascii", "strict")
            if line == "PING":
                send_line(conn, "PONG")
                continue
            if line == "INFO":
                pool = f" {POOL}" if POOL > 0 else ""
                send_line(conn, f"I 1 0 262144 8 1 1 0 0 0 {SLOTS} 262144 0 1{pool}")
                continue
            if line == "MEM":
                send_line(conn, "M 1 100 120 200 10 20 310 900 1000 100 400 50")
                continue
            if line == "CSTAT":
                send_line(conn, "C fake")
                continue
            if line == "TESTSTATE":
                with TRACE_LOCK:
                    send_line(conn, json.dumps(TRACE))
                continue
            if line.startswith("X "):
                record_cancel(int(line.split()[1]))
                continue
            if not line.startswith("GEN "):
                continue

            fields = line.split()
            req = int(fields[1])
            n_eos = int(fields[3])
            n_ids_at = 4 + n_eos
            n_ids = int(fields[n_ids_at])
            ids = [int(v) for v in fields[n_ids_at + 1:n_ids_at + 1 + n_ids]]

            grids = []
            if "MROPE" in fields:
                at = fields.index("MROPE")
                count = int(fields[at + 1])
                values = [int(value) for value in fields[at + 2:at + 2 + count * 3]]
                if len(values) != count * 3:
                    return
                grids = [tuple(values[index:index + 3])
                         for index in range(0, len(values), 3)]

            frame_ok = True
            if "VIMG" in fields:
                at = fields.index("VIMG")
                count = int(fields[at + 1])
                frame_ok = count == len(grids) and count <= 8
                for index in range(count):
                    header = read_exact(16)
                    if header is None:
                        return
                    magic, patches, nbytes = struct.unpack("<IIQ", header)
                    if nbytes > 512 * 1024 * 1024:
                        return
                    payload = read_exact(nbytes)
                    if payload is None:
                        return
                    expected = grids[index][0] * grids[index][1] * grids[index][2]
                    frame_ok = (frame_ok and magic == 0x56494D31 and
                                patches == expected and nbytes == patches * 1536 * 4)
            elif grids:
                frame_ok = False
            if not frame_ok:
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue

            seed = int(fields[fields.index("SAMPLE") + 5]) if "SAMPLE" in fields else 0
            if seed in (616161, 616162, 616163):
                cancellable(req, seed)
                continue

            if "424242" in fields:
                return
            if "31337" in fields:
                time.sleep(2)
            if "5150" in fields:  # aborted mid-decode (shared KV pool full)
                send_line(conn, f"T {req} 12675 -0.125")
                send_line(conn, f"D {req} error 0 0 0.0 0.0 0 0 0 0 0")
                continue
            if "9001" in fields:
                tokens = [
                    248069, 271, 248058, 198, 27, 1628, 27362, 67017, 29, 198,
                    27, 15704, 28, 8656, 29, 198, 98116, 198, 510, 15704, 29,
                    198, 27, 15704, 28, 13382, 29, 198, 18, 198, 510, 15704,
                    29, 198, 510, 1628, 29, 198, 248059,
                ]
                tokens = [(token, -0.1) for token in tokens]
            elif "9002" in fields:
                tokens = [27, 15704, 28, 41843, 29, 198, 37186, 73594, 29086,
                          198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059]
                tokens = [(token, -0.1) for token in tokens]
            elif "9003" in fields:
                tokens = [447, 67017, 29, 198, 27, 15704, 28, 8656, 29, 198,
                          98817, 198, 510, 15704, 29, 198, 510, 1628, 29, 198,
                          248059]
                tokens = [(token, -0.1) for token in tokens]
            else:
                # Think-resume scenarios. The tool template embeds </think>
                # (token 248069) in every prompt's tool instructions, so the
                # continuation is identified by its TAIL: the API appends the
                # closer ids [198, 248069, 271] to the prompt it resumes from,
                # and the post-close variant (9006) continues from a prompt
                # ending in its generated [248069, 271]. First runs end with
                # the template's generation prompt (the <think> opener),
                # never with the closer.
                tail3 = ids[-3:] if len(ids) >= 3 else []
                tail2 = ids[-2:] if len(ids) >= 2 else []
                # A continuation prompt ends with either the synthesized
                # closer ids [198, 248069, 271], the generated closer pair
                # [248069, 271] (9006), or 9008's generated [248069, 220]
                # (closer + space).
                resumed_prompt = (tail3 == [198, 248069, 271] or
                                  tail2 in ([248069, 271], [248069, 220]))
                think_only = [(198, -0.1), (27, -0.1), (1628, -0.1),
                              (27362, -0.1), (67017, -0.1), (29, -0.1)]
                if resumed_prompt and " 9005 " in line:
                    # 9005's FIRST resumed run (the un-rotated client seed
                    # 9005 still sits in the SAMPLE field): EOS without
                    # emitting any token. The API must retry the identical
                    # continuation with a rotated seed; the retried run
                    # misses this exact-seed branch and lands in the
                    # resumed_prompt branch below.
                    tokens = []
                elif resumed_prompt:
                    # Any other think-resume continuation: a complete answer.
                    tokens = [(12675, -0.125)]  # "Hi"
                elif "9004" in fields:
                    # Think-only EOS: the reply dies inside the unclosed
                    # <think> block (no </think>, no content, no tool call).
                    # Text: "\n<function=get_weather>" — plausible thinking
                    # that never closes the think block. The API must
                    # synthesize "\n</think>\n\n" and resume.
                    tokens = think_only
                elif "9005" in fields:
                    tokens = think_only
                elif "9006" in fields:
                    # EOS right after a bare </think>: reasoning closes but
                    # the answer never starts. The API must resume WITHOUT
                    # emitting a second closer; the continuation (tail
                    # [248069, 271]) lands in the resumed_prompt branch.
                    tokens = [(248069, -0.1), (271, -0.1)]
                elif "9008" in fields:
                    # </think> followed by a single space, then EOS: the
                    # answer is whitespace-only. split_reasoning only strips
                    # leading newlines, so " " passes as non-empty content
                    # and the whitespace-only reply would escape the resume
                    # detection while clients trim it away. The API must
                    # treat it as an empty answer and resume.
                    tokens = [(248069, -0.1), (220, -0.1)]
                elif "9009" in fields:
                    # A hallucinated tool call: complete <tool_call> block
                    # naming a function that is NOT in the client's tools
                    # list. The old parser entered Broken and silently
                    # swallowed everything after </think>, so clients saw a
                    # reasoning-only turn. The call must be surfaced in the
                    # standard tool_calls shape so the client agent can
                    # reject it with a tool error and the model self-correct.
                    # Text: "</think>\n\n<tool_call>\n<function=ghost_tool>\n<parameter=city>\n北京\n</parameter>\n</function>\n</tool_call>"
                    tokens = [(t, -0.1) for t in [
                        248069, 271, 248058, 198, 27, 1628, 28, 37372, 22076,
                        29, 198, 27, 15704, 28, 8656, 29, 198, 98116, 198,
                        510, 15704, 29, 198, 510, 1628, 29, 198, 248059,
                    ]]
                elif "9007" in fields:
                    # Think-only EOS with tool_choice=required: the API must
                    # NOT resume (the raw text has no answer split to
                    # evaluate), so exactly one generation happens and the
                    # reply is replayed as one think-only tool call.
                    tokens = think_only
                elif n_ids <= 2:
                    tokens = [(4130, -0.5), (284, -0.25)]  # " Par" + "is"
                elif 248069 in ids:
                    tokens = [(12675, -0.125)]  # enable_thinking=false -> "Hi"
                else:
                    tokens = [(248069, -0.75), (12675, -0.125)]  # "</think>Hi"
            for token, logprob in tokens:
                send_line(conn, f"T {req} {token} {logprob}")
            send_line(conn, f"D {req} done {n_ids} {len(tokens)} 1.0 2.0 0 0 0 0 0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18730)
    parser.add_argument("--slots", type=int, default=1, help="INFO kv_slots")
    parser.add_argument("--pool", type=int, default=0,
                        help="INFO kv_pool_tokens; 0 = omit (old engine)")
    args = parser.parse_args()
    global SLOTS, POOL
    SLOTS = args.slots
    POOL = args.pool
    with socket.socket() as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", args.port))
        server.listen(16)
        while True:
            conn, _ = server.accept()
            threading.Thread(target=handle, args=(conn,), daemon=True).start()


if __name__ == "__main__":
    main()
