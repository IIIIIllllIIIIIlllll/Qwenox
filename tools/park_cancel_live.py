#!/usr/bin/env python3
"""Observe and cancel a KV-pressure suspend on an already-running engine.

No service is started. Run only when the candidate engine is otherwise idle.
The required log must be the engine's live stderr log, visible from WSL.
Exit status: 0 PASS, 1 FAIL, 2 INCONCLUSIVE/setup/timeout.
"""
import argparse
import os
from pathlib import Path
import secrets
import selectors
import socket
import sys
import time

from qwentok import Tokenizer

ROOT = Path(__file__).resolve().parent.parent


def prompt(tk, label, nonce):
    prefix = tk.encode(f'<|im_start|>user\n[{label} {nonce}]\n')
    paragraph = (('A field researcher describes coastal weather, tidal pools, and seabirds. '
                  'Summarize the observations accurately and briefly.\n') if label == 'earlier' else
                 ('An engineer documents a railway bridge, its materials, and maintenance. '
                  'Summarize the observations accurately and briefly.\n'))
    body = tk.encode(paragraph)
    suffix = tk.encode('\nGive a concise summary.<|im_end|>\n<|im_start|>assistant\n')
    remaining = 3839 - len(prefix) - len(suffix)
    if not body or remaining <= 0:
        raise ValueError('tokenizer produced an invalid prompt')
    return prefix + (body * (remaining // len(body) + 1))[:remaining] + suffix


class LogTail:
    def __init__(self, path):
        self.path = path
        self.file = path.open('rb')
        st = os.fstat(self.file.fileno())
        self.identity = (st.st_dev, st.st_ino)
        self.file.seek(0, os.SEEK_END)
        self.offset = self.file.tell()
        # Ignore an existing incomplete line, rather than treating it as new.
        self.skip = False
        if self.offset:
            self.file.seek(-1, os.SEEK_END)
            self.skip = self.file.read(1) != b'\n'
        self.pending = b''

    def read(self):
        st = self.path.stat()
        if (st.st_dev, st.st_ino) != self.identity or st.st_size < self.offset:
            raise RuntimeError('engine log rotated or truncated; observation invalid')
        data = self.file.read()
        self.offset += len(data)
        self.pending += data
        lines = self.pending.split(b'\n')
        self.pending = lines.pop()
        if self.skip and lines:
            lines.pop(0)
            self.skip = False
        return [line.decode('utf-8', errors='replace').rstrip('\r') for line in lines]

    def close(self):
        self.file.close()


class Request:
    def __init__(self, sel, port, req, ids, n, deadline, drafter):
        self.req, self.tokens, self.done, self.buf = req, 0, None, b''
        self.sock = socket.create_connection(('127.0.0.1', port),
                                             timeout=max(0.01, deadline - time.monotonic()))
        try:
            wire = ['GEN', req, n, 0, len(ids), *ids, drafter]  # no EOS
            self.sock.sendall((' '.join(map(str, wire)) + '\n').encode())
            self.sock.setblocking(False)
            sel.register(self.sock, selectors.EVENT_READ, self)
        except BaseException:
            self.sock.close()
            raise

    def receive(self, sel):
        data = self.sock.recv(65536)
        if not data:
            raise RuntimeError(f'req {self.req}: disconnected before D')
        self.buf += data
        lines = self.buf.split(b'\n')
        self.buf = lines.pop()
        for raw in lines:
            fields = raw.decode('utf-8', errors='strict').split()
            if not fields:
                continue
            if fields[0] in ('T', 'D', 'W'):
                if len(fields) < 3 or fields[1] != str(self.req):
                    raise RuntimeError(f'req {self.req}: malformed or mismatched line {raw!r}')
                if fields[0] == 'T':
                    int(fields[2])
                    self.tokens += 1
                elif fields[0] == 'D':
                    self.done = fields[2]
                    print(f'req {self.req}: {raw.decode()} (received {self.tokens} tokens)', flush=True)
                    sel.unregister(self.sock)
                    break
            else:
                raise RuntimeError(f'req {self.req}: unexpected line {raw!r}')

    def close(self):
        self.sock.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--log', type=Path, required=True, help='WSL path to the candidate Windows engine stderr log')
    ap.add_argument('--tokenizer-dir', default=os.environ.get('TOK_DIR', str(ROOT / 'models' / 'tokenizer')),
                    help='WSL model tokenizer directory containing tokenizer.json (or TOK_DIR)')
    ap.add_argument('--port', type=int, default=18730)
    ap.add_argument('--drafter', type=int, choices=(0, 1, 4), default=0,
                    help='drafter: 0 serial (default), 1 MTP, 4 chain')
    ap.add_argument('--timeout', type=float, default=60, help='overall network/observation deadline in seconds (default 60)')
    ap.add_argument('--output-tokens', type=int, default=320, help='output limit for each main request (default 320)')
    args = ap.parse_args()
    if not 1 <= args.port <= 65535 or not 0 < args.timeout <= 120 or not 1 <= args.output_tokens <= 512:
        ap.error('port must be valid, timeout in (0,120], output-tokens in [1,512]')

    requests = []
    log = None
    sel = selectors.DefaultSelector()
    try:
        tk = Tokenizer(str(Path(args.tokenizer_dir).expanduser()))
        nonce = secrets.token_hex(8)
        a_ids = prompt(tk, 'earlier', nonce)
        b_ids = prompt(tk, 'later', nonce)
        if a_ids == b_ids:
            raise ValueError('prompts are identical')
        # Positive, distinct 63-bit request IDs correlate log lines, not connection IDs.
        base = secrets.randbits(60) + 1
        a_id, b_id, c_id = base, base + 1, base + 2
        log = LogTail(args.log)
        deadline = time.monotonic() + args.timeout
        print(f'log={args.log} start_offset={log.offset}; prompt lengths={len(a_ids)},{len(b_ids)}; '
              f'drafter={args.drafter}; req earlier={a_id} later={b_id}', flush=True)
        a = Request(sel, args.port, a_id, a_ids, args.output_tokens, deadline, args.drafter)
        requests.append(a)
        b = Request(sel, args.port, b_id, b_ids, args.output_tokens, deadline, args.drafter)
        requests.append(b)
        release = f'req {b_id}: KV released; waiting to resume'
        resume = f'req {b_id}: resumed from KV-pressure snapshot'
        saw_release = saw_resume = sent_x = False
        c = None
        while time.monotonic() < deadline:
            # Read all complete new log lines first, so a release+resume in one
            # batch cannot be mistaken for an observed cancellation window.
            for line in log.read():
                if line.endswith(release):
                    saw_release = True
                    print(f'LOG {line}', flush=True)
                if line.endswith(resume):
                    saw_resume = True
                    print(f'LOG {line}', flush=True)
            # Drain already-readable D lines before attempting X.
            # Windows select() rejects empty socket sets after all requests
            # complete, including between the main and follow-up requests.
            if sel.get_map():
                for key, _ in sel.select(timeout=0):
                    key.data.receive(sel)
            if saw_release and not saw_resume and not sent_x and b.done is None:
                try:
                    b.sock.sendall(f'X {b_id}\n'.encode())
                    sent_x = True
                    print(f'SENT X {b_id}', flush=True)
                except (BlockingIOError, OSError) as exc:
                    raise RuntimeError(f'could not send X: {exc}') from exc
            if a.done is not None and b.done is not None and c is None:
                c_ids = tk.encode(f'<|im_start|>user\n[after {nonce}] Say hello.\n'
                                  '<|im_end|>\n<|im_start|>assistant\n')
                c = Request(sel, args.port, c_id, c_ids, 8, deadline, args.drafter)
                requests.append(c)
            if c is not None and c.done is not None:
                # Capture any resume already written before reporting.
                for line in log.read():
                    if line.endswith(resume):
                        saw_resume = True
                        print(f'LOG {line}', flush=True)
                normal = lambda r: r.done in ('length', 'done') and r.tokens > 0
                if not normal(a) or not normal(c) or b.done not in ('cancel', 'length', 'done'):
                    print('FAIL: unexpected completion or no output on earlier/follow-up request')
                    return 1
                if not saw_release or not sent_x or saw_resume or b.done != 'cancel':
                    print('INCONCLUSIVE: suspend not observed, resumed too quickly, or X lost the completion race; '
                          f'release={saw_release} resume={saw_resume} X={sent_x} later={b.done}')
                    return 2
                print('PASS: release observed, X sent, later D cancel, earlier and follow-up completed; '
                      'log observation is not an atomic proof of state at X')
                return 0
            wait = min(0.01, max(0, deadline - time.monotonic()))
            if sel.get_map():
                for key, _ in sel.select(timeout=wait):
                    key.data.receive(sel)
            else:
                time.sleep(wait)
        print(f'INCONCLUSIVE: deadline expired; release={saw_release} resume={saw_resume} X={sent_x}')
        return 2
    except (OSError, ValueError, RuntimeError) as exc:
        print(f'INCONCLUSIVE: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
    finally:
        for r in requests:
            r.close()
        sel.close()
        if log is not None:
            log.close()


if __name__ == '__main__':
    sys.exit(main())
