#!/usr/bin/env python3
"""Ordinary published SSD-prefix A/B against an already-running engine.

No engine is started or stopped. Use an isolated engine/cache and no other
clients. Run save, restart the engine with the same cache, then run and compare.
The live continuation is the reference, not a fresh full prefill.
"""
import argparse
import json
import math
import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAVED = re.compile(r'kvsnap: saved (\d+) tokens -> (\S+)')
RESTORED = re.compile(r'kvsnap: restored (\d+) tokens from (\S+)')
RAM = re.compile(r'rckpt: restored (\d+) tokens')


def exchange(port, command, timeout):
    with socket.create_connection(('127.0.0.1', port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall((command + '\n').encode())
        with sock.makefile('rb') as f:
            while True:
                line = f.readline()
                if not line:
                    raise RuntimeError('engine disconnected before response')
                yield line.decode('utf-8', errors='strict').strip()


def sync(port, timeout):
    lines = exchange(port, 'KVSYNC', timeout)
    try:
        parts = next(lines).split()
    finally:
        lines.close()
    if len(parts) != 5 or parts[0] != 'K' or any(not x.isdecimal() for x in parts[1:]):
        raise RuntimeError('invalid KVSYNC response: ' + ' '.join(parts))
    return [int(x) for x in parts[1:]]


def gen(ids, n, req, port, timeout):
    command = ' '.join(map(str, ['GEN', req, n, 0, len(ids), *ids, 0]))
    tokens = []
    for line in exchange(port, command, timeout):
        parts = line.split()
        if not parts:
            continue
        if parts[0] in ('T', 'D'):
            if len(parts) < 3 or parts[1] != str(req):
                raise RuntimeError('invalid/mismatched response: ' + line)
            if parts[0] == 'T':
                if len(parts) != 3:
                    raise RuntimeError('invalid token response: ' + line)
                token = int(parts[2])
                if token < 0 or len(tokens) >= n:
                    raise RuntimeError('invalid or excess token: ' + line)
                tokens.append(token)
            else:
                if (len(parts) < 12 or parts[2] != 'length' or len(tokens) != n
                        or int(parts[3]) != len(ids) or int(parts[4]) != len(tokens)):
                    raise RuntimeError(f'unexpected finish: {line}; received {len(tokens)} tokens')
                cached = int(parts[10])
                if not 0 <= cached <= len(ids):
                    raise RuntimeError('invalid cached length: ' + line)
                return dict(tokens=tokens, reason=parts[2], cached=cached, done=line)
        elif parts[0] in ('E', 'ERR', 'ERROR'):
            raise RuntimeError('engine error: ' + line)
    raise RuntimeError('missing completion')


class LogWindow:
    def __init__(self, path):
        self.path = path
        with path.open('rb') as f:
            st = fstat(f)
            self.identity = (st.st_dev, st.st_ino)
            self.start = st.st_size
            if self.start:
                f.seek(self.start - 1)
                if f.read(1) != b'\n':
                    raise RuntimeError('log does not end at a line boundary')

    def read(self):
        with self.path.open('rb') as f:
            st = fstat(f)
            if (st.st_dev, st.st_ino) != self.identity or st.st_size < self.start:
                raise RuntimeError('log rotated or truncated')
            f.seek(self.start)
            data = f.read(st.st_size - self.start)
            if len(data) != st.st_size - self.start or (data and not data.endswith(b'\n')):
                raise RuntimeError('incomplete log window')
            end = self.path.stat()
            if (end.st_dev, end.st_ino) != self.identity or end.st_size < st.st_size:
                raise RuntimeError('log rotated or truncated during read')
        return data.decode('utf-8', errors='replace').splitlines()


def fstat(f):
    import os
    return os.fstat(f.fileno())


def entries(lines, pattern):
    return [dict(length=int(m.group(1)), name=m.group(2), line=line)
            for line in lines if (m := pattern.search(line))]


def continuation_evidence(result, lines):
    ssd = entries(lines, RESTORED)
    ram = [(int(m.group(1)), line) for line in lines if (m := RAM.search(line))]
    if len(ssd) + len(ram) > 1:
        raise RuntimeError('ambiguous restore logs; use an isolated engine')
    if ssd:
        tier, length = 'ssd', ssd[0]['length']
    elif ram:
        tier, length = 'ram', ram[0][0]
    else:
        tier, length = ('live', result['cached']) if result['cached'] else ('none', 0)
    if length != result['cached']:
        raise RuntimeError('restore log length disagrees with D cached length')
    return dict(tier=tier, length=length, ssd=ssd, ram=ram)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('cmd', choices=['save', 'run', 'compare'])
    ap.add_argument('--n', type=int, default=2048, help='data/qsa-oracle/<n>.json prompt')
    ap.add_argument('--tag', default='restore')
    ap.add_argument('--state', type=Path, required=True, help='new state file for save; existing file for run/compare')
    ap.add_argument('--log', type=Path, help='current engine log, required for save/run')
    ap.add_argument('--port', type=int, default=8732)
    ap.add_argument('--timeout', type=float, default=180)
    ap.add_argument('--save-tokens', type=int, default=16)
    ap.add_argument('--run-tokens', type=int, default=32)
    a = ap.parse_args()
    if not 1 <= a.port <= 65535 or not math.isfinite(a.timeout) or a.timeout <= 0:
        ap.error('invalid port or timeout')
    if not 1 <= a.save_tokens <= 64 or not 1 <= a.run_tokens <= 64:
        ap.error('token limits must be 1..64')
    if a.cmd != 'compare' and a.log is None:
        ap.error('--log is required for save/run')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.tag) or a.tag == 'live':
        ap.error('tag must contain only letters, digits, underscore or hyphen and not be live')

    if a.cmd == 'save':
        # Reserve the name before sending requests; never replace an existing file.
        with a.state.open('x', encoding='utf-8') as state_file:
            from qwentok import Tokenizer
            p = json.loads((ROOT / 'data/qsa-oracle' / f'{a.n}.json').read_text())['prompt_ids']
            if not p or any(type(x) is not int or x < 0 for x in p):
                raise ValueError('invalid oracle prompt')
            sync(a.port, a.timeout)
            window = LogWindow(a.log)
            seed = gen(p, a.save_tokens, 1, a.port, a.timeout)
            if seed['cached'] != 0:
                raise RuntimeError('seed was cached; use an isolated cold cache')
            sync_seed = sync(a.port, a.timeout)  # D precedes the asynchronous publication
            saved = entries(window.read(), SAVED)
            published = [e for e in saved if len(p) <= e['length'] <= len(p) + len(seed['tokens'])]
            if not published:
                raise RuntimeError('no published ordinary checkpoint covering the seed prompt')
            q = p + seed['tokens'] + Tokenizer().encode('\n\nContinue the text above.\n')
            window = LogWindow(a.log)
            live = gen(q, a.run_tokens, 2, a.port, a.timeout)
            sync_live = sync(a.port, a.timeout)
            live['cache'] = continuation_evidence(live, window.read())
            if live['cache']['tier'] not in ('live', 'ram') or not live['cached']:
                raise RuntimeError('baseline is not an ordinary live/RAM continuation')
            st = dict(version=1, n=a.n, Q=q, seed=seed, published=published,
                      sync_seed=sync_seed, sync_live=sync_live, run_tokens=a.run_tokens,
                      runs={'live': live})
            json.dump(st, state_file, ensure_ascii=False, indent=2)
            state_file.write('\n')
        print(json.dumps(dict(prompt=len(p), continuation=len(q), published=published,
                              live=live), ensure_ascii=False))
        print('Restart the engine with the same SSD cache before run; use its new log.')
    else:
        st = json.loads(a.state.read_text(encoding='utf-8'))
        if st.get('version') != 1 or 'live' not in st.get('runs', {}):
            raise ValueError('invalid or incomplete state')
        if a.cmd == 'run':
            sidecar = a.state.with_name(a.state.name + '.' + a.tag + '.json')
            if a.tag in st['runs'] or sidecar.exists():
                raise ValueError('tag already exists; refusing to overwrite')
            sync(a.port, a.timeout)
            window = LogWindow(a.log)
            result = gen(st['Q'], st['run_tokens'], 3, a.port, a.timeout)
            sync(a.port, a.timeout)
            result['cache'] = continuation_evidence(result, window.read())
            print(json.dumps(dict(tag=a.tag, observed=result), ensure_ascii=False), flush=True)
            hits = result['cache']['ssd']
            published = {(x['name'], x['length']) for x in st['published']}
            if (result['cache']['tier'] != 'ssd' or len(hits) != 1 or
                    (hits[0]['name'], hits[0]['length']) not in published):
                raise RuntimeError('no restore of a checkpoint published by the seed request')
            if result['tokens'] != st['runs']['live']['tokens'] or result['reason'] != st['runs']['live']['reason']:
                raise RuntimeError('SSD continuation differs from live baseline')
            st['runs'][a.tag] = result
            # Exclusive sidecar retains the original baseline and never overwrites prior data.
            with sidecar.open('x', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
                f.write('\n')
            print(json.dumps(dict(tag=a.tag, result=result, sidecar=str(sidecar)), ensure_ascii=False))
        else:
            sidecars = sorted(p for p in a.state.parent.iterdir()
                              if p.name.startswith(a.state.name + '.') and p.name.endswith('.json'))
            if not sidecars:
                raise RuntimeError('no restore results to compare')
            gold = st['runs']['live']
            for path in sidecars:
                r = json.loads(path.read_text(encoding='utf-8'))
                hits = r.get('cache', {}).get('ssd', [])
                published = {(x['name'], x['length']) for x in st['published']}
                ok = (r.get('tokens') == gold['tokens'] and r.get('reason') == gold['reason']
                      and r.get('cache', {}).get('tier') == 'ssd' and len(hits) == 1
                      and (hits[0]['name'], hits[0]['length']) in published
                      and r.get('cached') == hits[0]['length'])
                print(path, 'MATCH' if ok else 'FAIL')
                if not ok:
                    raise RuntimeError('comparison failed')
            print('kvsnap_ab ALL MATCH')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        print('kvsnap_ab FAIL:', exc, file=sys.stderr)
        sys.exit(1)
