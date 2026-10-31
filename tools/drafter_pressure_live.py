#!/usr/bin/env python3
"""Compare solo and overlapping requests against an already-running engine.

This program sends real requests; run only after other live tests have finished.
Every invocation creates a new results directory and never edits its inputs.
"""
import argparse
import json
import math
import os
import re
import secrets
import socket
import sys
import tempfile
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPTS = ROOT / 'logs/gpu-protocol-20261029-unique/prompts.json'
DEFAULT_LOG = Path('/mnt/c/Users/dogez/AppData/Local/Temp/qwenox-kv-candidate-2kPoRJWo/logs/95c5d5fc870d4d5581585538b7c527f2/engine.stderr.log')
LOG_PATTERN = re.compile(r'park|suspend|resum|rebuild|small[-_ ]state|waiting|starv|KV released|error|fail|abort|overflow', re.I)
STAT_NAMES = ('actual_drafter', 'rounds', 'committed', 'cached', 'proposed')


def write_json(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')


def request(label, ids, drafter, req_id, args, gate=None):
    result = dict(label=label, request_id=req_id, requested_drafter=drafter,
                  prompt_len=len(ids), max_tokens=args.max_tokens, tokens=[], events=[])
    started = time.monotonic()
    deadline = started + args.timeout
    sock = None
    try:
        sock = socket.create_connection(('127.0.0.1', args.port), timeout=args.timeout)
        if gate is not None:
            gate.wait(timeout=max(0.001, deadline - time.monotonic()))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('request deadline expired before send')
        sock.settimeout(remaining)
        fields = ['GEN', req_id, args.max_tokens, 0, len(ids), *ids, drafter]
        if args.mode == 'sampled':
            fields += ['SAMPLE', '0.8', 20, '0.95', 0, 777, 'SEEDED']
        result['sent_at'] = time.time()
        result['sent_monotonic'] = time.monotonic()
        sock.sendall((' '.join(map(str, fields)) + '\n').encode())
        buf = b''
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('request deadline expired waiting for D')
            sock.settimeout(remaining)
            chunk = sock.recv(65536)
            if not chunk:
                raise EOFError('engine disconnected before D; buffered bytes: ' + repr(buf))
            buf += chunk
            while b'\n' in buf:
                raw, buf = buf.split(b'\n', 1)
                line = raw.decode('utf-8', errors='replace')
                parts = line.split()
                if not parts:
                    result['events'].append(line)
                    continue
                if parts[0] in ('T', 'D'):
                    if len(parts) < 3 or parts[1] != str(req_id):
                        result['events'].append(line)
                        raise ValueError('invalid or mismatched response: ' + line)
                    if parts[0] == 'T':
                        if len(parts) != 3:
                            raise ValueError('invalid token response: ' + line)
                        token = int(parts[2])
                        if token < 0 or len(result['tokens']) >= args.max_tokens:
                            raise ValueError('invalid or excess token: ' + line)
                        result['tokens'].append(token)
                        continue
                    if (len(parts) < 12 or int(parts[3]) != len(ids)
                            or int(parts[4]) != len(result['tokens'])
                            or (parts[2] == 'length' and len(result['tokens']) != args.max_tokens)
                            or not 0 <= int(parts[10]) <= len(ids)):
                        raise ValueError('invalid completion counts: ' + line)
                    result['done'] = line
                    result['reason'] = parts[2]
                    result['stats'] = {name: int(parts[i]) if len(parts) > i else None
                                       for i, name in enumerate(STAT_NAMES, 7)}
                    result['finished_at'] = time.time()
                    result['finished_monotonic'] = time.monotonic()
                    return result
                result['events'].append(line)
    except Exception:
        result['error'] = traceback.format_exc()
        if gate is not None:
            gate.abort()
        return result
    finally:
        if sock is not None:
            sock.close()
        result['elapsed'] = round(time.monotonic() - started, 3)


def compare(baseline, concurrent):
    a, b = baseline['tokens'], concurrent['tokens']
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    tokens_match = a == b
    reasons_match = baseline.get('reason') == concurrent.get('reason')
    success = all(r.get('reason') in ('done', 'length') and r['tokens'] and not r.get('error') and
                  r.get('stats', {}).get('actual_drafter') == r['requested_drafter']
                  for r in (baseline, concurrent))
    return dict(passed=bool(success and tokens_match and reasons_match), tokens_match=tokens_match,
                first_difference=first, baseline_reason=baseline.get('reason'),
                concurrent_reason=concurrent.get('reason'), reasons_match=reasons_match)


def check_pressure(lines, request_ids):
    """Require an ordered suspend cycle for one of the overlapping requests."""
    legacy = (re.compile(r'captured \d+ tokens for KV-pressure suspend$', re.I),
              re.compile(r'KV released; waiting to resume$', re.I),
              re.compile(r'resumed from KV-pressure snapshot$', re.I))
    small_state = (re.compile(r'(?=.*\b(?:captur\w*)\b)(?=.*\bsmall[-_ ]state\b)', re.I),
                   re.compile(r'KV released; waiting to resume$', re.I),
                   re.compile(r'\brebuild\w*\b', re.I),
                   re.compile(r'\bresum(?:e|ed|ing)\b', re.I))
    formats = (('snapshot', legacy, ('capture', 'kv_released', 'resume')),
               ('small-state', small_state, ('capture', 'kv_released', 'rebuild', 'resume')))
    lengths = {name: len(stages) for name, _, stages in formats}
    evidence = {str(req_id): [] for req_id in request_ids}
    active_format = {}
    for line in lines:
        match = re.search(r'\breq(?:uest)?\s+(\d+):\s*(.*)', line.rstrip('\r'), re.I)
        if not match or match.group(1) not in evidence:
            continue
        req_id, message = match.groups()
        events = evidence[req_id]
        for name, patterns, stages in formats:
            if patterns[0].search(message):
                if not events or len(events) < lengths[active_format[req_id]]:
                    events[:] = [dict(stage=stages[0], line=line)]
                    active_format[req_id] = name
                break
            if (active_format.get(req_id) == name and 0 < len(events) < len(stages)
                    and patterns[len(events)].search(message)):
                events.append(dict(stage=stages[len(events)], line=line))
                break
    completed = [req_id for req_id, events in evidence.items()
                 if req_id in active_format and len(events) == lengths[active_format[req_id]]]
    return dict(passed=bool(completed), completed_request_ids=completed, evidence=evidence)


def private_snapshot_inventory(root):
    """Sample private files only; a transient file can escape this inventory."""
    files = {}
    for session in root.iterdir():
        if not session.name.startswith('kvsnap_req_') or not session.is_dir():
            continue
        for path in session.rglob('*'):
            if path.is_file():
                stat = path.stat()
                files[str(path.relative_to(root))] = dict(size=stat.st_size,
                    mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns,
                    inode=stat.st_ino)
    return files


def monitor_private_snapshots(root, stop, result):
    previous = result['initial_files']
    while True:
        stopping = stop.wait(0.05)
        try:
            current = private_snapshot_inventory(root)
            for path, metadata in current.items():
                if path not in previous or previous[path] != metadata:
                    result['observations'].append(dict(time=time.time(), path=path,
                        kind='appeared' if path not in previous else 'metadata_changed',
                        metadata=metadata))
            previous = current
        except OSError as exc:
            result['sampling_errors'].append(str(exc))
        if stopping:
            break
    result['last_sample_files'] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18730)
    parser.add_argument('--prompts', type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument('--log', type=Path, default=DEFAULT_LOG)
    parser.add_argument('--output-parent', type=Path, default=ROOT / 'logs')
    parser.add_argument('--timeout', type=float, default=180, help='absolute per-request deadline in seconds')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--mode', choices=('greedy', 'sampled'), default='greedy',
                        help='sampled uses SAMPLE 0.8 20 0.95 0 777 SEEDED')
    parser.add_argument('--drafter', type=int, choices=(0, 1, 4), default=1,
                        help='0 = plain decode; 1 and 4 = drafter modes')
    parser.add_argument('--snapshot-dir', type=Path,
                        help='isolated engine KVSNAP directory to sample (must already exist; '
                             'caller must configure and verify the engine uses it)')
    args = parser.parse_args()
    if not (1 <= args.port <= 65535 and 1 <= args.max_tokens <= 300
            and math.isfinite(args.timeout) and args.timeout > 0):
        parser.error('invalid port, timeout or max-tokens (must be 1..300)')

    # mkdtemp creates an exclusive directory in an existing parent, never replaces files.
    out = Path(tempfile.mkdtemp(prefix='drafter-pressure-', dir=args.output_parent))
    print('output:', out, flush=True)
    state = dict(port=args.port, prompts_file=str(args.prompts), log_file=str(args.log),
                 timeout=args.timeout, max_tokens=args.max_tokens, mode=args.mode,
                 drafter=args.drafter, results={}, checks={}, errors=[])
    log = None
    start_offset = None
    monitor = None
    stop_monitor = threading.Event()
    snapshot_evidence = dict(directory=str(args.snapshot_dir) if args.snapshot_dir else None,
        method='50ms metadata sampling; not filesystem event tracking',
        scope='kvsnap_req_* private session files', observations=[], sampling_errors=[],
        status='unverified',
        limitation='Transient writes can be missed; engine directory configuration and isolation '
                   'must be independently verified.')
    state['private_snapshot_evidence'] = snapshot_evidence
    try:
        if args.snapshot_dir is not None:
            if not args.snapshot_dir.is_dir():
                raise ValueError('snapshot directory must be an existing directory')
            snapshot_evidence['initial_files'] = private_snapshot_inventory(args.snapshot_dir)
            monitor = threading.Thread(target=monitor_private_snapshots,
                args=(args.snapshot_dir, stop_monitor, snapshot_evidence), daemon=True)
            monitor.start()
        log = args.log.open('rb')
        stat = os.fstat(log.fileno())
        start_offset = stat.st_size
        # Preserve the initial boundary; it must not be inferred from later writes.
        if start_offset:
            log.seek(start_offset - 1)
            boundary = log.read(1)
            if len(boundary) != 1:
                raise RuntimeError('unable to read engine log start boundary')
            start_is_line_boundary = boundary == b'\n'
        else:
            start_is_line_boundary = True
        state['log_start'] = dict(offset=start_offset, device=stat.st_dev, inode=stat.st_ino)
        prompts = json.loads(args.prompts.read_text(encoding='utf-8'))
        if (not isinstance(prompts, list) or len(prompts) != 2 or
                any(not isinstance(p, list) or not 3500 <= len(p) <= 4200 or
                    any(type(x) is not int or x < 0 for x in p) for p in prompts)):
            raise ValueError('expected exactly two token-id prompts of approximately 3839 tokens')
        state['prompt_lengths'] = [len(p) for p in prompts]
        # A random 59-bit base plus a counter gives distinct signed 64-bit IDs per run.
        base = secrets.randbits(59) << 4
        counter = 0

        def run(label, prompt, drafter, gate=None):
            nonlocal counter
            counter += 1
            return request(label, prompt, drafter, base + counter, args, gate)

        def record(r):
            state['results'][r['label']] = r
            write_json(out / (r['label'] + '.json'), r)
            print(f"{r['label']} req={r['request_id']} reason={r.get('reason')} "
                  f"tokens={len(r['tokens'])} D={r.get('done')} error={r.get('error')}", flush=True)

        for d in (args.drafter,):
            for i in range(2):
                record(run(f'd{d}-solo-{i}', prompts[i], d))
            gate = threading.Barrier(2)
            # Assign IDs before starting threads; request() itself never shares mutable state.
            jobs = []
            for i in range(2):
                counter += 1
                jobs.append((f'd{d}-concurrent-{i}', prompts[i], d, base + counter, args, gate))
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(request, *job) for job in jobs]
                for future in as_completed(futures):
                    record(future.result())
            pair = [state['results'][f'd{d}-concurrent-{i}'] for i in range(2)]
            overlap = all('sent_monotonic' in r and 'finished_monotonic' in r for r in pair) and (
                max(r['sent_monotonic'] for r in pair) < min(r['finished_monotonic'] for r in pair))
            state['checks'][f'd{d}-overlap'] = bool(overlap)
            print(f'd{d} client request overlap: {bool(overlap)}', flush=True)
            for i in range(2):
                key = f'd{d}-prompt-{i}'
                state['checks'][key] = compare(state['results'][f'd{d}-solo-{i}'], pair[i])
                print(key, json.dumps(state['checks'][key]), flush=True)
                print('cache lengths (solo/concurrent):',
                      state['results'][f'd{d}-solo-{i}'].get('stats', {}).get('cached'),
                      pair[i].get('stats', {}).get('cached'),
                      '(concurrent tier cannot be attributed by these logs)', flush=True)
    except Exception:
        state['errors'].append(traceback.format_exc())
        print(state['errors'][-1], file=sys.stderr, flush=True)
    finally:
        if monitor is not None:
            stop_monitor.set()
            monitor.join()
        if snapshot_evidence['observations']:
            snapshot_evidence['status'] = 'private_file_activity_observed'
        print('private snapshot evidence:', json.dumps(snapshot_evidence), flush=True)
        if log is not None:
            try:
                if start_offset is None or 'log_start' not in state:
                    raise RuntimeError('engine log start boundary was not established')
                # A fresh handle is required: WSL may cache the old size on the
                # handle opened before a Windows process appended to the file.
                with args.log.open('rb') as end_log:
                    current = os.fstat(end_log.fileno())
                    state['log_path_at_end'] = dict(device=current.st_dev, inode=current.st_ino,
                                                    size=current.st_size)
                    if (current.st_dev, current.st_ino) != (stat.st_dev, stat.st_ino):
                        raise RuntimeError('engine log rotated; cannot use the new file')
                    end = current.st_size
                    state['log_end_offset'] = end
                    if end < start_offset:
                        raise RuntimeError('engine log truncated during the run')
                    end_log.seek(start_offset)
                    data = end_log.read(end - start_offset)
                    if len(data) != end - start_offset:
                        raise RuntimeError('engine log was truncated while reading')
                    current_path = args.log.stat()
                    if ((current_path.st_dev, current_path.st_ino) != (stat.st_dev, stat.st_ino)
                            or current_path.st_size < end):
                        raise RuntimeError('engine log rotated or truncated while reading')
                with (out / 'engine-offset.log').open('xb') as f:
                    f.write(data)
                # Only complete lines wholly written during this run are evidence.
                if not start_is_line_boundary:
                    data_for_checks = data.partition(b'\n')[2]
                else:
                    data_for_checks = data
                lines = data_for_checks.split(b'\n')[:-1]
                lines = [line.decode('utf-8', errors='replace') for line in lines]
                request_ids = [r['request_id'] for label, r in state['results'].items()
                               if label.startswith(f'd{args.drafter}-concurrent-')]
                state['checks']['pressure-cycle'] = check_pressure(lines, request_ids)
                print('pressure-cycle', json.dumps(state['checks']['pressure-cycle']), flush=True)
                matches = [line for line in lines if LOG_PATTERN.search(line) or 'captured' in line.lower()]
                with (out / 'engine-pressure-errors.log').open('x', encoding='utf-8') as f:
                    for line in matches:
                        f.write(line + '\n')
                print(f'log byte range [{start_offset}, {end}); pressure/error lines ({len(matches)}):', flush=True)
                for line in matches:
                    print(line, flush=True)
            except Exception:
                state['errors'].append(traceback.format_exc())
                print(state['errors'][-1], file=sys.stderr, flush=True)
            finally:
                log.close()
        state['passed'] = (not state['errors'] and len(state['checks']) == 4 and
                           not snapshot_evidence['observations'] and
                           all(v if isinstance(v, bool) else v['passed'] for v in state['checks'].values()))
        write_json(out / 'results.json', state)
        print(f"{'PASS' if state['passed'] else 'FAIL'} (request/pressure checks only; "
              f"private no-write is {snapshot_evidence['status']}): {out}", flush=True)
    return 0 if state['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
