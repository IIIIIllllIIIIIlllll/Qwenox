"""Reject foreign MTP page encodings while retaining a valid trunk restore.

Run after kv16_e2e.py completed, using the same engine and model files:
  python tools/kv16_mtp_guard.py --results results-live --output results-mtp-guards
"""
import argparse
import json
from pathlib import Path
import shutil
import struct
import sys
from types import SimpleNamespace

from kv16_e2e import Suite

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
prior = json.loads((args.results / 'results.json').read_text())
assert prior.get('passed')
suite = Suite(SimpleNamespace(engine=prior['engine'], model=prior['models'],
                              output=args.output))
try:
    for precision in ('bf16', 'fp16'):
        first = prior['runs'][precision + '-guards-save']['requests'][0]['tokens']
        q = suite.prompt(2051) + first + [198, 1000, 198]
        source = args.results / (precision + '-guards-valid')
        good = suite.out / (precision + '-good')
        bad = suite.out / (precision + '-bad-mtp')
        shutil.copytree(source, good)
        shutil.copytree(source, bad)
        pages = list(bad.glob('*.kvm'))
        suite.check(bool(pages), precision + ' has separate MTP page files')
        for path in pages:
            with path.open('r+b') as stream:
                stream.seek(36)
                stream.write(struct.pack('<i', 16 if precision == 'bf16' else 2))
        with suite.run(precision + '-reference-serial', precision, good, paged=2) as engine:
            gold = engine.gen(q, 64, drafter=0)
            engine.sync()
            suite.check(gold['cached'] > 2051, precision + ' reference restores trunk')
        with suite.run(precision + '-reject-foreign-mtp', precision, bad, paged=2) as engine:
            actual = engine.gen(q, 64, drafter=1)
            engine.sync()
            log = engine.logtext()
            suite.check(actual['cached'] > 2051 and 'kvsnap: restored' in log,
                        precision + ' wrong MTP encoding preserves valid trunk restore')
            suite.check('(mtp live)' not in log and actual['done'][7] == '0',
                        precision + ' wrong MTP encoding disables speculation')
            suite.check(actual['tokens'] == gold['tokens'],
                        precision + ' rejected MTP matches serial continuation exactly')
    suite.report['passed'] = True
    suite.save()
    print('ALL PASS (MTP format guards)', flush=True)
finally:
    suite.save()
