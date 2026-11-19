#!/usr/bin/env python3
"""Check the actual CVEC export blocks used by both launch scripts."""
import argparse
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ('QWENOX_CVEC', 'QWENOX_CVEC_MODE', 'QWENOX_CVEC_SCALE',
        'QWENOX_CVEC_LAYERS', 'QWENOX_CVEC_DIR', 'QWENOX_CVEC_PROBE')
BASH = 'bash'


class CvecLaunchConfigTest(unittest.TestCase):
    def export(self, name, enabled, extras):
        source = (ROOT / name).read_text(encoding='utf-8')
        start = source.index('unset QWENOX_CVEC QWENOX_CVEC_MODE')
        end = source.index('export QWENOX_QSA_KV_BF16', start)
        env = {k: v for k, v in os.environ.items() if k not in KEYS}
        if extras:
            env.update(QWENOX_CVEC_DIR='single:20', QWENOX_CVEC_PROBE='20')
        setup = ("set -eu\nCVEC='vector with spaces.gguf'\n" if enabled else "set -eu\nCVEC=''\n")
        setup += "CVEC_MODE=add\nCVEC_SCALE=0.1\nCVEC_LAYERS='16 32'\n"
        dump = '\n'.join('printf \'%s\\n\' "${' + key + '-UNSET}"' for key in KEYS)
        result = subprocess.run([BASH, '-c', setup + source[start:end] + '\n' + dump],
                                env=env, capture_output=True, text=True, check=True)
        return dict(zip(KEYS, result.stdout.splitlines(), strict=True))

    def test_enabled_keeps_explicit_options(self):
        for name in ('start_win.sh', 'tools/serve_common.sh'):
            with self.subTest(launcher=name):
                self.assertEqual(self.export(name, True, True), dict(zip(KEYS, (
                    'vector with spaces.gguf', 'add', '0.1', '16 32', 'single:20', '20'))))

    def test_enabled_does_not_invent_options(self):
        for name in ('start_win.sh', 'tools/serve_common.sh'):
            with self.subTest(launcher=name):
                actual = self.export(name, True, False)
                self.assertEqual(actual['QWENOX_CVEC_DIR'], 'UNSET')
                self.assertEqual(actual['QWENOX_CVEC_PROBE'], 'UNSET')

    def test_disabled_clears_options(self):
        for name in ('start_win.sh', 'tools/serve_common.sh'):
            with self.subTest(launcher=name):
                self.assertEqual(set(self.export(name, False, True).values()), {'UNSET'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bash', default=(r'C:\Program Files\Git\bin\bash.exe'
                                         if os.name == 'nt' else 'bash'))
    args = parser.parse_args()
    BASH = args.bash
    unittest.main(argv=[__file__], verbosity=2)
