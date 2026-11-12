"""CPU-only fault injection for the production HIP wrappers and watchdog."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cxx', default=os.environ.get('CXX'))
    args = parser.parse_args()
    cxx = args.cxx or shutil.which('clang++') or shutil.which('g++')
    if not cxx:
        parser.error('C++17 compiler required; pass --cxx PATH')
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='hip-watchdog-') as directory:
        binary = Path(directory) / ('test.exe' if os.name == 'nt' else 'test')
        cmd = [cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
               str(root / 'tools/hip_watchdog_test.cpp'), '-o', str(binary)]
        if os.name != 'nt':
            cmd.append('-pthread')
        subprocess.run(cmd, check=True, timeout=120)
        for profile in ('0', '1'):
            for mode in ('idle', 'healthy', 'disabled', 'sync', 'device', 'event',
                         'copy', 'async', 'nested'):
                result = subprocess.run([str(binary), mode], capture_output=True,
                                        text=True, timeout=10,
                                        env={**os.environ, 'QWENOX_SYNC_PROF': profile})
                want = 0 if mode in ('idle', 'healthy', 'disabled') else 124
                if result.returncode != want:
                    raise AssertionError((mode, profile, result.returncode, result.stderr))
                print(f'PASS {mode} sync-prof={profile}')
        if os.name == 'nt':
            # Exercise the exact production termination callback, including a
            # report writer that never returns. This does not call HIP.
            source = (root / 'src/gpu/parts/51_host_cfg.inc').read_text(encoding='utf-8')
            start = source.index('[](const hipwatch::Call& call) {')
            end = source.index('\n  });', start)
            callback = source[start:end] + '\n  }'
            runtime = Path(directory) / 'runtime.cpp'
            runtime.write_text('''
#include <windows.h>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include "hip_watchdog.h"
namespace crashdiag {
void note(const char*, ...) {}
void report_fatal(const char*, const char*) {
  if (getenv("TEST_BLOCK_REPORT")) Sleep(INFINITE);
}
}
int main() {
  auto callback = ''' + callback + ''';
  callback({__FILE__, "test sync", __LINE__, hipwatch::Clock::now()});
  return 3;
}
''', encoding='utf-8')
            subprocess.run([cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root / 'src'), str(runtime), '-o', str(binary)],
                           check=True, timeout=120)
            for blocked in (False, True):
                env = dict(os.environ)
                env.pop('TEST_BLOCK_REPORT', None)
                if blocked:
                    env['TEST_BLOCK_REPORT'] = '1'
                result = subprocess.run([str(binary)], capture_output=True, text=True,
                                        timeout=10, env=env)
                assert result.returncode == 124, (result.returncode, result.stderr)
                assert 'HIP watchdog: test sync' in result.stderr
                print(f'PASS production hard-exit blocked-report={blocked}')


if __name__ == '__main__':
    main()
