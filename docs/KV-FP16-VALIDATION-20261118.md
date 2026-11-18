# FP16 / BF16 KV validation, 2026-11-18

All target-machine checks passed. FP16 and BF16 both occupy 16 bits / 2 bytes
per element, but use different encodings. The snapshot format tags are FP16
`16` and BF16 `2`; the encoding also changes the fingerprint, page names,
MTP-section names and checkpoint names. Equal row sizes never authorize reuse.

Target: Windows 11 build 26200, Radeon 8060S (`gfx1151`), AMD driver
32.0.32015.7029, TheRock gfx1151 10.1.0. The candidate uses local source based on
`1e28ab6` plus the FP16 changes. The remote checkout's existing configuration
edits were preserved; testing used an isolated source/build/cache directory.

Model files, all from `C:\Users\Mark\Models\HGN`:

- `qwen38-flash-next-v2.hgn`
- `qwen38-flash-next-ngram.hgn`
- `qwen38-flash-next-mtp.hgn`

The same model paths and numerical settings were used for both precisions.
WMMA, graphs, paging, GDN streaming and MTP gamma 4 were enabled. BF16 used
its existing BTV path; FP16 used native FP16 WMMA with row-major K/V.

The target-built engine SHA-256 is
`42f374cd7a10d664b44352ec78864d4fd3b1a73ba97ff45885eb87d3211ccbcc`.
The retrieved local binary is `build/qwenox-engine-fp16-tested.exe`.

| Test | BF16 | FP16 |
|---|---|---|
| Full gfx1151 kernel suite, including existing regressions | PASS | PASS |
| 2051-token prompt, serial decode, 64-token restored continuation | PASS | PASS |
| 2051 / 8192 / 32768-token prompts, MTP, 64-token restored continuation | PASS | PASS |
| Continuous save → scrambled paged restore → ordinary paged restore | PASS | PASS |
| MTP remains live after valid snapshot restore | PASS | PASS |
| Foreign precision in the same directory: zero cache hits | PASS | PASS |
| Shared-directory cold computation equals independent clean-directory control | PASS | PASS |
| Original files preserved; distinct fingerprints, names and header tags | PASS | PASS |
| Switch back: original snapshots remain usable, continuation tokens match | PASS | PASS |
| Two concurrent shared-prefix / COW requests: 96 tokens match serial controls | PASS | PASS |
| Wrong checkpoint tag: reject and recompute, 64 tokens match cold control | PASS | PASS |
| Wrong trunk-page tag: reject and recompute, 64 tokens match cold control | PASS | PASS |
| Wrong MTP-page tag: retain trunk, disable speculation, match serial control | PASS | PASS |

The real-model suites recorded 132 passing checks across 52 independent engine
starts and 68 generation requests. A live-continuation oracle keeps the same
connection across requests and asserts reuse of the generated prefix. Every
save completes a `KVSYNC` barrier before shutdown. Restore tests require an
actual SSD hit in both the protocol response and engine log. Comparisons are
within one precision; outputs between FP16 and BF16 need not match.

The full GPU suite includes 13 FP16 checks and 14 existing Q8 checks. FP16
native WMMA maximum absolute errors against an independent CPU attention
reference were 3.218e-5 (dense) and 1.094e-5 (sparse), below 8e-5. Storage
encoding, RNE ties, page seams, graph replay with changing device bases,
decode references and masked future NaNs passed. A separate Windows host
harness using production snapshot code and simulated device memory verified
1300-token trunk/MTP save/restart/restore byte-for-byte, and cross-format
rejection before KV writes.

Reproduce on the model/GPU host after building the updated engine:

```powershell
python tools/kv16_e2e.py --engine build/qwenox-engine-win.exe `
  --model C:/Users/Mark/Models/HGN/qwen38-flash-next-v2.hgn `
  --model C:/Users/Mark/Models/HGN/qwen38-flash-next-ngram.hgn `
  --model C:/Users/Mark/Models/HGN/qwen38-flash-next-mtp.hgn `
  --output results-live
python tools/kv16_mtp_guard.py --results results-live --output results-mtp-guards
```

Output directories must be new. The drivers choose dedicated loopback ports,
set explicit environments and use only their own caches and child processes.
The recorded target directory is
`C:\Users\Mark\Workspace\Qwenox\build\kv16-validation-20261118-1755`.

Raw evidence is retrieved under `build/fp16-gfx1151-validation/`:

- `kernel-suite.log`: target compiler output and complete GPU/host regression results.
- `results-live/results.json`, `e2e-live.log`: 48 starts, 122 checks, token IDs,
  cache-hit counts, settings and individual engine logs.
- `results-mtp-guards/results.json`, `mtp-guards.log`: four starts and ten
  MTP-format defense checks, with individual logs.
- `tested-source-manifest.json`: SHA-256 hashes of the candidate sources.
- `decoded-samples.json`: generated text decoded with the target model tokenizer.

The same evidence is packed in `build/kv16-validation-logs.zip` locally and
`kv16-validation-logs.zip` in the remote candidate directory. Snapshot payloads
are retained only inside the isolated remote test directories.

The remote `results/` and `e2e.log` are an earlier exploratory run whose
reference reopened the connection. The single-slot server clears live history
on connection changes, so those results are excluded. The authoritative matrix
is `results-live/` with persistent-connection live-prefix assertions.

This validates the exercised numerical and cache paths. Perplexity/KLD,
contexts beyond 32K, concurrency above two and throughput comparisons were
not part of this correctness matrix. The default remains BF16.
