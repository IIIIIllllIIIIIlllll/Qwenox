# FP16 KV cache

Set `KV_TYPE=fp16` in `service.conf` or select FP16 in the web launcher,
Windows settings panel or terminal settings, then restart the engine.
The default remains `bf16`. Rebuild the engine and launcher/API together.
For direct engine invocation, set `QWENOX_KV_TYPE=fp16` (`f16` is also
accepted by the engine); this takes precedence over `QWENOX_QSA_KV_BF16`.

The trunk and MTP layer store K and V as IEEE binary16, rounded directly
from FP32 with round-to-nearest-even. Each 512-element K or V row occupies
1024 bytes, the same row size as BF16. FP16 has a finer mantissa and a
narrower range than BF16 (maximum finite magnitude 65504); it is not a
lossless encoding of arbitrary FP32 values.

Decode expands FP16 to FP32 in the existing split attention kernel.
With `QWENOX_QSA_WMMA=1` (enabled by the launchers), dense and sparse
prefill/verify use native FP16 WMMA for QK and PV, with FP32 accumulation
and softmax. Q and softmax probabilities are rounded to FP16 for WMMA.
Without WMMA, attention uses FP32 arithmetic over expanded FP16 values.
The scalar `QWENOX_QSA_LOOP` fallback also supports FP16.

FP16 keeps only row-major K/V, without a persistent transposed V cache or
BF16 twin. BTV, union and WMMA6 are therefore inactive in FP16 mode, and
multi-segment verification uses per-segment attention launches. At 262144
pooled tokens and 12 trunk layers plus MTP, K/V arrays occupy 6.50 GiB,
excluding guard pages and the indexer. Default BF16 with transposed V
occupies 9.50 GiB for those arrays. These are cache sizes, not estimates of
total engine memory or performance. Indexer and GDN precision are unchanged.

Paging, scrambled physical pages, shared-prefix copy-on-write, dynamic
graph bases, MTP views and SSD snapshots preserve FP16 bytes. Snapshot
format tag 16 distinguishes FP16 from BF16 tag 2, despite their identical
row sizes. Existing BF16/Q8 fingerprints and payload layouts are preserved;
snapshots of different encodings cannot be restored into one another.

`bash build_win.sh test` (or `bash build.sh test` on Linux) includes FP16
regressions covering IEEE encodings and ties-to-even, untouched rows,
scalar/batch identity, reversed pages and page seams, captured-store replay,
dense/sparse decode against explicitly expanded FP32 caches, native WMMA
against a CPU attention reference, paged/contiguous identity, and verify
future-row masking including poisoned NaNs. The tests preserve the random
inputs used by existing regressions.

The host snapshot harness supports both formats, without a model or GPU:

```bash
bash tools/kvsnap_host/run.sh . /tmp/kvsnap-bf16-test bf16
bash tools/kvsnap_host/run.sh . /tmp/kvsnap-fp16-test fp16
```

Validation on Windows, 2026-11-18:

- The gfx1151 engine, API and all three settings interfaces compiled.
- The full kernel suite passed on gfx1151 Radeon 8060S and gfx1103 Radeon 780M,
  including 13 FP16 checks and 14 existing Q8 checks. Maximum absolute FP16
  WMMA error against the CPU reference was 3.22e-5 (dense) / 1.10e-5 (sparse).
- A Windows host harness using the production snapshot code and simulated
  device memory passed 1300-token save/restart/restore for trunk and MTP,
  for both BF16 and FP16, and rejected cross-format restores before writes.
- Configuration validation, all three shell launchers and terminal settings
  accepted bf16/fp16/q8 and rejected unknown values.
- Real-model HGN v2 + ngram + MTP validation passed on the target gfx1151:
  52 isolated engine starts and 132 recorded checks. Both precisions passed
  2051/8192/32768-token snapshot save/restart/restore with 64 continuation
  tokens matching their own live state, including MTP and changed page layouts.
- Both directions of switching precision in the same snapshot directory
  rejected foreign entries, preserved their files, matched a clean cold run,
  and restored their own snapshots after switching back. Two concurrent
  requests matched their serial controls for each precision.
- Deliberately wrong checkpoint/page format tags triggered rejection and
  correct cold recomputation. Wrong MTP page tags preserved the valid trunk,
  disabled speculation, and matched serial continuation exactly.

See [KV-FP16-VALIDATION-20261118.md](KV-FP16-VALIDATION-20261118.md) for the
test matrix, candidate hash, raw logs and reproduction commands. The real-model
drivers are `tools/kv16_e2e.py` and `tools/kv16_mtp_guard.py`; each requires a new
output directory and stops only the engine processes it starts.

These checks establish implementation and snapshot correctness for the tested
configurations. They do not require FP16 and BF16 to generate identical tokens.
KLD/perplexity, contexts beyond 32K, wider concurrency and performance tuning
remain separate evaluations before changing the BF16 default.
