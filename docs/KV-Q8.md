# Experimental Q8 KV cache

Build the updated engine and launcher/API first. Set `KV_TYPE=q8` in the
environment or service.conf and restart the engine. The launcher default is
`bf16`. Direct engine invocations use `QWENOX_KV_TYPE=q8`;
this overrides the legacy `QWENOX_QSA_KV_BF16` switch.

The trunk and MTP layer store K and V in symmetric INT8 groups of 32
consecutive dimensions, with one independent FP16 scale per group. A group
occupies 34 bytes; one 512-element K or V row occupies 544 bytes instead of
1024. No scale is shared across tokens, so appends and speculative rollbacks
do not requantize accepted history.

Decode expands values to FP32 in the existing split attention kernel.
Prefill and verify expand fragments to BF16 inside row-major WMMA attention.
Q8 does not allocate the extra full-context BF16 transposed V cache; BTV,
union and WMMA6 optimizations therefore do not apply. Multi-segment verify
uses the per-segment attention launches. The sparse indexer and GDN states
keep their existing storage precision.

Paging, shared-prefix copy-on-write, MTP views and SSD snapshot payloads
include the packed scales. Snapshot tag 8 and its schema fingerprint isolate
Q8 from BF16/FP32 caches. The default BF16 snapshot format is preserved.

At 262144 pooled tokens and 12 trunk layers plus MTP, row-major Q8 K/V uses
about 3.45 GiB, excluding guard pages and indexer. BF16 K/V plus the default
transposed V uses about 9.50 GiB. These numbers are cache-array sizes, not
whole-engine memory estimates.

`bash build_win.sh test` (or `bash build.sh test` on Linux) includes Q8
storage and attention regressions: CPU quantization reference, scalar/batch
identity, reversed pages and page seams, dense/sparse decode against explicit
FP32 dequantization, WMMA against BF16 dequantization, and verify future-row
masking. These check implementation correctness, not model quality.

Functional validation on Windows gfx1151, 2026-11-18, using the HGN v2 base,
ngram and MTP weights:

- Engine, API and benchmark builds completed; the full kernel test suite
  passed, including all ten Q8 checks.
- Real model generation completed with a short prompt and a 2098-token
  prompt spanning the first sparse-attention boundary, using both contiguous
  KV and scrambled physical pages (`--maxctx 4096`).
- MTP speculative generation exercised captured verify graphs and rollback.
- Two concurrent requests completed; concurrent MTP was exercised separately
  from RAM-checkpoint/COW coverage.
- SSD save/restart/restore produced the same 16 continuation token IDs as
  the live state, including MTP state. Lowering the RAM-checkpoint minimum
  to 256 in the test also exercised partial-page copy-on-write.

The Windows candidate binaries are `build/qwenox-engine-q8.exe` and
`build/qwenox-api-q8.exe`. The API supervisor launches the standard engine
filename; build/install the updated standard engine before using Q8 through
the normal service launcher. Test logs are under `build/q8-*.log` and
`build/q8-server-test.out` on the validation machine.

This is a functional implementation. KLD and long-context quality evaluation
are pending; speed and model quality must be evaluated before enabling it
as the default. Model weights are unaffected.
