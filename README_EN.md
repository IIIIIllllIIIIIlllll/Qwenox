# Qwenox

![Strix Halo — Qwen3.8-Flash-Next](media/strix_banner_21x9_v2.png)

*中文版:[README.md](README.md)*

Qwenox is a local inference engine for Qwen3.8-Flash-Next (177B MoE, qwen4_exp architecture). It is written
for AMD Strix Halo (Ryzen AI Max+ 395, gfx1151) and similar 128 GB unified-memory APUs, and runs on both
Linux and Windows. It serves an OpenAI-compatible API and the Anthropic Messages API (streaming, tool calls, image input)
and a web console.

Key design points:

- **The weights fit.** Routed experts are 4-bit and read directly by the GPU kernels. The PLE n-gram table
  (27–48 GiB) stays on disk and is read on demand, so it takes no memory. Resident weights are 60–77 GiB
  depending on the weight file; the remaining memory holds a 256K context.
- **Speculative decoding.** ngram drafting first, with the MTP draft head as fallback. Greedy is checked
  token by token; sampling resamples against the target distribution, so the output distribution matches
  non-speculative decoding. On by default, no flags needed.
- **No repeated prefill.** A paged KV pool plus two levels of prompt cache (in-memory checkpoints at message
  boundaries, and SSD snapshots that survive restarts). Edit-and-resend, multi-turn continuation and
  switching between conversations reuse what was already computed.

## Measured Performance

Development machine: **GMK EVO-X2 (AMD Ryzen AI Max+ 395, Strix Halo / gfx1151)**, 122 GiB RAM.

Linux (ROCm 10.1, production config, prefill with MOE_A4=3, PREFILL_CHUNK=16384):

| Metric | hgn v2 |
| --- | --- |
| Prefill (2K) | ~1610–1651 tok/s (measured 10-18) |
| Prefill (8K) | 1902.7 tok/s (measured 10-18) |
| Prefill (32K) | 1938.5 tok/s (measured 10-18) |
| Prefill (64K) | 1936.4 tok/s (measured 10-18) |
| Prefill (128K) | 1840.8 tok/s (measured 10-18) |
| Prefill weighted average | 1874.5 tok/s (7 files, 2 samples each, measured 10-18) |
| Prefill (256K context) | ~1590 native, ~1580 with YaRN factor 2 (~0.5% overhead, measured 10-07) |
| Decode (speculative, greedy, γ=4) | ~51 tok/s at 8K, ~50 at 64K (real text, 10-07) |
| Decode (no speculation) | ~34 tok/s at 8K (measured 10-07, hgn v2 weights; weights with 8-bit dense read more per token, ~12–16% slower) |
| Average power draw | ~120 W |
| Peak (instantaneous) power draw | ~130 W (bursts for a few seconds, then settles back to ~120 W) |

Windows (same machine, 96 GiB GPU memory set in the BIOS, tray production config, prefill chunk 8192, measured
10-13):

| Metric | hgn v2 | GGUF UD-IQ4_XS + MTP |
| --- | --- | --- |
| Prefill (32K) | ~1560 tok/s | ~1440 tok/s |
| Decode (speculative, 2K prompt) | 58.5 tok/s | 46.8 tok/s |
| Decode (no speculation) | 31.4 tok/s | 27.7 tok/s |
| Cold start (whole weight file read into VRAM) | ~35 s | ~50–60 s |

Speculative decode speed depends heavily on how repetitive the text is. The tables use real-text prompts;
highly repetitive content such as code or templates gets a high draft hit rate and exceeds 60 tok/s in the
same configuration; sampling (adaptive γ) is slightly below greedy. All dates are 2026; on Linux, ROCm 7.14
and 10.1 measured identical. Multi-request numbers are in "Concurrency".

## Supported Weights

| Weights | Platform | Quality (KLD vs BF16, lower is better) | Download / resident | When to use |
| --- | --- | --- | --- | --- |
| **hgn v2** (recommended) | Linux, Windows | 0.107 | ~112 GiB / 62.1 GiB | Default: fastest, best results in practice |
| GGUF UD-Q4_K_XL | Linux | 0.051 | 104 GiB / ~77 GiB | Better quality on Linux |
| GGUF UD-IQ4_XS | Windows (Linux untested) | 0.100 | 87 GiB / 60.4 GiB (63.0 with MTP) | Only to save VRAM, and only without MTP; quality on par with v2, slower than v2 |
| hgn w4b | Linux, Windows | 0.163 | 118 GiB / 68.0 GiB | Already-downloaded older files; not recommended for new downloads |

- Resident memory is the weight footprint measured on Windows, including the MTP draft head and excluding the PLE
  n-gram table, which stays on disk (hgn 47.7 GiB, GGUF 26.8 GiB); that is why the download is larger. The v2 body
  already contains the MTP draft head, so the separate `mtp.hgn` costs no extra VRAM; the GGUF MTP head is a
  separate file and adds to the total.
- UD-Q4_K_XL has lower KLD than v2 mainly because its dense part (attention, GDN, shared expert, lm_head) is
  8-bit while v2's is 4-bit; the cost is twice the dense bytes per token, so non-speculative decode is ~16% slower.
- UD-IQ4_XS stores most routed experts as IQ3_S, narrower than v2's 4-bit, with 8-bit dense. Its only advantage
  is lower VRAM use, and only without MTP: without MTP it uses 3.2 GiB less than v2, with the Q8_0 MTP draft head
  it uses 1.3 GiB more, and without MTP decode is only about half the speculative speed. Quality is roughly on par
  with v2: KLD 0.100 vs 0.107 and top-1 agreement 89.7% vs 88.8% are slightly better, while PPL (3.44 vs 3.42) and
  the worst-0.1% KLD (3.11 vs 2.91) are slightly worse. It is slower than v2: decode ~12%, prefill ~6%, and
  multi-request throughput ~15% lower.
- KLD uses wikitext-2, 64 chunks × 512, the same method as unsloth / llama.cpp; see [KLD.md](docs/KLD.md).
- Other quant levels in the Unsloth repo (IQ1 / IQ3_XXS / Q2 / Q3 / Q5 / Q6 etc.) are not supported.

Downloads use the Hugging Face CLI (`pip install -U huggingface_hub`). The commands below put everything
into the project's `models/` directory, which is where the launchers look by default.

### hgn v2

Source: [peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next)
(published by halogen, loaded directly by this engine).

| File | Size | Notes |
| --- | --- | --- |
| `qwen38-flash-next-v2.hgn` | 62.1 GiB | Main model |
| `qwen38-flash-next-ngram.hgn` | 47.7 GiB | PLE n-gram table, read from disk on demand |
| `qwen38-flash-next-mtp.hgn` | 1.4 GiB | 8-bit MTP draft head |
| `qwen38-flash-next-vision.hgn` | 0.84 GiB | Vision tower; skip it for text-only and set `VISION_FILE=""` |
| `tokenizer/` | — | Tokenizer |

```bash
hf download peonist-ai/halogen-qwen3.8-flash-next --local-dir ./models --include \
  qwen38-flash-next-v2.hgn qwen38-flash-next-ngram.hgn \
  qwen38-flash-next-mtp.hgn qwen38-flash-next-vision.hgn "tokenizer/*"
```

The other files in that repo (`w4b`, `overlay`, `ht43`) are not needed.

### GGUF

Source: [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).
These are the same files llama.cpp uses; if you already have them, symlink them into `models/`.

| File | Size | Notes |
| --- | --- | --- |
| `UD-Q4_K_XL/…-0000{1..4}-of-00004.gguf` | 103.7 GiB | Main model (Linux), all 4 shards |
| `UD-IQ4_XS/…-0000{1..3}-of-00003.gguf` | 87.2 GiB | Main model (Windows), all 3 shards |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.6 GiB | MTP draft head |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (repo root) | 3.9 GiB | MTP draft head; choose one of the shared-Q8_0, Q8_0, and Q4_K_M sidecars |
| `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf` (repo root) | 2.6 GiB | MTP draft head; Q5_0 HC-up tensors are converted to q8g32 at load; tested |
| `mmproj-BF16.gguf` | 0.85 GiB | Vision tower; `qwen38-flash-next-vision.hgn` from the hgn repo also works |

The GGUF repo has no `tokenizer.json`; take it from the original model repo (it is the same file as in the
hgn repo's `tokenizer/`).

```bash
# Linux: UD-Q4_K_XL
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir ./models --include \
  "UD-Q4_K_XL/*" MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf mmproj-BF16.gguf
# Windows: UD-IQ4_XS (vision tower from the hgn repo; this is the combination tested on Windows)
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir ./models --include \
  "UD-IQ4_XS/*" mtp-Qwen3.8-Flash-Next-Q8_0.gguf
hf download peonist-ai/halogen-qwen3.8-flash-next --local-dir ./models --include \
  qwen38-flash-next-vision.hgn
# Both need the tokenizer
hf download Qwen/Qwen3.8-Flash-Next tokenizer.json --local-dir ./models/tokenizer
```

The shards land in subdirectories; put their paths into `service.conf` as described in "Quick Start".

## Quick Start

Linux and Windows share the same source tree, the same `models/` directory and the same `service.conf`.
Download the weights as described above, put their paths into `service.conf`, then build and start for your
OS. Once running, the API is at `http://<host>:8731/v1` (standard OpenAI interface, streaming supported; Anthropic clients use `http://<host>:8731/v1/messages`), and
`http://<host>:8731/` in a browser opens the console.

### Weight paths

`service.conf` has one section for hgn and one for GGUF; only the one in use is read at startup. An
environment variable of the same name takes precedence over the file.

hgn v2:

```bash
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/qwen38-flash-next-v2.hgn}"
NGRAM_FILE="${NGRAM_FILE:-$MODEL_DIR/qwen38-flash-next-ngram.hgn}"
OVERLAY_FILE="${OVERLAY_FILE-}"
```

For GGUF use the actual subdirectories from the download; `GGUF_FILE` is the first shard:

```bash
# Linux: UD-Q4_K_XL
GGUF_FILE="${GGUF_FILE:-$MODEL_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
GGUF_MTP_FILE="${GGUF_MTP_FILE-$MODEL_DIR/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
GGUF_VISION_FILE="${GGUF_VISION_FILE-$MODEL_DIR/mmproj-BF16.gguf}"
# Windows: UD-IQ4_XS
GGUF_FILE="${GGUF_FILE:-$MODEL_DIR/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
GGUF_MTP_FILE="${GGUF_MTP_FILE-$MODEL_DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
GGUF_VISION_FILE="${GGUF_VISION_FILE-$MODEL_DIR/qwen38-flash-next-vision.hgn}"
```

An empty `GGUF_MTP_FILE` disables speculation. On Windows you can also fill these in on the console's
`#/engine` page instead of editing the file; saving writes back to the same `service.conf`. Loading and
conversion rules are in [GGUF.md](docs/GGUF.md).

### Linux

Requirements: ROCm (HIP 7.x or newer), a gfx1151 GPU, ≥ 100 GiB available memory. Build dependencies:
rocBLAS, hipBLASLt, rocPRIM; the API front end also needs libpng, libjpeg, libwebp.

```bash
bash build.sh        # builds engine, API front end, benchmark into build/, and the TUI settings tool at the root
bash start_hgn.sh    # start with hgn weights
bash start_gguf.sh   # or with GGUF weights
```

Run `./qwenox-settings` for the cross-platform terminal settings UI instead of editing `service.conf` directly —
the UI is bilingual (Chinese/English; switch on page 6 "Language").
`bash build.sh settings` builds just that tool. See [SETTINGS-TUI.md](docs/SETTINGS-TUI.md) for keys and check scope.

The two launchers differ only in which weights they read: they load the engine, then start the API once it is
ready. They run in the foreground and Ctrl+C stops both; over SSH, run them inside tmux. If anything is
missing they list every missing file and exit; `--check` checks the configuration without starting. Logs are
in `logs/<date-time>-<PID>/`. See [QUICKSTART_EN.md](docs/QUICKSTART_EN.md); build troubleshooting is in
[BUILD_EN.md](docs/BUILD_EN.md).

### Windows

Same features as Linux (engine, OpenAI API, multimodal, concurrency). Requirements:

- 96 GiB of GPU memory set in the BIOS. VRAM decides how much context you can open; see "Windows VRAM limit".
- Running needs only the AMD graphics driver, no ROCm install.
- Building needs Git for Windows (for Git Bash) and the TheRock Windows package (default
  `C:\therock-dist-windows-gfx1151-10.1.0`, or point `THEROCK=` elsewhere); see [BUILD_EN.md](docs/BUILD_EN.md).

Build in Git Bash (or double-click `build_win.bat`):

```bash
bash build_win.sh    # engine, benchmark, API tray app, and the root launcher start_win.exe
```

For daily use, double-click `start_win.exe` in the project root. It puts an icon in the system tray, opens no
console window and does not load the model by itself. Double-click the tray icon to open the web console; on
the `#/engine` page choose the weights (hgn or GGUF), edit the configuration and start the engine. The
right-click menu can also start/stop the engine, copy the API address and open logs. Logs are in `logs\`. If
you prefer the command line, `bash start_win.sh` in Git Bash behaves like the Linux launchers. For a terminal
settings UI, run `qwenox-settings.exe` at the project root in Windows Terminal, PowerShell, or CMD; it edits only
`service.conf` and never starts the service. See [SETTINGS-TUI.md](docs/SETTINGS-TUI.md).

Copying the built `build/` and `start_win.exe` together with `models/` to another gfx1151 Windows machine is
enough to run there; that machine needs neither Git nor TheRock. Porting details are in
[PORTING-WINDOWS_EN.md](docs/PORTING-WINDOWS_EN.md).

## Common Settings

All settings are in `service.conf` (on Windows also editable on the console's `#/engine` page); an
environment variable of the same name takes precedence. Restart the engine after editing.

| Setting | Default | Meaning |
| --- | --- | --- |
| `MAX_CONTEXT` | `262144` | Input + output limit for a single request |
| `KV_POOL_TOKENS` | `0` | Total KV shared by all concurrent requests; 0 means equal to `MAX_CONTEXT`; each extra 256K costs ~9.6 GiB |
| `PARALLEL` | `4` | Number of requests running at once, 1–8 |
| `PREFILL_CHUNK` | `0` | Tokens per prefill segment for long prompts; 0 means 16384 on Linux, 6144 on Windows |
| `ROPE_FACTOR` | `1` | 1 is native 256K; set 2 and `MAX_CONTEXT=524288` for YaRN 512K |
| `MTP_GAMMA` | `0` | Draft length; 0 is automatic, 1–8 fixes it |
| `MOE_A4` | `0` | 4-bit prefill speedup (hgn v2 only), see below; 0 off, 3 on |

YaRN applies to the whole instance and slightly changes the output distribution even on short text (KLD
0.023 vs native), so turn it on only when a single request really needs more than 256K. The 512K setup,
shared-pool admission rules and validation scope are in [YARN-512K.md](docs/YARN-512K.md).

### 4-bit prefill speedup (MOE_A4)

Optional, off by default. When on, hgn v2 weights quantize the routed experts' input activations to int4
(one scale per 128 elements) during prefill, and the expert GEMMs run on INT4 WMMA
(`v_wmma_i32_16x16x16_iu4`, twice the F16 WMMA rate on gfx1151). Set `MOE_A4=3` in `service.conf`, or
tick "4-bit prefill speedup" on the V2 weights page of the Windows settings panel. The value is a
bit mask: 1 quantizes gate/up only, 2 down only, 3 both. GGUF and w4b weights ignore it.

**What it affects**: prefill only, i.e. any input processed more than 16 tokens at a time: the prompt,
new messages in a multi-turn chat, tool-call results. Generated tokens go through decode, and MTP verify
and concurrent batching do not use this path either, so generation computes exactly as with the switch
off. The cost is lower precision in what the model reads in; the longer the input and the more a task
depends on reading it exactly, the more of the work is affected.

| | off | `=1` | `=2` | `=3` |
| --- | --- | --- | --- | --- |
| 64K prefill (Windows, tok/s) | 1609 | 1719 | 1615 | 1752 (+8.9%) |
| KLD vs BF16 | 0.1075 | 0.1280 | 0.1226 | 0.1427 |

**Tests done** (all hgn v2 on Windows):

- **KLD**: wikitext-2, 64 chunks × 512, same method as [KLD.md](docs/KLD.md). This method scores every
  position through prefill, so the A4 error covers every scored token; it measures the loss in reading
  precision, not in generation quality.
- **Correctness**: with the switch off, output is token-for-token identical to the engine before this
  feature; with `=3`, two concurrent multi-chunk long prompts with short requests cutting in mid-chunk
  produce the same greedy output as running alone (`tools/a4_verify_win.sh`).
- **Long-context retrieval** (`bench/codeneedle`, see [bench/README.md](bench/README.md)): this
  project's own source, shuffled into 16K / 32K / 64K haystacks, with 144 questions that each have a
  single answer (which file reads an environment variable, its default value, which file defines a
  function, a constant's value). Off and `=3` both score 99.3% greedy; 143 of 144 outputs are
  byte-identical and no question flips between right and wrong; the log-probability of the same answer
  moves by under 0.01 on average, with no growth with length. Haystack prefill 1580 → 1726 tok/s.

**Not covered by these tests**:

- The retrieval baseline is near the ceiling, so these tests can rule out an obvious regression but
  cannot detect a small loss.
- Nothing beyond 64K, and no questions that need several places in the input combined.
- No end-to-end tasks such as code generation or tool calling. The hipfire author compared both sides
  of the switch on their model with HumanEval, GSM8K and MMLU-Pro and found the difference within
  noise; that is a useful reference but not a measurement on this project.

Recommendation: turn it on for heavy long-input workloads where time to first token matters; leave it off
when precision matters or when unsure.

## Concurrency

Several requests can run at the same time; a new one does not wait for the previous one to finish.

- **One shared KV page pool.** `PARALLEL` slots (1–8, default 4) take 256-token pages from one pool on
  demand, like llama.cpp's shared context. The pool defaults to 256K in total: four 64K requests, or one
  200K plus a few short ones. `MAX_CONTEXT` limits a single request; the total across all concurrent
  requests is the pool size `KV_POOL_TOKENS`.
- **Batched decode.** The MTP verify steps of all active requests are merged into one batch and computed in
  the same step, so more requests share each round's weight reads. The draft length γ is chosen
  automatically from the number of active requests.
- **Long prompts don't block others.** Prefill yields the GPU between layers about every 300 ms, giving
  each decoding session a round; other sessions stall at most ~0.6 s. When several long prompts arrive
  together, the one with less remaining work finishes first.
- **When the pool runs out:** in-memory checkpoints are evicted first, then idle slots' KV; if that is still
  not enough, the later request is aborted (the API returns an error). An earlier request is never pushed
  out by a later one. Requests beyond `PARALLEL` queue for a slot.
- **Reproducible.** With v2 weights, greedy requests and sampled requests with a `seed` produce exactly the
  same output under concurrency as when run alone. On GGUF, near-tie tokens occasionally differ; set
  `QWENOX_OCC_FORCE=4` for strict equality at a small throughput cost.

API concurrent throughput (Windows, coding prompts, greedy, MTP on, 2 × 1200 tokens per request, total tok/s
across all requests):

| Concurrent requests | 1 | 4 | 8 |
| --- | --- | --- | --- |
| hgn v2 | 55.3 | 83.2 | 97.0 |
| GGUF UD-IQ4_XS | 47.0 | 70.7 | 82.4 |

Total throughput rises with the number of requests, but each request gets less: about 12 tok/s per request
at 8. Configuration, scheduling and verification are in [CONCURRENCY.md](docs/CONCURRENCY.md).

### Concurrency across workloads

The table above is a same-prompt code comparison. MTP acceptance has a large effect on concurrent
throughput, so here is a second set of measurements across workloads (2026-10-23, Qwenox API, gfx1151
Windows, greedy, MTP on, prompt cache off, all requests started together, streaming; total throughput is
measured over the window in which all requests are active, not by summing per-request speeds):

| Workload | Concurrent | MTP acceptance | Per-request steady | Total |
| --- | --- | --- | --- | --- |
| Synthetic text (256 prompt / 512 output) | 2 | 47.4% | 30.0 tok/s | 58.4 tok/s |
| HTML code (Tetris + Breakout, full generation) | 2 | 77–83% | 43.5 tok/s | 83.5 tok/s |
| Creative writing (eight ~800-character stories) | 8 | 49.1% | 10.7 tok/s | 85.9 tok/s |
| HTML code (eight pages, 1024-token cap each) | 8 | 69.4% | 13.7 tok/s | 109.7 tok/s |

- Total throughput rises with concurrency but with diminishing returns: on the high-acceptance code
  workload, going from 2 to 8 requests raises total throughput only from 83.5 to 109.7 tok/s (about
  1.3×), while per-request speed falls from 43.5 to 13.7 tok/s. Eight slots suit throughput-oriented
  background work; 2–4 are a better fit for interactive use.
- MTP acceptance is what makes concurrency pay off: at 8 requests, creative writing at 49.1% reaches
  85.9 tok/s, while HTML code at 69.4% reaches 109.7 tok/s. Higher acceptance means each verify round
  advances more tokens and amortizes the weight reads over more requests.
- With 8 requests arriving together, TTFT rises linearly from about 0.34 s to 2.47 s, roughly +300 ms per
  request; prefill is time-sliced rather than run 8-way in parallel. Per-request ITL also rises from
  about 22–24 ms on 2-way code to 62–97 ms, with a standard deviation of 82–145 ms, so generation is
  less smooth.
- Token counts come from server `usage`; acceptance comes from the chain drafter in `/reqstat`
  (accepted/proposed). Two-way creative writing (43–50% acceptance) measured about 57 tok/s over the
  overlap window and 28.5 tok/s per request, consistent with the synthetic run.
- The 97.0 tok/s at 8 requests in the table above and the 109.7 tok/s here are not the same parameter
  set (different prompts, output lengths and acceptance); do not compare them by subtraction.

## Windows VRAM limit

On Windows, VRAM decides how much context you can open. A process can only reliably use one block of device
memory allocated at startup, capped at about 95 GiB; weights, the KV page pool, the prefill workspace and the
vision tower all have to fit in it. Anything that does not fit at run time falls back to shared (host) memory,
with no guarantee of speed or stability. Actual use of that block (measured 2026-10-13, tray production config:
MTP on, vision tower on, prefill chunk 8192):

| Configuration | hgn v2 | UD-IQ4_XS + MTP | UD-IQ4_XS without MTP | hgn w4b |
| --- | --- | --- | --- | --- |
| 256K pool, 4 slots | 82.6 GiB | 83.9 GiB | 79.4 GiB | 88.8 GiB |
| 512K pool (256K per request), 2 slots | 92.2 GiB | 93.5 GiB | — | does not fit (6 GiB over) |
| 512K pool (256K per request), 8 slots | 93.8 GiB | just over (~70 MiB) | — | — |
| YaRN 512K (512K per request), 2 slots | 93.3 GiB | 94.5 GiB | — | — |

- **v2 goes up to 512K**: either two 256K requests resident at once (`KV_POOL_TOKENS=524288`), or one 512K request
  with YaRN (`MAX_CONTEXT=524288`, `ROPE_FACTOR=2`). 8 slots also fit, but with only ~1 GiB to spare; image-heavy
  multimodal requests allocate more at run time, so 2–4 slots are recommended at 512K.
- **UD-IQ4_XS with MTP uses ~1.3 GiB more than v2**; at a 512K pool use at most 2–4 slots.
- **w4b is limited to 256K.**
- The table was measured at prefill chunk 8192; the default 6144 uses about 1.7 GiB less in every row, with PP
  within ~0.5% of 8192. 16384 does not fit.
- The launcher refuses pools above 512K.
- UD-Q4_K_XL is 77 GiB resident; extrapolating from the table, 256K alone would need ~97 GiB, so it is not
  supported on Windows.
- Task Manager's "Dedicated GPU memory" shows the whole block reserved at startup: the use above plus up to 4 GiB
  of headroom (capped at 95 GiB), plus ~1.3 GiB, i.e. ~88.6 GiB for v2 at 256K and ~95.7 GiB for any 512K
  configuration. The engine also uses 3–4 GiB of host memory.

### Slots and VRAM

VRAM goes into the page pool, not into the slot count. Measured with hgn v2 at the default prefill chunk 6144
(2026-10-14, MTP on, vision tower on, `MAX_CONTEXT=262144`; "actual use" as in the table above, the other two
columns are Task Manager readings in GB):

| Pool | Slots | Actual use | Dedicated GPU memory | Shared GPU memory |
| --- | --- | --- | --- | --- |
| 256K | 1 | 79.7 GiB | 85.8 | 1.9 |
| 256K | 2 | 80.3 GiB | 86.3 | 1.9 |
| 512K | 1 | 89.8 GiB | 95.7 | 2.1 |
| 512K | 2 | 90.5 GiB | 95.7 | 2.5 |
| 512K | 4 | 91.0 GiB | 95.6 | 2.5 |

- Going from 1 to 2 slots adds ~0.6 GiB, each further slot ~0.27 GiB. Growing the pool from 256K to 512K adds
  ~10 GiB regardless of slots.
- When comparing one slot with two, keep `KV_POOL_TOKENS` the same; if the pool changed too, the ~10 GiB
  difference is the pool.
- All device memory is allocated at startup. With a 512K pool, 2 × 32K and 2 × 100K concurrent requests ran to
  completion without any of the numbers above growing and without spilling into shared memory.
- At a 512K pool, 1 slot and several slots show the same "Dedicated GPU memory": the reserved block is already at
  the 95 GiB cap, so the slot count does not show up there.
- To re-measure: stop the production service, then run `bash tools/vram_conc_win.sh` in Git Bash (default: 512K
  pool, 1/2/4 slots, ~5 minutes per configuration; `PARS="1 2" POOL=0 SHORT=1` measures a 256K pool, startup and
  one 8K request only). The summary goes to `logs/vram_conc/summary.txt`.

## Crash diagnostics

When the engine exits unexpectedly, the API bundles the logs, crash reports (with a small minidump on Windows), system and
driver information and the Windows event log into one diagnostic package under `logs/diag/`. On Windows the tray shows a
notification; click it to select the zip in Explorer. If nothing crashed but the service hangs or misbehaves, right-click
the tray icon → "Export diagnostics…". On Linux the start script collects the package automatically, or run
`build/qwenox-api --collect-diag`. Send the zip to the developers; it contains no request content. See
[CRASHDIAG.md](docs/CRASHDIAG.md) (in Chinese).

## Documentation

- [QUICKSTART_EN.md](docs/QUICKSTART_EN.md) — build, start, configure
- [BUILD_EN.md](docs/BUILD_EN.md) — build environment and troubleshooting
- [GGUF.md](docs/GGUF.md) — how GGUF weights are loaded, with measurements
- [CONVERT_EN.md](docs/CONVERT_EN.md) — model conversion tool
- [HGN-FORMAT_EN.md](docs/HGN-FORMAT_EN.md) — the `.hgn` weight container format
- [CONCURRENCY.md](docs/CONCURRENCY.md) — concurrent requests: configuration, scheduling, verification
- [YARN-512K.md](docs/YARN-512K.md) / [YARN-512K-RESULTS.md](docs/YARN-512K-RESULTS.md) — YaRN 512K and measurements
- [MTP_EN.md](docs/MTP_EN.md) / [NGRAM_EN.md](docs/NGRAM_EN.md) — speculative decoding
- [KLD.md](docs/KLD.md) — quality testing (same method as unsloth / llama.cpp)
- [BENCHMARK_EN.md](docs/BENCHMARK_EN.md) — benchmark tools
- [bench/README.md](bench/README.md) — task-level evaluation (long-context code-needle A/B, in Chinese)
- [PORTING-WINDOWS_EN.md](docs/PORTING-WINDOWS_EN.md) — Windows port and measurements
- [POWER.md](docs/POWER.md) — power/temperature reading accuracy by platform, and why per-CPU watts is unavailable on APUs (in Chinese)
- [CRASHDIAG.md](docs/CRASHDIAG.md) — crash reports, diagnostic packages, minidump symbolization, self-test (in Chinese)

Tests: `bash build.sh test` runs the kernel unit tests (no model loaded, expect ALL PASS);
`python3 tools/bpw.py` reports the bpw of each weight file under `models/`.

## Acknowledgements

The implementation approach draws on peonist-ai's [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server); the `.hgn` weight container format is halogen's checkpoint container format (see [HGN-FORMAT_EN.md](docs/HGN-FORMAT_EN.md)). Thanks to the halogen authors for their work.

GGUF support draws heavily on [gufo](https://github.com/gufo-org/gufo) (MIT license): the routed-expert F16 WMMA GEMM kernel is ported from its RoutedF16GEMMKernel (`src/gpu/parts/26_kernels_moe_gguf.inc`), the LUT-decode variant for hgn q4cp / GGUF IQ4 follows the same pipeline (`27_kernels_moe_lut.inc`), and the GGUF-to-engine tensor transform semantics follow its reference.cpp (`src/gguf_map.h`). Prefill optimizations such as HC gate fusion, producer epilogues writing the next GEMM's input directly, and the multi-row Q8 dense kernel also follow gufo's approach (comparison in [GUFO-GAP.md](docs/GUFO-GAP.md)).

The 4-bit prefill speedup (`MOE_A4`) draws on warpfront's [hipfire](https://github.com/warpfront/hipfire) (Apache-2.0 license): quantizing routed-expert activations to int4 per block and running the expert GEMMs on INT4 WMMA comes from hipfire's IU4 route, reimplemented here for the hgn v2 i4r format (`k_quant_a4` / `k_moe_iu4`, `src/gpu/parts/27_kernels_moe_lut.inc`); the packed epilogue of the LUT expert GEMM also follows hipfire's approach. The task evaluations shared by the hipfire author helped us weigh this switch. Thanks to the hipfire author.

The ngram speculative drafting idea and the two-level prompt cache draw on [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT license); GGUF weights and the vision tower (mmproj) reuse the same files as llama.cpp, and the IQ3_S grid table comes from llama.cpp. See the "Attribution" section of [NGRAM_EN.md](docs/NGRAM_EN.md) for the ngram notice.

## License

[AGPL-3.0](LICENSE)
