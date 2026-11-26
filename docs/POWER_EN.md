# Power Management and the Decode Cold-Start (DVFS Ramp)

> Root-caused on gfx1151 (Strix Halo APU, Windows 11), 2026-11-26.
> TL;DR: **the ~10% decode slowdown during the first ~35 s after an engine
> restart or a few idle minutes is a platform power-management frequency
> ramp (DVFS), not a code regression, not cold PLE pages, not missing
> warmup. Engine-side warm-up load cannot bypass it (measured).
> The correct knob lives in the OS.**

## Symptoms and elimination

- Same engine, same seeded request, back-to-back: decode 34 → 34.5 → 34.9 →
  **37.1 plateau** after ~35 s of sustained load; back to ~33 after ~4 idle minutes.
- **Not PLE page cache**: the 2nd identical request (its n-gram rows already
  cached) is not faster, and performance drops again after a 4-minute idle
  (cache would not expire).
- **Not warm-up load**: after 13 s of ~1900 tok/s prefill (near-peak GEMM),
  decode is still ~33. An engine warm pass does not fix this.
- **Confirmed power management**: switching to the High performance power
  scheme makes the first request run at 37.2 tok/s immediately, even after
  4 idle minutes. No ramp.

## Per-platform knobs

### Windows

```bat
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   :: High performance
powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e   :: back to Balanced
```

Recommended for dedicated inference boxes. Cost: slightly higher idle power
and fan noise.

### Linux

```bash
# CPU: amd_pstate active mode + performance governor
echo active | sudo tee /sys/devices/system/cpu/amd_pstate/status
for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance | sudo tee $g; done
# GPU: pin highest DPM level (amdgpu, applies to Strix Halo)
echo high | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level
```

`power_dpm_force_performance_level=high` pins the GPU clocks and is the
direct Linux equivalent of the Windows High performance scheme (even
stronger). Use `auto` to restore.

**Shortcut (Windows)**: the web launcher's header has a one-click
"Power → Set High Performance" button (hidden when already on High
performance or on Linux; uses the PowrProf API, no admin rights needed).

## Secondary: PLE cold faults after an engine restart

Unrelated to the power ramp but in the same time window: after a **process
restart** (not after idle), the PLE n-gram table (47.7 GiB, mmap'd) has a
cold page cache, and the first ~1-2K decoded tokens see ~500-2000 hard
faults/s. On NVMe this costs a few percent and heals itself quickly; it is
much smaller than the power ramp. Keep the table on the fastest disk
(it is on NVMe here). A startup pre-fault is only worth it if the first
few seconds of the first request after restart matter to you.

## Anti-patterns

- **No engine "warm decode at startup" against the ramp**: 13 s of heavy
  prefill was measured ineffective — the ramp is driven by power-policy
  time/idle, not by utilization.
- **No keep-alive heartbeat load**: it steals bandwidth and power while
  hiding the real knob.
