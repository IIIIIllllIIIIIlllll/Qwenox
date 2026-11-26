# 功耗与温度读数：哪些可信、为什么

控制台"功耗""温度"卡片的数据源是 `src/api/power.cpp`。目标平台只有 gfx1151
（Strix Halo），本文记录实测结论，解释为什么只显示整封装功耗、为什么单独的
CPU 功耗拿不到。

## TL;DR（gfx1151, RYZEN AI MAX+ 395 / Radeon 8060S）

| 读数 | 状态 |
|---|---|
| 整机封装 (ASIC_P) | ✅ 真实，唯一可信的功耗 |
| GPU 功耗 (GFX_P) | ❌ 残桩（GPU 满载 99% 也只报 11W），不上报 |
| CPU 功耗 (CPU_P) | ❌ 残桩（CPU 满载只报 3W），不上报 |
| SoC 功耗 | ❌ 无此传感器 |
| GPU / SoC 温度 | ✅ |
| CPU 温度 | ❌ 恒 0，隐藏 |

**单独的 CPU 功耗对纯用户态不可得**（原因见文末），软件只显示能确认真实的读数，
残桩一律隐藏，而不是展示恒为 0 的假数据。

## 读取路径

- **Windows**: ADL PMLog（运行时装载 `atiadlxx.dll`，调 `ADL2_New_QueryPMLogData_Get`）。
  这是 APU/核显上唯一可用的 AMD 官方用户态通道；OverdriveN/5 在 APU 上读不到功耗。
- **Linux**: amdgpu hwmon sysfs `power1_average`（µW）+ `tempN_input`（按 label 认
  edge/junction），CPU 温度走 k10temp（缺则 zenpower）的 Tctl/Tdie。

读数是传感器瞬时值，纯驱动/sysfs 调用，不影响推理。读不到的字段 API 返回 `null`，
前端自动隐藏该行。

## 实测证据（Windows，ADL PMLog）

测试方法：`tools/npuprobe/pmlog.cpp`（`pmlog dump` 打印全部支持的传感器），
三组对照——空闲 / 16 线程 CPU 满载 / GPU 满载（hipMemset 循环，ACT_GFX 99–100%）。

### gfx1151 (RYZEN AI MAX+ 395)

| 场景 | ASIC_P(23) | GFX_P(30) | CPU_P(33) |
|---|---|---|---|
| 空闲 | 17 W | 1 W | 0 W |
| CPU 满载 | 72 W | 1 W | 3 W |
| GPU 满载 | 79 W | 11 W | 0 W |

- 只有 ASIC_P 随负载真实变化，且数值合理（空闲 ~17W，满载 ~75W）。
- GFX_P(30) 是残桩：GPU 满载也只报 11W；此平台无电压/电流传感器（21/31）可交叉验证。
- CPU_P(33) 是残桩：CPU 满载只报 3W。
- 传感器 71/72 虽报 supported，但在 ADL 头文件里是 `PMLOG_RESERVED_13/14`（保留位），无意义。
- T_CPU(32) 恒 0；T_GFX(28)/T_SOC(29) 正常。

### 佐证：8745HS (Radeon 780M，非目标平台)

| 场景 | ASIC_P(23) | GFX_P(30) | GFX_V(21)×GFX_I(31) | CPU_P(33) |
|---|---|---|---|---|
| 空闲 | 10–22 W | 18–25 W | 1.26V×20A = 25.2 W | 2 W |
| CPU 满载（GPU 闲置） | 44 W | 36–48 W | 1.26V×33A = 41.5 W | 4 W |
| GPU 满载 | 49 W | 39 W | 0.97V×41A ≈ 39 W | 3 W |

- 这颗芯片上 GFX_P(30) **逐点等于 GFX_V×GFX_I**，是真实传感器——但纯 CPU 满载
  （GPU 800MHz/6% 活动）时它也冲到 40W+：CPU 核与 GPU 共用一条供电轨，
  传感器 30 是"核心轨功耗（CPU+GPU 合计）"，不是 GPU 单独功耗。
- CPU_P(33) 同样是残桩（满载 ≤6W）。
- 结论：即使在传感器"真实"的 APU 上，ADL 也给不出 CPU 或 GPU 的单独功耗。

## 代码行为

`power.cpp` 只读 gfx1151 上确认真实的传感器：

1. **整机封装**：ASIC_P(23)，回退 SSPAIRED_ASICPOWER(46)、BOARD_POWER(73)。
2. **温度**：T_GFX(28)（回退 EDGE(8)）、HOTSPOT(27)、T_CPU(32)、T_SOC(29)，
   只收 1–150 °C 区间的值，恒 0 的坏传感器（gfx1151 的 T_CPU）自动隐藏。
3. **GFX_P(30)/CPU_P(33) 不读**：gfx1151 上是残桩。接口里的 `gfx_watts`/`cpu_watts`
   恒为 -1，前端隐藏对应行。

AMD 驱动更新后如果行为变了，重跑一次 `pmlog dump`（空闲/CPU 满载/GPU 满载三组）
即可重新判定要不要放开某个传感器。

## 为什么不做单独的 CPU 功耗

CPU 真实功耗的唯一硬件来源是 SMU（电源管理协处理器）或每核能量 MSR
（`0xC001029A` 每核 / `0xC001029B` 整封装，Zen 能量单位 2⁻¹⁶ J），
**两条路都必须在内核态（ring 0）访问**。Windows 上的可行性与代价：

| 途径 | 结论 |
|---|---|
| ADL / ADLX | 本项目现用；APU 上无 CPU 功耗（ADLX 只有 GPU 指标） |
| WMI 电源计 / EMI 能量计数器 | 桌面/迷你机不存在或恒 0（电池设备，接交流电） |
| SRUM / E3 能量估算 | 按模型估算的每进程能耗，非实时功率，精度不够 |
| Ryzen Master 驱动 (`AMDRyzenMasterDriverV32`) | 设备节点存在但 ACL 拒绝非管理员；IOCTL 接口私有且按版本演进，脆弱 |
| WinRing0 / inpoutx64（RyzenAdj、旧版 LibreHardwareMonitor） | 已在微软易受攻击驱动黑名单上，开内核隔离（HVCI）的系统拒绝加载 |
| PawnIO（新版 LibreHardwareMonitor/SMUDebugTool） | 可行：签名驱动 + 脚本化 RDMSR/SMN，不在黑名单；需管理员装一次 |
| 自写 KMDF 驱动 | 技术上几十行（RDMSR 差分即可），但需 EV 证书 + 微软 Attestation 签名 |
| HWiNFO64 / AIDA64 共享内存 | 可行：它们自带驱动读 SMU，发布到共享内存，读取方免权限；但依赖第三方软件常驻 |

综上：单独的 CPU 功耗在 APU 上对纯用户态不可得，本项目的取舍是
"只显示确认真实的读数，残桩一律隐藏"，而不是展示恒为 0 的假数据。

## 复测工具

`tools/npuprobe/pmlog.cpp`（Windows，只依赖 KERNEL32/USER32，拷到任何装了
AMD 驱动的机器即可跑）：

```bat
pmlog.exe dump                     :: 打印全部支持的传感器（index=数值）
pmlog.exe watch 60 pmlog.log       :: 每 250ms 采样时钟/活动率/功耗到日志
```

判定流程：空闲 dump 一次 → CPU 满载（PowerShell `1..16 | % { Start-Job { while($true){} } }`）
dump 两次 → GPU 满载（跑推理或 hipMemset 循环）dump 两次，看哪些传感器跟随哪类负载。

---

## 附：解码冷启动归因——DVFS 频率爬升（2026-11-26 实测）

引擎重启或空闲数分钟后，前 ~35 秒 decode 慢 ~10%（34 → 37.1 平台期），
根因是**平台电源管理的频率爬升**，与代码、PLE 页缓存、预热缺失均无关：

- **排除 PLE 冷页**：同一 seeded prompt 的第 2 个请求（其 n-gram 行已在页
  缓存）并不变快；且 4 分钟空闲（缓存不失效）后性能重新回落。
- **排除"重负载热身"**：13 秒 ~1900 tok/s prefill（接近峰值 GEMM）之后，
  decode 依然只有 ~33。**引擎启动跑 warm pass 解决不了它。**
- **坐实电源管理**：`powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`
  （高性能）后，闲置 4 分钟首个请求立即 37.2 tok/s，无 ramp。

各平台旋钮：Windows 用上面的高性能 GUID（`381b4222-f694-41f0-9685-ff5bb260df2e`
回平衡）；Linux 用 `amd_pstate=active` + performance governor，并用
`echo high > /sys/class/drm/card*/device/power_dpm_force_performance_level`
直接锁 GPU DPM（效果更强的等价旋钮，`auto` 恢复）。

次要项：**进程重启后**（非空闲后）PLE 表（47.7 GiB mmap）页缓存为空，前
~1-2K decode token 伴随 ~500-2000 次/秒硬缺页，NVMe 上代价几个百分点、
自愈快，量级远小于电源 ramp；表放最快的盘即可（本机已在 NVMe）。

反模式：不要做引擎内"启动 warm decode"对抗 ramp（实测无效，ramp 由电源
策略的时间/空闲驱动而非利用率）；不要 keep-alive 心跳负载（偷带宽费电）。
