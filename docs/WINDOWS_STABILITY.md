# Windows 稳定性候选版本

## 排查结论与限制

此版本针对 FAEX1 / Ryzen AI MAX+ 395 / Radeon 8060S 的整机冻结报告加入防护。
五份诊断包跨越 0.0.9 和 0.1.0；不能把它们当作同一构建的五次复现。
11 月 3 日的在线诊断中，一个 GPU 工作线程最后记录了 `hipStreamSynchronize`，
约 81 秒后仍未看到后续调用（旧日志不记录返回时刻）；健康检查线程每两秒继续记录 `hipSetDevice`。
这支持“GPU 路径停滞”的排查方向，但不能区分内核/驱动阻塞、GPU 工作未完成等原因。

系统日志有非正常重启和 LiveKernelEvent 193。部分报告重复引用更早的 dump，
不能将 WER 上报时间直接当作此次冻结发生时间。
微软说明 [0x193 是 DirectX 图形内核触发的 live dump](https://learn.microsoft.com/en-us/windows-hardware/drivers/debugger/bug-check-0x193--video-dxgkrnl-livedump)，
[事件 41 本身不能确定异常重启的根因](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/event-id-41-restart)。
原始诊断包、进程 dump、账号和机器地址不随源码发布。

## 防护行为

- 健康检查、取消和在线诊断的连接不再无条件进入 HIP。`MEM` 仍查询驱动，不能用它充当无阻塞健康检查。
- 模型加载完成、进入服务模式时启用独立 CPU 看门狗。Windows 默认 120 秒，Linux 默认关闭。
- 跟踪实际 HIP 拷贝、异步拷贝和 stream/device/event 同步调用；空闲、排队和长请求本身不触发。
- 每个线程独立计时；其他请求或健康检查的活动不会掩盖卡住的调用。
- 超时尝试保存 `hip-timeout` 报告，退出码 124。诊断尝试另有 5 秒退出兜底，不调用 HIP reset/free/synchronize，不自动重新加载模型。
- `QWENOX_HIP_TIMEOUT=0` 关闭；30–3600 设置秒数。这是引擎环境变量，不是 `STALL_TIMEOUT`（后者用于启动加载检查）。
- 不覆盖所有 HIP 入口、CPU 死锁或内核冻结；Windows 已无法调度用户线程时，看门狗也无法执行。不能保证阻止需要断电恢复的整机冻结。

## 临时低压力入口

退出原 Qwenox 托盘程序后，双击完整 Windows 包中的 `start_win_stability.bat`，再从托盘启动引擎。
该入口保留模型路径，只为这一轮 API/引擎进程覆盖以下值：

| 配置 | 值 |
|---|---:|
| MAX_CONTEXT / KV_POOL_TOKENS | 65536 |
| PARALLEL | 1 |
| PREFILL_CHUNK / CONC_PREFILL_CHUNK | 1024 |
| KVSNAP_MAX_GB | 0 |
| RCKPT_MAX | 2 |
| MAX_IMAGES | 1 |
| QWENOX_HIP_TIMEOUT | 120 |

代价是上下文缩短、并发降低、SSD 快照复用关闭；未声称速度提升或已证实可消除冻结。
它不减少模型权重本身的占用，不删除现有快照，不修改系统 TDR、BIOS、驱动或 `service.conf`。
退出托盘后重新使用 `start_win.exe` 即恢复原配置。若已有 API 运行，脚本拒绝启动，避免复用旧环境。

## 验证

`python tools/hip_watchdog_test.py --cxx <C++17 compiler>` 编译实际拷贝/同步包装器，
用模拟阻塞 HIP 调用验证超时，覆盖 profiling 开/关两种模式。
另运行 `tools/ple_win_iocp_test.py`、`tools/kv_admission_test.py`、`tools/prefill_cancel_test.py`。

CPU 测试不代表 GPU 内核、模型输出、吞吐量或数小时整机稳定性验证。
在具备 TheRock、MSVC/Windows SDK 和模型的环境中，仍需完整 Windows 构建及目标硬件验证。
先用低压力入口验证短请求与正常退出，再逐步恢复上下文、并发、快照，每轮仅改变一项。
出现再次冻结或退出码 124 后保留新诊断包，再分析相应 Windows WATCHDOG dump；不要把重启循环当作恢复策略。

本次主机验证：Windows CPU 故障注入 20 项、PLE IOCP 17 项、KV admission 与 prefill cancel 回归均通过；修改后的 diagpack.cpp 已单独编译。完整 GPU 引擎尚未编译/实机验证，本机缺少完整 TheRock 开发库及 MSVC/Windows SDK。

## FAEX1 在线基线（2026-11-09）

已在目标 FAEX1 上通过 HTTP 实测现有 **v0.1.0 / 91b111a**，不是候选引擎。
原配置为双槽、262144 上下文、524288 KV 池、4096 prefill，SSD 快照关闭，MOE_A4=0。
测试未修改配置或重启进程，结束后确认配置相同、同一引擎进程仍运行且请求队列为空。

- 10 次串行短请求通过，第一轮请求总耗时约 0.43–0.62 秒。
- 流式响应完整结束，首段正文约 0.10 秒。
- 输入 1558 / 6166 / 24598 token 的请求均成功，第一轮总耗时约 1.16 / 4.11 / 11.27 秒。长输入包含重复文本及前缀复用；后续轮次大量命中缓存，不能当作独立冷启动 prefill 吞吐量基准。
- 两路并发短请求均完成；中断流式连接后恢复空闲，下一条请求成功。
- 额外进行 180 秒、每轮间隔 3 秒的混合流式/非流式测试，共 40 条请求，全部通过。
- 测试前后诊断包与原始结果仅保存在测试端，没有提交用户日志或进程 dump。

这轮没有复现冻结，但覆盖的是短期、低占空比、合成文本负载。它不验证数小时稳定性、
完整 256K 上下文、多模态、快照开启、不同驱动或候选版看门狗的实机表现。

可重复运行（在空闲服务上执行；会生成真实推理请求）：

```text
python tools/windows_live_stability.py --base http://HOST:8731 --output logs/live-validation --soak-seconds 180 --pause-seconds 3
```

脚本不更改服务配置，异常即停止后续测试；生成上限为 64 token，取消用例上限 512。
输出目录包含服务信息，应保留在本地。`tools/windows_build_probe.ps1` 可在目标 Windows
上生成只读编译环境/硬件清单，帮助准备候选引擎构建，不安装工具或调整系统设置。
