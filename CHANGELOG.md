# 更新日志

版本号记在根目录 `VERSION` 文件里，编译时写进程序。查看当前版本有三种方式：托盘菜单第一行、控制台总览页"版本"、`GET /health` 返回的 `version` 字段。引擎本身可以运行 `qwenox-engine-win.exe --version`。

## v0.1.0 — 2026-10-30

### 新功能

- **DFlash2 独立 sidecar**：不再要求同时加载 MTP 权重；selector codebook 和草稿矩阵支持 Q8_0 / F32 / F16，BF16 sidecar 保留逐位一致的 mmap 快路径；可用 `QWENOX_DRAFTER=dflash2` 选择草稿头。
- **QSA 检索预算可扩大到 4096 token**：`QSA_INDEXER_BUDGET=4096`，用于需要更大长上下文检索预算的场景；默认仍为 2048。
- **Windows 支持 WebP 图片输入**：静态链接固定版本的 libwebp 解码器，不增加运行 DLL。
- **控制台新增 Frutiger Aero 界面风格**。
- **Windows 处理器亲和性**：`AFFINITY` 支持双 CCD 上下半区或自定义 mask，设置面板和网页控制台均可配置。

### 修复

- **fork cut 异常后服务永久挂起**：失败请求现在完整清理 cut / prefill 状态，避免后续 GEN 被旧状态卡住；同时修复 Windows PLE `issue()` / `wait_all()` 与 `run()` 共用 IOCP 时的竞态（#41）。
- **Linux PLE 双 flight 并发争用**：序列化同一 io_uring ring 的提交与完成收割，避免读错行、prefill 永久挂起或进程中止（#38）。
- **投机验证跨并发和动态 γ 的确定性**：验证与拒绝前缀重放默认使用逐行不变的计算路径，避免 greedy / fixed-seed 输出随批大小改变；`QWENOX_SPEC_ROW_INVARIANT=0` 可恢复旧快速路径。
- 强制选择 gfx1151 APU，避免多适配器系统误选其他 GPU。

### 其他

- 补充不同负载下的并发解码实测与确定性开关文档。
- 版本号 0.1.0。

## v0.0.9 — 2026-10-23

### 新功能

- **崩溃诊断**：崩溃报告、Windows minidump、飞行记录、系统/驱动/事件日志诊断包；托盘可一键导出，详见 [CRASHDIAG.md](docs/CRASHDIAG.md)。
- **DFlash2 投机解码**：新增 drafter 2，支持 greedy 和采样，单槽。
- **KV 快照增强**：新会话在系统提示中段分叉时，下一次从分叉点恢复；RAM/SSD 按耗时选层、两段淘汰、暂停点落盘；网络盘文件多时不再卡推理。
- **PLE n-gram 大表可选 bf16 无损版**：dtype 0，引擎按行宽自动识别。
- **GGUF**：专家权重支持 IQ3_XXS / IQ2_S / Q2_0；dense 权重保留原格式（decode 不再读 q8g32 重量化）；新增 GGUF → i4r 专家补丁生成器和启动器开关 `GGUF_I4R_PATCH`；down f16 pairs 和 LUT prefill 合并 tile 默认开。
- **API**：新增 `/v1/tokenize`、`/v1/detokenize`；按请求生成 ZIP 日志（`REQLOG`）；用量统计改为全历史小时聚合；请求体上限提升到 128 MiB。
- **控制台/设置**：新增"构成主义"界面风格；用量柱状图悬浮显示当天请求/token/缓存命中；单请求上限新增 64K–512K 预设，超过 256K 自动开 YaRN。
- **Benchmark**：新增增量 prefill 测试段（KV 延续阶梯，`BENCH_INCREMENT_*`）；Summary 去重；正确转发 `MOE_A4`。
- **Windows**：设置面板/托盘改 PerMonitorV2 + 全量 DPI 缩放；新增 `release_win.sh` / `release_win.bat` 发行打包脚本。

### 性能

- **Windows 128K prefill**：PLE 两 chunk 预取（逐 bit 不变，稳态 +0.4%，6 s 级磁盘停顿从第 3 块起完全隐藏）+ paged KV 恒等页表快路径（逐 bit 不变，稳态 +0.86%，`qsa:flash` −5.5%）。
- Windows PLE gather 默认 8 路多泵（`QWENOX_PLE_RINGS` 可调）。
- decode 融合减少 kernel 启动；采样草稿接入合批草稿池（默认开）；采样 MTP 轮按草稿置信度逐轮定长（`QWENOX_SPEC_CONF`，默认 0.9）。
- GGUF prefill MoE 尾部优化；dense 原格式减少 decode 重量化；LUT 合并 tile 默认开。
- hipBLASLt 算法按 P 分桶（A1），改善 agent 短增量 prefill。
- KVSNAP 按耗时选层，减少快照开销。

### 修复

- Windows 崩溃诊断实机修正：访问冲突报告、C++ 异常原因、PDB 链接参数、engine 链接 advapi32。
- 客户端文本里的字面控制串按普通文本编码，避免 F1 特殊 token 注入。
- 未声明可调用工具的请求禁止输出 `<tool_call>`。
- 客户端断连立即中断 prompt 预填。
- 修复 GCC dangling-reference 误报；托盘版本与状态标题合并。
- 修正 kprof 的 PLE 归属（`ple_wait`/`ple_dev` 分段）。
- gfx1151 只采信整封装功耗，残桩传感器不再上报。

### 其他

- NPU 卸载移出主线存档到 `attic/`；Linux NPU 路径因 amdxdna 强制 IOMMU SVA 与 prefill 带宽需求冲突而封死。
- `QWENOX_GG_A4` 标记为废弃、禁止使用，仅留档。
- README 正文标点统一为中文全角；Linux prefill 实测数据更新为 hgn v2、`MOE_A4=3`。
- 版本号 0.0.9。

## v0.0.8 — 2026-10-15

### 新功能

- **Anthropic Messages 接口**：新增 `/v1/messages` 与 `/v1/messages/count_tokens`，Anthropic 官方 SDK 和走这套协议的客户端可以直接连。
- **支持 Unsloth UD-IQ4_XS GGUF**：纯 GGUF 权重可在 Windows 上启动，带 MTP 投机解码。
- **默认权重改为 V2 HGN**。
- **KV 快照目录可自选**：`KVSNAP_DIR`，设置面板里可以选，支持放在网络盘上。
- **运行时可视化**：控制台可打开实时视图 `/viz-live`，以及 3D 模型结构视图 `/viz-model`（随解码过程变化，带数据流箭头）。
- **控制台**：可切换界面风格（卡带 / 打孔机）；总览页新增温度卡（CPU / GPU / 热点 / SoC）和版本号。
- **托盘**：新增"开机自启"和"系统提示"两个开关；菜单第一行显示版本号。
- **prefill 4-bit 加速（MOE_A4）**：可选开关，默认关，在设置面板 V2 权重页勾选。长文本 prefill 更快，64K 以内的实测输出质量感知不到差别。
- **网页可在磁盘上覆盖**：项目根目录 `web/`（或 `WEB_DIR`）下有同名文件时优先使用，改前端页面不用重新编译。
- **版本号**：新增 `VERSION` 文件，`/health` 返回 `version` / `build`，引擎支持 `--version`。

### 性能

- 多路并发解码：多个请求同时生成时合批执行投机验证和草稿，并按当前并发路数选择 MTP 步长，并发吞吐明显提升。
- dense 层 WMMA 默认开启（`QWENOX_DENSE_WMMA=0` 可关）。
- 两路长 prompt 同时 prefill 时，剩余少的先跑，只为轻量任务让出算力；2×64K 的首字时间约 100 s → 87 s。
- v2 权重的 dense 反量化提速，MoE 采用混合 tiling（输出逐位不变）。
- Windows 上加载 GGUF 后，引擎常驻的主机内存从 8.0 GiB 降到 2.9 GiB。
- Windows 默认 prefill 分块从 8192 改为 6144。

### 修复

- Windows 并发服务崩溃（PLE 预读与解码共用同一个完成端口，互相抢完成包）。
- 投机验证的 graph 捕获后，事件句柄悬空导致的随机崩溃。
- 并发时输出与单独运行的结果不一致。
- 两路长 prefill 同时进行时 PLE 预取串号。
- API 比引擎先启动时，首个请求崩溃，以及并发槽位恒为 1。
- 偶发"只有思维链、没有正文"的空回复。
- 设置面板：切换权重版本时提示文字叠在一起；脚本转换出的 MTP 头被误判为错误文件；显存估算按 Windows 实测校准。
- 托盘探测引擎是否存活时不再刷屏引擎日志。
- 控制台动效不再被 Windows 的"减少动画效果"设置关掉。
- 删除从未实现、调用恒返回 500 的 `/cache` 端点。

### 其他

- 项目统一更名为 Qwenox；Linux 下引擎产物名为 `build/qwenox-engine`。

## v0.0.7 及更早

见 git 历史（tag `v0.0.1` – `v0.0.7-fix-2`）。
