# 投机解码优化与 halogen 对标调查（2026-11-09）

本文记录 2026-11-09 针对 halogen-flash-server（peonist-ai，闭源）的一轮对标调查与
随之而来的 Qwenox 投机解码优化。全部数字出自同一台 GMK EVO-X2（Ryzen AI Max+ 395，
gfx1151，Windows），同一份 hgn v2 权重，可复现（见末节）。

## 1. halogen 的解码为什么快——逆向调查结论

halogen 引擎以闭源容器镜像分发（`ghcr.io/peonist-ai/halogen-flash-server`，约 0.94 GB，
不含权重），其 EULA 明确允许逆向与基准复测。镜像拆解（引擎 `flash_serve`，ELF 内嵌
gfx1151 GPU 代码对象，1789 个内核符号完整可见）加上官方文档，结论是其解码速度由
三层叠加，没有魔法：

1. **串行基线不离谱**：官方自测串行 greedy 只有 37.6 tok/s（1.5k 上下文）/ 34.1（32k）。
   4.16 bpw 量化 + MoE 每 token 只读激活专家，在 Strix Halo 的带宽上这个数合理。
2. **多路投机解码是倍率主体**：MTP 草稿头（置信度门控链长，默认 GATE_P=0.90，上限 7）
   + prompt lookup（请求自身文本的 n-gram 接续，默认 3,3，接在头草稿链尾、同轮联合验证）
   + grammar 起草（JSON schema 编译为 GPU token mask）。temperature 0 时输出与串行
   greedy 逐字节一致——投机是纯加速，不是质量权衡。同一构建按文本类型差 2 倍
   （散文 ~54 tok/s，代码 ~73，极限重复 ~94）。
3. **单芯片专用内核**：全部内核为 gfx1151 手写模板（I4R/HT 自定义量化格式的 MoE 专家
   内核、warp 亲和分组路由、稀疏注意力 indexer），无任何可移植层开销，只支持这一款芯片。

## 2. 本仓库已合并的优化（main, merge 0799534）

| 改动 | 开关 | 默认 | 说明 |
|---|---|---|---|
| greedy 草稿置信度门控 | `QWENOX_GREEDY_CONF=θ` | 0（关） | halogen MTP_GATE_P 对应物：每步草稿后在 GPU 上算草稿分布 top-1 概率（新内核 `k_top1_prob`），跌破 θ 收链。实测 θ=0.9 接受率升至 73% 但总速反降（本机 verify 轮固定成本高，Γ 已收敛到最优），保留备用 |
| 联合起草 | `QWENOX_JOINT` | **开** | halogen PLD 语义：ngram 轮首未命中走 MTP 轮时，用扩展历史（chist + cur + 草稿）再查一次 ngram 表，接续接在链尾同轮 verify。输出与串行逐字节一致。实测 4 场景加权 +1.9%，桥接回重复区的场景 +2.6~2.8 tok/s |
| bench 自适应 γ 档 | `MTP_GAMMA>8` | — | bench 专用：`MTP_GAMMA=99` 走 GammaCtl 自适应（配置解析不收负数，用 >8 代替） |
| 串行测量 | `BENCH_SERIAL=1` | 关 | 纯 AR 循环 tok/s；配 `QWENOX_PERF=1` 出 embed/ple/attn/moe/head 五段 ms/token |
| chain 路径基准 + 新场景 | `BENCH_CHAIN=1` | 关 | bench 走 serve 同款 `spec_loop_chain`（ngram+MTP+joint 全栈），并追加 repeat-edit 与中文三场景（zh-code/zh-prose/zh-repeat） |

**不要设 `QWENOX_DRAFT_VOCAB`（草稿词表前缀截断）用于中文场景。** 它把草稿 argmax 限制在
lm_head 前 N 行，中文 token 在词表高位（Qwen 词表 CJK 在 10 万+ id），中文草稿必然被拒：
zh-prose 接受率从 41.9% 崩到 17.7%，八场景加权从 61.4 掉到 46.8 tok/s。halogen 的对应机制
（draft-vocab.lists）实为 198759/248320 ≈ 80% 全词表，不是激进截断。英文-only 部署想用可以
（约 +3.6%），多语言勿用。

## 3. 单流实测矩阵（本机，hgn v2，greedy，192 tok × 2 采样，chain 路径）

| 场景 | tok/s | 接受率 | halogen 官方（Linux） |
|---|---|---|---|
| python-code | 73.5 | 79.4% | 73（写代码） |
| zh-code | 80.4 | 84.4% | — |
| repeat-edit（174 tok prompt） | 77.7 | 94.3% | 94（极限重复） |
| zh-repeat | 83.6 | 90.8% | — |
| creative-writing | 44.9 | 48.7% | 54（散文） |
| zh-prose | 41.5 | 41.9% | — |
| common-sense-qa | 61.7 | 60.3% | — |
| **八场景加权** | **61.7** | 67.7% | — |

对照出厂基准默认（固定 γ4、spec_loop、英文三场景）：48.3 → 61.7，**+28%**。

其余口径：prefill 加权 ~1550-1650 tok/s（halogen 1821 @8K，约 -10%）；纯串行 33.5 tok/s
（halogen 37.6，-11%）。注意双方数字各自出于 Linux/Windows 与不同 prompt 集，是同硬件
同权重的对照，不是同台对垒。

### 串行底子的带宽归因（QWENOX_PERF 分段 + hgn 头部字节账）

每 token 必读 3.67 GB（dense 2.13 + 激活专家 10×2.53MB×48 层 1.22 + lm_head 0.32）。
实测各段：head 196 GB/s（lm_head 单发大 GEMV，证明 fabric 能扛 ~200）＞ attn 107 ＞
**moe 86 GB/s（最差，48 层 × rot→gate_up→silu→down→rot 五段链 + 随机专家离散读）**。
decode 图捕获本身值 ~6.5 ms/token（直发 36.3 ms → 图 29.9 ms）。
33.5→37.6 的差距约 12%，其中 ~7 个百分点是 Windows/Linux 驱动栈（同引擎 Linux 34 /
Windows 31.4），内核层可追的余量约 5 个百分点（MoE 五段链融合方向）。

## 4. 并发解码调查（PARALLEL=4，KV 池 524288，生产采样参数）

| 并发 | 单流 tok/s | 聚合 tok/s |
|---|---|---|
| 1 | 51.9 | 51.9 |
| 2 | 34.3 | 68.5 |
| 3 | 25.7 | 72.5 |
| 4 | 21.4 | 82.4（见顶） |
| 8（排队） | 15.4 | 81.5 |

**会合机制本身是健康的**：4 路并发下 `[vr] batches=85, multi=85, solo=0`（85/85 轮全部
跨槽合批，69 轮满 4 段），OccGamma 自选 γ≈3~4，逐深度接受率 0.76~0.95。

**瓶颈是会合锁步**：roundprof 显示每轮 vpool 等待 105~135 ms、draft（含草稿会合等待）
18~37 ms、yield 3~8 ms、commit ~2 ms。单路一轮 ~67 ms，四路联合一轮 ~150 ms 干 4 倍
的活——批量验证本身划算（共享权重读取），但每轮两个会合点 + GPU 轮转权串行化的
commit/yield 凑出约 65 ms/轮的同步开销。

旋钮 A/B 全部探平（每项重启实测）：`QWENOX_OCC_FORCE=5` → 78.7（γ 不是瓶颈）；
`QWENOX_NO_DR=1` → 73.6（草稿会合是对的）；`VR_WAIT_MS=15 + DR_WAIT_MS=10` → 78.6
（等待几乎不因超时，是到达错峰）。

### 后续结构方案（按切口大小排序，未实施）

1. **消掉两会合之间持轮转权做纯 host 工作的窗口**：worker 从 dr_pool 出来要重新
   acquire 轮转权，然后握着它拼 verify 向量（纯 CPU 工作）再进 vseg_pool。把 vt 构造
   移出持权窗口，4 路每轮可省约 10~20 ms 串行段。切口最小，先做。
2. **单会合化**：草稿+验证合并为一个会合点，同步点减半。改动大，需重设计 turn 纪律。
3. **草稿预跑流水线**：验证执行者跑第 N 轮验证时顺手跑各槽第 N+1 轮首步草稿
   （验证结果已在其手里），worker 醒来直接拿草稿。收益最大，复杂度最高。

## 5. 复现

```bash
# 单流八场景（chain 全栈 + 自适应 γ；联合起草默认开）
set MTP_GAMMA=99 && set BENCH_CHAIN=1 && set BENCH_DECODE_TOKENS=192
set BENCH_INCREMENT=0 && set BENCH_PREFILL_REPEATS=1
build\qwenox-bench.exe run

# 串行底子 + 分段计时
set BENCH_SERIAL=1 && set QWENOX_PERF=1
# 并发：serve（PARALLEL=4）+ 并发客户端；roundprof 用 QWENOX_SYNC_PROF=1
```

调查素材（本 workspace 之外的参考）：halogen 镜像 rootfs 与 gfx1151 GPU 代码对象、
其 `docs/FLAGS.md`（全部投机参数的默认值与语义）在对标调查机上有完整留存。

## 6. halogen 反汇编深读（gfx1151 代码对象，2026-11-09 下午）

用 TheRock 的 llvm-objdump 反汇编了 halogen 镜像里的 gfx1151 GPU 代码对象
（.hip_fatbin 内的 amdgcn ELF，21 MB .text，1789 个内核全部可读）。硬事实：

1. **融合巨型内核**：`q4moe::k_q4moe<...>` 一族单个内核 100~270 KB（约 4 万条
   指令），整个 MoE 专家通道（gate_up→silu→down）融合为一次启动。本引擎同路径
   是 5 个串行小内核 ×48 层。
2. **张量核打带宽仗**：98 个内核含 WMMA（`v_wmma_f32_16x16x16_bf16`）——融合
   MoE 族 31 个、ht dense GEMV 族 56 个。融合内核内部是标准 GEMM 分级管线：
   global（128 位宽加载）→ LDS → WMMA。带宽受限下张量核的价值是发射压力低、
   在途加载多。
3. **VOPD 双发射**：全二进制 357,574 条 `v_dual_*` 指令（gfx1151 的双发射
   FP32 管道），编译器深度展开所有可并行的 FP32 对。
4. **不用 HIP Graph**：`flash_serve` 无 hipGraph 符号，纯 hipLaunchKernel +
   stream/event。Linux 启动延迟低 + 单内核更大更少，不需要图捕获（本引擎在
   Windows 靠 decode 图填平 host 发射开销，实测值 ~6.5 ms/token）。
5. verify 多行走**调优过的 hipBLASLt GEMM**（`flash-tune.plan` 为片上实测调优；
   日志明写 "down experts need the GEMM path (hipBLASLt) at n > 1"）；接受/残差
   采样全程 GPU 内核（`k_accept_prob_f32` / `k_resid_sample_f32`），零 host 往返。

**散文 54 tok/s 的分解**：= 串行 37.6 × 投机倍率 1.436。倍率端靠低轮成本
（GEMM 级 verify + GPU 端接受）+ `SPEC_ADAPT` 低接受率窗口自动关草稿头跑串行；
串行端靠融合 WMMA 内核。**可追的单点：本引擎 decode/verify（P≤48）的专家路径
仍是标量 FMA GEMV（`k_q4cp_gemv_gg`），而 prefill 已有 i4r LUT WMMA 路径——把
验证轮的 MoE 路由到 WMMA 同时改善串行底子和投机轮成本。**（未开工，2026-11-09
记录为下一步候选）

## 7. gufo 口径核查（营销数字 vs 其自身文档）

gufo（gufo-org/gufo，MIT）头条 "Flash-Next 最高 116.93 tg / 245.81 聚合 8 用户"
出自其 `docs/models/qwen3.8-flash-next/BENCHMARKS.md` 的长上下文 sweep：**prompt
是 20-token 模式的无限重复，模型逐字续写，MTP 接受率 ~100%**，其文档自己注明
"the MTP column is an upper bound for repeated text, not chat or code"。同文档的
诚实口径：聊天散文 MTP greedy **44.5~45.2 tok/s**（67% 接受）、串行 AR 35.2、
现实并发（gufo-tp 文档，2K+128）4 路聚合 ~86。**三方现实文本口径对比：串行
33.5(Qwenox)/35.2(gufo)/37.6(halogen)；散文投机 44.9/44.5-45.2/54；代码
73.5-80.4(Qwenox 最高）；4 路聚合 82.4/~86/未公布。** 结论：剔除复读机口径后
三方同档，halogen 的真实剩余优势是散文的串行底子（大半为 Linux/Windows 驱动差）。
另注：gufo 的草稿词表是 "Latin draft vocabulary"——与本引擎 §2 记录的
DRAFT_VOCAB 中文翻车是同一类偏置。
