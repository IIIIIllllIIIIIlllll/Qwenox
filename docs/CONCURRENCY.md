# 并发请求（PARALLEL）

面向轻负载的并发：几条短任务（知识问答、小改写）同时进来时不必排队等前一条生成完。
它不是批量 decode：GPU 同一时刻只算一条序列，各序列按轮转交替执行，所以总吞吐基本
不变（单条 decode 约 31 tok/s @7K），收益是后来的请求不用等先来的请求全部结束。

## 配置

`service.conf`：

| 项 | 默认 | 说明 |
|---|---|---|
| `PARALLEL` | 4 | 同时运行的请求数，1–8。1 = 旧的单路行为 |
| `KV_PAGED` | 1 | 必须为 1（`PARALLEL>1` 时启动器会拒绝 0） |
| `KV_POOL_TOKENS` | 0 | 共享 KV 页池大小，取 `max(KV_POOL_TOKENS, MAX_CONTEXT)`；默认即一条 256K |

引擎读 `QWENOX_PARALLEL`，API 启动时从引擎 `INFO` 的 `kv_slots` 字段得知路数并开同样
多的引擎连接。每多一路约多占 0.27 GiB 显存（Windows 实测：512K 页池下 2 路 → 8 路多 1.6 GiB，v2 与
UD-IQ4_XS 相同），主要是 GDN 状态（113 MB）、会合 verify 的行窗口备份（48.4 MB）和 MTP 流状态。
MTP 层的 KV/indexer keys 是共享页池里的一层（B0），和主干 KV 一样按页取用，不再每路按
MAX_CONTEXT 预分配（旧版每路多 0.5 GiB（BF16）/ 1 GiB（FP32）；PARALLEL=4 实测省 1.63 GiB）。
rckpt 检查点连同 MTP 层的页一起钉住，恢复后继续 MTP 投机（`tools/b0_verify.sh` 验证）。

## 语义

- **共享上下文**：所有槽位从同一个 KV 页池（256 token 一页）按需取页，类似 llama.cpp
  默认的 unified KV。例如 256K 池可以是 4 条 64K，也可以是 1 条 200K 加 3 条短的。
- **调度**：每个连接一个读线程（`PING/INFO/X` 随时响应），每个 `GEN` 一个工作线程；
  FIFO 票据锁拥有 GPU，持有者在让出点（spec 轮开头、prefill 分块边界、串行 decode
  每个 token）发现有人排队就交出。
- **与单跑是否一致**：v2 hgn 上，每条序列的 greedy 输出与它单独运行逐位一致；带 seed
  的采样请求（API 显式传 `seed`，引擎协议 `SEEDED`）同样一致。没给 seed 的采样请求本来就
  随机，不保证。GGUF 上 greedy 在近似平局处偶尔会不同（Windows UD-IQ4_XS 实测：10 个请求里
  有 1 个，在第 27 个 token 上“are:”与“are G…”的差别，大约一半的并发运行出现）：并发时
  MTP 投机的 γ 按同时在跑的人数选（OccGamma），而 GGUF 路径的输出对 γ 不严格不变。需要
  严格一致时设 `QWENOX_OCC_FORCE=4`，greedy 的 γ 在任何并发下都固定为 4（吞吐会低一些）。
- **会合 verify 的行窗口备份**：多条序列的 MTP verify 合成一批跑完后，GPU 会先交出去，
  各序列再依次回来提交 / 回滚，期间别的请求（prefill、ngram verify、单段 verify）可能已经
  改写了共享缓冲里它们那几行。引擎在交出前按槽备份这些行，回来时先拷回（每槽 48.4 MB，
  `QWENOX_VR_RESTORE=0` 可关，关掉后并发输出会与单跑分叉）。`QWENOX_VR_CHECK=1` 调试用：
  两边算哈希，被改写时日志打 `[vr-check] ... CLOBBERED`。
- **并发时的 prefill 分段**：PARALLEL>1 时 prompt 按 min(`CONC_PREFILL_CHUNK`, 单路分段) 分段
  （默认 8192，引擎变量 `QWENOX_CONC_PREFILL_CHUNK`；单路分段 = `PREFILL_CHUNK`，Linux 默认 16384），
  只缩不放。`PREFILL_CHUNK` 管工作区显存和单条 prompt 的 PP，`CONC_PREFILL_CHUNK` 管并发响应：
  分段越小，长 prompt prefill 期间别的会话卡得越短（32K prompt 实测最长卡顿 16384 时 830 ms、
  8192 时 624 ms），代价是这条 prompt 的首 token 约晚 8%。它在启动时一次定死，只有一个请求时也按它分段。
  分段不逐位一致，对 BF16 真值的 KLD 相同（`tools/d1a_kld.sh`）。0 = 并发时也用单路分段。
- **多条长 prompt 同时 prefill**（10-13）：prefill 只在两种情况下把 GPU 让出来：
  - 有 decode/verify 小轮次在等：层间每 300 ms 出让一次，每次给它一轮；
  - 有另一条剩余更少的 prompt 在等：在分段边界让出。剩余一样多时先来的先跑完。

  所以两条同长的 64K prompt 不再逐段轮流，而是一条跑完再跑另一条：总耗时约等于两条单跑之和，
  先到的那条首 token 约提前一半。短 prompt 插进长 prompt 时，最多等一个分段。
  回退开关：`QWENOX_CONC_LEND_LIGHT_ONLY=0` 恢复"有人排队就出让"，`QWENOX_PREFILL_SRPT=0`
  恢复逐段轮流。同一版修了 PLE 下一段预取跨槽误用的问题：两路同时起步时，分段起点相同，
  先进来的那路会读到另一路的 PLE 行。验证脚本是 `tools/prefill_sched_win.sh`（`SMOKE=1` 只跑约 2 分钟冒烟）。
- **超出路数**：第 N+1 个请求在引擎（和 API）里排队，直到有槽位空出；排队中可以取消。
- **槽位复用**：新请求优先选 hist 是其严格前缀的空闲槽位（多轮对话免 prefill），其次
  空槽，再次连接已断开的最久未用槽，最后最久未用槽。
- **池不够时**（按顺序）：先淘汰 RAM 检查点；再丢空闲槽位的 KV（最久未用优先）；
  还不够时，处于可挂起安全点、且快照长度达到 `QWENOX_KVSNAP_PARK_MIN` 的较晚请求会
  先 capture 到 KVSNAP、释放 KV 页，之后按 FIFO 恢复；恢复 ticket 会跨暂时性容量不足保留，
  不会被新请求插队。先来的请求永远不会因为后来的请求被牺牲。
- **park 边界**：plain decode 和 vision 请求的中途 park 仍关闭；短状态、快照目录不可用、
  writer 失败、取消/断连或超过 `QWENOX_PARK_MAX_ATTEMPTS` 时沿用有界的 abort/error fallback。
  `QWENOX_KVSNAP_MIN` 控制普通回合结束后的长期 cache，`QWENOX_KVSNAP_PARK_MIN` 单独控制
  临时 park snapshot；后者未设置时继承前者。
- **API 表现**：透明恢复的请求对客户端继续输出；无法 park 或被硬中断的请求如果已经流出 token，
  返回 503（`aborted by the engine ... shared KV context full`），还没出 token 则返回 400。
- kvsnap（SSD 快照）与 rckpt（RAM 检查点）照常工作，检查点同样钉住池里的页。MTP checkpoint
  只有 trunk pages 与非零 MTP sections 都发布后才对 restore 可见。

日志关键字：`[kvpage] pool dry: dropped idle slot`、`aborting the later request`、
`serve: req N failed:`。

## 确定性修复（与并发一起提交）

验证时发现，单 token decode 的 MoE down 投影（`moe()` / `mtp_moe()`）用 float
`atomicAdd` 累加 10 个专家，累加顺序取决于 wave 调度。空闲 GPU 上通常可复现，但时序
一变（另一条序列插在中间、调试同步）就会在最后一位上不同，几千 token 后翻转近似平局
的 token。同一配置单路连跑 3 次就出现过一次在第 2441 个 token 分叉。

现在单 token 路径与小批量 verify 一样走 `k_q4cp_gemv_gd_topk_h16<10>`（专家部分和留在
寄存器里按专家顺序求和，无原子操作）。速度不变；输出与旧二进制在长生成中会有差异
（累加顺序不同），但同一二进制下 unpaged / 分页 / 乱序分页 / 调试同步 / 并发全部逐位一致。
`QWENOX_MOE_DOWN_ATOMIC=1` 可切回旧路径做对比。

## 验证

需要先停掉生产服务。一键脚本，最后一行 PASS/FAIL：

```bash
bash build.sh
bash tools/conc_verify.sh
```

三段依次起测试引擎（端口 8732）：

1. **单路**：一组请求（drafter 0/1/3/4、采样、3K 长 prompt、三轮对话）串行跑。
2. **4 路**：同一组请求由 4 个客户端线程并发发送，必须与单路逐位一致（token、结束原因、
   spec 统计）；再测 `INFO kv_slots`、负载下 `PING` 延迟 < 0.5 s、排队中取消、运行中取消、
   同连接重复 `GEN` 被拒、客户端断连释放槽位。
3. **池溢出**：2 路 + `--maxctx 16384`（池只够一条 16K 序列）。A（7K prompt + 3000 token）
   先跑，B 后来：B 在 decode 中被中断，A 与单独运行逐位一致；A2（12K）运行时 B2 连 prefill
   都放不下，B2 直接失败；之后 B2 单独重试成功。

可选：`OLD_BIN=build/qwenox.old` 额外对拍旧二进制的单路结果（注意上面的确定性修复会让
旧/新在长输出上合法地不同）；`STAGES="ovf"` 只跑某几段；`CONC_ENV="..."` 附加引擎环境。

Windows（Git Bash）：

```bash
bash build_win.sh engine
bash tools/conc_fix_regress_win.sh     # 约 70 分钟，结尾 CONC_FIX PASS/FAIL
bash tools/conc_diverge_win.sh         # 排查用：单跑参考 + 若干并发配置，只比 token，打表
```

`conc_fix_regress_win.sh` 跑 v2 4 路 / 8 路多次、带 seed 的采样、GGUF 4 路 / 8 路、
`d2_s2_ids.sh`、vseg 自检和 API 吞吐（`QWENOX_VR_RESTORE` 开 / 关对照）。GGUF 按上面“与单跑
是否一致”一节，用 `QWENOX_OCC_FORCE=4` 判定，按人数选 γ 的那次只报告。
`conc_diverge_win.sh` 的用法见文件头（`CONFIGS="base|QWENOX_OCC_FORCE=4"`、`CONC_SEEDED=1`、
GGUF 用 `MODEL_ARGS=<gguf> GGUF_MTP=<sidecar>`）。
