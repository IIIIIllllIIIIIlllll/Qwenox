# GGUF 权重（hgn → GGUF 迁移）

目标：引擎直接加载 llama.cpp / Unsloth 的 `Qwen3.8-Flash-Next-UD-Q4_K_XL` GGUF，用户不再需要
维护 hgn 这类两个超大的专用权重文件。G3.3 起可以完全不用 hgn 启动。

## 用法：纯 GGUF（G3.3，不需要任何 .hgn）

Linux 上 hgn 和 GGUF 各有一个启动器，都读项目根目录 `models/`：

```bash
bash start_hgn.sh    # hgn 权重
bash start_gguf.sh   # GGUF 权重（本文）
```

把 Unsloth 的 Qwen3.8-Flash-Next UD-Q4_K_XL 全部文件放进 `models/`（与 llama.cpp 用的是同一份文件，
可以直接软链过去），另需 `models/tokenizer/tokenizer.json`（与 hgn 共用）：

```text
models/
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf  … 00004-of-00004.gguf   # 4 个分片都要
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf                                     # MTP sidecar
  mmproj-BF16.gguf                                                            # 视觉塔
  tokenizer/tokenizer.json
```

对应 service.conf 的"GGUF 权重"一段（同名环境变量优先）：

| 配置项 | 默认 | 说明 |
|---|---|---|
| `GGUF_FILE` | `$MODEL_DIR/…-00001-of-00004.gguf` | 第 1 个分片；其余分片须在同一目录，启动器逐个检查 |
| `GGUF_MTP_FILE` | `$MODEL_DIR/mtp-…-Q8_0.gguf` | 设 `""` 不做 MTP 投机 |
| `GGUF_VISION_FILE` | `$MODEL_DIR/mmproj-BF16.gguf` | 设 `""` 纯文本 |

缺任何一个文件时启动器列出全部缺失项并退出 1，不会启动引擎；`bash start_gguf.sh --check` 只检查。
`start_gguf.sh` 只接受 `.gguf`，`start_hgn.sh` 只接受 `.hgn`；两者都会清掉外部残留的 `QWENOX_GGUF*`，
权重格式只由所用的启动器决定。

- 启动器执行 `qwenox <第 1 个分片>.gguf ...`，引擎看到 `.gguf` 基座即为纯 GGUF 启动，等价于
  `QWENOX_GGUF=<该分片> QWENOX_GGUF_DENSE=1`，基座 Checkpoint 为空，全部张量来自 GGUF。MTP sidecar 由启动器
  以 `QWENOX_GGUF_MTP` 显式传入（空串 = 不用）；直接运行引擎且没设 `QWENOX_GGUF_MTP` 时，引擎自动使用分片
  目录里唯一的 `mtp-*.gguf`。
- PLE n-gram 表直接用 GGUF 里的 IQ4_NL（90 B/行，约 27 GiB；hgn fp8 为 160 B/行 47.7 GiB），
  prefill 在 GPU 上解量化（`k_ple_iq4nl_dequant`），decode 在 CPU 上解量化；`PLE_URING=1` 的 io_uring 批量读
  改为指向表所在的分片。
- Linux 与 Windows 都支持（Windows 2026-10-13 起，见下文"UD-IQ4_XS（Windows）"）。

## 用法：混合模式（过渡，仍需 hgn；只用于验证脚本直接运行引擎，启动器会清掉这些变量）

```bash
D=~/App/llama.cpp/models/Qwen3.8-Flash-Next-UD-Q4_K_XL
export QWENOX_GGUF=$D/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf   # G2：路由专家来自 GGUF
export QWENOX_GGUF_DENSE=1                                                 # G3.1：其余非专家张量也来自 GGUF
export QWENOX_GGUF_MTP=$D/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf          # G3.1/G3.2：MTP 头（含路由专家）
# G3.2：视觉塔在命令行给 --vision-tower $D/mmproj-BF16.gguf
```

| 变量 | 作用 |
|---|---|
| `QWENOX_GGUF=<第 1 个分片>` | trunk 48 层路由专家直接读 GGUF（hipHostRegister 原地映射，不复制），跑 `26_kernels_moe_gguf.inc` 的 WMMA kernel（移植自 gufo，MIT） |
| `QWENOX_GGUF_DENSE=1` | 加载期把 dense / norm / router / HC / embed / lm_head / PLE 投影从 GGUF 转成引擎格式（`src/gguf_map.h`），按 hgn 名覆盖 hgn 记录 |
| `QWENOX_GGUF_MTP=<sidecar>` | MTP 头从 Q8_0 sidecar 读：非专家张量需 `QWENOX_GGUF_DENSE=1`；路由专家（Q8_0 gate/up/down）只要同时设了 `QWENOX_GGUF` 就原地映射，走同一套 MoE kernel |
| `QWENOX_GGUF_MTP_EXPERTS=0` | 调试用：MTP 路由专家仍取 hgn |
| `--vision-tower <mmproj.gguf>` | 路径以 `.gguf` 结尾时按 llama.cpp mmproj（clip / qwen3vl_merger）加载，转成与 hgn 视觉文件相同的 bf16 张量（`gguf_map::build_vision`） |
| `QWENOX_GGUF_PLE=1` | 混合模式下 PLE n-gram 表也用 GGUF 的 IQ4_NL（默认保留 hgn 的 fp8 表，KLD 更好，见下） |
| `QWENOX_GGUF_DENSE_FILTER=a,b,...` | 调试用：只有名字包含其中某个子串的张量来自 GGUF（`!` 前缀表示取反），用于二分定位 |

## 转换规则（gguf_map.h）

- Q8_0 矩阵 → dtype 8 q8g32 planar（无损重排）；F32 矩阵 → bf16；F32 向量 → dtype 1 f32。
- GGUF 的 GDN value head 是 "tiled" 排列：引擎 head h = GGUF head `(h%3)*16 + h/3`，
  作用于 attn_qkv 的 v 行、attn_gate、ssm_alpha/beta、ssm_a、dt_bias、conv1d v 通道、ssm_out 输入列。
- `A_log = log(-ssm_a)`；零中心 RMSNorm 权重在 GGUF 里已 +1（ssm_norm 除外），加载时减 1。
- indexer q_proj(512) + k_proj(128) 拼成 index_qk_proj(640)；MTP eh_proj = [embedding 列 | hidden 列]。
- 逐张量对照检查：`tools/g3_map_check.cpp`（vs 生产 hgn）。
- 视觉塔：Conv3d patch embed 被转换器拆成两个时间切片 `v.patch_embd.weight` / `.weight.1`
  （各 [1152][3][16][16]），拼回引擎的 [1152][1536]（列 = c*512 + t*256 + y*16 + x）；
  其余是改名（`attn_qkv`→`attn.qkv`，`ffn_up/down`→`mlp.linear_fc1/2`，`ln1/2`→`norm1/2`，
  `post_ln`→`merger.norm`，`mm.0/2`→`merger.linear_fc1/2`）。F32 张量是 bf16 原值的上转，转回 bf16 无损。

## 结果（2026-09-25，gfx1151，`tools/g3_verify.sh` PASS）

| | hgn | G2（专家 GGUF） | G3.1（+dense GGUF） | llama.cpp Q4_K_XL |
|---|---|---|---|---|
| KLD vs BF16（64×512） | 0.1631 | 0.1494 | **0.0453** | 0.049 |
| top1 一致 | 86.61% | 87.20% | **93.25%** | |
| pp 8K @ chunk 2048 | 786 | 1215 | 1160 | |
| pp 32K @ chunk 16384 | 1242 | 1397 | 1392 | |
| decode @ 32K（tok/s） | 30.1 | 25.4 | 21.0 → 24.8（decode 提速后） | |
| MTP 投机 commit/round（8K，γ=3） | 3.78 | | 3.71 → 3.82 | |

- 质量提升主要来自非专家张量（hgn 的 dense 是 4-bit q4cp，GGUF 是 Q8_0）。
- 同一 1024 token 上逐 token decode 与 batched prefill 的 PPL：G3 6.814 / 6.760（0.8%），
  hgn 自身 6.905 / 6.862（0.6%），属于两条计算路径的正常差异。
- 不设 GGUF 变量时与改动前提交的 hgn 结果逐位相同（REF 要用同一提交编出的二进制，见 g3_verify.sh 注释）。
- **decode / 投机提速（2026-09-25 下午）。** 原先两个短板：P=1 时 WMMA 专家 kernel 16 行只用 1 行；
  Q8_0 dense 每 token 读取字节约为 q4cp 的 2 倍。已做：
  1. 小 P（≤8，`QWENOX_GG_GEMV`）的路由专家走原始块 FP32 GEMV `moe_gg_gemv`（26_kernels_moe_gguf.inc），
     P=1 每层 158 µs（WMMA 约 225）。
  2. 2≤P≤8 时做专家去重（`QWENOX_GG_UQ=0` 关闭）：投机验证的几个 token 共享的专家只读一次，
     与不去重的路径逐位相同（`tools/moe_gguf_gemv_test.cu --share 0.5`：MoE 1.07–1.14×；
     实测投机端到端只有约 +1%，在噪声内，没有共享时开销 <1%）。
  3. q8g32 dense GEMV：P=1 `k_q8g32_gemv_lpr`；P>1 `k_q8g32_gemv_mlpr`（每组权重解码一次，复用于 P 行），
     bf16 x 且 cols≥4096 用每行一个 block 的 `k_q8g32_gemv_brow`（HC down 39→23 µs，out_proj 131→93 µs）。
     基准 `tools/q8g32_gemv_bench.cu`。
  4. rows<64 的 bf16 矩阵（HC inject）：P=1 `k_bf16_gemv_row`，P>1 `k_bf16_gemv_row_mp`，替代 rocBLAS（66→4 µs）。

  | 512 token prompt，同机 | hgn | G3 原始 | G3 现在 |
  |---|---|---|---|
  | decode（tok/s） | 32.3 | 23.0 | 26.2 |
  | 投机 8K γ=3（tok/s） | 51.5 | 38.0 | 45.7–46.8 |

  剩余差距主要是 Q8_0 dense 的字节数（约为 q4cp 的 2 倍），带宽已接近上限；
  只有重新量化（牺牲 KLD）才能消除，暂不做。

- **G3.2（MTP 路由专家 + 视觉塔）。**
  MTP sidecar 的路由专家是 Q8_0 gate/up/down（主体是 Q4_K/Q5_K + Q5_1），MoE kernel 加了 Q8_0 gate/up
  实例（WMMA 与小 P GEMV 两条路径，`tools/moe_gguf_*test* --layer 48` PASS）。P=1 每步比 hgn q4cp 多约 100 µs
  （Q8_0 字节多），占一轮投机 <1%。

  | 投机 γ=3，同机 | MTP 专家来自 hgn | 来自 sidecar |
  |---|---|---|
  | 8K prompt tok/s（commit/round） | 46.0 / 46.2（3.82） | 46.3 / 46.1（3.82） |
  | 512 prompt tok/s（commit/round） | 33.9（2.88） | 35.0（2.99） |

  视觉塔：333 个张量与 hgn 视觉文件逐位相同；离线 `--vision-test` 前向的 35 个层 dump 逐字节相同
  （`tools/g3_vision_verify.sh`）。加载 1.2 s（hgn 1.0 s）。

- **G3.3（纯 GGUF 启动）。** 和混合模式唯一的数值差别是 PLE n-gram 表：GGUF 里只有 IQ4_NL，
  比 hgn 的 fp8 粗，KLD 0.0453 → 0.0511。此时引擎和 llama.cpp 用的是逐字节相同的权重（llama.cpp 0.049），
  剩下的 +0.002 来自计算路径（bf16 KV 等）。表变小后 prefill 反而略快（页缓存压力小）。

  | 同机，`tools/g3_pure_verify.sh` | G3.2（hgn fp8 PLE 表） | 纯 GGUF（IQ4_NL PLE 表） |
  |---|---|---|
  | KLD vs BF16（64×512） | 0.0453 | 0.0511 |
  | top1 一致 | 93.25% | 92.64% |
  | 1024 token PPL prefill / decode | 6.814 / 6.760 | 6.862 / 6.802 |
  | pp 8K @ chunk 2048（两轮） | 1126–1155 | 1196–1206 |
  | pp 32K @ chunk 16384 | 1388 | 1403 |
  | decode @ 32K（tok/s） | 24.8 | 24.8 |
  | 投机 8K γ=3 tok/s（commit/round；hgn 51.3 / 3.78） | 46（3.82） | 45.9–46.1（3.82） |

  纯 GGUF 与"hgn 基座 + GGUF 全覆盖 + `QWENOX_GGUF_PLE=1`"的 PPL 逐位相同，说明没有任何张量还在读 hgn。

## 同机完整对比（`tools/bench_full.sh`）

两个启动器各自的生产配置（`start_hgn.sh` / `start_gguf.sh --check` 的环境变量与权重参数，prefill chunk 16384），
2026-09-25 重启后全新编译。离线项用 `tools/pp_prod.sh` / `tools/kld_engine.sh`，API 项由启动器真正起服务
（256K 上下文、4 路并发，只关 KVSNAP），数字取自 API / 引擎日志的逐请求统计。

| 项目 | hgn | GGUF |
|---|---|---|
| prefill 8K（tok/s） | 1056 | 1370 |
| prefill 32K（整体 / 末 chunk） | 1250 / 1242 | 1435 / 1441 |
| prefill 64K（整体 / 末 chunk） | 1242 / 1209 | 1409 / 1370 |
| decode @32K，不投机（tok/s） | 30.5 | 25.0 |
| MTP 投机 8K prompt + 256，γ=3（tok/s，commit/round） | 51.9（3.84） | 46.6（3.82） |
| KLD vs BF16 / top1 / PPL | 0.163 / 86.61% / 3.468 | 0.0511 / 92.64% / 3.348 |
| API 就绪后内存 device / RSS（GiB） | 32.6 / 69.4 | 34.8 / 81.1 |
| API 中文散文 512 tok（tok/s，草稿接受率） | 30.1（0.40） | 26.2（0.37） |
| API 代码 512 tok（tok/s，草稿接受率） | 31.1（0.41） | 32.3（0.55） |
| API 8K prompt 首 token（s，prefill tok/s） | 7.6（1016） | 6.2（1237） |
| API 32K prompt 首 token（s，prefill tok/s） | 27.2（1209） | 24.3（1353） |
| API 4 路并发 ×256 tok 总吞吐（tok/s） | 29.8 | 28.5 |

GGUF prefill 快 13–30%、KLD 只有 hgn 的 1/3；decode 慢约 18%（Q8_0 dense 字节多）。
对话类文本草稿接受率只有 0.4 左右，投机收益远小于离线 8K 文档 prompt（commit/round 3.8）。
pp_prod.sh 现在把启动器命令行里的全部权重参数传给引擎，hgn 的投机用的是生产的 8-bit `mtp.hgn`
（以前只传主模型 + overlay，用的是 overlay 内置的 4-bit 草稿头，51.5 / 3.78）。

## hgn 路由专家也走 WMMA：LUT 解码 kernel（2026-09-25，`tools/lut_verify.sh` PASS）

以前 hgn 没吃到 G1 MoE kernel 的收益：它只认 GGUF 块格式，hgn（q4cp）prefill 仍是
dequant + hipBLASLt（`QWENOX_MOE_LT`）/ 标量 `k_moe_w4`。q4cp 与 IQ4_NL 结构几乎一样
（32 元素一组、4-bit 查表索引、fp16 scale），所以写了一个 LUT 解码族 kernel
`src/gpu/parts/27_kernels_moe_lut.inc`：`k_moe_lut<kQ4CP | kIQ4NL | kIQ4XS, pair>`，
流水线/tiling/epilogue 与 `k_moe_gg` 相同，解码 = LDS half2 对表（一个字节查出两个元素）× scale。

- **默认开启**（hgn prefill，P > moe_naive_max）；`QWENOX_MOE_Q4W=0` 恢复旧路径且与旧提交逐位相同。
  开启时 `QWENOX_MOE_LT` 不再生效，它的 dequant/gather 缓冲（`d_moexg` 等，chunk 16384 约 0.8 GiB、
  Windows chunk 8192 约 0.4 GiB）也不分配，`devarena_estimate` 同步。
- decode（P ≤ 16 的 grouped GEMV）完全不变：逐 token decode mean_nll 与旧二进制逐位相同。
- IQ4_NL / IQ4_XS 模板已在 `tools/moe_lut_test.cu --synth iq4nl|iq4xs` 对 CPU 参考通过；
  引擎的 GGUF 路径还没接（手头没有 IQ4 专家的 GGUF 文件），以后给 Windows 用 UD-IQ4_XS 时再接。

kernel（`tools/moe_lut_test.cu --hgn`，真实 q4cp 权重）：P=2048 10.6 ms/层（旧 `k_moe_w4` 28.3，
GGUF Q4_K 11.8）；P=16384 0.168 ms/token（旧 LT 路径 0.291，GGUF 0.173）。

引擎（`start_hgn.sh` 生产配置，同一二进制 off / on）：

| 项目 | off（旧路径） | on（LUT WMMA） | GGUF 参考 |
|---|---|---|---|
| KLD vs BF16 / top1 | 0.1631 / 86.61% | 0.1628 / 86.81% | 0.0511 / 92.64% |
| prefill 8K @ chunk 2048（tok/s） | 816 | 1215（+49%） | 1206 |
| prefill 8K 单 chunk | 1096 | 1361（+24%） | 1370 |
| prefill 32K @ chunk 16384 | 1237 | 1422（+15%） | 1435 |
| decode @32K（tok/s） | 30.5 | 31.2 | 25.0 |
| MTP 投机 8K γ=3（tok/s，commit/round） | 51.6（3.84） | 49.3（3.67） | 46.6（3.82） |

投机每轮耗时不变，commit/round 的差别来自 prefill 数值略变后生成的文本不同（单次 256 token，
脚本门槛 ≥ 0.95×off）。1024 token prefill vs decode PPL 6.875 / 6.905（< 1%）。

结论：hgn 的 prefill 速度追平 GGUF，decode 保持 hgn 的优势，显存比 UD-Q4_K_XL 少约 11 GiB——
Windows（96 GiB 上限）继续用 hgn 即可拿到这份 prefill 提速，重新编译即可。质量（KLD）仍是 hgn 的
0.163，下一步是用 imatrix 重新量化出高质量 hgn（dense 8-bit）。

## 不换格式也能拿到 GGUF 的质量：高质量 hgn

GGUF 的质量优势主要来自 dense 8-bit（Q8_0）。同样的 dense 8-bit + imatrix 专家量化可以直接做进
hgn（Windows 需要），见 [HGN-HQ.md](HGN-HQ.md)。

## UD-IQ4_XS（Windows，2026-10-13）

Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS`（3 个分片，87.2 GiB，第 1 个分片只有元数据）与 UD-Q4_K_XL
是同一架构、同一套张量名，差别只在块类型。名字里的 IQ4_XS 只占很少一部分，实际分布（读分片头统计）：

| 张量 | 类型 | 大小 |
|---|---|---|
| gate/up 专家，47 层 | IQ3_S（110 B/256） | 31.6 GiB |
| gate/up 专家，blk.2 | IQ4_XS | 0.8 GiB |
| down 专家，43 层 | IQ4_NL | 18.9 GiB |
| down 专家，blk.2/4/30/46/47 | Q8_0 | 4.2 GiB |
| 注意力/SSM/HC/共享专家/embed | Q8_0 | ≈4.3 GiB |
| lm_head（output.weight） | Q6_K | 0.49 GiB |
| PLE n-gram 表 | IQ4_NL | 26.8 GiB（按需读盘，不进显存） |

主干权重常驻 60.4 GiB（专家 55.4 + dense 4.8 + norm 等）。分片里没有 MTP 头，MTP 用单独的 `mtp-*.gguf`
sidecar（没有时不投机），加上 Q8_0 sidecar 后权重为 63.0 GiB。v2 hgn 主体（62.1 GiB）自带 MTP 草稿头，所以
两者实际比较的是：不带 MTP 时 UD-IQ4_XS 的 arena 用量比 v2 少 3.2 GiB，带 MTP 时多 1.3 GiB（256K 4 路：
79.4 / 83.9 vs v2 82.6 GiB；512K 页池 2 路 93.5 vs 92.2 GiB；2026-10-13 实测）。显存上 UD-IQ4_XS 只在不用
MTP 时占优。

MTP sidecar 可选 `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` 或 `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`。Q8_0 版
（4.1 GB，dense 全 Q8_0，路由专家 Q8_0）比 Linux 上用的 `…-shared-Q8_0.gguf` 多带了
`token_embd` / `output`，引擎不读这两个，草稿头沿用主干的 embed 和 lm_head：`token_embd` 两边逐字节相同
（md5 一致），说明出自同一个 checkpoint。Q4_K_M 版中的三个 `hc_*_up` 为 Q5_0，现由通用 CPU 反量化路径
解码，并在加载时重量化为 q8g32；routed-expert gate/up/down 分别为 Q4_K/Q4_K/Q8_0，不经过原生 Q5_0
dense GPU kernel。分片目录里有多个 `mtp-*.gguf` 时引擎不会自动挑，必须显式指定（`GGUF_MTP_FILE` 或
`QWENOX_GGUF_MTP`）。

实现要点：

- **IQ3_S 不需要新的解码框架**：`iq3s_grid` 的 512×4 个字节全部是 {1,3,…,15}，所以每 32 个元素等价于
  "4-bit 码（低 3 位 = g>>1，第 3 位 = 符号）+ 16 项码本 ±(2c+1) + 一个 scale `d·(1+2·sc)`"。WMMA prefill
  复用 `k_moe_lut`（`kIQ3S`）：commit 阶段用 LDS 里的 grid 把原始块展开成与 IQ4 相同的 16 个配对码字节，
  decode/WMMA/epilogue 全部不变。grid 表见 `src/gguf_iq3s_grid.h`（llama.cpp，MIT）。
- 小 P（decode/投机验证）GEMV：`26_kernels_moe_gguf.inc` 新增 `UnitW<kIq4nl / kIq4xs / kIq3s>`；IQ4 码本
  在寄存器里用 `v_perm` 查表，IQ3_S grid 在 `__constant__`。点积写成显式 `fmaf` 链：专家去重路径要求与
  普通路径逐位相同，隐式 fp-contract 在两个 kernel 里不一定一样。
- 逐层混合类型：`moe_ggx_up/down`（27 末尾）按每层的 `tgu/tdn` 派发——K-quant/Q5_0/Q5_1/Q8_0 走 `k_moe_gg`，
  IQ 走 `k_moe_lut`。
- lm_head Q6_K：加载时按 ggml 语义反量化，再每 32 个元素重量化为 q8g32（`gguf_map::add_q8` 的非 Q8_0 分支）。
  8-bit 码对 6-bit 源，误差不超过源量化步长的 1/8；decode logits GEMV 读取字节与 Q8_0 dense 一致。
- Windows：分片用 `OVERLAPPED | NO_BUFFERING` 句柄打开（同 hgn），专家与 dense 一起拷进 devarena（没有
  hipHostRegister 直读路径）；`devarena_estimate` 计入 GGUF 专家；PLE 表的 IOCP 批量读指向表所在分片。
- 服务启动：设置页/网页选"GGUF"（service.conf 写 `# start_win: weights=gguf`）后，托盘 `qwenox-win` 与
  `bash start_win.sh` 都改用 `GGUF_FILE`（第 1 个分片）启动；`GGUF_MTP_FILE` 置空即不投机，
  `GGUF_VISION_FILE` 可以填 mmproj `*.gguf`，也可以直接用现有的 `*-vision.hgn`。

验证：`bash tools/iq4xs_verify_win.sh`（约 30 分钟，结尾 `IQ4XS_VERIFY PASS/FAIL`）——编译、真实权重
kernel 对拍（混合类型层 0/2/30/47）、v2 路径与旧提交逐位相同、GGUF prefill/GEMV PPL 一致、PPL 低于 v2、
服务端到端（weights=gguf 临时 conf + 测试端口）、decode/prefill 速度。CPU 解码参考对 ggml 逐位：
`tools/gguf_dequant_ref_check.cpp`（g++ 即可，见文件头）。

结果（Windows，2026-10-13，同一二进制，`data/qsa-oracle/2048.tokens`）：

| 项目 | UD-IQ4_XS（GGUF） | v2 hgn |
|---|---|---|
| kernel 对拍（层 0/2/30/47，真实权重 vs CPU） | WMMA up rel-L2 ≤ 6.3e-4，GEMV ≤ 2e-7，去重路径逐位相同 | — |
| v2 路径（新二进制 vs 6eb7c9e） | — | PPL 与 greedy 128 ids 逐字相同 |
| PPL（生产内核组合，chunk 2048 / chunk 4） | **39.62** / 39.50（相差 0.30%） | 48.12 |
| KLD vs BF16（`data/kld/bf16_c512.kld`，64×512，生产 env）：Mean / top-1 / 99.9% / PPL(Q) | **0.1002** / 89.72% / 3.109 / 3.444 | 0.1075 / 88.79% / 2.910 / 3.416 |
| decode（2048 prompt，`--gen 128`，均无 MTP） | 27.7 tok/s | 31.4 tok/s |
| prefill（qwenox-bench，1 遍）2K / 8K / 32K / 64K / 128K | 1164 / 1318 / 1443 / 1444 / 1418 | 546¹ / 1167 / 1562 / 1566 / 1539 |
| prefill 加权平均 | 1415.6 tok/s | 1504.3 tok/s |
| 冷启动（`--ppl`，maxctx 4096） | 46 s，arena 69.8 GiB | 30 s，arena 72.1 GiB |
| 服务端到端（weights=gguf，32K，单路） | 拉起 + 中文对话正常，device 74.1 GiB | — |

¹ 第一个文件，含预热，不计。

结论：质量与 v2 基本持平。对 BF16 的 KLD 平均值和 top-1 一致率略好，PPL(Q) 和 99.9% 尾部略差；
上面单个 2048-token 文本的 PPL 差 18%，不能代表整体（单段文本的 PPL 对个别位置很敏感）。不带 MTP 时
显存更省。速度方面 decode 慢
约 12%，32K 以上 prefill 慢约 8%，原因和 UD-Q4_K_XL 相同：dense 是 Q8_0，每 token 读的字节约为 v2 的 2 倍。
MTP（`tools/iq4xs_verify_win.sh` 的 mtp 步骤，greedy γ=4，`--spec-gen 256`，maxctx 16384）：

| prompt | UD-IQ4_XS + mtp Q8_0 sidecar | v2 + mtp.hgn |
|---|---|---|
| 2048 | 46.8 tok/s，commit/round 4.02 | 58.5 tok/s，commit/round 4.02 |
| 8192 | 47.1 tok/s，commit/round 4.02 | 49.8 tok/s，commit/round 3.67 |

GGUF 投机输出与普通 greedy（`--gen 256`，25.2 tok/s）前 256 个 token 完全一致；投机约为普通 decode 的
1.86×。草稿头接受率不低于 v2，剩下的差距来自每轮验证的 dense 字节（同上）。服务端带 sidecar 启动
（`mtp: draft head enabled`）对话正常。

并发 / 显存 / prefill 分段（`tools/iq4xs_conc_win.sh`，Windows，2026-10-13）：

- **并发功能正常**：生产配置（256K、`PARALLEL=8`、MTP、视觉塔）8 路同时跑 `conc_verify.py` 的 10 个请求
  （MTP 投机、采样、多轮续写、3K prompt）全部正常结束；INFO 槽数、负载下 PING（0 ms）、排队/运行中取消、
  断连全部通过。和单路相比逐 token 不完全相同（5/10 个请求在第 5–45 个 token 分叉，之后是通顺的同义改写）：
  GGUF 的 dense / 多行专家 kernel 不是行无关的，批大小不同舍入就不同。v2 的 `k_ht_wmma` 专门做成了行无关。
- **合批吞吐有台阶**（`QWENOX_VBENCH`，同一步算 P 行，ms）：

  | P | 1 | 4 | 8 | 10 | 24 | 40（8 段） |
  |---|---|---|---|---|---|---|
  | GGUF | 37.9 | 54.0 | 72.6 | 103.0 | 184.2 | 239.5 |
  | v2 | 33.4 | 48.3 | 65.7 | 77.8 | 113.1 | 136.5 |

  P≤8 只比 v2 慢约 10%；P>8 时 dense q8g32 改走"整块反量化成 bf16 + GEMM"，这一项占了台阶的大头（专家改走
  GEMV 的消融 `QWENOX_GG_GEMV=64`：P=10 103→94.5，P=40 反而更慢 313）。不开 MTP 的 8 路 decode（每步 8 行）
  不受影响；8 路 + MTP（每步 16–40 行）比 v2 慢 30–75%。→ 已按 gufo 的思路修，见下一节。

### 多行 dense + 专家 GEMV（2026-10-13，参考 gufo 0.11）

gufo（同机同模型，MIT，已致谢）在 9–48 行用一个多行 kernel 直接读 Q8_0 权重（整数 WMMA），并让它与单行
kernel 的运算顺序一致。这里按 v2 `k_ht_wmma` 的做法实现（f16 WMMA，不量化激活）：

- **`k_q8g32_wmma`**（28_kernels_hgn_v2.inc / `q8g32_wmma`）：q8g32 码是 int8，转 f16 无误差，只有 x→f16 一次
  舍入；每 32 列一个 scale 在 f32 里乘。波次布局（KW）和 K 切分（gps / ns）只由张量形状决定，与 P、行切片、
  行在批里的位置都无关 → **P=1..48 同一行逐位相同**（`tools/q8_wmma_test.cu`：对 CPU double 相对误差 ≤1e-6，
  移位/倒序/切片逐位）。所有 P≤48 的多行调用（verify、合批 verify / draft、小 prefill 尾块、lm_head）都走它；
  P=1 的 decode 前向仍用带宽调优的 GEMV（WMMA 在 P=1 慢 15–20%）。开关 `QWENOX_DENSE_WMMA_Q8=0`（默认跟随
  `QWENOX_DENSE_WMMA`），调参 `QWENOX_Q8W_KW / _BLOCKS / _U`（只影响速度），调试 `QWENOX_Q8W_SKIP=RxC,...`
  （按形状退回旧 kernel）、`QWENOX_Q8W_XSTAT=N`（打印 x 的量级 / f16 次正规比例）。
- **专家去重 GEMV 扩到 48 行**（26：`kUqMaxP` 8→48，成员按 8 个一组在寄存器里算）：`moe_gguf` 在 P≤48 时
  都走 GEMV 族（`QWENOX_GG_GEMV`，默认 48），每个 (token, slot) 的运算与 P 无关，与普通 GEMV 逐位相同。

实测（Windows，`tools/gguf_batch_verify_win.sh`，同一二进制新 / 旧 = `QWENOX_DENSE_WMMA_Q8=0 QWENOX_GG_GEMV=8`）：

| 同一步算 P 行（ms） | 1 | 5 | 8 | 9 | 16 | 24 | 40 | 40（8 段） | 48 |
|---|---|---|---|---|---|---|---|---|---|
| 旧 | 37.9 | 61.3 | 74.1 | 96.3 | 133.5 | 183.5 | 270.3 | 238.3 | 296.7 |
| 新 | 44.2¹ | 61.9 | 73.7 | 80.4 | 105.3 | 136.2 | 204.4 | 188.7 | 225.2 |

¹ vbench 的 P=1 走的是 prefill 路径（多行 kernel）；真正的 decode 前向仍用旧 GEMV，AR decode 27.9 = 旧 27.8 tok/s。

- 9–48 行快 17–25%；8 路 + MTP 每轮 16–40 行正好落在这个区间。
- **API 端到端并发吞吐**（`tools/gguf_serve_bench_win.sh`：conc_bench 代码题，greedy，每路 2 × 1200 token，
  PARALLEL=8、MTP 开、64K 上下文，聚合 tok/s）：

  | 并发 | GGUF 旧 | **GGUF 新** | v2 hgn |
  |---|---|---|---|
  | 1 | 48.0 | 47.0 | 55.3 |
  | 4 | 57.1 | **70.7（+24%）** | 83.2 |
  | 8 | 65.3 | **82.4（+26%）** | 97.0 |

  草稿接受率三者相同（0.66–0.69），差距全在每轮 verify 的耗时上。和 v2 剩下的差距（8 路 −15%，1 路 −15%）
  是 dense 字节数：Q8_0 每个权重 1.06 字节，v2 是 4-bit 加旋转（约 0.5 字节）。
- 单用户 MTP（γ4，2048 / 8192 prompt）46.4 / 45.9 vs 旧 47.8 / 47.5 tok/s（verify 5 行持平，差异主要是接受率
  4.02→3.94 的文本噪声）。
- 质量：以旧路径 chunk 512 为基准的相对 KLD（16×512 token）——chunk 24（每块都走 ≤48 行路径）旧 0.0252 /
  新 0.0268，chunk 4 旧 0.0272 / 新 0.0258，相互在误差条内；chunk 2048 PPL 新 39.6154 = 旧 39.6159。
  （小 chunk 的 PPL 本身是混沌的：只把某一个形状换成新 kernel，PPL 就在 35.7–44.3 之间跳，不能当门槛。）
- 逐位：`QWENOX_VSEG_SELFTEST` 3,4,5 与 8×5（批 verify == 逐段 verify）PASS；γ1/3/4/7 的 greedy 投机输出相同
  （γ2 在第 257 个 token 处差 1 个，新旧一样）。
- 已知：Windows 上 `conc_verify` 的"4 / 8 路并发 == 单路串行"连 v2 生产模型也不成立（改动前就这样，分叉
  在第 7–64 个 token，之后是通顺的同义文本），不是 GGUF 引入的；`vseg` 自检逐位说明合批 verify 本身没问题，
  分叉来自调度路径的其他差异，待单独排查。
- **显存**（生产配置，引擎 `memory[ready]`）：chunk 8192 → device **89.3 GiB**（arena 估算 85.3 + 4 slack，
  权重 63.0 含 MTP 草稿头），HIP 剩 17.5 / 107.9 GiB；chunk 16384 → arena 估算 92.5 GiB，加 slack 后被 95 GiB
  上限截断，device 95.0 GiB，HIP 剩 11.4 GiB。arena 启动时一次性预留，运行中不再增长。
- **主机内存**（`tools/gguf_vision_mem_win.sh`，生产配置起服务，进程工作集）：显存之外，引擎在系统内存里常驻
  约 2.9 GiB（HIP 运行时、tokenizer、RAM 检查点等，与 v2 相同；跑请求后 +1 GiB 左右是 RAM 检查点 / 图片缓冲）。
  加载期间峰值约 10.5 GiB：dense 转换的 ~5 GiB 主机副本 + 经文件映射读入的源页面，转换完立即释放。
  早先版本这 ~5 GiB 映射页面会一直挂在工作集里（任务管理器显示"使用中" 8–9 GiB），现在加载完对分片映射调用
  `VirtualUnlock` 移出工作集（`gguf::File::trim_resident`），页面转入 standby 缓存，可随时回收。
  另有约 2.6 GiB 显示为 GPU"共享内存"（HIP 自己的系统内存分配），GGUF 与 v2 相同。PLE 表（26.8 GiB）按需读盘，
  只进 standby 缓存，不算"使用中"。
- **多模态**：GGUF 主干 + 现有 `qwen38-flash-next-vision.hgn` 视觉塔，经 API 发图片，描述正确，与 v2 一致
  （视觉塔前向约 6 s / 1152×640 图）。mmproj `*.gguf` 视觉塔在 Linux 上已验证与 vision.hgn 逐位相同
  （`tools/g3_vision_verify.sh`），Windows 上待有 mmproj 文件后用 `tools/gguf_vision_mem_win.sh` 复测。
- **prefill 分段**（32K 单请求，去掉预热）：chunk 4096 / 8192 / 16384 = 1399 / 1436 / 1419 tok/s。稳态
  chunk 8192 → 16384 只快约 2.5%（5.51 → 10.75 s / 块），第一块有 PLE 冷读。Windows 默认 8192 就合适；
  `PARALLEL>1` 时实际分段是 min(`CONC_PREFILL_CHUNK`, `PREFILL_CHUNK`)。
- **单文件 GGUF**：文件名形如 `…-0000i-of-0000N.gguf` 时按分片打开全部 N 个，否则按单文件打开（MTP sidecar
  一直走这条路）。`llama-gguf-split --merge` 合并后的文件可以直接填进 `GGUF_FILE`。

注：验证中 v2 + mtp.hgn 在 2048 prompt 上偶发过一次段错误（`verify graphs captured` 之后，崩在 HIP 运行时
`hipEventSynchronize` 内部）。同一命令用 6eb7c9e 基线与新二进制各重跑 2 次都正常，输出逐 token 相同，
因此与本次改动无关，原因待查。

## IQ2_XXS / IQ2_XS 混合量化专家

外部混合量化（实测：`Dongfangshuo-Qwen3.8-Flash-Next-NSFW-IQ3_XXS`，2026-11-04）会在部分层的
`ffn_{gate,up}_exps` 里使用 **IQ2_XXS / IQ2_XS**。这两种类型现已支持原生加载、CPU 解量化、
小 P GEMV（含专家去重）和大 P LUT/WMMA，不需要把专家重量化成其他格式。

该模型 48 层中：10 层 gate/up 为 IQ2_XS（0,2,4,5,9,10,14,19,25,39），9 层为 IQ2_XXS
（1,3,6,7,11,13,16,27,31），其余 29 层为 IQ3_S/IQ2_S/IQ3_XXS。down 类型为 IQ4_NL/Q2_0。
实现布局为 IQ2_XXS 256 元素 / 66 B、IQ2_XS 256 / 74 B；码本来自 llama.cpp，CPU 与 GPU
共用同一 ggml 解码语义。IQ2_XXS 每 32 元素一个 scale，IQ2_XS 每 16 元素一个 scale。

验证（真实权重与 CPU 参考对拍）：

- `tools/gguf_dequant_ref_check.cpp`：IQ2_XXS/IQ2_XS 各 4096 个随机块与 ggml 原函数逐位相同；
  原有 IQ3_S/Q6_K 对照也保持零 mismatch。
- 第 0 层（IQ2_XS/IQ4_NL）与第 1 层（IQ2_XXS/Q2_0）：P=2048 的 64/128-slot WMMA 均 PASS，
  gate/up 最大 rel-L2 分别为 5.406e-4 / 5.824e-4（门槛 1e-2）。
- 同两层 P=1/4/48 GEMV 均 PASS，最大 rel-L2 < 2e-7；P=4/48 的 dedup 与普通路径 hid/out
  均逐位相同。
- 第 12 层 IQ2_S 与第 35 层 IQ3_XXS 的 GEMV/WMMA 回归均 PASS。
- 完整 48 层纯 GGUF 已成功加载并推理：2047-token PPL 为 10.6162（chunk 2048）/ 9.98484
  （chunk 4，相差 5.95%；小 chunk PPL 的混沌性见上文，不能单独作门槛）。普通 greedy 生成
  128 token、98 个不同 token，28.9 tok/s。
- Q8_0 MTP sidecar 可加载并启用草稿头：37.4 tok/s、commit/round 3.07。与普通 greedy 的前
  54 个生成 token 一致，之后因 verify/decode 数值路径差异分叉，不要求逐 bit 相同。

## Q5_0 down（UD-IQ4_XS 四分片，2026-11-05）

Unsloth `Qwen3.8-Flash-Next-UD-IQ4_XS` 四分片版本的 48 层 routed-expert down 中，33 层使用
**Q5_0**，其余为 13 层 IQ4_NL 和 2 层 Q2_0。Q5_0 现已支持原生 GPU 加载，不需要离线重量化：
CPU `gguf::dequant_row` 按 ggml 的 32 元素 / 22 B 块解码；小 P GEMV 复用 Q5_1 点积，以
`mn=-16d` 表示 `d(q-16)`，普通与专家去重路径共用同一实现；大 P 走 `k_moe_gg` WMMA，
直接解码 `(q-16)d`，不经过 LUT。22 B 块的非对齐读取严格限制在实际块内。

真实权重与 CPU 参考对拍结果：

- `tools/gguf_dequant_ref_check.cpp`：Q5_0 的 4096 个随机块、131072 个输出与 llama.cpp
  原函数逐位相同；IQ2_XXS/IQ2_XS/IQ3_S/Q6_K 回归也全部零 mismatch。
- 第 0 层（IQ3_XXS/Q5_0）P=1/4/48 GEMV 均 PASS，out 最大 rel-L2 为 3.783e-7；
  P=4/48 的 dedup 与普通路径 hid/out 均逐位相同。
- 第 0 层与第 47 层（IQ4_XS/Q5_0）P=2048 WMMA 均 PASS，Q5_0 down 最大 rel-L2 为
  2.009e-4（门槛 1e-2），hid/pairs 非有限值均为 0。
- 第 7/8 层 IQ4_NL/Q2_0 down 与 Q8_0 MTP sidecar 的 GEMV/WMMA 回归均 PASS。
- 完整四分片纯 GGUF（禁用 MTP、KVSNAP 和 RAM 检查点）已成功加载并生成 128 token：
  权重 60.6 GiB，device arena 70.20 GiB；2048-token prefill 为 711.2 tok/s，普通 decode
  为 28.8 tok/s，进程正常退出。

### Q4_K_M MTP sidecar（2026-11-06）

Q5_0 的支持按张量槽位区分：routed-expert down 使用原生 GPU 路径；dense 映射槽位在加载时经 CPU
反量化并重量化为 q8g32；routed-expert gate/up 尚未支持 Q5_0。Q4_K_M MTP 的三个 Q5_0 都位于 dense
HC-up，因此属于已支持范围。

实测 UD-IQ4_XS 主权重与 Q4_K_M MTP sidecar 组合，`qwenox-bench` 的模型加载、完整 prefill、增量
prefill 和 MTP decode 全部 PASS，日志出现 `mtp: draft head enabled`。在 maxctx 262144、gamma=4、
每场景生成 128 token、单轮采样下，加权 prefill 为 **1534.4 tok/s**，MTP decode 为 **35.6 tok/s**，
草稿接受率为 **47.6%**，最终结果为 `Benchmark complete: PASS`。

## 验证脚本

- `tools/g2_verify.sh` — G2（专家）
- `tools/g3_verify.sh` — G3.1：KLD、decode/prefill PPL、hgn 路径逐位不变、MTP、速度
- `tools/g3_vision_verify.sh` — G3.2 视觉塔：张量逐位对照 + 端到端前向 dump 逐字节对照
- `tools/g3_pure_verify.sh` — G3.3 纯 GGUF：start_gguf.sh 配置、KLD、与混合启动逐位相同、decode/prefill、
  hgn 路径逐位不变、MTP、速度
- `tools/launcher_verify.sh` — 两个启动器：--check 命令行/环境、缺文件报错、格式互斥、端到端起停
- `tools/bench_full.sh` — hgn vs GGUF 完整性能：prefill 8K/32K/64K、decode、投机、KLD、API 端到端，
  汇总表写到 logs/bench_full.md（`FORMATS=gguf` 只测一种，`SKIP="kld api"` 跳过几项）
- `tools/g3_smoke.sh [ENV=...]` — 512 token PPL 冒烟（可带过滤器等 env）
- `tools/lut_verify.sh` — hgn LUT MoE kernel：开关关闭与旧二进制逐位相同、decode 不变、KLD、
  prefill/decode PPL、MTP、速度（`BIN=` 新二进制，`REF=` 旧提交二进制）
- `tools/moe_lut_test.cu` — LUT kernel（q4cp 真实权重 `--hgn` / 合成 IQ4_NL、IQ4_XS `--synth`）vs CPU 参考
- `tools/gguf_batch_verify_win.sh` — GGUF 多行 dense + 专家 GEMV：kernel 单测、vbench、decode、PPL、vseg 逐位、并发
- `tools/gguf_serve_bench_win.sh` — 经 API 的并发吞吐（conc_bench 1/4/8 路），GGUF 新 / 旧 / v2
- `tools/gguf_vision_mem_win.sh` — Windows：主机内存曲线（`tools/win_memsampler.ps1`）+ 图片请求 + mmproj vs vision.hgn 前向逐位
- `tools/q8_wmma_test.cu`（`bash tools/ht_wmma_test.sh q8`）— `k_q8g32_wmma` 精度 / 行无关 / 切片 / 速度
- `tools/iq4xs_conc_win.sh` — UD-IQ4_XS（Windows）：并发（conc_verify 单路 vs 8 路 + 控制面）、生产配置显存、prefill 分段
- `tools/iq4xs_verify_win.sh` — UD-IQ4_XS（Windows）：kernel 对拍、v2 逐位不变、PPL 一致性/质量、服务端到端、速度
- `tools/gguf_dequant_ref_check.cpp` — gguf.h CPU 解码（Q5_0、IQ2_XXS、IQ2_XS、IQ3_S、Q6_K）vs ggml 原函数逐位
- `tools/moe_gguf_test.cu` — MoE kernel（含 IQ 类型，经 `moe_ggx_*`）vs CPU 参考
- `tools/moe_gguf_gemv_test.cu` — 小 P 专家 GEMV（含去重路径逐位对照）vs CPU 参考
- `tools/q8g32_gemv_bench.cu` — q8g32 dense GEMV 各变体（P=1 与 P=4）
