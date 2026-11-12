# Qwenox

![Strix Halo — Qwen3.8-Flash-Next](media/strix_banner_21x9_v2.png)

*English: [README_EN.md](README_EN.md)*

Qwenox 是 Qwen3.8-Flash-Next（177B MoE，qwen4_exp 架构）的本地推理引擎，为 AMD Strix Halo
（Ryzen AI Max+ 395，gfx1151）这类 128 GB 统一内存的 APU 编写，Linux 和 Windows 都能运行。
对外提供 OpenAI 兼容 API 和 Anthropic Messages API（流式输出、工具调用、图片输入）以及网页控制台。

几个关键做法：

- **权重放得下**：路由专家 4-bit，GPU kernel 直接读；PLE n-gram 表（27–48 GiB）留在磁盘按需读取，
  不占内存。常驻权重 60–77 GiB（视权重文件），剩下的内存给 256K 上下文。
- **投机解码**：ngram 起草优先，MTP 草稿头兜底。greedy 逐位校验，采样按目标分布重采样，输出分布与
  不投机时一致。默认开启，无需参数。
- **不重复 prefill**：分页 KV 页池加两级 prompt 缓存（消息边界的内存检查点、可跨重启恢复的 SSD 快照），
  编辑重发、多轮续写、切换对话都直接复用已算过的部分。

## 性能实测

开发机器为 **GMK EVO-X2（AMD Ryzen AI Max+ 395，Strix Halo / gfx1151）**，122 GiB 内存。

Linux（ROCm 10.1，生产配置，prefill 开 MOE_A4=3，PREFILL_CHUNK=16384）：

| 指标 | hgn v2 |
| --- | --- |
| Prefill（2K） | 约 1610–1651 tok/s（10-18 实测） |
| Prefill（8K） | 1902.7 tok/s（10-18 实测） |
| Prefill（32K） | 1938.5 tok/s（10-18 实测） |
| Prefill（64K） | 1936.4 tok/s（10-18 实测） |
| Prefill（128K） | 1840.8 tok/s（10-18 实测） |
| Prefill 加权平均 | 1874.5 tok/s（7 个文件，每个 2 次采样，10-18 实测） |
| Prefill（256K 上下文） | 原生约 1590、YaRN factor 2 约 1580 tok/s，开销约 0.5%（10-07 实测） |
| Decode（投机，greedy，γ=4） | 8K 约 51、64K 约 50 tok/s（10-07 真实文本） |
| Decode（不投机） | 8K 约 34 tok/s（10-07 实测，hgn v2 权重；dense 为 8-bit 的权重每 token 读量大，约低 12–16%） |
| 平均功耗 | 约 120 W |
| 瞬时最大功耗 | 约 130 W（爆发持续几秒后回落至 120 W 左右） |

Windows（同一台机器，BIOS 划分 96 GiB 显存，托盘程序生产配置，prefill chunk 8192，10-13 实测）：

| 指标 | hgn v2 | GGUF UD-IQ4_XS + MTP |
| --- | --- | --- |
| Prefill（32K） | 约 1560 tok/s | 约 1440 tok/s |
| Decode（投机，2K prompt） | 58.5 tok/s | 46.8 tok/s |
| Decode（不投机） | 31.4 tok/s | 27.7 tok/s |
| 冷启动（整份权重读进显存） | 约 35 s | 约 50–60 s |

投机 decode 的速度随文本重复度变化很大。上表用真实文本 prompt 测得；代码、模板这类高重复内容
草稿命中率高，同配置可达 60 tok/s 以上；采样（自适应 γ）比 greedy 略低。表中日期均为 2026 年；
Linux 上 ROCm 7.14 与 10.1 实测无差异。多路并发的吞吐见「并发」一节。

## 支持的权重

| 权重 | 平台 | 质量（KLD vs BF16，越低越好） | 下载体积 / 常驻内存 | 什么时候用 |
| --- | --- | --- | --- | --- |
| **hgn v2**（推荐） | Linux、Windows | 0.107 | 约 112 GiB / 62.1 GiB | 默认选择：速度最快，实际效果最好 |
| GGUF UD-Q4_K_XL | Linux | 0.051 | 104 GiB / 约 77 GiB | Linux 上要更好的质量 |
| GGUF UD-IQ4_XS | Windows（Linux 未实测） | 0.100 | 87 GiB / 60.4 GiB（加 MTP 63.0） | 只为省显存，而且只在不用 MTP 时成立；质量与 v2 相当，速度不如 v2 |
| hgn w4b | Linux、Windows | 0.163 | 118 GiB / 68.0 GiB | 兼容已下载的旧版文件，不建议新下载 |

- 常驻内存是 Windows 上实测的权重占用，含 MTP 草稿头，不含 PLE n-gram 表。这张表留在磁盘上按需读取
  （hgn 47.7 GiB，GGUF 26.8 GiB），所以下载体积大于常驻内存。v2 主体里已经带了 MTP 草稿头，另外的
  `mtp.hgn` 不再多占显存；GGUF 的 MTP 草稿头是单独的文件，要另算。
- UD-Q4_K_XL 的 KLD 比 v2 低，主要因为 dense 部分（注意力、GDN、shared expert、lm_head）是 8-bit，
  v2 是 4-bit；代价是 dense 每 token 多读一倍字节，不投机 decode 慢约 16%。
- UD-IQ4_XS 的路由专家大多是 IQ3_S，比 v2 的 4-bit 更窄，dense 仍是 8-bit。它唯一的优势是显存占用低，
  但这只在不用 MTP 时成立：不带 MTP 比 v2 少 3.2 GiB，带上 Q8_0 MTP 草稿头反而多 1.3 GiB，而不带 MTP
  decode 只有投机时的一半左右。质量与 v2 基本持平：KLD 0.100 对 0.107、top-1 一致率 89.7% 对 88.8%
  略好，PPL（3.44 对 3.42）和最差 0.1% 位置的 KLD（3.11 对 2.91）略差。速度不如 v2：decode 慢约 12%，
  prefill 慢约 6%，多路并发吞吐低约 15%。
- KLD 的测法是 wikitext-2 64 chunk × 512，与 unsloth / llama.cpp 相同，见 [KLD.md](docs/KLD.md)。
- Unsloth 仓库里的其它档位（IQ1/IQ3_XXS/Q2/Q3/Q5/Q6 等）没有适配。

下载用 Hugging Face CLI（`pip install -U huggingface_hub`）。下面的命令都下载到项目的 `models/` 目录，
启动器默认从这里读取。

### hgn v2

来源：[peonist-ai/halogen-qwen3.8-flash-next](https://huggingface.co/peonist-ai/halogen-qwen3.8-flash-next)
（halogen 发布的权重，本引擎直接加载）。

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `qwen38-flash-next-v2.hgn` | 62.1 GiB | 主模型 |
| `qwen38-flash-next-ngram.hgn` | 47.7 GiB | PLE n-gram 表，留在磁盘按需读取 |
| `qwen38-flash-next-mtp.hgn` | 1.4 GiB | 8-bit MTP 草稿头 |
| `qwen38-flash-next-vision.hgn` | 0.84 GiB | 视觉塔；只用文本可以不下载，并设 `VISION_FILE=""` |
| `tokenizer/` | — | 词表 |

```bash
hf download peonist-ai/halogen-qwen3.8-flash-next --local-dir ./models --include \
  qwen38-flash-next-v2.hgn qwen38-flash-next-ngram.hgn \
  qwen38-flash-next-mtp.hgn qwen38-flash-next-vision.hgn "tokenizer/*"
```

仓库里的其它文件（`w4b`、`overlay`、`ht43`）不用下载。

### GGUF

来源：[unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)。
与 llama.cpp 用的是同一份文件，已经下载过的可以直接软链到 `models/`。

| 文件 | 大小 | 说明 |
| --- | --- | --- |
| `UD-Q4_K_XL/…-0000{1..4}-of-00004.gguf` | 103.7 GiB | 主模型（Linux），4 个分片都要 |
| `UD-IQ4_XS/…-0000{1..3}-of-00003.gguf` | 87.2 GiB | 主模型（Windows），3 个分片都要 |
| `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2.6 GiB | MTP 草稿头 |
| `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`（仓库根目录） | 3.9 GiB | MTP 草稿头，与 shared-Q8_0、Q4_K_M 版三选一 |
| `mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`（仓库根目录） | 2.6 GiB | MTP 草稿头；Q5_0 HC-up 在加载时转为 q8g32，已实测 |
| `mmproj-BF16.gguf` | 0.85 GiB | 视觉塔；也可以用 hgn 仓库的 `qwen38-flash-next-vision.hgn` |

GGUF 不带 `tokenizer.json`，要从原模型仓库另取（和 hgn 仓库 `tokenizer/` 里的是同一份）。

```bash
# Linux:UD-Q4_K_XL
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir ./models --include \
  "UD-Q4_K_XL/*" MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf mmproj-BF16.gguf
# Windows:UD-IQ4_XS(视觉塔用 hgn 仓库的 vision.hgn,Windows 上实测过)
hf download unsloth/Qwen3.8-Flash-Next-GGUF --local-dir ./models --include \
  "UD-IQ4_XS/*" mtp-Qwen3.8-Flash-Next-Q8_0.gguf
hf download peonist-ai/halogen-qwen3.8-flash-next --local-dir ./models --include \
  qwen38-flash-next-vision.hgn
# 两者都需要词表
hf download Qwen/Qwen3.8-Flash-Next tokenizer.json --local-dir ./models/tokenizer
```

分片下载后在子目录里，路径按「快速开始」一节填进 `service.conf`。

## 快速开始

Linux 和 Windows 用同一份源码、同一个 `models/` 目录和同一份 `service.conf`。按上一节下载权重，
把路径填进 `service.conf`，再按自己的系统编译、启动。启动后 API 在 `http://<主机>:8731/v1`
（标准 OpenAI 接口，支持流式；Anthropic 客户端用 `http://<主机>:8731/v1/messages`），浏览器打开 `http://<主机>:8731/` 是控制台。

### 填写权重路径

`service.conf` 里 hgn 和 GGUF 各有一段，启动时只读用到的那一段；同名环境变量优先于文件里的值。

hgn v2:

```bash
MODEL_FILE="${MODEL_FILE:-$MODEL_DIR/qwen38-flash-next-v2.hgn}"
NGRAM_FILE="${NGRAM_FILE:-$MODEL_DIR/qwen38-flash-next-ngram.hgn}"
OVERLAY_FILE="${OVERLAY_FILE-}"
```

GGUF 按下载后的实际子目录填写，`GGUF_FILE` 填第 1 个分片：

```bash
# Linux:UD-Q4_K_XL
GGUF_FILE="${GGUF_FILE:-$MODEL_DIR/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf}"
GGUF_MTP_FILE="${GGUF_MTP_FILE-$MODEL_DIR/MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf}"
GGUF_VISION_FILE="${GGUF_VISION_FILE-$MODEL_DIR/mmproj-BF16.gguf}"
# Windows:UD-IQ4_XS
GGUF_FILE="${GGUF_FILE:-$MODEL_DIR/UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf}"
GGUF_MTP_FILE="${GGUF_MTP_FILE-$MODEL_DIR/mtp-Qwen3.8-Flash-Next-Q8_0.gguf}"
GGUF_VISION_FILE="${GGUF_VISION_FILE-$MODEL_DIR/qwen38-flash-next-vision.hgn}"
```

`GGUF_MTP_FILE` 置空就不做投机。Windows 上也可以不手改文件，在控制台 `#/engine` 页填写，
保存后写回同一份 `service.conf`。加载和转换规则见 [GGUF.md](docs/GGUF.md)。

### Linux

要求：ROCm（HIP 7.x 及以上），GPU 为 gfx1151，可用内存 ≥ 100 GiB。编译依赖 rocBLAS、hipBLASLt、
rocPRIM，API 前端另需 libpng、libjpeg、libwebp。

```bash
bash build.sh        # 编译引擎、API 前端、benchmark 到 build/，TUI 设置工具到根目录
bash start_hgn.sh    # 用 hgn 权重启动
bash start_gguf.sh   # 或用 GGUF 权重启动
```

不想直接编辑 `service.conf` 时，可以运行 `./qwenox-settings` 使用跨平台终端设置界面（中英双语，
第 6 页「语言」切换）；也可以只运行
`bash build.sh settings` 单独编译它。按键和检查范围见 [SETTINGS-TUI.md](docs/SETTINGS-TUI.md)。

两个启动器只是读的权重不同：先加载引擎，就绪后再启动 API。它们在前台运行，Ctrl+C 同时停止两者；
远程登录时放在 tmux 里。缺文件时列出全部缺失项并退出，加 `--check` 只检查配置、不启动。
日志在 `logs/<日期时间>-<PID>/`。详见 [QUICKSTART.md](docs/QUICKSTART.md)，编译排错见
[BUILD.md](docs/BUILD.md)。

### Windows

功能和 Linux 一致（引擎、OpenAI API、多模态、并发）。要求：

- BIOS 里给 GPU 划分 96 GiB 显存。能开多大的上下文由显存决定，见「Windows 的显存上限」一节。
- 运行只需要 AMD 显卡驱动，不需要安装 ROCm。
- 编译需要 Git for Windows（提供 Git Bash）和 TheRock 的 Windows 包（默认
  `C:\therock-dist-windows-gfx1151-10.1.0`，放在别处时用 `THEROCK=` 指定），见 [BUILD.md](docs/BUILD.md)。

在 Git Bash 里编译（也可以双击 `build_win.bat`）：

```bash
bash build_win.sh    # 引擎、benchmark、API 托盘程序、根目录入口 start_win.exe
```

日常运行双击根目录的 `start_win.exe`。它在任务栏右下角放一个托盘图标，不开控制台窗口，也不会
自动加载模型。双击托盘图标打开控制台网页，在 `#/engine` 页选择权重（hgn 或 GGUF）、修改配置，
再启动引擎；右键菜单里也能启停引擎、复制 API 地址、查看日志。日志在 `logs\`。习惯命令行的话，
在 Git Bash 里运行 `bash start_win.sh` 也可以，行为和 Linux 启动器相同。习惯终端设置界面的话，
在 Windows Terminal / PowerShell / CMD 中运行根目录的 `qwenox-settings.exe`；它只编辑 `service.conf`，
不启动服务，使用说明见 [SETTINGS-TUI.md](docs/SETTINGS-TUI.md)。

编好的 `build/`、`start_win.exe` 连同 `models/` 拷到另一台 gfx1151 Windows 机器上可以直接运行，
那台机器不需要装 Git 或 TheRock。移植细节见 [PORTING-WINDOWS.md](docs/PORTING-WINDOWS.md)。

## 常用配置

配置集中在 `service.conf`（Windows 上也可以在控制台 `#/engine` 页修改），同名环境变量优先，
修改后重启引擎生效。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `MAX_CONTEXT` | `262144` | 单条请求的输入 + 输出上限 |
| `KV_POOL_TOKENS` | `0` | 所有并发请求共用的 KV 总量，0 表示等于 `MAX_CONTEXT`；每多 256K 约多占 9.6 GiB |
| `PARALLEL` | `4` | 同时运行的请求数，1–8 |
| `PREFILL_CHUNK` | `0` | 长 prompt 每段多少 token，0 表示 Linux 16384、Windows 6144 |
| `ROPE_FACTOR` | `1` | 1 为原生 256K；设为 2 并把 `MAX_CONTEXT` 改为 `524288` 即 YaRN 512K |
| `MTP_GAMMA` | `0` | 草稿长度，0 表示自动，1–8 表示固定 |
| `MOE_A4` | `0` | prefill 4-bit 加速（只对 hgn v2 生效），见下文；0 关，3 开 |

YaRN 是整个实例的设置，开启后短文本的输出分布也会略有变化（与原生 KLD 0.023），只在单条请求确实
需要超过 256K 时才开。512K 的配置方法、共享池准入规则和验证范围见 [YARN-512K.md](docs/YARN-512K.md)。

### prefill 4-bit 加速（MOE_A4）

可选开关，默认关。打开后，hgn v2 权重在 prefill 时把路由专家的输入激活按每 128 个元素量化成 int4，
专家 GEMM 改用 INT4 WMMA（`v_wmma_i32_16x16x16_iu4`，gfx1151 上的算力是 F16 WMMA 的 2 倍）。
`service.conf` 里设 `MOE_A4=3`，或在 Windows 设置面板的 V2 权重页勾选“prefill 4-bit 加速”。
取值是位掩码：1 只量化 gate/up，2 只量化 down，3 两者都量化。GGUF 和 w4b 权重忽略这个开关。

**影响范围**：只作用于 prefill，也就是一次处理超过 16 个 token 的输入：prompt、多轮对话里新加入的
消息、工具调用返回的结果。生成的 token 走 decode，MTP verify 和并发合批也不经过这条路径，
所以生成阶段的计算与关闭时相同。代价是模型读入的内容精度降低，对输入越长、越依赖读准输入的场景
影响面越大。

| | 关 | `=1` | `=2` | `=3` |
| --- | --- | --- | --- | --- |
| 64K prefill（Windows，tok/s） | 1609 | 1719 | 1615 | 1752（+8.9%） |
| KLD vs BF16 | 0.1075 | 0.1280 | 0.1226 | 0.1427 |

**做过的测试**（均为 hgn v2、Windows）：

- **KLD**：wikitext-2 64 chunk × 512，与 [KLD.md](docs/KLD.md) 同口径。这个测法每个位置都按 prefill
  打分，A4 的误差覆盖全部被测 token，反映的是读入精度的损失，不是生成质量的损失。
- **正确性**：开关关闭时，输出与加入此功能前的引擎逐 token 相同；`=3` 时，两条多段长 prompt
  并发运行、中途插入短请求，greedy 输出与单独运行逐 token 相同（`tools/a4_verify_win.sh`）。
- **长上下文检索**（`bench/codeneedle`，说明见 [bench/README.md](bench/README.md)）：用本项目源码
  拼成 16K / 32K / 64K 的长文，段落打乱，问 144 道答案唯一的题（环境变量在哪个文件读取、默认值是
  多少，函数定义在哪个文件，常量的值）。关和 `=3` 两轮 greedy 准确率都是 99.3%，144 题里 143 题
  输出逐字相同，没有一题一对一错；同一答案的对数概率平均变化不超过 0.01，不随长度增大；
  长文 prefill 1580 → 1726 tok/s。

**测试没覆盖到的**：

- 检索题的基线已接近满分，只能排除明显的退化，小幅损失测不出来。
- 没测 64K 以上的长度，没测需要串联多处信息的推理题。
- 没测代码生成、工具调用这类端到端任务。hipfire 作者在他们的模型上用 HumanEval、GSM8K、MMLU-Pro
  对比过开关两边，差异在噪声范围内，可作参考，但不是本项目的实测。

建议：需要处理大量长输入、更在意首字速度时打开；对精度敏感，或者不确定时保持关闭。

## 并发

多条请求可以同时进行，不用等前一条生成完。

- **共享一个 KV 页池**：`PARALLEL` 个槽位（1–8，默认 4）从同一个页池按需取页，每页 256 token，
  类似 llama.cpp 的共享上下文。页池默认总共 256K，可以是 4 条 64K，也可以是 1 条 200K 加几条短的。
  `MAX_CONTEXT` 只限制单条请求，所有并发请求的总量由页池大小 `KV_POOL_TOKENS` 决定。
- **decode 合批**：各路的 MTP verify 合成一批，在同一步里算完，路数越多每轮的权重读取被分摊得越多。
  草稿长度 γ 按同时在跑的路数自动选择。
- **长 prompt 不卡住别人**：prefill 在层与层之间每约 300 ms 让出一次 GPU，给正在 decode 的会话一轮，
  别的会话最长卡顿约 0.6 s；多条长 prompt 同时到达时，剩余较少的先跑完。
- **页池不够时**：先淘汰内存检查点，再释放空闲槽位的 KV，还不够就中断后来的请求（API 返回错误）。
  先来的请求不会被后来的请求挤掉。超出 `PARALLEL` 的请求排队等槽位。
- **结果可复现**：v2 权重上，greedy 请求和带 `seed` 的采样请求，并发时的输出与单独运行逐位相同。
  GGUF 在近似平局的 token 上偶尔会不同；需要严格一致时设 `QWENOX_OCC_FORCE=4`，吞吐会略降。

API 并发吞吐（Windows，写代码题，greedy，MTP 开，每路 2 × 1200 token，所有请求合计 tok/s）：

| 同时请求数 | 1 | 4 | 8 |
| --- | --- | --- | --- |
| hgn v2 | 55.3 | 83.2 | 97.0 |
| GGUF UD-IQ4_XS | 47.0 | 70.7 | 82.4 |

合计吞吐随路数上升，但每一路分到的速度会下降：8 路时每路约 12 tok/s。配置、调度和验证方法见
[CONCURRENCY.md](docs/CONCURRENCY.md)。

### 不同负载下的并发实测

上面的表是代码题的横向对比。MTP 接受率对并发吞吐影响很大，补一组不同负载的实测
（2026-10-23，Qwenox API，gfx1151 Windows，greedy，MTP 开，prompt cache 关，所有请求同时发起、
流式输出；合计吞吐取所有请求同时在跑的窗口，不是单路速度相加）：

| 负载 | 并发 | MTP 接受率 | 单路稳态 | 合计吞吐 |
| --- | --- | --- | --- | --- |
| 合成文本（256 prompt / 512 output） | 2 | 47.4% | 30.0 tok/s | 58.4 tok/s |
| HTML 代码（俄罗斯方块 + 打砖块，完整生成） | 2 | 77–83% | 43.5 tok/s | 83.5 tok/s |
| 创意写作（8 个约 800 字故事） | 8 | 49.1% | 10.7 tok/s | 85.9 tok/s |
| HTML 代码（8 个页面，每路 1024 token 上限） | 8 | 69.4% | 13.7 tok/s | 109.7 tok/s |

- 合计吞吐随路数上升，但边际递减：高接受率代码负载从 2 路的 83.5 tok/s 到 8 路的 109.7 tok/s，
  只提升约 1.3 倍，单路却从 43.5 降到 13.7 tok/s。8 路适合吞吐优先的后台任务；交互式使用 2–4 路
  更合适。
- MTP 接受率是并发放大的关键：同样是 8 路，创意写作 49.1% 只有 85.9 tok/s，HTML 代码 69.4% 能到
  109.7 tok/s。接受率越高，每轮 verify 推进的 token 越多，越能把多路的权重读取摊薄。
- 8 路同时到达时，TTFT 从约 0.34 s 线性升到约 2.47 s，基本每多一路多约 300 ms，prefill 是分时
  处理的，不是 8 路并行。单路 ITL 均值也从 2 路代码的约 22–24 ms 升到 62–97 ms，标准差 82–145 ms，
  生成更不平稳。
- token 数取自服务端 `usage`，接受率取自 `/reqstat` 的 chain drafter（accepted/proposed）。2 路
  创意写作（43–50% 接受率）重叠期约 57 tok/s、单路约 28.5 tok/s，与合成文本一致。
- 上表 8 路的 97.0 tok/s 与本组 109.7 tok/s 不是同一组参数（题目、输出长度、接受率都不同），
  不要直接相减比较。

## Windows 的显存上限

Windows 上能开多大的上下文由显存决定。单个进程只能可靠使用启动时一次性申请的一整块显存，上限约
95 GiB，权重、KV 页池、prefill 工作区、视觉塔都要装进这块。运行中如果不够，超出的部分会落到共享内存
（主机内存），速度和稳定性都没有保证。下表是这块显存的实际用量（2026-10-13 实测，托盘程序的生产配置：
MTP 开、视觉塔开、prefill chunk 8192）：

| 配置 | hgn v2 | UD-IQ4_XS + MTP | UD-IQ4_XS 无 MTP | hgn w4b |
| --- | --- | --- | --- | --- |
| 256K 页池，4 路 | 82.6 GiB | 83.9 GiB | 79.4 GiB | 88.8 GiB |
| 512K 页池（单条 256K），2 路 | 92.2 GiB | 93.5 GiB | — | 放不下（超 6 GiB） |
| 512K 页池（单条 256K），8 路 | 93.8 GiB | 刚好超出（约 70 MiB） | — | — |
| YaRN 512K（单条 512K），2 路 | 93.3 GiB | 94.5 GiB | — | — |

- **v2 可以开到 512K**：可以是 2 条各 256K 的请求同时驻留（`KV_POOL_TOKENS=524288`），也可以是 YaRN
  下单条 512K（`MAX_CONTEXT=524288`、`ROPE_FACTOR=2`）。8 路并发也放得下，但只剩约 1 GiB 余量；图片多
  的多模态请求在运行时还要申请显存，512K 时建议 2–4 路。
- **UD-IQ4_XS 带 MTP 比 v2 多占约 1.3 GiB**，512K 页池最多 2–4 路。
- **w4b 只能开 256K**。
- 上表在 prefill chunk 8192 下测得；默认的 6144 每项约少 1.7 GiB，PP 与 8192 只差约 0.5%。16384 放不下。
- 页池超过 512K 时启动器会拒绝启动。
- UD-Q4_K_XL 常驻 77 GiB，按上表推算 256K 就要约 97 GiB，Windows 上不支持。
- 任务管理器的“专用 GPU 内存”显示的是启动时预留的整块：上表用量加最多 4 GiB 余量（封顶 95 GiB），
  再加约 1.3 GiB，所以 v2 256K 时约 88.6 GiB，512K 配置时都是约 95.7 GiB。引擎另占 3–4 GiB 主机内存。

### 并发路数与显存

显存主要花在页池上，不在路数上。下表是 hgn v2 在默认 prefill chunk 6144 下的实测（2026-10-14，MTP 开、
视觉塔开、`MAX_CONTEXT=262144`；“实际用量”同上表口径，另两列是任务管理器的读数，单位 GB）：

| 页池 | 路数 | 实际用量 | 专用 GPU 内存 | 共享 GPU 内存 |
| --- | --- | --- | --- | --- |
| 256K | 1 | 79.7 GiB | 85.8 | 1.9 |
| 256K | 2 | 80.3 GiB | 86.3 | 1.9 |
| 512K | 1 | 89.8 GiB | 95.7 | 2.1 |
| 512K | 2 | 90.5 GiB | 95.7 | 2.5 |
| 512K | 4 | 91.0 GiB | 95.6 | 2.5 |

- 从 1 路开到 2 路多约 0.6 GiB，之后每多一路约 0.27 GiB。页池从 256K 加到 512K 多约 10 GiB，与路数无关。
- 比较“单路和双路”时要保持 `KV_POOL_TOKENS` 不变；同时改了页池，差出来的约 10 GiB 是页池的。
- 显存在启动时一次分配完。512K 页池下 2 条 32K、2 条 100K 同时跑完，上表各项都不再增长，也没有溢出到
  共享内存。
- 512K 页池时 1 路和多路的“专用 GPU 内存”读数相同：预留块已顶到 95 GiB 上限，看不出路数的差别。
- 复测：停掉生产服务后在 Git Bash 里运行 `bash tools/vram_conc_win.sh`（默认 512K 页池 1/2/4 路，约 5 分钟/配置；
  `PARS="1 2" POOL=0 SHORT=1` 测 256K 页池且只测启动和一条 8K），汇总在 `logs/vram_conc/summary.txt`。

## 崩溃诊断

引擎意外退出时，API 自动把日志、崩溃报告（Windows 附小 minidump）、系统与驱动信息、Windows 事件日志打成一个诊断包，
放在 `logs/diag/`。Windows 托盘会弹提示，点一下就在资源管理器里选中这个 zip；没崩溃但卡住或输出异常时，右键托盘图标
→"导出诊断包…"。Linux 下由启动脚本自动打包，也可以手动运行 `build/qwenox-api --collect-diag`。遇到问题时把 zip 发给开发者即可，
包里不含请求内容。详见 [CRASHDIAG.md](docs/CRASHDIAG.md)。

## 文档

- [QUICKSTART.md](docs/QUICKSTART.md) — 编译、启动、配置
- [BUILD.md](docs/BUILD.md) — 编译环境与排错
- [GGUF.md](docs/GGUF.md) — GGUF 权重的加载方式与实测
- [CONVERT.md](docs/CONVERT.md) — 模型转换工具
- [HGN-FORMAT.md](docs/HGN-FORMAT.md) — `.hgn` 权重容器格式
- [CONCURRENCY.md](docs/CONCURRENCY.md) — 并发请求的配置、调度与验证
- [YARN-512K.md](docs/YARN-512K.md) / [YARN-512K-RESULTS.md](docs/YARN-512K-RESULTS.md) — YaRN 512K 与实测
- [MTP.md](docs/MTP.md) / [NGRAM.md](docs/NGRAM.md) — 投机解码
- [KLD.md](docs/KLD.md) — 质量测试（与 unsloth / llama.cpp 同口径）
- [BENCHMARK.md](docs/BENCHMARK.md) — 性能测试工具
- [bench/README.md](bench/README.md) — 任务级评测（长上下文代码捞针 A/B）
- [PORTING-WINDOWS.md](docs/PORTING-WINDOWS.md) — Windows 移植与实测
- [POWER.md](docs/POWER.md) — 功耗/温度读数的可信度与平台差异（为什么 APU 上没有单独的 CPU 功耗）
- [CRASHDIAG.md](docs/CRASHDIAG.md) — 崩溃报告、诊断包、minidump 符号化与自测

测试：`bash build.sh test` 跑 kernel 单测（不加载模型，预期 ALL PASS）；`python3 tools/bpw.py` 统计
`models/` 下各权重的 bpw。

## 致谢

本项目的实现方式借鉴了 peonist-ai 的 [halogen-flash-server](https://github.com/peonist-ai/halogen-flash-server)；`.hgn` 权重容器格式即 halogen 的 checkpoint 容器格式（见 [HGN-FORMAT.md](docs/HGN-FORMAT.md)）。感谢 halogen 作者的工作。

GGUF 支持大量参考了 [gufo](https://github.com/gufo-org/gufo)（MIT 许可证）：路由专家 F16 WMMA GEMM kernel 移植自其 RoutedF16GEMMKernel（`src/gpu/parts/26_kernels_moe_gguf.inc`），hgn q4cp / GGUF IQ4 的 LUT 解码版沿用同一条流水线（`27_kernels_moe_lut.inc`），GGUF 与引擎张量间的变换语义参考其 reference.cpp（`src/gguf_map.h`）；prefill 的 HC 门控融合、生产者 epilogue 直写下一 GEMM 输入、多行 Q8 dense kernel 等优化也借鉴了 gufo 的做法（对照分析见 [GUFO-GAP.md](docs/GUFO-GAP.md)）。

prefill 4-bit 加速（`MOE_A4`）借鉴了 warpfront 的 [hipfire](https://github.com/warpfront/hipfire)（Apache-2.0 许可证）：路由专家激活按块量化成 int4、专家 GEMM 走 INT4 WMMA 的做法来自 hipfire 的 IU4 路由，本项目按 hgn v2 的 i4r 格式重新实现（`k_quant_a4` / `k_moe_iu4`，`src/gpu/parts/27_kernels_moe_lut.inc`）；LUT 专家 GEMM 的 packed epilogue 也参考了 hipfire 的写法。hipfire 作者分享的任务评测结果帮助我们判断了这个开关的取舍。感谢 hipfire 作者。

ngram 投机解码的起草思路与两级 prompt 缓存借鉴了开源项目 [llama.cpp](https://github.com/ggml-org/llama.cpp)（MIT 许可证），GGUF 权重与视觉塔（mmproj）直接复用 llama.cpp 的同一份文件；IQ3_S 码表取自 llama.cpp；ngram 声明详见 [NGRAM.md](docs/NGRAM.md) 的「来源与声明」一节。

## 许可证

[AGPL-3.0](LICENSE)
