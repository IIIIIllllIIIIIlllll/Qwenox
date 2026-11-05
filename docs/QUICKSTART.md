# 编译和启动

*English: [QUICKSTART_EN.md](QUICKSTART_EN.md)*

在 Linux / ROCm 机器上进入项目目录,先编译,再按手上的权重格式二选一启动:

```bash
bash build.sh
bash start_hgn.sh    # hgn 权重(tools/flashnext2hgn.py 转换得到)
bash start_gguf.sh   # 或 GGUF 权重(Unsloth UD-Q4_K_XL,与 llama.cpp 同一份文件)
```

`build.sh` 一次编译 GPU 引擎和原生 API,输出 `build/qwenox-engine`、`build/qwenox-api`。
两个启动器除了读哪份权重之外完全相同:先加载引擎,等就绪后启动 API。
默认 256K 上下文、MTP gamma 3。按 **Ctrl+C** 同时停止本次启动的 API 和引擎。
启动不会退出终端;保持 SSH 会话打开,或在 tmux 中运行。旧的 `start.sh`
只打印上面的选择提示并退出。

启动后 API 默认在 `http://<主机>:8731/v1`。
日志按每次启动保存在 `logs/日期时间-PID/engine.log` 和 `api.log`。

## 模型文件

两个启动器都从项目根目录的 `models/` 读取权重(`service.conf` 的
`MODEL_DIR`,默认 `./models`),`tokenizer/` 两种格式共用(只需
`tokenizer.json`,取自原模型 HF 仓库)。只需放自己要用的那一种:

```text
models/
  tokenizer/tokenizer.json                              # 两种都要
  # --- bash start_hgn.sh ---
  qwen38-flash-next-w4b.hgn                             # 主模型
  qwen38-flash-next-w4b.overlay.hgn                     # overlay
  qwen38-flash-next-mtp.hgn                             # 8-bit MTP 草稿
  qwen38-flash-next-vision.hgn                          # 视觉塔
  # --- bash start_gguf.sh ---
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf     # 主模型,4 个分片都要
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
  Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf
  mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf               # MTP 草稿 sidecar
  mmproj-BF16.gguf                                      # 视觉塔
```

缺文件时启动器不会启动,而是一次列出全部缺失的文件和配置项,退出码 1,例如:

```text
错误：start_gguf.sh（GGUF 权重）缺少以下文件：
  分片 3/4：./models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf
```

`start_hgn.sh` 只接受 `.hgn`、`start_gguf.sh` 只接受 `.gguf`,放错会提示
改用另一个启动器。文件名不同时修改 `service.conf` 里对应的一段:

- hgn:`MODEL_FILE`、`OVERLAY_FILE`、`MTP_FILE`、`VISION_FILE`。不需要
  overlay 可设 `OVERLAY_FILE=""`;`MTP_FILE` 是独立的 8-bit MTP 投机草稿
  权重,设 `MTP_FILE=""` 则退回 overlay 内置的 4-bit 草稿头。用
  `tools/flashnext2hgn.py` 转换自有模型时会自动生成这套文件,见 CONVERT.md。
- GGUF:`GGUF_FILE`(填第 1 个分片,其余分片须在同一目录)、
  `GGUF_MTP_FILE`、`GGUF_VISION_FILE`。设 `GGUF_MTP_FILE=""` 则不做 MTP 投机
  (仍有 ngram 草稿)。精度与性能对比见 GGUF.md。

两者纯文本服务都可以把视觉塔设为空(`VISION_FILE=""` / `GGUF_VISION_FILE=""`)。
`TOKENIZER_DIR` 两者共用。

所有相对路径都以脚本所在项目目录为基准,不受打开终端的位置影响。
也可临时指定配置,无需修改文件(同名环境变量优先):

```bash
MODEL_DIR=/data/models VISION_FILE="" bash start_hgn.sh
GGUF_VISION_FILE="" bash start_gguf.sh
```

## 投机解码

默认 drafter 是 chain(ngram 优先、MTP 兜底),引擎侧默认生效,无需任何
参数;贪心与采样请求都走 chain。HTTP API 不暴露 drafter 选择,请求体里
发 `drafter` 字段会被静默忽略。环境变量 `QWENOX_DRAFTER=ngram` 纯 ngram、
`=mtp` 纯 MTP、`=serial` 串行基线。

`MTP_GAMMA=1 bash start_hgn.sh`(或 `start_gguf.sh`)可试一轮草稿长度 1,范围 1–8,默认 3;
修改后需重启引擎,不需要编译。参数和接受率的含义见 MTP.md。

## 其他常用命令

```bash
bash start_hgn.sh --check    # 仅检查文件、端口、内存等,不启动服务(start_gguf.sh 同)
bash build.sh engine         # 只编译引擎
bash build.sh api            # 只编译 API
bash build.sh test           # 编译并运行 kernel 单测,不加载模型
```

端口、监听地址、上下文、MTP、内存上限集中在 `service.conf`。默认 API
监听 `0.0.0.0:8731`,可从局域网访问;只需本机访问时改为
`API_HOST="127.0.0.1"`。

引擎内部端口默认只监听 `127.0.0.1:8730`，不接受外机直连；监听 IPv4 地址
由 `ENGINE_HOST` 配置，端口由 `ENGINE_PORT` 配置。通常只对外开放 API 即可。
仅确需直接连接引擎时，才显式将 `ENGINE_HOST` 改为本机网卡 IP 或 `0.0.0.0`
（会允许外机直连，需自行限制网络访问）。直接运行引擎时使用
`--serve --host 127.0.0.1 --port 8730`；省略 `--host` 也只监听本机。

运行时内存可通过 `curl http://127.0.0.1:8731/memory` 查询。返回值区分
HIP 设备分配、`mmap + hipHostRegister` 的专家权重、pinned host 内存和
进程 RSS；`gpu_accessible_committed_bytes` 是引擎自身可准确记账的合计，
不会把未注册、可回收的文件 mmap 页缓存误算为显存。

### Repeat penalties（llama.cpp 语义）

chat / completions / responses / Anthropic Messages 请求都可使用 `repeat_last_n`
（整数，默认 64）和 `repeat_penalty`（有限数值，默认 1.0，必须 > 0）。只给其中一个
字段时，另一个取默认值。两个字段也都支持下文的服务端覆盖。

只有请求体**显式包含**其中任一字段时才进入 llama.cpp penalties 模式；由覆盖表注入
也算显式包含。此时 repeat、presence、frequency penalties 使用同一个由 **prompt +
已接受生成 token** 组成的窗口。两个 repeat 字段都没出现时，旧请求的
`presence_penalty` / `frequency_penalty` 行为保持不变。

- `repeat_last_n=0` 禁用本请求的整个 penalties 路径（包括 presence/frequency）；
  `-1` 使用本请求的全部历史；其余有效范围是 `1..2147483647`。
- `repeat_penalty=1.0` 虽是中性值，显式写出仍会进入上述共享窗口模式。
- sampled 解码使用 dense correction。显式 repeat 在 greedy（`temperature=0`）下也会
  生效，引擎需要时回退到串行解码，API 不会因为是 greedy 而拒绝。连接不支持该能力的
  旧引擎时，显式 repeat 请求返回 502；未显式 repeat 的旧请求不受影响。

### 控制台与服务端参数覆盖

浏览器打开 `http://<主机>:8731/` 即是控制台,包括健康状态、内存记账、
"采样 / 思考参数(服务端覆盖)"和"对话测试"(流式调用
`/v1/chat/completions`,思考内容单独显示)。

覆盖表用来防止客户端传错采样参数或思考开关。可覆盖的字段:
`enable_thinking`、`reasoning_effort`、`preserve_thinking`、`temperature`、
`top_p`、`top_k`、`min_p`、`presence_penalty`、`frequency_penalty`、
`repeat_last_n`、`repeat_penalty`、`max_tokens`。每个字段三种模式:

- 跟随客户端:不干预(表里没有该字段)。
- 默认值:仅在客户端没传该字段时使用。
- 强制:无论客户端传什么都改成此值;`max_tokens` 的强制是上限,客户端
  给得更大才压到此值。强制 `temperature=0` 时请求里的 `logprobs` 会被去掉。

覆盖在请求解析之前改写请求体,对 chat / responses / completions 都生效
(responses 的思考强度是 `reasoning.effort`;思考字段对 completions 无意义)。
被改写的字段写进响应头 `X-Qwenox-Overrides`(流式同样有)。当前表也出现在
`/health` 的 `server_overrides`。

接口:`GET /admin/overrides` 读取,`POST /admin/overrides` 整表替换,`{}`
表示清空。示例:

```bash
curl -X POST http://127.0.0.1:8731/admin/overrides -H 'Content-Type: application/json' \
  -d '{"enable_thinking":{"mode":"force","value":true},
       "reasoning_effort":{"mode":"default","value":"medium"},
       "temperature":{"mode":"force","value":0.6}}'
```

表持久化在 `data/api-overrides.json`(相对启动目录,重启后自动加载);
`qwenox-api --overrides FILE` 可改路径,`--overrides ''` 表示只放内存。默认
任何能访问 API 的人都能修改覆盖表(与 API 本身同样开放)。需要限制时在
启动 API 的环境里设置 `QWENOX_API_ADMIN_KEY=<密钥>`,之后 POST 必须带
`X-Admin-Key: <密钥>` 或 `Authorization: Bearer <密钥>`;控制台会显示
密钥输入框。一键验证:`bash tools/api_override_verify.sh`(测试端口
8732/8733,需先停掉生产服务)。

引擎使用 `tools/run_capped.sh`,默认 86 GiB 内存上限、启动前至少
100 GiB 可用内存;另一个引擎或同项目启动任务运行时会拒绝重复启动。
加载阶段持续 60 秒没有日志、引擎 I/O 或 GPU GTT 分配进展会停止;最长
启动时间 360 秒。退出只清理本次启动的进程。

### 自定义网页（不用重编）

控制台和可视化页面编译时嵌入 API，源文件在 `src/api/static/`：`index.html`、
`dashboard.html`、`app.js`、`skin-cassette.css`、`skin-punch.css`、`skin-constructivist.css`、
`skin-aero.css`、`aero-landscape.svg`、`skin-faded.css`、`qwen-faded.png`、`viz-live.html`、
`viz-model.html`。项目根（`service.conf` 所在目录）的 `web/` 里有同名文件时，API
返回这份文件。每次请求都现读，改完刷新浏览器就生效，不用重编，也不用重启。

1. 新建 `web/`，把要改的文件从 `src/api/static/` 拷进去再改。只认上面这几个文件名，
   放别的文件不会被提供。
2. 刷新浏览器。响应头带 `X-Qwenox-Static: disk` 说明用的是磁盘上的文件。
3. 想恢复内置版本，删掉 `web/` 里对应的文件即可。

`web/` 不入库，`git pull` 不会和自己的修改冲突。代价是升级后 `web/` 里的旧文件
仍会顶替新版内置页面，新功能可能在页面上看不到或出错。API 启动时如果 `web/` 存在，
日志会列出当前被覆盖的文件（`qwenox-api: web: ... 当前覆盖: ...`），升级后对照新版
`src/api/static/` 合并或删掉。

`service.conf` 的 `WEB_DIR` 可以换目录，相对路径以项目根为基准。开发时设
`WEB_DIR=src/api/static`，改源码即生效。

### Anthropic Messages 接口

除 OpenAI 接口外,API 还提供 Anthropic 协议:`POST /v1/messages` 和
`POST /v1/messages/count_tokens`。Claude Code 等 Anthropic 客户端把基址设为
`http://<主机>:8731` 即可(例如 `ANTHROPIC_BASE_URL=http://<主机>:8731`),
`x-api-key` / `anthropic-version` / `anthropic-beta` 请求头接受但不校验,
`model` 字段不参与选模型。

请求转成 chat 请求体后走与 `/v1/chat/completions` 相同的模板、工具解析和
服务端覆盖(覆盖表按 chat 字段生效)。字段对应:

| Anthropic | 本服务 |
|---|---|
| `system`(字符串或 text 块) | system 消息 |
| `text` / `image`(base64)/ 纯文本 `document` 块 | 用户内容;URL 图片不拉取,返回 400 |
| `tool_use` / `tool_result` 块 | assistant `tool_calls` / tool 消息 |
| 历史中的 `thinking` 块 | `reasoning_content`(`redacted_thinking` 忽略) |
| 末条 assistant 消息 | 预填:回复从它续写,不回显预填内容,不输出思考 |
| `thinking.type` enabled/adaptive/disabled | `enable_thinking`;`budget_tokens` 忽略 |
| `output_config.effort` | `reasoning_effort` |
| `tools[].input_schema` | 函数参数;无 `input_schema` 的服务端工具(web_search 等)忽略 |
| `tool_choice` auto/any/tool/none | auto/required/指定函数/none |
| `stop_sequences` | `stop`;命中时 `stop_reason=stop_sequence` 并回填 `stop_sequence` |

`temperature`、`top_p`、`top_k`、`max_tokens` 同名透传;本服务的 chat 扩展字段
(`min_p`、`seed`、`repeat_last_n`、`repeat_penalty`、`enable_thinking`、
`tool_call_format` 等)也可直接写在请求里。

响应按 Anthropic 格式返回 `thinking`(`signature` 为空串)、`text`、`tool_use`
内容块;`stop_reason` 为 `end_turn` / `max_tokens` / `stop_sequence` / `tool_use`;
`usage.cache_read_input_tokens` 是 KV 前缀缓存命中的 token 数,
`input_tokens` 为其余部分。流式按 `message_start`、`content_block_start` /
`content_block_delta`(`thinking_delta`、`text_delta`、`input_json_delta`)/
`content_block_stop`、`message_delta`、`message_stop` 输出,排队和 prefill 期间发
`ping`。错误使用 `{"type":"error","error":{"type":...,"message":...}}`,流已开始后
以 `error` 事件发出。`count_tokens` 按同一模板渲染后返回 `input_tokens`。

## 编译环境

验证环境:Ubuntu,`gfx1151` GPU,ROCm HIP 7.x / AMD clang。
GPU 编译固定使用 `-O3 -Werror`,链接 rocBLAS 和 hipBLASLt,还需要
rocPRIM 头文件。API 使用 C++17 / `-O2 -Werror`,nlohmann/json 已随仓库提供，
另依赖 libpng、libjpeg、libwebp 和 pthread。Ubuntu 安装命令:

```bash
sudo apt install build-essential libpng-dev libjpeg-dev libwebp-dev
```

脚本优先查找 PATH 中的 `hipcc`,否则使用 `/opt/rocm/bin/hipcc`。
其他 ROCm 位置或 GPU 可显式指定:

```bash
HIPCC=/opt/rocm/bin/hipcc GPU_ARCH=gfx1151 bash build.sh
```

编译在 8 GiB 内存限额和超时保护下进行(引擎 600 秒、API 各目标
120 秒)。在较慢的机器上超时不代表源码错误;确认编译器仍有进展,
再根据机器能力调整 `build.sh` 的超时。细节见 BUILD.md。
