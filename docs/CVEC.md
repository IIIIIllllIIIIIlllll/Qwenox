# 控制向量（refusal projection）

qwenox 可以加载 llama.cpp 格式的控制向量 GGUF，在每层 hyper-connection（HC）残差上做
**投影消融**（project）或**加法引导**（add）。向量不修改权重，启动时决定，运行期零额外开销
（关闭时）或每层一次残差读写（开启时）。

参考向量：[Cudecnik/Qwen3.8-Flash-Next-refusal-projection](https://huggingface.co/Cudecnik/Qwen3.8-Flash-Next-refusal-projection)
（`Qwen3.8-Flash-Next-refusal-projection.gguf`，约 480 KB；参考实现在 Cudila/llama.cpp 的
`cvec-hooks` 分支）。qwenox 的语义、层映射和参数与它对齐。

## 开启方法

service.conf（同名环境变量优先）：

```bash
CVEC="${CVEC-}"                       # 控制向量 GGUF 路径；留空 = 关闭
CVEC_MODE="${CVEC_MODE:-project}"     # project | add
CVEC_SCALE="${CVEC_SCALE:-1.0}"
CVEC_LAYERS="${CVEC_LAYERS:-4 44}"    # 闭区间 "<起> <止>"，1..47
```

Windows 设置面板、终端设置（`bash build.sh settings`）与网页启动器/API 控制台都有对应项，
且都放在**权重**页/组：它是模型资产而非运行参数，三种权重格式通用，并随权重方案一起
保存/切换。层区间用**双端滑条**调节（网页/面板拖动两端；终端里 `←/→` 移动当前端、
`↑/↓` 一次 5 层、`Tab` 换端）。落盘格式仍是 `"4 44"`。改完重启引擎生效（与 KV_TYPE 一致）。

直接运行引擎时用环境变量：

| 变量 | 默认 | 说明 |
|---|---|---|
| `QWENOX_CVEC` | 空（关闭） | 控制向量 GGUF 路径 |
| `QWENOX_CVEC_MODE` | `project` | `project` 或 `add` |
| `QWENOX_CVEC_SCALE` | `1.0` | 必须 > 0 |
| `QWENOX_CVEC_LAYERS` | `4 44` | 闭区间，`1 <= 起 <= 止 <= 47` |
| `QWENOX_CVEC_DIR` | `per-layer` | 或 `single:<层号>`：整个区间都用该层的方向 |
| `QWENOX_CVEC_PROBE` | 未设 | 诊断：打印该层投影前/后的 cos（会禁用 decode 图） |

启动日志确认加载成功：

```text
qwenox: cvec /path/…-refusal-projection.gguf: mode=project dir=per-layer layers=4..44 scale=1, 41/41 directions (norm 0.999998..1.000001)
```

没有这一行就是没加载（`--check` 也会打印 `ENV QWENOX_CVEC=…`）。

## 语义

每层 mlp HC 写入之后、下一层读取之前，对 4 条 HC 残差分支逐 token：

```text
project:  h ← h − s·(h·v)·v      (v 单位化，s = scale × |原始向量|)
add:      h ← h + s·v            (s = scale，v 保持原样)
```

- 张量 `direction.K` 用于第 K 层（0 基）的输出残差（与 llama.cpp 控制向量的层编号一致）；
  `CVEC_LAYERS` 是闭区间。
- project 加载时把向量单位化并把原范数并入 scale；实测参考文件范数为 1.0，两者等价。
- 参考配置：project、per-layer、scale 1.0、层 4..44。加法文件（`…-refusal-additive.gguf`）
  只配 add 模式、层 16..32、scale 0.05–0.15——**不要**用 1.0，会输出乱码。
- 只作用于 trunk；MTP / DFlash2 草稿头不另加向量（草稿的验证由目标模型完成，与参考一致）。
- 与权重格式无关：hgn v1/v2 与 GGUF 都能用。

## 实现与开销

- 投影内核每层每个 (token, 分支) 一个 block：两级归约求 `h·v`，再统一回写；残差是 bf16
  （`QWENOX_GR_BF16`，启动器默认开）时按 bf16 舍入。参考实现是 f32 残差，所以 bf16 下
  post cos 不会精确为 0（通常 1e-3 量级），不影响功能。
- project 保持融合写开启：prefill 的 mlp 残差写与下一层 norm 是同一个 kernel
  （`k_gr_scatter_norm_*`），投影插在“写回 bf16 残差”与“求平方和/norm”之间。
  每 token 每分支一次块内 dot，不增加残差流量。
- add 使用独立的写回 → `k_cvec_apply` → norm 路径。启动时关闭 project 专用的
  scatter+norm 融合，避免 prefill/投机验证执行投影公式，并保持与独立路径一致。
  `QWENOX_HC_FUSE_WRITE`（把写回折进 moe reduce，默认关）在所有 CVEC 模式下都关闭。
- 开销（project 的既有测量）：decode 每层每 token 多一次 4×2560 的读+写（约 80 KB），实测 29.3 → 29.4 tok/s
  （在噪声内）。prefill 8K 实测 1187/1183/1226（关）vs 1146/1176/1230 tok/s（开），
  在噪声内。显存额外约 0.5 MB（47×2560 f32）。add 的独立路径增加 kernel 调用，
  尚未对其吞吐量作专门复测。
- 启用后 SSD/RAM 快照指纹会混入模式、scale、层区间、向量内容及各层有效 scale 的哈希：开/关切换或换向量文件后
  旧快照不会被恢复（KV/GDN 状态与残差投影不兼容）。关闭时指纹与之前完全一致。

## 验证

- 投影正确性：`QWENOX_CVEC_PROBE=<层号>` 启动后发一条普通 prompt，日志出现
  `cvec probe L20 (project): br0 pre=… post=… … PASS`（project 且 post |cos| < 2e-2）。
  probe 会禁用 decode 图，并让 prefill 走独立的投影 kernel（融合 kernel 里没有 host 可读的
  中间值），只用于诊断。
- 数值影响：`tools/kld_engine.sh --tokens <ids> logs/stock.kld stock` 生成无向量基准，
  再用 `REF=logs/stock.kld` 带向量跑一次，记录 mean/p99 KL（参考实现自报 0.186，重尾）。
- 速度与输出：`tools/cvec_eval.py --base http://127.0.0.1:8731` 抓短生成的 decode tok/s、
  输出前 80 字与 JSON，用于开/关对比。
- 回归测试：`bash build.sh test` / `bash build_win.sh test` 包含：
  - f32/bf16 × add/project 的逐元素与正交性检查。
  - 六种融合写回/norm 变体的 project/关闭对照；覆盖 1、3、4、5、9 token，
    同时与独立 CPU 公式比对残差和 norm 输出，包含 T=4 的尾批。
  - 实际加载两个同方向、不同范数的 GGUF，检查有效 scale、指纹隔离和关闭时的零指纹。
  - 两套 shell 启动链路的导出代码：开启时保留显式 DIR/PROBE、未配置时保持未设，关闭时清理。

## 注意

- 投影会移除模型的安全拒答行为，输出由使用者负责；这是行为改变而非精度问题。
- 调节手段只有 scale 与层区间；没有按请求开关（与 llama.cpp 的启动参数一致）。
- 向量文件缺失、维度不是 2560、层区间越界、模式或 scale 非法时引擎直接报错退出，不会静默降级。
