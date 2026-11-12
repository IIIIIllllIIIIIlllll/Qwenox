# 跨平台 TUI 设置工具

`qwenox-settings` 是 Linux / Windows 共用的终端设置程序，只读取、检查和写回
`service.conf`，不启动引擎或 API。它与 Windows 的 `settings_win.exe` GUI、网页控制台共用
同一份配置；原 GUI 不受此工具影响。

## 编译

Linux：

```bash
bash build.sh settings
```

Windows（Git Bash 中编译）：

```bash
bash build_win.sh settings-tui
```

该目标是纯主机程序；默认 TheRock 路径不可用时，可以单独指定任意兼容的 C++17 编译器，例如：

```bash
CXX="/c/Program Files/AMD/ROCm/7.1/bin/clang++.exe" bash build_win.sh settings-tui
```

两个平台的默认 `all` 构建也会生成该工具。产物位于项目根目录：

- Linux：`./qwenox-settings`
- Windows：`./qwenox-settings.exe`

只依赖 C++17 标准库和系统控制台，不需要 ncurses、HIP 或 ROCm 运行库。

## 使用

在 Linux 终端中运行：

```bash
./qwenox-settings
```

Windows 请在 Windows Terminal、PowerShell 或 CMD 中运行：

```powershell
.\qwenox-settings.exe
```

原生 Windows 程序不能直接附着到 Git Bash 的 MSYS 伪终端；如果一定要在 Git Bash 里运行，
可使用 Git for Windows 附带的 `winpty ./qwenox-settings.exe`。

编辑其他位置或其他项目的配置：

```bash
./qwenox-settings --conf /path/to/service.conf
```

只检查、不进入界面、不写文件：

```bash
./qwenox-settings --check
./qwenox-settings --conf /path/to/service.conf --check
```

`--check` 发现失败项时退出码为 1，否则为 0。权重视为必填，因此模型尚未下载时文件检查会失败，
这是预期行为。

## 按键

| 按键 | 功能 |
| --- | --- |
| `↑` / `↓` | 选择字段；检查页滚动；语言页选择语言 |
| `←` / `→` | 切换页面；编辑文本时移动光标 |
| 数字 `1`–`6` | 直接切换到对应页面 |
| `Enter` | 编辑文本，或切换布尔值/选项；编辑时确认；有路径候选时先接受补全；语言页应用所选语言 |
| `Tab` / `Shift+Tab` | 非编辑状态下切换下一项 / 上一项；路径编辑时 Tab 补全/循环候选 |
| `Backspace` / `Delete` | 删除光标前 / 后的字符 |
| `Home` / `End` | 移到文本开头 / 末尾 |
| `Ctrl+U` | 清空当前文本 |
| `F2` / `Ctrl+S` | 保存 |
| `Esc` / `q` | 取消编辑或退出；有未保存修改时需要确认 |

### 界面语言

界面支持中文和英文，默认按以下顺序决定：`service.conf` 的 `# start_win: lang=zh|en`
注释标记 > 环境变量 `UI_LANG=zh|en` > 系统界面语言（Windows 取系统 UI 语言；Linux 按
`LANGUAGE` / `LC_ALL` / `LANG` 判断）。

在第 6 页「语言 / Lang」用 `↑`/`↓` 选择、`Enter` 或空格应用，切换**立即生效**并写入
lang 标记，不需要 F2 保存。该标记与 Windows 设置面板、托盘共用，一处切换三处生效。
`--check` 输出和 `--help` 帮助同样随语言。

### 路径自动补全

权重文件、MTP、视觉塔、i4r 补丁、tokenizer、KV 快照目录和请求日志目录支持 Tab 补全：

1. 按 Enter 开始编辑路径，再按 Tab 枚举当前目录的一层候选。
2. 只有一个候选时直接补全；多个候选时先补到最长公共前缀并显示列表。
3. 继续按 Tab 循环候选，按 Enter 接受补全并留在编辑状态；再按 Enter 确认字段。
4. 目录候选会追加路径分隔符，接受后再按 Tab 即可枚举下一级。tokenizer、快照和日志目录只列出目录。
5. 移动光标或继续输入会关闭旧候选；Esc 仍取消整个字段编辑。

相对路径以 `service.conf` 所在目录为基准，支持绝对路径、`$MODEL_DIR` / `${MODEL_DIR}` 和 `~/`。
Windows 匹配不区分大小写并接受两种路径分隔符，Linux 保持大小写敏感。补全只读取当前目录，不递归扫描。

保存前存在失败项时会要求再次确认。保存后保持当前界面，重启服务后配置才会生效。

## 配置范围

界面分为六页：

1. **权重**：hgn v1 / v2 / GGUF、主模型、n-gram、overlay、MTP、视觉塔、GGUF i4r 补丁、tokenizer。
2. **推理**：上下文、并发、KV 页池、prefill 分段、QSA、YaRN、MOE_A4、MTP gamma、图片上限。
3. **缓存/启动**：SSD/RAM 检查点、PLE io_uring、Linux 内存与超时、Windows CPU 亲和性。
4. **端口/日志**：引擎与 API 地址、运行时可视化、完整请求日志。
5. **检查**：文件、数值、跨参数约束、平台差异和环境变量覆盖。
6. **语言**：切换界面中英文；立即生效并写入 `# start_win: lang=` 标记（Windows 设置面板和托盘读取同一标记）。

标有 `[Linux]` 或 `[Windows]` 的字段在两端都能编辑，便于提前为另一平台准备同一份配置，但只会在
对应平台生效。未出现在界面里的高级配置行会原样保留。

权重选择继续写 `# start_win: weights=v1|v2|gguf` 标记，以兼容现有 Windows GUI 和网页控制台。

## 检查与保存边界

静态检查包括必填文件、HGN/GGUF magic、数值范围、YaRN 上限、分页 KV 与并发关系、端口格式和
环境变量覆盖。Windows 会检查 512K KV 上限；Linux 只提示该配置不能直接用于 Windows。

它不替代启动器的完整环境检查。端口是否被占用、ROCm/运行库、systemd 用户会话和实际可用内存，
仍应在启动前执行：

```bash
bash start_hgn.sh --check
# 或
bash start_gguf.sh --check
```

配置优先级仍为 **环境变量 > service.conf > 内置默认**。检查页会列出非空环境变量覆盖；此时修改文件
不会覆盖当前进程环境中的值。可选文件留空表示禁用。

写回由现有 `svcconf::save_conf` 完成：保留注释、空行和未知键，已知键保持 bash 可 `source` 的格式，
写前生成 `service.conf.bak`。程序用单实例锁避免两个 TUI 同时编辑同一份配置，但不要同时在其他设置
界面中保存同一文件。
