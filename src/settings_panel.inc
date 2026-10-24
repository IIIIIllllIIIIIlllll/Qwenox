// settings_panel.inc — settings_win.exe 的设置面板（由 settings_win.cpp include，单 TU）。
// 源自归档的旧启动器面板 attic/launcher/launch_panel.inc，已去掉"启动服务"
// 流程：只编辑 / 检查 / 保存 service.conf，不启动服务（启动用 start_win.exe）。
// Win32 原生窗口，零新运行时依赖；不用 .rc/对话框模板（llvm-rc 不可用），
// CreateWindowExW 自建窗口 + 子控件，代码内布局。
//
// 面板分 4 页，顶部 4 个 BS_PUSHLIKE|BS_AUTORADIOBUTTON 切换（不引入 comctl32）：
//          各版本只显示自己相关的文件行；GGUF 由 qwenox-win 按 weights=gguf 标记
//          直接启动（GGUF_FILE 首分片 + GGUF_MTP_FILE + GGUF_VISION_FILE）。
//          V2 另有"prefill 4-bit 加速"勾选框（MOE_A4，勾 = 3 / 不勾 = 0，会降低
//          prompt 部分精度，检查区出 [警告] 说明代价；V1/GGUF 不显示、原值保留）。
//   第 2 页 推理设置：MAX_CONTEXT / PARALLEL / 并发总池（KV_POOL_TOKENS）/
//          KV_PAGED / PREFILL_CHUNK / KV 快照（KVSNAP_MAX_GB + KVSNAP_DIR）/
//          YaRN 勾选（ROPE_FACTOR 按上限自动推导）/
//          处理器亲和性勾选（不限制 / 亲和到 CCD 1，落盘 AFFINITY ""/upper-half）。
//          GGUF 仅保存（本机服务由 start_win.exe 启动，只支持 hgn；GGUF 配置供 Linux
//          bash start_gguf.sh 使用）。
//          ROPE_ORIGINAL_CTX 不做 UI 行（用户不该改）：校验仍从 g_conf/环境读，
//          也不在 kManagedKeys 里，save_conf 永远不会改写 conf 中的该行。
//   第 3 页 端口与监听：ENGINE_HOST / ENGINE_PORT / API_HOST / API_PORT。
//   第 4 页 显存检查：HIP 可用显存、权重合计、并发总池/工作区/并发估算、
//          合计对照结论，以及其它页的失败汇总；"重新检查"按钮在本页。
// 每页底部有独立的多行只读检查区（[通过]/[失败]/[警告]/[提示] 前缀）。
// 底部按钮行（保存 / 关闭 / 语言切换）在所有页之外，恒可见。
// "保存" 不做检查门禁：写盘成功即关闭面板，失败弹错误框保持打开。
//
// 左侧立绘条：media/qwen-charactor-alpha.png 加载成功时窗口左侧加宽
// g_art_w（210px），所有控件经 add_ctrl 统一右移，WM_PAINT 里 GDI+ 绘制
// （PNG alpha 合成，底部对齐）；图片缺失时 g_art_w=0，退回经典紧凑布局。
//
// 依赖 settings_win.cpp 中已定义的：g_root、g_root_w、g_conf、load_conf()、cfg()、
// cfg_optional()、to_w()、msgbox()、file_exists()、dir_exists()、same_file()、
// port_free()、port_listening()、app_icon()、TR 宏、kIconIco，
// 以及 engine_net::address()（settings_win.cpp 已 include engine_net.h）。
//
// 入口：panel_run()
//   打开时按 conf 的 API_PORT 探测服务是否在监听（g_service_running）：
//   运行中则端口占用检查降级为提示、显存余量结论降级为对照。
// 返回 kPanelClose（保存后关闭 / 关闭按钮 / 窗口关闭）/ kPanelRelang（切换了
// 界面语言，调用方重开面板）。
//
// 界面语言：中英双语，文案处用 TR(中文, English) 就地双语。优先级：
//   service.conf 的 "# start_win: lang=zh|en" 标记 > 环境变量 UI_LANG >
//   GetUserDefaultUILanguage()（中文→zh，其它→en）。
// 面板右下角语言按钮写 lang 标记并返回 kPanelRelang。
// 检查级别前缀本地化为 [通过]/[失败]/[警告]/[提示] 与 [PASS]/[FAIL]/[WARN]/[HINT]，
// 判定用 Lv 枚举（不再字符串比较）。控制台模式的诊断输出未翻译。
//
// 权重版本选择持久化为注释行 "# start_win: weights=v1|v2|gguf"（保存时与键值
// 同一次写入 upsert，见 svcconf::save_conf 的 marker 参数）；无标记时按
// NGRAM_FILE 是否与 MODEL_FILE 同文件推断 V1/V2。

#include <commdlg.h>
#include <objbase.h>    // CoInitializeEx / CoCreateInstance（KV 快照目录选择框）
#include <shobjidl.h>   // IFileOpenDialog + FOS_PICKFOLDERS
#include <set>

namespace {

// ---- service.conf 读写 -------------------------------------------------------

const char* kWeightsMarker = "# start_win: weights=";
const char* kLangMarker = "# start_win: lang=";

// 界面语言开关：panel_run 每次打开时按 lang 标记 > UI_LANG > 系统 UI 语言解析。
// TR 宏在 settings_win.cpp 顶部定义（设置工具文案也用），就地双语
//（两个分支必须一个窄串对窄串、一个宽串对宽串）。
bool g_lang_en = false;

std::string conf_path() { return g_root + "\\service.conf"; }

using svcconf::read_text_file;

// 读 "# start_win: <prefix><value>" 行的值；没有返回空串。
std::string conf_marker_value(const char* prefix) {
    std::string text;
    if (!read_text_file(conf_path(), &text)) return "";
    const size_t p = text.find(prefix);
    if (p == std::string::npos) return "";
    size_t b = p + strlen(prefix);
    size_t e = text.find_first_of("\r\n", b);
    std::string v = text.substr(b, e == std::string::npos ? std::string::npos : e - b);
    const size_t a = v.find_first_not_of(" \t");
    const size_t z = v.find_last_not_of(" \t");
    return a == std::string::npos ? "" : v.substr(a, z - a + 1);
}

std::string conf_weights_marker() { return conf_marker_value(kWeightsMarker); }

// 写 "prefix+value" 标记行：已有该前缀行则改写，否则追加到文件末尾。
void conf_write_marker_line(const char* prefix, const char* value) {
    const std::string path = conf_path();
    std::string text;
    read_text_file(path, &text);
    std::string out;
    bool written = false;
    size_t pos = 0;
    while (pos <= text.size()) {
        size_t eol = text.find('\n', pos);
        const bool last = eol == std::string::npos;
        std::string line = text.substr(pos, last ? std::string::npos : eol - pos);
        pos = last ? text.size() + 1 : eol + 1;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!written && line.compare(0, strlen(prefix), prefix) == 0) {
            out += std::string(prefix) + value;
            written = true;
        } else {
            out += line;
        }
        if (!last) out += '\n';
    }
    if (!written) {
        if (!out.empty() && out.back() != '\n') out += '\n';
        out += prefix + std::string(value) + "\n";
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(out.data(), 1, out.size(), f);
    fclose(f);
}

// 界面语言解析：lang 标记 > UI_LANG > 系统 UI 语言（中文 → zh，其它 → en）。
void panel_lang_resolve() {
    const std::string m = conf_marker_value(kLangMarker);
    if (m == "en") return void(g_lang_en = true);
    if (m == "zh") return void(g_lang_en = false);
    if (const char* e = getenv("UI_LANG")) {
        if (_stricmp(e, "en") == 0) return void(g_lang_en = true);
        if (_stricmp(e, "zh") == 0) return void(g_lang_en = false);
    }
    g_lang_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_CHINESE;
}

// 面板管理的 conf 键（也是环境变量覆盖提示的范围）与可选文件键：
// 定义在 service_conf.h（svcconf 命名空间，与 gdec-api 共享）。
// ROPE_ORIGINAL_CTX 刻意不在其中：面板不做它的 UI 行，save_conf 也不改写它。
using svcconf::kManagedKeys;
using svcconf::is_optional_file_key;

// 环境变量覆盖提示归属的页：0 权重 / 1 推理 / 2 端口。
int page_of_key(const std::string& k) {
    if (k == "MAX_CONTEXT" || k == "PARALLEL" || k == "KV_POOL_TOKENS" || k == "KV_PAGED" ||
        k == "PREFILL_CHUNK" || k == "ROPE_FACTOR" || k == "KVSNAP_MAX_GB" ||
        k == "KVSNAP_DIR" || k == "AFFINITY")
        return 1;
    if (k == "ENGINE_HOST" || k == "ENGINE_PORT" || k == "API_HOST" || k == "API_PORT")
        return 2;
    return 0;
}

// 写回 service.conf：实现在 service_conf.h（svcconf::save_conf）——注释/空行/
// 未知键原样保留；已知键整行替换为 KEY="${KEY:-值}"（可选文件键为
// KEY="${KEY-值}"）；键缺失追加到文件末尾；写前备份为 service.conf.bak；
// marker 参数可在同一次写入里 upsert 注释标记行（如权重版本标记）。
// panel_apply_and_save 直接调用 svcconf::save_conf（带 weights 标记）。

// ---- 小工具 -------------------------------------------------------------------

std::string from_w(const std::wstring& w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()),
                                      nullptr, 0, nullptr, nullptr);
    std::string s(static_cast<size_t>(n), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), static_cast<int>(w.size()), &s[0], n,
                        nullptr, nullptr);
    return s;
}

std::string trim_copy(const std::string& s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// 不退出的数值解析（面板检查用；launch_win.cpp 的 cfg_int/parse_double 会 fail 退出）
bool pn_int(const std::string& s, long* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    const long n = strtol(s.c_str(), &end, 10);
    if (!end || *end) return false;
    *out = n;
    return true;
}

bool pn_double(const std::string& s, double* out) {
    if (s.empty()) return false;
    char* end = nullptr;
    errno = 0;
    const double n = strtod(s.c_str(), &end);
    if (errno || !end || *end || end == s.c_str() || n != n) return false;
    *out = n;
    return true;
}

bool file_size_of(const std::string& path, unsigned long long* size) {
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &d)) return false;
    *size = (static_cast<unsigned long long>(d.nFileSizeHigh) << 32) | d.nFileSizeLow;
    return true;
}

// HGN1 头（104B）+ 记录表（160B/条，布局见 src/hgn.h）：只读文件头，
// 返回指定 tensor 的 data_size。用于把磁盘分页的 PLE n-gram 表从权重
// 文件大小里剔除 —— 它按需 gather，不驻留显存（v1 内嵌、v2 独立文件）。
#pragma pack(push, 1)
struct HgnHeader {
    char magic[4];
    uint32_t version, tensor_count, reserved;
    uint64_t records_offset, data_offset, file_size;
    char model_name[64];
};
struct HgnRecord {
    char name[96];
    uint32_t dtype, ndims;
    uint64_t dims[4];
    uint64_t data_offset, data_size, extra;
};
#pragma pack(pop)
static_assert(sizeof(HgnHeader) == 104 && sizeof(HgnRecord) == 160,
              "HGN1 layout must match src/hgn.h");

unsigned long long hgn_tensor_bytes(const std::string& path, const char* name) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    HgnHeader h{};
    unsigned long long bytes = 0;
    if (fread(&h, sizeof h, 1, f) == 1 && memcmp(h.magic, "HGN1", 4) == 0 &&
        h.tensor_count <= 4096 &&
        _fseeki64(f, (long long)h.records_offset, SEEK_SET) == 0) {
        for (uint32_t i = 0; i < h.tensor_count; i++) {
            HgnRecord r{};
            if (fread(&r, sizeof r, 1, f) != 1) break;
            if (strncmp(r.name, name, sizeof r.name) == 0) {  // name[96] 定长，可能无 NUL
                bytes = r.data_size;
                break;
            }
        }
    }
    fclose(f);
    return bytes;
}

// GGUF：在一个分片文件里找张量 name，返回它的数据字节数（0 = 没找到/不认识的
// 类型）。只读头部：KV 段逐项跳过（数组按元素类型跳），张量表按块大小算字节。
// 块大小表与 src/gguf.h block_info 一致（只列本项目会遇到的类型）。
unsigned long long gguf_tensor_bytes_in(const std::string& path, const char* name) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return 0;
    unsigned long long out = 0;
    auto rd = [&](void* p, size_t n) { return fread(p, 1, n, f) == n; };
    auto skip = [&](unsigned long long n) { return _fseeki64(f, (long long)n, SEEK_CUR) == 0; };
    auto scalar_size = [](uint32_t t) -> unsigned {
        switch (t) {
            case 0: case 1: case 7: return 1;
            case 2: case 3: return 2;
            case 4: case 5: case 6: return 4;
            case 10: case 11: case 12: return 8;
            default: return 0;
        }
    };
    auto skip_str = [&]() {
        unsigned long long n = 0;
        return rd(&n, 8) && n < (1ull << 32) && skip(n);
    };
    char magic[4];
    uint32_t ver = 0;
    unsigned long long nt = 0, nkv = 0;
    bool ok = rd(magic, 4) && memcmp(magic, "GGUF", 4) == 0 && rd(&ver, 4) && ver >= 2 &&
              rd(&nt, 8) && rd(&nkv, 8) && nt < 100000 && nkv < 100000;
    for (unsigned long long i = 0; ok && i < nkv; i++) {
        uint32_t t = 0;
        ok = skip_str() && rd(&t, 4);
        if (!ok) break;
        if (t == 8) {
            ok = skip_str();
        } else if (t == 9) {
            uint32_t at = 0;
            unsigned long long n = 0;
            ok = rd(&at, 4) && rd(&n, 8);
            if (!ok) break;
            if (at == 8) {
                for (unsigned long long j = 0; ok && j < n; j++) ok = skip_str();
            } else {
                const unsigned es = scalar_size(at);
                ok = es && skip(n * es);
            }
        } else {
            const unsigned es = scalar_size(t);
            ok = es && skip(es);
        }
    }
    for (unsigned long long i = 0; ok && i < nt; i++) {
        unsigned long long n = 0;
        ok = rd(&n, 8) && n < 4096;
        if (!ok) break;
        std::string nm(n, '\0');
        uint32_t nd = 0, type = 0;
        unsigned long long ne[4] = {1, 1, 1, 1}, off = 0;
        ok = rd(&nm[0], n) && rd(&nd, 4) && nd <= 4;
        for (uint32_t d = 0; ok && d < nd; d++) ok = rd(&ne[d], 8);
        ok = ok && rd(&type, 4) && rd(&off, 8);
        if (!ok || nm != name) continue;
        unsigned be = 0, bb = 0;
        switch (type) {
            case 0: be = 1; bb = 4; break;      // F32
            case 1: case 30: be = 1; bb = 2; break;  // F16 / BF16
            case 7: be = 32; bb = 24; break;    // Q5_1
            case 8: be = 32; bb = 34; break;    // Q8_0
            case 12: be = 256; bb = 144; break; // Q4_K
            case 13: be = 256; bb = 176; break; // Q5_K
            case 14: be = 256; bb = 210; break; // Q6_K
            case 20: be = 32; bb = 18; break;   // IQ4_NL
            case 21: be = 256; bb = 110; break; // IQ3_S
            case 23: be = 256; bb = 136; break; // IQ4_XS
            default: break;
        }
        if (be) out = ne[0] * ne[1] * ne[2] * ne[3] / be * bb;
        break;
    }
    fclose(f);
    return out;
}

// GGUF 多分片：逐个分片找张量（name-00001-of-0000N.gguf 命名，同 gguf_shards_size）。
unsigned long long gguf_tensor_bytes(const std::string& first, const char* name) {
    const size_t of = first.find("-of-");
    size_t dash = std::string::npos;
    if (of != std::string::npos && of > 0) dash = first.rfind('-', of - 1);
    if (dash == std::string::npos) return gguf_tensor_bytes_in(first, name);
    const int n = atoi(first.substr(of + 4).c_str());
    for (int i = 1; i <= n && i < 1000; i++) {
        char idx[16];
        snprintf(idx, sizeof idx, "%05d", i);
        const unsigned long long b =
            gguf_tensor_bytes_in(first.substr(0, dash + 1) + idx + first.substr(of), name);
        if (b) return b;
    }
    return 0;
}

std::string sniff_magic(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    char b[4] = {};
    const size_t n = fread(b, 1, 4, f);
    fclose(f);
    return n == 4 ? std::string(b, 4) : "";
}

std::string gib(double bytes) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f", bytes / 1073741824.0);
    return buf;
}

// 绝对路径转项目相对（./...，正斜杠），bash 启动器兼容；不在项目下则原样返回。
std::string make_relative(std::string path) {
    for (auto& c : path)
        if (c == '\\') c = '/';
    std::string root = g_root;
    for (auto& c : root)
        if (c == '\\') c = '/';
    if (path.size() > root.size() + 1 && _strnicmp(path.c_str(), root.c_str(), root.size()) == 0 &&
        path[root.size()] == '/')
        return "." + path.substr(root.size());
    return path;
}

// ---- 面板取值 -----------------------------------------------------------------

enum WeightsVer : int { kVerV1 = 0, kVerV2 = 1, kVerGguf = 2 };

struct PanelVals {
    int weights_ver = kVerV1;
    std::string model, ngram, overlay, mtp, vision, tok;
    bool no_overlay = false, no_mtp = false, no_vision = false;
    std::string gguf, gguf_mtp, gguf_vision;
    bool no_gguf_mtp = false, no_gguf_vision = false;
    std::string max_context, parallel, kv_pool, prefill_chunk;
    bool kv_paged = true;
    std::string rope_factor, rope_original;  // rope_original 无 UI 行，从 conf/环境读
    std::string kvsnap_max_gb;
    std::string kvsnap_dir;  // KVSNAP_DIR：空 = 引擎默认 data\kvsnap
    std::string moe_a4 = "0";  // MOE_A4：0 关，1–3 开（只对 V2 生效）
    std::string affinity;
    std::string engine_host, engine_port, api_host, api_port;
};

// MOE_A4（prefill MoE 4-bit 激活，见 service.conf）：1–3 为开，其它一律视为关。
bool moe_a4_on(const std::string& s) { return s == "1" || s == "2" || s == "3"; }

PanelVals panel_vals_from_conf() {
    PanelVals v;
    const std::string model_dir = cfg("MODEL_DIR", "models");
    v.model = cfg("MODEL_FILE", model_dir + "\\qwen38-flash-next-w4b.hgn");
    v.ngram = cfg("NGRAM_FILE", v.model);
    const bool ngram_follows = v.ngram == v.model;
    if (ngram_follows) v.ngram.clear();  // 空 = 跟随主模型
    v.overlay = cfg_optional("OVERLAY_FILE", "");
    v.no_overlay = v.overlay.empty();
    v.mtp = cfg_optional("MTP_FILE", model_dir + "\\qwen38-flash-next-mtp.hgn");
    v.no_mtp = v.mtp.empty();
    v.vision = cfg_optional("VISION_FILE", model_dir + "\\qwen38-flash-next-vision.hgn");
    v.no_vision = v.vision.empty();
    v.gguf = cfg("GGUF_FILE",
                 model_dir + "\\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf");
    v.gguf_mtp =
        cfg_optional("GGUF_MTP_FILE", model_dir + "\\mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf");
    v.no_gguf_mtp = v.gguf_mtp.empty();
    v.gguf_vision = cfg_optional("GGUF_VISION_FILE", model_dir + "\\mmproj-BF16.gguf");
    v.no_gguf_vision = v.gguf_vision.empty();
    v.tok = cfg("TOKENIZER_DIR", model_dir + "\\tokenizer");
    v.max_context = cfg("MAX_CONTEXT", "262144");
    v.parallel = cfg("PARALLEL", "1");
    v.kv_pool = cfg("KV_POOL_TOKENS", "0");
    v.kv_paged = cfg("KV_PAGED", "1") != "0";
    v.prefill_chunk = cfg("PREFILL_CHUNK", "0");
    v.rope_factor = cfg("ROPE_FACTOR", "1");
    v.rope_original = cfg("ROPE_ORIGINAL_CTX", "262144");
    v.kvsnap_max_gb = cfg("KVSNAP_MAX_GB", "20");
    v.kvsnap_dir = cfg_optional("KVSNAP_DIR", "");
    v.moe_a4 = cfg("MOE_A4", "0");
    v.affinity = cfg("AFFINITY", "");
    v.engine_host = cfg("ENGINE_HOST", "127.0.0.1");
    v.engine_port = cfg("ENGINE_PORT", "8730");
    v.api_host = cfg("API_HOST", "0.0.0.0");
    v.api_port = cfg("API_PORT", "8731");
    // 版本：标记优先；无标记时 NGRAM_FILE 解析后与 MODEL_FILE 不同 → V2，否则 V1
    const std::string marker = conf_weights_marker();
    if (marker == "v1") v.weights_ver = kVerV1;
    else if (marker == "v2") v.weights_ver = kVerV2;
    else if (marker == "gguf") v.weights_ver = kVerGguf;
    else v.weights_ver = ngram_follows ? kVerV1 : kVerV2;
    return v;
}

// 面板值 → conf 键值。只写所选版本相关的权重键，其余保持 conf 原值不动。
// V1：NGRAM_FILE 写字面量 $MODEL_FILE（bash 默认展开 = 与主模型同文件）。
// V2：OVERLAY_FILE 写空（${KEY-} 无冒号风格，空=禁用）；NGRAM_FILE 为独立文件。
// GGUF：hgn 各键不动，只写 GGUF_* 与公共键。
std::map<std::string, std::string> panel_vals_to_conf(const PanelVals& v) {
    std::map<std::string, std::string> m;
    if (v.weights_ver == kVerGguf) {
        m["GGUF_FILE"] = v.gguf;
        m["GGUF_MTP_FILE"] = v.no_gguf_mtp ? "" : v.gguf_mtp;
        m["GGUF_VISION_FILE"] = v.no_gguf_vision ? "" : v.gguf_vision;
    } else {
        m["MODEL_FILE"] = v.model;
        if (v.weights_ver == kVerV2) {
            m["NGRAM_FILE"] = v.ngram;
            m["OVERLAY_FILE"] = "";
        } else {
            m["NGRAM_FILE"] = "$MODEL_FILE";
            m["OVERLAY_FILE"] = v.no_overlay ? "" : v.overlay;
        }
        m["MTP_FILE"] = v.no_mtp ? "" : v.mtp;
        m["VISION_FILE"] = v.no_vision ? "" : v.vision;
    }
    std::string tok = v.tok;
    while (!tok.empty() && (tok.back() == '\\' || tok.back() == '/')) tok.pop_back();
    m["TOKENIZER_DIR"] = tok;
    m["MAX_CONTEXT"] = v.max_context;
    m["PARALLEL"] = v.parallel;
    m["KV_POOL_TOKENS"] = v.kv_pool;
    m["KV_PAGED"] = v.kv_paged ? "1" : "0";
    m["PREFILL_CHUNK"] = v.prefill_chunk;
    m["ROPE_FACTOR"] = v.rope_factor;
    m["KVSNAP_MAX_GB"] = v.kvsnap_max_gb;
    std::string kdir = v.kvsnap_dir;
    while (!kdir.empty() && (kdir.back() == '\\' || kdir.back() == '/')) kdir.pop_back();
    m["KVSNAP_DIR"] = kdir;
    m["MOE_A4"] = v.moe_a4;
    m["AFFINITY"] = v.affinity;
    m["ENGINE_HOST"] = v.engine_host;
    m["ENGINE_PORT"] = v.engine_port;
    m["API_HOST"] = v.api_host;
    m["API_PORT"] = v.api_port;
    return m;
}

// ---- 环境检查（按页拆分） -------------------------------------------------------

struct CheckLines {
    std::string text;  // \r\n 分隔，前缀 [通过]/[失败]/[警告]/[提示]
    std::vector<std::string> fatals;  // 失败行正文（第 4 页汇总用）
    bool fatal = false;
};

// 级别用枚举判定，前缀文本按语言渲染（[失败] / [FAIL] 等）。
enum Lv : int { kLvPass, kLvFail, kLvWarn, kLvHint };

const char* lv_text(Lv lv) {
    switch (lv) {
        case kLvPass: return TR("通过", "PASS");
        case kLvFail: return TR("失败", "FAIL");
        case kLvWarn: return TR("警告", "WARN");
        default: return TR("提示", "HINT");
    }
}

void check_line(CheckLines& c, Lv lv, const std::string& msg) {
    c.text += "[" + std::string(lv_text(lv)) + "] " + msg + "\r\n";
    if (lv == kLvFail) {
        c.fatal = true;
        c.fatals.push_back(msg);
    }
}

void check_weight(CheckLines& c, const char* label, const std::string& path,
                  bool required, bool warn_skip) {
    if (!file_exists(path)) {
        if (warn_skip)
            check_line(c, kLvWarn, std::string(label) + TR("：找不到 ", ": ") + path +
                                       TR("，引擎将跳过", " not found; engine will skip"));
        else if (required)
            check_line(c, kLvFail,
                       std::string(label) + TR("：找不到 ", ": not found: ") + path);
        return;
    }
    unsigned long long size = 0;
    file_size_of(path, &size);
    // 不按大小判定：合法 sidecar 可以很小（转换脚本产出的 MTP 仅 ~93 MB），
    // 真坏的文件引擎加载时会报错
    // magic 嗅探：.hgn → HGN1（src/hgn.h），.gguf → GGUF；其它扩展名跳过
    const size_t dot = path.find_last_of('.');
    const std::string ext =
        dot == std::string::npos ? "" : path.substr(dot + 1);
    if (_stricmp(ext.c_str(), "hgn") == 0 || _stricmp(ext.c_str(), "gguf") == 0) {
        const std::string want = _stricmp(ext.c_str(), "hgn") == 0 ? "HGN1" : "GGUF";
        const std::string magic = sniff_magic(path);
        if (magic != want) {
            check_line(c, kLvFail,
                       std::string(label) + TR("：不是有效的 .", ": not a valid .") + ext +
                           TR(" 文件（magic 非 ", " file (magic != ") + want + TR("）", "): ") +
                           path);
            return;
        }
        check_line(c, kLvPass,
                   std::string(label) + TR("：", ": ") + path + TR("（", " (") + gib((double)size) +
                       TR(" GiB，", " GiB, ") + want + TR("）", ")"));
    } else {
        check_line(c, kLvWarn, std::string(label) + TR("：", ": ") + path + TR("（", " (") +
                                   gib((double)size) +
                                   TR(" GiB，扩展名未知，未做格式嗅探）",
                                      " GiB, unknown extension; no format sniffing)"));
    }
}

// 第 1 页：权重（按所选版本检查对应的文件集合 + tokenizer）。
void check_weights_page(CheckLines& c, const PanelVals& v) {
    if (v.weights_ver == kVerGguf) {
        check_weight(c, TR("GGUF 主分片", "GGUF main shard"), v.gguf, true, false);
        if (v.gguf.find("-00001-of-") != std::string::npos)
            check_line(c, kLvPass,
                       TR("GGUF 分片命名：填的是第 1 个分片，其余分片须在同一目录",
                          "GGUF shards: the first shard is entered; keep the rest in the same "
                          "directory"));
        else
            // 单文件 GGUF（未分片，或 llama-gguf-split --merge 合并后）是合法的：
            // 引擎只在文件名形如 -0000i-of-0000N.gguf 时按分片打开，否则按单文件打开
            check_line(c, kLvHint,
                       TR("GGUF_FILE 按单文件加载（文件名不含 -00001-of-XXXXX）；若是多分片模型，"
                          "请填第 1 个分片",
                          "GGUF_FILE is loaded as a single file (no -00001-of-XXXXX in the name); "
                          "for a multi-shard model enter the first shard"));
        if (v.no_gguf_mtp)
            check_line(c, kLvPass, TR("GGUF MTP：不使用", "GGUF MTP: not used"));
        else
            check_weight(c, "GGUF MTP", v.gguf_mtp, true, false);
        if (v.no_gguf_vision)
            check_line(c, kLvPass, TR("GGUF 视觉塔：不使用", "GGUF vision tower: not used"));
        else
            check_weight(c, TR("GGUF 视觉塔", "GGUF vision tower"), v.gguf_vision, true, false);
        check_line(c, kLvHint,
                   TR("GGUF 支持 Unsloth UD-Q4_K_XL / UD-IQ4_XS（Windows 已验证 UD-IQ4_XS）",
                      "GGUF: Unsloth UD-Q4_K_XL / UD-IQ4_XS (UD-IQ4_XS verified on Windows)"));
    } else {
        check_weight(c, TR("主模型", "Main model"), v.model, true, false);
        if (v.weights_ver == kVerV2) {
            if (v.ngram.empty())
                check_line(c, kLvFail,
                           TR("n-gram 表：V2 权重必须填独立文件",
                              "n-gram table: V2 weights require a separate file"));
            else if (same_file(v.model, v.ngram))
                check_line(c, kLvFail,
                           TR("n-gram 表：V2 必须是与主模型不同的独立文件",
                              "n-gram table: V2 needs a file separate from the main model"));
            else
                check_weight(c, TR("n-gram 表", "n-gram table"), v.ngram, true, false);
            check_line(c, kLvPass,
                       TR("overlay：V2 不使用（保存时置空，引擎本就会跳过）",
                          "overlay: unused with V2 (cleared on save; the engine skips it anyway)"));
            if (moe_a4_on(v.moe_a4))
                check_line(c, kLvWarn,
                           TR("prefill 4-bit 加速已开启：prompt 处理约快 9%，代价是精度降低——"
                              "prompt 部分逐 token KLD 0.108 → 0.143；生成的 token 走 decode，不受影响",
                              "4-bit prefill speedup on: prompt processing ~9% faster at the cost of "
                              "precision (prompt per-token KLD 0.108 -> 0.143); generated tokens go "
                              "through decode and are unaffected"));
        } else {
            check_line(c, kLvPass,
                       TR("n-gram 表：与主模型同文件（V1 内置）",
                          "n-gram table: same file as the main model (built into V1)"));
            if (v.no_overlay)
                check_line(c, kLvPass, TR("overlay：不使用", "overlay: not used"));
            else
                check_weight(c, "overlay", v.overlay, false, true);
        }
        if (v.no_mtp)
            check_line(c, kLvPass, TR("MTP 草稿：不使用", "MTP draft: not used"));
        else
            check_weight(c, TR("MTP 草稿", "MTP draft"), v.mtp, true, false);
        if (v.no_vision)
            check_line(c, kLvPass, TR("视觉塔：不使用", "Vision tower: not used"));
        else
            check_weight(c, TR("视觉塔", "Vision tower"), v.vision, true, false);
    }
    if (!dir_exists(v.tok))
        check_line(c, kLvFail, TR("tokenizer：目录不存在 ", "tokenizer: directory missing: ") + v.tok);
    else if (!file_exists(v.tok + "\\tokenizer.json"))
        check_line(c, kLvFail,
                   TR("tokenizer：找不到 ", "tokenizer: missing ") + v.tok + "\\tokenizer.json");
    else
        check_line(c, kLvPass, TR("tokenizer：", "tokenizer: ") + v.tok + "\\tokenizer.json");
}

// 第 2 页：推理设置（上下文/并发/YaRN）的数值校验（与启动流程同一套规则）。
void check_context_page(CheckLines& c, const PanelVals& v) {
    // 说明行（不参与判定，恒在列表最上方）
    c.text += TR("· 单请求上限：一条请求的输入+输出最大 token 数，模型原生 262144。\r\n",
                 "· Per-request context: max input+output tokens per request; the model is "
                 "natively 262144.\r\n");
    c.text += TR("· 并发总池：所有并发共享的 KV 缓存总量（类似 vLLM 的 KV cache），单请求上限不能超过它；\r\n",
                 "· KV pool: total KV cache shared by all concurrent streams (like vLLM's KV "
                 "cache); a per-request context must fit within it;\r\n");
    c.text += TR("· 并发合计占用超过总池时新请求会报错。默认跟随单请求上限；Windows 最大 512K。\r\n",
                 "· when streams together exceed the pool, new requests error out. Defaults to "
                 "the per-request cap; max 512K on Windows.\r\n");
    c.text += TR("· YaRN：单请求上限超过原生 262144 时开启，扩展倍数按上限自动推导。\r\n",
                 "· YaRN: enable when the per-request cap exceeds native 262144; the scale "
                 "factor is derived from the cap automatically.\r\n");
    c.text += TR("· KV 快照目录：留空 = 缓存放仓库内 data\\kvsnap；填目录则缓存放它下面的 "
                 "qwenox-kvsnap 子目录（自动创建，可整个删）。\r\n",
                 "· KV snapshot folder: empty = cache goes to data\\kvsnap inside the repo; a "
                 "path puts the cache in its qwenox-kvsnap subfolder (auto-created, safe to "
                 "delete wholesale).\r\n");
    long maxctx = 0, parallel = 1, kv_pool = 0, prefill = 0, rope_orig = 0;
    double factor = 1;
    bool nums_ok = true;
    auto need_int = [&](const char* name, const std::string& s, long lo, long hi, long* out) {
        if (!pn_int(s, out) || *out < lo || *out > hi) {
            check_line(c, kLvFail, std::string(name) + TR(" 必须是 ", " must be an integer ") +
                                       std::to_string(lo) + "-" + std::to_string(hi) +
                                       TR(" 的整数（当前为 \"", " (current: \"") + s +
                                       TR("\"）", "\")"));
            nums_ok = false;
            return false;
        }
        return true;
    };
    need_int("MAX_CONTEXT", v.max_context, 1024, 1 << 20, &maxctx);
    need_int("PARALLEL", v.parallel, 1, 8, &parallel);
    need_int("KV_POOL_TOKENS", v.kv_pool, 0, 1 << 24, &kv_pool);
    need_int("PREFILL_CHUNK", v.prefill_chunk, 0, 1 << 20, &prefill);
    long kvsnap = 20;
    need_int("KVSNAP_MAX_GB", v.kvsnap_max_gb, 0, 1 << 16, &kvsnap);
    // ROPE_ORIGINAL_CTX 无 UI 行，但校验照做（conf 里填坏时启动器同样会拒）
    need_int("ROPE_ORIGINAL_CTX", v.rope_original, 1, 1 << 30, &rope_orig);
    if (!pn_double(v.rope_factor, &factor) || factor < 1) {
        check_line(c, kLvFail,
                   TR("ROPE_FACTOR 必须 >=1（当前为 \"", "ROPE_FACTOR must be >=1 (current: \"") +
                       v.rope_factor + TR("\"）", "\")"));
        nums_ok = false;
    }
    if (!nums_ok) return;
    if (parallel > 1 && !v.kv_paged)
        check_line(c, kLvFail, TR("PARALLEL>1 需要 KV_PAGED=1", "PARALLEL>1 requires KV_PAGED=1"));
    if (factor > 1 && maxctx > factor * rope_orig)
        // 勾选 YaRN 时 factor 按上限自动推导，理论上不会触发；留作兜底
        check_line(c, kLvFail,
                   "MAX_CONTEXT=" + std::to_string(maxctx) +
                       TR(" 超出 YaRN 上限 ROPE_FACTOR×ROPE_ORIGINAL_CTX=",
                          " exceeds the YaRN limit ROPE_FACTOR×ROPE_ORIGINAL_CTX=") +
                       std::to_string((long)(factor * rope_orig)) +
                       TR("；调小 MAX_CONTEXT 或开启 YaRN（并发容量用 KV_POOL_TOKENS 扩）",
                          "; lower MAX_CONTEXT or enable YaRN (grow capacity via "
                          "KV_POOL_TOKENS)"));
    if (factor <= 1 && maxctx > rope_orig)
        check_line(c, kLvWarn, TR("未开启 YaRN 且 MAX_CONTEXT=", "YaRN is off and MAX_CONTEXT=") +
                                   std::to_string(maxctx) + TR(" 超过原生 ", " exceeds native ") +
                                   std::to_string(rope_orig) +
                                   TR("，超出部分的位置编码未验证；建议开启 YaRN",
                                      "; positions beyond it are unverified; consider enabling "
                                      "YaRN"));
    // 单请求不能超过 KV 缓存总量（vLLM 语义；kv_pool=0 = 跟随上限，不触发）
    if (kv_pool > 0 && maxctx > kv_pool)
        check_line(c, kLvFail, TR("单请求上限 MAX_CONTEXT=", "Per-request context MAX_CONTEXT=") +
                                   std::to_string(maxctx) +
                                   TR(" 不能大于并发总池 KV_POOL_TOKENS=",
                                      " exceeds the KV pool KV_POOL_TOKENS=") +
                                   std::to_string(kv_pool) +
                                   TR("；调小上限或调大总池", "; lower the cap or grow the pool"));
    // Windows 设备内存是硬上限 95 GiB 的 arena：v2 权重 512K 池实测用 92.2（2 路）–93.8（8 路）
    // GiB，已是放得下的上限。有效值 = max(池, 上限)
    // （0 = 跟随，上限超 512K 时跟随逻辑同样会把池顶超）。
    const long pool_eff = kv_pool > maxctx ? kv_pool : maxctx;
    if (pool_eff > 524288)
        check_line(c, kLvFail,
                   TR("并发总池有效值 ", "Effective KV pool ") + std::to_string(pool_eff) +
                       TR(" 超过 512K（524288 token）：Windows 95 GiB 显存 arena"
                          " 放不下更大的池；调小 MAX_CONTEXT 或 KV_POOL_TOKENS",
                          " exceeds 512K (524288 tokens): the 95 GiB Windows VRAM arena cannot "
                          "fit a larger pool; lower MAX_CONTEXT or KV_POOL_TOKENS"));
    if (!c.fatal)
        check_line(c, kLvPass,
                   TR("数值校验：上下文/并发/YaRN 参数合法",
                      "Numeric checks: context / concurrency / YaRN parameters valid"));
    // KV 快照目录（KVSNAP_DIR）：引擎语义见 engine_sup.cpp / start_win.sh —— 留空不动
    // QWENOX_KVSNAP_DIR（默认 data\kvsnap）；填了则实际快照目录是其下的
    // qwenox-kvsnap 子目录，启动时自动创建。
    std::string kd = v.kvsnap_dir;
    while (!kd.empty() && (kd.back() == '\\' || kd.back() == '/')) kd.pop_back();
    if (kd.empty())
        check_line(c, kLvPass, TR("KV 快照目录：未设置，快照在仓库内 data\\kvsnap",
                                  "KV snapshot folder: unset; snapshots go to data\\kvsnap "
                                  "inside the repo"));
    else if (kvsnap == 0)
        check_line(c, kLvHint,
                   TR("KV 快照上限为 0：快照功能关闭，目录设置暂不生效",
                      "KV snapshot cap is 0: snapshots are off, the folder has no effect yet"));
    else {
        check_line(c, kLvPass, TR("KV 快照目录：", "KV snapshot folder: ") + kd +
                                   TR("（实际快照目录 ", " (effective snapshot directory ") + kd +
                                       TR("\\qwenox-kvsnap，不存在时启动自动创建）",
                                          "\\qwenox-kvsnap, auto-created at startup if missing)"));
        // 目标卷剩余空间对照容量上限：目录尚未创建时向上找第一个已存在的祖先查
        //（"D:\新建\目录" → "D:\新建" → "D:"，同卷不影响剩余空间）
        ULARGE_INTEGER free_b{};
        bool have_space = false;
        for (std::string probe = kd;;) {
            if (GetDiskFreeSpaceExA(probe.c_str(), &free_b, nullptr, nullptr)) {
                have_space = true;
                break;
            }
            const size_t cut = probe.find_last_of("\\/");
            if (cut == std::string::npos) break;
            probe.erase(cut);
        }
        if (have_space && (double)free_b.QuadPart < (double)kvsnap * 1073741824.0)
            check_line(c, kLvWarn,
                       TR("KV 快照目标盘剩余 ", "KV snapshot target disk has ") +
                           gib((double)free_b.QuadPart) +
                           TR(" GiB，低于容量上限 ", " GiB free, below the cap of ") +
                           std::to_string(kvsnap) +
                           TR(" GiB：快照存不满容量就会因空间紧张被 LRU 逐出",
                              " GiB: snapshots will be LRU-evicted before filling the cap"));
    }
}

// 第 3 页：监听地址与端口。service_running 时占用检查降级为提示。
void check_ports_page(CheckLines& c, const PanelVals& v, bool service_running) {
    // 说明行（不参与判定，恒在列表最上方）
    c.text += TR("· 引擎是内部行协议，由 API 自动桥接，无需直连；浏览器/客户端连 API 端口。\r\n",
                 "· The engine speaks an internal line protocol bridged by the API; browsers/"
                 "clients connect to the API port.\r\n");
    c.text += TR("· 监听地址：127.0.0.1 = 仅本机（引擎推荐），0.0.0.0 = 对外开放。\r\n",
                 "· Bind address: 127.0.0.1 = local only (recommended for the engine), "
                 "0.0.0.0 = open to the network.\r\n");
    sockaddr_in tmp{};
    if (v.engine_host.empty() || !engine_net::address(v.engine_host, 1, &tmp))
        check_line(c, kLvFail,
                   TR("ENGINE_HOST 必须是 IPv4 点分地址，不支持主机名（当前为 \"",
                      "ENGINE_HOST must be an IPv4 dotted address; hostnames are not supported "
                      "(current: \"") +
                       v.engine_host + TR("\"）", "\")"));
    else if (v.engine_host != "127.0.0.1")
        check_line(c, kLvWarn, "ENGINE_HOST=" + v.engine_host +
                                   TR(" 会把引擎行协议暴露到网络，仅确需外部直连时这样设",
                                      " exposes the engine's line protocol to the network; set "
                                      "this only if external direct access is required"));
    if (v.api_host.empty() || !engine_net::address(v.api_host, 1, &tmp))
        check_line(c, kLvFail,
                   TR("API_HOST 必须是 IPv4 点分地址（当前为 \"",
                      "API_HOST must be an IPv4 dotted address (current: \"") +
                       v.api_host + TR("\"）", "\")"));
    long eport = 0, aport = 0;
    bool nums_ok = true;
    auto need_int = [&](const char* name, const std::string& s, long* out) {
        if (!pn_int(s, out) || *out < 1 || *out > 65535) {
            check_line(c, kLvFail, std::string(name) + TR(" 必须是 1-65535 的整数（当前为 \"",
                                                          " must be an integer 1-65535 (current: \"") +
                                       s + TR("\"）", "\")"));
            nums_ok = false;
        }
    };
    need_int("ENGINE_PORT", v.engine_port, &eport);
    need_int("API_PORT", v.api_port, &aport);
    if (nums_ok) {
        if (eport == aport) {
            check_line(c, kLvFail,
                       TR("ENGINE_PORT 与 API_PORT 必须不同", "ENGINE_PORT and API_PORT must differ"));
        } else if (service_running) {
            check_line(c, kLvHint, TR("端口 ", "Ports ") + std::to_string(eport) + "/" +
                                       std::to_string(aport) +
                                       TR("：由当前服务占用（正常）", ": held by the running service (normal)"));
        } else {
            if (!port_free((int)eport))
                check_line(c, kLvFail,
                           TR("端口 ", "Port ") + std::to_string(eport) + TR(" 已被占用", " is in use"));
            if (!port_free((int)aport))
                check_line(c, kLvFail,
                           TR("端口 ", "Port ") + std::to_string(aport) + TR(" 已被占用", " is in use"));
            if (port_free((int)eport) && port_free((int)aport))
                check_line(c, kLvPass, TR("端口 ", "Ports ") + std::to_string(eport) + "/" +
                                           std::to_string(aport) + TR(" 空闲", " are free"));
        }
    }
    if (!c.fatal) check_line(c, kLvPass, TR("监听地址与端口合法", "Bind addresses and ports are valid"));
}

// GGUF 分片（name-00001-of-00004.gguf …）合计大小；单文件/找不到时分片数为 0/1。
unsigned long long gguf_shards_size(const std::string& first, int* count) {
    *count = 0;
    unsigned long long total = 0;
    const size_t of = first.find("-of-");
    size_t dash = std::string::npos;
    if (of != std::string::npos && of > 0) dash = first.rfind('-', of - 1);
    if (dash == std::string::npos) {
        unsigned long long s = 0;
        if (file_size_of(first, &s)) {
            total = s;
            *count = 1;
        }
        return total;
    }
    const std::string pattern = first.substr(0, dash) + "-*" + first.substr(of);
    WIN32_FIND_DATAA fd{};
    HANDLE hf = FindFirstFileA(pattern.c_str(), &fd);
    if (hf == INVALID_HANDLE_VALUE) return 0;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
            total += (static_cast<unsigned long long>(fd.nFileSizeHigh) << 32) | fd.nFileSizeLow;
            ++*count;
        }
    } while (FindNextFileA(hf, &fd));
    FindClose(hf);
    return total;
}

// 第 4 页：显存环境检查完整报告。service_running 时余量结论降级为对照提示。
void check_vram_page(CheckLines& c, const PanelVals& v, bool service_running) {
    // 显存：动态加载 HIP（启动器不链 amdhip64，DLL 缺失只降级不阻塞）
    size_t hip_free = 0, hip_total = 0;
    bool hip_ok = false;
    // 全路径 + LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR：相对路径 "build\..." 跟工作目录
    // 走（快捷方式/其他 cwd 下加载失败），且 amdhip64_7.dll 的依赖 DLL 也在
    // build\ 下，标准搜索顺序找不到——该 flag 把 DLL 自身目录纳入依赖搜索。
    const std::string hip_dll = g_root + "\\build\\amdhip64_7.dll";
    if (HMODULE hip = LoadLibraryExA(hip_dll.c_str(), nullptr,
                                     LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
                                         LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
        auto p_init = reinterpret_cast<int (*)(unsigned int)>(GetProcAddress(hip, "hipInit"));
        auto p_info =
            reinterpret_cast<int (*)(size_t*, size_t*)>(GetProcAddress(hip, "hipMemGetInfo"));
        if (p_init && p_info && p_init(0) == 0 && p_info(&hip_free, &hip_total) == 0)
            hip_ok = true;
    }
    if (hip_ok)
        check_line(c, kLvHint, TR("HIP 显存：总量 ", "HIP VRAM: total ") + gib((double)hip_total) +
                                   TR(" GiB，当前可用 ", " GiB, currently free ") +
                                   gib((double)hip_free) + " GiB");
    else
        check_line(c, kLvWarn,
                   TR("无法查询显存（build\\amdhip64_7.dll 缺失或驱动异常），"
                      "下面的估算请自行对照",
                      "Cannot query VRAM (build\\amdhip64_7.dll missing or driver problem); "
                      "compare the estimates below manually"));

    // 数值（第 2 页）解析失败时按默认粗估，不阻塞报告
    long maxctx = 262144, parallel = 1, kv_pool = 0, prefill = 0;
    pn_int(v.max_context, &maxctx);
    pn_int(v.parallel, &parallel);
    pn_int(v.kv_pool, &kv_pool);
    pn_int(v.prefill_chunk, &prefill);
    if (maxctx < 1) maxctx = 262144;
    if (parallel < 1) parallel = 1;
    if (kv_pool < 0) kv_pool = 0;
    if (prefill < 0) prefill = 0;
    // GDEC_PREFILL_CHUNK(conf 里的 GDEC_* 会导出到引擎环境)才是真正的 prefill
    // 分段;它比 PREFILL_CHUNK 大时以它为准,工作区估算才贴合引擎实际分配。
    if (const auto it = g_conf.find("GDEC_PREFILL_CHUNK"); it != g_conf.end()) {
        long g = 0;
        if (pn_int(it->second, &g) && g > prefill) prefill = g;
    }

    // 权重文件总大小：按第 1 页所选版本的文件集合，去重
    unsigned long long weights = 0;
    if (v.weights_ver == kVerGguf) {
        int shards = 0;
        weights += gguf_shards_size(v.gguf, &shards);
        if (shards > 0)
            check_line(c, kLvHint, TR("GGUF 权重：", "GGUF weights: ") + std::to_string(shards) +
                                       TR(" 个分片合计 ", " shards totaling ") +
                                       gib((double)weights) + " GiB");
        else
            check_line(c, kLvHint,
                       TR("GGUF 主分片找不到，权重按 0 估算",
                          "GGUF first shard not found; weights estimated as 0"));
        const std::string side[] = {v.no_gguf_mtp ? "" : v.gguf_mtp,
                                    v.no_gguf_vision ? "" : v.gguf_vision};
        unsigned long long extra = 0;
        for (const std::string& f : side) {
            unsigned long long s = 0;
            if (!f.empty() && file_size_of(f, &s)) extra += s;
        }
        // 非 shared 版 MTP sidecar 自带 token_embd / output，引擎不读（草稿头沿用主干的
        // embed 与 lm_head），不进显存：mtp-…-Q8_0.gguf 3.85 GiB 文件实测只占 2.6 GiB。
        if (!side[0].empty()) {
            const unsigned long long unread = gguf_tensor_bytes_in(side[0], "token_embd.weight") +
                                              gguf_tensor_bytes_in(side[0], "output.weight");
            if (unread < extra) extra -= unread;
        }
        if (extra) {
            weights += extra;
            check_line(c, kLvHint, TR("GGUF sidecar（MTP/视觉塔）：", "GGUF sidecars (MTP/vision): ") +
                                       gib((double)extra) + " GiB");
        }
        // n-gram 表（per_layer_token_embd，IQ4_NL）在分片里，按需读盘不驻留显存：
        // 从驻留口径里减掉（与 v1 内嵌表同理）
        const unsigned long long ple = gguf_tensor_bytes(v.gguf, "per_layer_token_embd.weight");
        if (ple && ple < weights) {
            weights -= ple;
            check_line(c, kLvHint,
                       TR("n-gram 表（GGUF 内）：", "n-gram table (in the GGUF): ") +
                           gib((double)ple) +
                           TR(" GiB，磁盘分页按需读取，不驻留显存",
                              " GiB, disk-paged on demand, not VRAM-resident"));
        }
    } else {
        // n-gram 表是磁盘分页的（引擎按需 gather，只有小 staging 驻留），
        // 单列且不计入显存估算：v2 是独立文件（直接不算进 weights）；
        // v1 内嵌在主权重文件里，从 HGN 头读出表大小后从驻留口径里减掉。
        unsigned long long ngram_bytes = 0;
        if (v.weights_ver == kVerV2 && !v.ngram.empty() && file_exists(v.ngram) &&
            !same_file(v.ngram, v.model))
            file_size_of(v.ngram, &ngram_bytes);
        const unsigned long long ngram_embedded =
            (v.weights_ver == kVerV1 && file_exists(v.model))
                ? hgn_tensor_bytes(v.model, "layers.1.ple.ngram_embedding.weight")
                : 0;
        // 主模型自带 MTP 草稿头（v2 主体即是）时，MTP 文件只替换同名张量、不额外占显存
        //（v2 带不带 mtp.hgn 实测 weight arena 都是 62.1 GiB）
        const bool model_has_mtp =
            file_exists(v.model) && hgn_tensor_bytes(v.model, "mtp.fc_hidden.weight") > 0;
        const std::string files[] = {v.model,
                                     v.weights_ver == kVerV2 ? "" : (v.no_overlay ? "" : v.overlay),
                                     (v.no_mtp || model_has_mtp) ? "" : v.mtp,
                                     v.no_vision ? "" : v.vision};
        std::vector<std::string> uniq;
        for (const std::string& f : files) {
            if (f.empty() || !file_exists(f)) continue;
            bool dup = false;
            for (const std::string& u : uniq)
                if (same_file(f, u)) {
                    dup = true;
                    break;
                }
            if (!dup) uniq.push_back(f);
        }
        for (const std::string& u : uniq) {
            unsigned long long s = 0;
            if (file_size_of(u, &s)) weights += s;
        }
        check_line(c, kLvHint, TR("权重文件：", "Weight files: ") + std::to_string(uniq.size()) +
                                   TR(" 个合计 ", " totaling ") + gib((double)weights) + " GiB");
        if (model_has_mtp && !v.no_mtp && !v.mtp.empty())
            check_line(c, kLvHint,
                       TR("主模型已带 MTP 草稿头：MTP 文件只替换同名张量，不另计显存",
                          "The main model already has the MTP draft head: the MTP file only "
                          "replaces those tensors and adds no VRAM"));
        if (ngram_bytes)
            check_line(c, kLvHint,
                       TR("n-gram 表：", "n-gram table: ") + gib((double)ngram_bytes) +
                           TR(" GiB，磁盘分页按需读取，不驻留显存",
                              " GiB, disk-paged on demand, not VRAM-resident"));
        if (ngram_embedded)
            check_line(c, kLvHint,
                       TR("n-gram 表（内嵌）：", "n-gram table (embedded): ") +
                           gib((double)ngram_embedded) +
                           TR(" GiB，磁盘分页按需读取，不驻留显存",
                              " GiB, disk-paged on demand, not VRAM-resident"));
        // 显存估算按驻留口径：减掉内嵌表（v2 的独立表本就未计入）
        if (ngram_embedded < weights) weights -= ngram_embedded;
    }

    // 显存估算（系数按 src/gpu/parts/40_model.inc 分配代码核算，并与 2026-10-13 Windows
    // 实测 arena 用量对齐：v2 256K/4 路 82.6、512K 池/2 路 92.2、/8 路 93.8、YaRN 512K/2 路
    // 93.3 GiB，估算误差 < 0.5 GiB）：
    // - 页池：41,408 B/token = 12 QSA 层 + 1 MTP 层的 BF16 KV + BTV 转置 V + 索引键（含 f16 twin）
    // - 单条上限：4,288 B/token = d_iscores（INDEX_BATCH 1024 × 4 B）+ d_ikg/d_ikgh，按 maxctx 分配
    // - 每路 0.27 GiB：GDN 状态 + 会合 verify 行窗口备份 + MTP 流状态（2 路→8 路实测 +1.6 GiB）
    // - 固定 0.5 GiB：其余小缓冲
    const double kKvPoolBytesPerToken = 41408.0;
    const double kKvCtxBytesPerToken = 4288.0;
    // 0 = 跟随 MAX_CONTEXT；std::max 与 windows.h 的 max 宏冲突，手写比较
    const double pool_tokens = (double)(kv_pool > maxctx ? kv_pool : maxctx);
    const double kv_bytes = pool_tokens * kKvPoolBytesPerToken;
    const double ctx_bytes = (double)maxctx * kKvCtxBytesPerToken;
    const double chunk = prefill > 0 ? (double)prefill : 6144.0;  // 引擎 Windows 默认
    const double ws_bytes = chunk / 8192.0 * 6.9 * 1073741824.0;  // 8192≈6.9 GiB 实测
    const double gdn_bytes = parallel * 0.27 * 1073741824.0;      // 每路状态
    const double misc_bytes = 0.5 * 1073741824.0;
    const double total = (double)weights + kv_bytes + ctx_bytes + ws_bytes + gdn_bytes + misc_bytes;
    check_line(c, kLvHint, TR("并发总池：", "KV pool: ") + std::to_string((long)pool_tokens) +
                               TR(" token × 40.4 KiB ≈ ", " tokens × 40.4 KiB ≈ ") + gib(kv_bytes) +
                               " GiB");
    check_line(c, kLvHint, TR("索引打分缓冲：单条上限 ", "Index score buffers: per-request cap ") +
                               std::to_string(maxctx) +
                               TR(" token × 4.2 KiB ≈ ", " tokens × 4.2 KiB ≈ ") + gib(ctx_bytes) +
                               " GiB");
    check_line(c, kLvHint, TR("prefill 工作区：chunk ", "Prefill workspace: chunk ") +
                               std::to_string((long)chunk) + " ≈ " + gib(ws_bytes) + " GiB");
    check_line(c, kLvHint, TR("每路状态（GDN、verify 备份等）：", "Per-slot state (GDN, verify backup, etc.): ") +
                               std::to_string(parallel) + TR(" 路 × 0.27 ≈ ", " slots × 0.27 ≈ ") +
                               gib(gdn_bytes) + " GiB");
    if (!hip_ok) {
        // 查不到显存也要给出估算总量 —— 它不依赖 HIP 查询，只有对比结论需要
        check_line(c, kLvHint, TR("显存估算 ≈ ", "VRAM estimate ≈ ") + gib(total) +
                                   TR(" GiB（显存查询失败，请自行对照）",
                                      " GiB (VRAM query failed; compare manually)"));
    } else if (!service_running) {
        if (total > (double)hip_free)
            check_line(c, kLvFail, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB 超过当前可用 ", " GiB exceeds the free ") +
                                       gib((double)hip_free) + TR(" GiB，启动会 OOM", " GiB; startup would OOM"));
        else if ((double)hip_free - total < 4.0 * 1073741824.0)
            check_line(c, kLvWarn, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB，余量不足 4 GiB", " GiB, less than 4 GiB of headroom"));
        else if (total > 95.0 * 1073741824.0)
            check_line(c, kLvWarn, TR("预计占用 ", "Estimated use ") + gib(total) +
                                       TR(" GiB 超过引擎 arena 上限 95 GiB，超出部分回退 hipMalloc",
                                          " GiB exceeds the 95 GiB engine arena cap; the excess "
                                          "falls back to hipMalloc"));
        else
            check_line(c, kLvPass, TR("显存估算 ≈ ", "VRAM estimate ≈ ") + gib(total) +
                                       TR(" / 可用 ", " / free ") + gib((double)hip_free) + " GiB");
    } else if (hip_ok) {
        check_line(c, kLvHint, TR("服务运行中：估算 ≈ ", "Service running: estimate ≈ ") + gib(total) +
                                   TR(" GiB，仅对照总量 ", " GiB, compared against total ") +
                                   gib((double)hip_total) + " GiB");
    }
}

struct AllChecks {
    CheckLines weights, context, ports, vram;
    bool fatal() const { return weights.fatal || context.fatal || ports.fatal || vram.fatal; }
};

AllChecks panel_checks(const PanelVals& v, bool service_running) {
    AllChecks a;
    check_weights_page(a.weights, v);
    check_context_page(a.context, v);
    check_ports_page(a.ports, v, service_running);
    check_vram_page(a.vram, v, service_running);
    // 环境变量覆盖提示，归入对应页
    for (const char* key : kManagedKeys) {
        const char* e = getenv(key);
        if (!e || !*e) continue;
        CheckLines& dst = page_of_key(key) == 1   ? a.context
                          : page_of_key(key) == 2 ? a.ports
                                                  : a.weights;
        check_line(dst, kLvWarn, std::string(key) +
                                     TR(" 正被环境变量覆盖（=\"", " is overridden by an environment variable (=\"") +
                                     e + TR("\"），面板改动不生效", "\"); panel changes will not take effect"));
    }
    return a;
}

// ---- 面板窗口 -----------------------------------------------------------------

// PanelResult（kPanelClose/kPanelRelang）在 settings_win.cpp 前向声明段中完整定义
//（main() 在本文件之后调用 panel_run()，需要先知道枚举值）。

enum PanelId : int {
    kBtnSave = 100, kBtnCancel, kBtnRecheck,
    kTab1, kTab2, kTab3, kTab4,
    kWVer1, kWVer2, kWVer3,
    kEdModel, kEdNgram, kEdOverlay, kEdMtp, kEdVision, kEdTok,
    kEdGguf, kEdGgufMtp, kEdGgufVision,
    kBrModel, kBrNgram, kBrOverlay, kBrMtp, kBrVision,
    kBrGguf, kBrGgufMtp, kBrGgufVision,
    kOffOverlay, kOffMtp, kOffVision, kOffGgufMtp, kOffGgufVision,
    kEdMaxctx, kCbMaxctxPreset, kCbParallel, kEdKvPool, kCkKvPaged, kEdPrefill,
    kCkYarn, kEdKvsnap, kEdKvsnapDir, kBrKvsnapDir, kCkAffinity,
    kEdEHost, kEdEPort, kEdAHost, kEdAPort,
    kEdOut1, kEdOut2, kEdOut3, kEdReport, kEdSummary, kBtnLang,
    kCkMoeA4,
};

struct FileRow {
    HWND label = nullptr, off = nullptr, edit = nullptr, browse = nullptr;
};

struct PanelCtl {
    HWND tabs[4] = {};
    HWND wver[3] = {};
    FileRow row_model, row_ngram, row_overlay, row_mtp, row_vision;
    FileRow row_gguf, row_gguf_mtp, row_gguf_vision, row_tok;
    HWND tok_hint = nullptr, ver_note = nullptr;
    HWND moe_a4 = nullptr;  // V2 专属勾选框：prefill 4-bit 加速（MOE_A4）
    HWND maxctx = nullptr, maxctx_preset = nullptr, parallel = nullptr, kv_pool = nullptr,
         kv_paged = nullptr, prefill = nullptr;
    HWND yarn = nullptr;  // "YaRN 长上下文扩展" 勾选框（ROPE_FACTOR 自动推导，无编辑行）
    HWND kvsnap = nullptr;
    HWND kvsnap_dir = nullptr, kvsnap_dir_br = nullptr;
    HWND affinity = nullptr;
    HWND ehost = nullptr, eport = nullptr, ahost = nullptr, aport = nullptr;
    HWND out1 = nullptr, out2 = nullptr, out3 = nullptr, report = nullptr, summary = nullptr;
};
PanelCtl g_pc;
HWND g_panel = nullptr;
PanelResult g_panel_result = kPanelClose;
bool g_panel_done = false;
bool g_service_running = false;     // 打开时探测：服务运行中（端口占用/显存检查降级为提示）
int g_weights_ver = kVerV1;         // 当前显示的权重版本（布局用）
std::string g_moe_a4_init = "0";    // 打开时 conf 的 MOE_A4（勾着保存时保留手设的 1/2）
bool g_pool_touched = false;        // 用户手动改过并发总池框（false = 跟随单请求上限显示）
bool g_pool_guard = false;          // 程序自己在写总池框（防 EN_CHANGE 误置 touched）
AllChecks g_last_checks;
std::vector<HWND> g_page_ctls[4];   // 每页的控件句柄（切页时整体 ShowWindow）
Gdiplus::Image* g_art = nullptr;    // 左侧立绘（media/qwen-charactor-alpha.png）
Gdiplus::Image* g_bg_img = nullptr; // 全窗背景（media/panel-bg.png，banner 派生）
int g_art_w = 0;                    // 立绘条宽度；图片缺失时为 0，退回经典紧凑布局
HDC g_bg_dc = nullptr;              // 烘焙好的全窗背景（背景图+立绘），erase 时 BitBlt
HBITMAP g_bg_bmp = nullptr;
HBRUSH g_panel_bg = nullptr;        // 面板底色刷（夜空深蓝，背景图缺失时的兜底）
HBRUSH g_edit_bg = nullptr;         // 输入框深色卡片刷
constexpr COLORREF kPanelBg = RGB(16, 20, 34);
constexpr COLORREF kPanelFg = RGB(232, 236, 246);  // 深色背景上的浅色文字
constexpr COLORREF kEditBg = RGB(24, 30, 50);      // 输入框卡片：比压暗层略亮

// 布局常量：内容区 644×504（立绘加载成功时窗口左侧再加 g_art_w 宽的立绘条，
// 所有控件经 add_ctrl 统一右移），内容右缘 636。
// label 列要装下 "单请求上限 (token)"（高 DPI 下约 185px），故 kLblW=190。
const int kCliW = 644, kCliH = 504;
const int kLblX = 12, kLblW = 190, kEdX = 202;
const int kOffW = 88;                 // "不使用" checkbox（勾选框 + 3 个汉字）
const int kEdPlainW = 354;            // 无 checkbox 文件行的 EDIT 宽（202..556）
const int kBrX = 564, kBrW = 72;      // 浏览按钮
const int kHintX = 328, kHintW = 308; // 第 2 页行尾提示（edit 202+120=322 右侧）
const int kListBottom = 452;          // 各页检查区下缘（底部按钮行在其下）

// ---- DPI ---------------------------------------------------------------------
// 上面所有布局常量都按 96-DPI 像素书写；建窗口/控件、GDI+ 绘制时经 sc()
// 换算成当前窗口 DPI 的物理像素。进程按 PerMonitorV2 声明（manifest 或
// main() 里的运行时兜底），窗口跨屏时由 WM_DPICHANGED 更新 g_dpi 并重排。
int g_dpi = 96;
int sc(int v) { return MulDiv(v, g_dpi, 96); }

// 控件在 96-DPI 坐标系里的基准位置（含立绘条偏移）：WM_DPICHANGED 时
// 不重建窗口、不丢用户输入，按这张表把所有控件重排到新 DPI。
struct CtlBase {
    HWND h;
    RECT r;
};
std::vector<CtlBase> g_ctl_base;

void remember_ctl(HWND h, int x, int y, int w, int ht) {
    if (!h) return;
    const RECT r{x, y, x + w, y + ht};
    for (CtlBase& b : g_ctl_base)
        if (b.h == h) {
            b.r = r;
            return;
        }
    g_ctl_base.push_back({h, r});
}

// 摆放控件并记录基准位置。flags 里的显示/隐藏标志原样保留；统一补
// SWP_NOZORDER/NOACTIVATE，控件层不需要改 z 序。
void place(HWND h, int x, int y, int w, int ht, UINT flags) {
    if (!h) return;
    remember_ctl(h, x, y, w, ht);
    SetWindowPos(h, nullptr, sc(x), sc(y), sc(w), sc(ht),
                 flags | SWP_NOZORDER | SWP_NOACTIVATE);
}

// 统一字体：不用 DEFAULT_GUI_FONT（MS Shell Dlg 在不同机器上替换表不同，
// 有的落到宋体，西文更宽会溢出固定宽度布局）。YaHei UI 自 Vista 起全版本自带。
// 字高 14 是 96-DPI 基准，按当前 DPI 换算（200% → 28px，物理字号不变）。
HFONT g_panel_font = nullptr;
int g_panel_font_dpi = 0;

HFONT panel_font() {
    if (!g_panel_font || g_panel_font_dpi != g_dpi) {
        // 旧 HFONT 不 DeleteObject：WM_DPICHANGED 里控件还短暂持有它；
        // 每次切换多一个字体对象，量级可忽略。
        g_panel_font = CreateFontW(-sc(14), 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE,
                                   DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                                   CLEARTYPE_QUALITY, DEFAULT_PITCH | FF_DONTCARE,
                                   L"Microsoft YaHei UI");
        if (!g_panel_font) g_panel_font = static_cast<HFONT>(GetStockObject(DEFAULT_GUI_FONT));
        g_panel_font_dpi = g_dpi;
    }
    return g_panel_font;
}

HWND add_ctrl(HWND parent, const wchar_t* cls, DWORD style, int x, int y, int w, int h,
              const wchar_t* text, int id) {
    // 深色主题下按钮全部 owner-draw（WM_DRAWITEM 里 GDI+ 深色圆角卡片）；
    // 页签的 BS_PUSHLIKE 单选改成 BS_OWNERDRAW，选中态由 show_page 手动维护
    if (g_panel_bg && wcscmp(cls, L"Button") == 0) {
        const DWORD ty = style & BS_TYPEMASK;
        if (ty == BS_PUSHBUTTON || ty == BS_DEFPUSHBUTTON || (style & BS_PUSHLIKE))
            style = (style & ~(BS_TYPEMASK | BS_PUSHLIKE)) | BS_OWNERDRAW;
    }
    const int bx = x + g_art_w;  // 立绘条偏移保持在 96-DPI 坐标里，由 sc() 统一放大
    HWND c = CreateWindowExW(0, cls, text, WS_CHILD | WS_VISIBLE | style,
                             sc(bx), sc(y), sc(w), sc(h), parent,
                             reinterpret_cast<HMENU>(static_cast<INT_PTR>(id)),
                             GetModuleHandleW(nullptr), nullptr);
    // 勾选框/单选钮：去主题退回经典绘制，由 WM_CTLCOLORSTATIC 的浅色文字 +
    // 透明底接管（主题样式的文字是深色，夜空背景上看不清）
    if (g_panel_bg && wcscmp(cls, L"Button") == 0 &&
        ((style & BS_TYPEMASK) == BS_AUTOCHECKBOX ||
         (style & BS_TYPEMASK) == BS_AUTORADIOBUTTON))
        SetWindowTheme(c, L"", L"");
    // 编辑框滚动条深色化（DarkMode_Explorer 主题类，Win10 1809+；失败则保持默认）
    if (g_panel_bg && wcscmp(cls, L"Edit") == 0)
        SetWindowTheme(c, L"DarkMode_Explorer", nullptr);
    if (c) {
        remember_ctl(c, bx, y, w, h);
        SendMessageW(c, WM_SETFONT, reinterpret_cast<WPARAM>(panel_font()), TRUE);
    }
    return c;
}

HWND add_label(HWND parent, int x, int y, const wchar_t* text, int w = kLblW) {
    return add_ctrl(parent, L"Static", SS_LEFTNOWORDWRAP, x, y + 3, w, 18, text, 0);
}

// 建控件并登记进某页（page=-1 表示恒可见，不参与切页）
HWND pctl(int page, HWND parent, const wchar_t* cls, DWORD style, int x, int y, int w, int h,
          const wchar_t* text, int id) {
    HWND c = add_ctrl(parent, cls, style, x, y, w, h, text, id);
    if (page >= 0) g_page_ctls[page].push_back(c);
    return c;
}

HWND plabel(int page, HWND parent, int x, int y, const wchar_t* text, int w = kLblW) {
    return pctl(page, parent, L"Static", SS_LEFTNOWORDWRAP, x, y + 3, w, 18, text, 0);
}

void set_edit(HWND h, const std::string& s) { SetWindowTextW(h, to_w(s).c_str()); }

std::string get_edit(HWND h) {
    const int n = GetWindowTextLengthW(h);
    std::wstring w(static_cast<size_t>(n) + 1, L'\0');
    GetWindowTextW(h, &w[0], n + 1);
    w.resize(static_cast<size_t>(n));
    return trim_copy(from_w(w));
}

bool get_check(HWND h) { return SendMessageW(h, BM_GETCHECK, 0, 0) == BST_CHECKED; }

void set_optional_row_state(HWND off, HWND edit, HWND browse) {
    const bool disabled = get_check(off);
    EnableWindow(edit, !disabled);
    if (browse) EnableWindow(browse, !disabled);
}

// YaRN 勾选 → ROPE_FACTOR = max(2, ceil(单请求上限 / ROPE_ORIGINAL_CTX))；
// 不勾 → 1。上限/原生值解析失败时按默认粗估（校验页会单独报 [失败]）。
std::string derive_rope_factor(bool yarn_on, const std::string& max_context,
                               const std::string& rope_original) {
    if (!yarn_on) return "1";
    long mc = 262144, ro = 262144;
    if (!pn_int(max_context, &mc) || mc < 1) mc = 262144;
    if (!pn_int(rope_original, &ro) || ro < 1) ro = 262144;
    long f = (mc + ro - 1) / ro;
    if (f < 2) f = 2;
    return std::to_string(f);
}

PanelVals panel_collect() {
    PanelVals v;
    v.weights_ver = get_check(g_pc.wver[2])   ? kVerGguf
                    : get_check(g_pc.wver[1]) ? kVerV2
                                              : kVerV1;
    v.model = get_edit(g_pc.row_model.edit);
    v.ngram = get_edit(g_pc.row_ngram.edit);
    v.no_overlay = get_check(g_pc.row_overlay.off);
    v.overlay = get_edit(g_pc.row_overlay.edit);
    v.no_mtp = get_check(g_pc.row_mtp.off);
    v.mtp = get_edit(g_pc.row_mtp.edit);
    v.no_vision = get_check(g_pc.row_vision.off);
    v.vision = get_edit(g_pc.row_vision.edit);
    v.gguf = get_edit(g_pc.row_gguf.edit);
    v.no_gguf_mtp = get_check(g_pc.row_gguf_mtp.off);
    v.gguf_mtp = get_edit(g_pc.row_gguf_mtp.edit);
    v.no_gguf_vision = get_check(g_pc.row_gguf_vision.off);
    v.gguf_vision = get_edit(g_pc.row_gguf_vision.edit);
    v.tok = get_edit(g_pc.row_tok.edit);
    v.max_context = get_edit(g_pc.maxctx);
    const int sel = static_cast<int>(SendMessageW(g_pc.parallel, CB_GETCURSEL, 0, 0));
    v.parallel = std::to_string(sel >= 0 ? sel + 1 : 1);
    v.kv_pool = get_edit(g_pc.kv_pool);
    // 并发总池的跟随语义：框值 == 单请求上限（含程序跟随填充）或显式 0/跟随状态 → 存 0
    if (v.kv_pool == v.max_context || v.kv_pool == "0" || !g_pool_touched) v.kv_pool = "0";
    v.kv_paged = get_check(g_pc.kv_paged);
    v.prefill_chunk = get_edit(g_pc.prefill);
    v.rope_original = cfg("ROPE_ORIGINAL_CTX", "262144");  // 无 UI 行，conf/环境原样读
    v.rope_factor = derive_rope_factor(get_check(g_pc.yarn), v.max_context, v.rope_original);
    v.kvsnap_max_gb = get_edit(g_pc.kvsnap);
    v.kvsnap_dir = get_edit(g_pc.kvsnap_dir);
    // MOE_A4：勾选框只在 V2 显示。勾着保存时保留 conf 里手设的 1/2，否则写 3（两段都量化）；
    // V1/GGUF 时引擎本就忽略该项，原值照留（非法值归 0，免得启动器拒绝启动）。
    const std::string a4_keep = moe_a4_on(g_moe_a4_init) ? g_moe_a4_init : "0";
    if (v.weights_ver == kVerV2)
        v.moe_a4 = get_check(g_pc.moe_a4) ? (a4_keep != "0" ? a4_keep : "3") : "0";
    else
        v.moe_a4 = a4_keep;
    // 亲和性开关化：勾 = upper-half（CCD 1），不勾 = 不限制；conf 里手写的
    // 任何非空 mask 在面板上都显示为勾，保存时规格化为 upper-half
    v.affinity = get_check(g_pc.affinity) ? "upper-half" : "";
    v.engine_host = get_edit(g_pc.ehost);
    v.engine_port = get_edit(g_pc.eport);
    v.api_host = get_edit(g_pc.ahost);
    v.api_port = get_edit(g_pc.aport);
    return v;
}

// 第 4 页底部：其它页的失败汇总（GGUF 选中时也在这里说明启动受限原因）。
void refresh_outputs(const PanelVals& v, const AllChecks& c) {
    set_edit(g_pc.out1, c.weights.text);
    set_edit(g_pc.out2, c.context.text);
    set_edit(g_pc.out3, c.ports.text);
    set_edit(g_pc.report, c.vram.text);
    const CheckLines* pages[3] = {&c.weights, &c.context, &c.ports};
    const char* names[3] = {TR("第 1 页", "Page 1"), TR("第 2 页", "Page 2"),
                            TR("第 3 页", "Page 3")};
    size_t n = 0;
    for (const CheckLines* p : pages) n += p->fatals.size();
    std::string s;
    if (n == 0) {
        s = TR("其它页没有失败项。\r\n", "No failures on the other pages.\r\n");
    } else {
        s = TR("其它页还有 ", "The other pages still have ") + std::to_string(n) +
            TR(" 项失败：\r\n", " failure(s):\r\n");
        for (int i = 0; i < 3; ++i)
            for (const std::string& f : pages[i]->fatals)
                s += std::string(names[i]) + TR("：", ": ") + f + "\r\n";
    }
    if (v.weights_ver == kVerGguf)
        s += TR("已选 GGUF 权重：服务按 GGUF 首分片启动（其余分片须在同一目录）。\r\n",
                "GGUF weights selected: the service starts from the first GGUF shard (keep the "
                "other shards in the same folder).\r\n");
    set_edit(g_pc.summary, s);
}

void panel_recheck() {
    const PanelVals v = panel_collect();
    g_last_checks = panel_checks(v, g_service_running);
    refresh_outputs(v, g_last_checks);
}

bool panel_apply_and_save(std::string* error) {
    const PanelVals v = panel_collect();
    const auto conf_map = panel_vals_to_conf(v);
    // 权重版本标记与键值同一次写入 upsert（同一份 .bak 备份）；
    // 错误文案与本地 save_conf 包装同一风格
    const std::string path = conf_path();
    const char* ver = v.weights_ver == kVerV2    ? "v2"
                      : v.weights_ver == kVerGguf ? "gguf"
                                                  : "v1";
    if (!svcconf::save_conf(path, conf_map, error, kWeightsMarker, ver)) {
        *error = std::string(TR("无法写入 ", "Cannot write ")) + path;
        return false;
    }
    return true;
}

bool panel_browse(HWND edit) {
    wchar_t buf[MAX_PATH] = L"";
    std::wstring initdir;
    const std::string cur = get_edit(edit);
    if (!cur.empty()) {
        // 编辑框里多是 "./models/x.hgn" 这类相对路径（正斜杠）；整串塞进
        // lpstrFile 会被对话框当成非法文件名，GetOpenFileNameW 直接失败
        // （FNERR_INVALIDFILENAME），表现为点了没反应。先规范成绝对路径，
        // 拆成 初始目录 + 纯文件名；解析不了就只给初始目录。
        std::wstring w = to_w(cur);
        for (auto& c : w)
            if (c == L'/') c = L'\\';
        wchar_t abs[MAX_PATH];
        const DWORD n = GetFullPathNameW(w.c_str(), MAX_PATH, abs, nullptr);
        if (n > 0 && n < MAX_PATH) {
            const std::wstring a(abs, n);
            const size_t slash = a.find_last_of(L'\\');
            if (slash != std::wstring::npos) {
                initdir = a.substr(0, slash);
                wcsncpy_s(buf, a.substr(slash + 1).c_str(), _TRUNCATE);
            }
        }
    }
    if (initdir.empty()) initdir = g_root_w;
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = g_panel;
    ofn.lpstrFilter = g_lang_en
                           ? L"hgn weights (*.hgn)\0*.hgn\0GGUF weights (*.gguf)\0*.gguf\0All files (*.*)\0*.*\0"
                           : L"hgn 权重 (*.hgn)\0*.hgn\0GGUF 权重 (*.gguf)\0*.gguf\0所有文件 (*.*)\0*.*\0";
    ofn.lpstrFile = buf;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrInitialDir = initdir.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR | OFN_DONTADDTORECENT;
    ofn.lpstrTitle = TR(L"选择权重文件", L"Select weights file");
    if (!GetOpenFileNameW(&ofn)) {
        const DWORD err = CommDlgExtendedError();  // 0 = 用户取消，不算失败
        if (err)
            msgbox(std::string(TR("无法打开文件选择框，错误码 ",
                                  "Cannot open the file dialog, error code ")) +
                   std::to_string(err),
                   MB_OK | MB_ICONWARNING);
        return false;
    }
    set_edit(edit, make_relative(from_w(buf)));
    return true;
}

// 目录选择框（KV 快照目录用）：IFileOpenDialog + FOS_PICKFOLDERS，Win10 现代样式、
// 可手输路径。CoInitializeEx 随每次选择配对；COM 已按其它线程模型初始化
//（RPC_E_CHANGED_MODE）时照常继续、不负责 Uninitialize。取消/创建失败静默返回。
bool panel_browse_dir(HWND edit) {
    const HRESULT hr_init = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    bool ok = false;
    IFileOpenDialog* dlg = nullptr;
    if (SUCCEEDED(CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                   IID_IFileOpenDialog, reinterpret_cast<void**>(&dlg)))) {
        dlg->SetOptions(FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST);
        dlg->SetTitle(TR(L"选择 KV 快照目录", L"Select KV snapshot folder"));
        // 当前值作为初始目录（先规范成绝对路径；解析或转换失败就按默认目录开）
        const std::string cur = get_edit(edit);
        if (!cur.empty()) {
            std::wstring w = to_w(cur);
            for (auto& c : w)
                if (c == L'/') c = L'\\';
            wchar_t abs[MAX_PATH];
            const DWORD n = GetFullPathNameW(w.c_str(), MAX_PATH, abs, nullptr);
            if (n > 0 && n < MAX_PATH) {
                IShellItem* seed = nullptr;
                if (SUCCEEDED(SHCreateItemFromParsingName(abs, nullptr, IID_PPV_ARGS(&seed)))) {
                    dlg->SetFolder(seed);
                    seed->Release();
                }
            }
        }
        IShellItem* item = nullptr;
        if (SUCCEEDED(dlg->Show(g_panel)) && SUCCEEDED(dlg->GetResult(&item))) {
            PWSTR path = nullptr;
            if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &path))) {
                set_edit(edit, make_relative(from_w(path)));
                ok = true;
                CoTaskMemFree(path);
            }
            item->Release();
        }
        dlg->Release();
    }
    if (SUCCEEDED(hr_init)) CoUninitialize();
    return ok;
}

// 把一个文件行排到指定 y（SWP_SHOWWINDOW），或整行隐藏
void place_row(const FileRow& r, int y, bool visible) {
    const UINT flags = visible ? SWP_SHOWWINDOW : SWP_HIDEWINDOW;
    const int ex = (r.off ? kEdX + kOffW + 8 : kEdX) + g_art_w;
    const int ew = 556 + g_art_w - ex;
    place(r.label, kLblX + g_art_w, y + 3, kLblW, 18, flags);
    if (r.off) place(r.off, kEdX + g_art_w, y, kOffW, 20, flags);
    place(r.edit, ex, y, ew, 22, flags);
    if (r.browse) place(r.browse, kBrX + g_art_w, y - 1, kBrW, 24, flags);
}

// 第 1 页布局：按版本只显示相关文件行，顺序重排，隐藏其余
void layout_page1(int ver) {
    g_weights_ver = ver;
    const FileRow* order[9];
    int n = 0;
    if (ver == kVerGguf) {
        order[n++] = &g_pc.row_gguf;
        order[n++] = &g_pc.row_gguf_mtp;
        order[n++] = &g_pc.row_gguf_vision;
        order[n++] = &g_pc.row_tok;
    } else if (ver == kVerV2) {
        order[n++] = &g_pc.row_model;
        order[n++] = &g_pc.row_ngram;
        order[n++] = &g_pc.row_mtp;
        order[n++] = &g_pc.row_vision;
        order[n++] = &g_pc.row_tok;
    } else {
        order[n++] = &g_pc.row_model;
        order[n++] = &g_pc.row_overlay;
        order[n++] = &g_pc.row_mtp;
        order[n++] = &g_pc.row_vision;
        order[n++] = &g_pc.row_tok;
    }
    const FileRow* all[9] = {&g_pc.row_model,  &g_pc.row_ngram,  &g_pc.row_overlay,
                             &g_pc.row_mtp,    &g_pc.row_vision, &g_pc.row_gguf,
                             &g_pc.row_gguf_mtp, &g_pc.row_gguf_vision, &g_pc.row_tok};
    int y = 76;
    for (int i = 0; i < n; ++i) {
        place_row(*order[i], y, true);
        y += 26;
    }
    for (const FileRow* r : all) {
        bool shown = false;
        for (int i = 0; i < n; ++i)
            if (order[i] == r) shown = true;
        if (!shown) place_row(*r, 0, false);
    }
    // tok_hint / ver_note 是透明底 Static（HOLLOW_BRUSH），自身重绘不擦底：
    // 改字或挪位后须让父窗口把新旧位置的并集贴回背景再画控件，否则旧文字
    // 残留、与新文字叠印。先记下挪动前的位置。
    HWND parent = GetParent(g_pc.ver_note);
    auto rect_in_parent = [parent](HWND c) {
        RECT r;
        GetWindowRect(c, &r);
        MapWindowPoints(nullptr, parent, reinterpret_cast<POINT*>(&r), 2);
        return r;
    };
    RECT dirty = rect_in_parent(g_pc.tok_hint);
    RECT tmp = rect_in_parent(g_pc.ver_note);
    UnionRect(&dirty, &dirty, &tmp);
    tmp = rect_in_parent(g_pc.moe_a4);
    UnionRect(&dirty, &dirty, &tmp);
    tmp = rect_in_parent(g_pc.out1);
    UnionRect(&dirty, &dirty, &tmp);
    place(g_pc.tok_hint, kEdX + g_art_w, y + 2, 434, 18,
          SWP_SHOWWINDOW | SWP_NOCOPYBITS);
    place(g_pc.ver_note, kEdX + g_art_w, y + 24, 434, 18,
          SWP_SHOWWINDOW | SWP_NOCOPYBITS);
    // V2 专属：prefill 4-bit 加速勾选框排在版本说明下一行，检查区随之下移一行
    const bool a4_row = ver == kVerV2;
    place(g_pc.moe_a4, kEdX + g_art_w, y + 46, 434, 20,
          SWP_NOCOPYBITS | (a4_row ? SWP_SHOWWINDOW : SWP_HIDEWINDOW));
    const int out_y = a4_row ? y + 72 : 254;
    place(g_pc.out1, 12 + g_art_w, out_y, 624, kListBottom - out_y,
          SWP_SHOWWINDOW | SWP_NOCOPYBITS);
    SetWindowTextW(g_pc.ver_note, ver == kVerGguf
                                      ? TR(L"GGUF：填第 1 个分片；MTP/视觉塔可留空不用",
                                           L"GGUF: enter the first shard; MTP/vision may be left off")
                                      : ver == kVerV2
                                            ? TR(L"V2：主体+独立 n-gram 文件；overlay 无效（保存置空）",
                                                 L"V2: main + separate n-gram file; overlay ignored (cleared on save)")
                                            : TR(L"V1：w4b 单文件，n-gram 内置；overlay/MTP/视觉塔可选",
                                                 L"V1: single w4b file, built-in n-gram; overlay/MTP/vision optional"));
    tmp = rect_in_parent(g_pc.tok_hint);
    UnionRect(&dirty, &dirty, &tmp);
    tmp = rect_in_parent(g_pc.ver_note);
    UnionRect(&dirty, &dirty, &tmp);
    tmp = rect_in_parent(g_pc.moe_a4);
    UnionRect(&dirty, &dirty, &tmp);
    tmp = rect_in_parent(g_pc.out1);
    UnionRect(&dirty, &dirty, &tmp);
    RedrawWindow(parent, &dirty, nullptr,
                 RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
}

void show_page(int idx) {
    for (int i = 0; i < 4; ++i) {
        for (HWND c : g_page_ctls[i]) ShowWindow(c, i == idx ? SW_SHOW : SW_HIDE);
        // owner-draw 页签没有自动互斥，选中态手动维护
        SendMessageW(g_pc.tabs[i], BM_SETCHECK, i == idx ? BST_CHECKED : BST_UNCHECKED, 0);
        InvalidateRect(g_pc.tabs[i], nullptr, FALSE);
    }
    if (idx == 0) layout_page1(g_weights_ver);  // 重新隐藏非当前版本的文件行
}

// 把「背景 + 立绘」按当前 g_dpi 烘焙成一张全客户区缓存位图：WM_ERASEBKGND
// 直接 BitBlt 它来完成擦除 —— 透明文字标签下方永远是真实背景像素，
// 既不闪烁也不会残影。WM_DPICHANGED 只重烘焙，不重读图片（g_art_w 不变，
// 已记录的控件基准坐标仍然有效）。
void bake_panel_bg() {
    if (g_bg_dc) { DeleteDC(g_bg_dc); g_bg_dc = nullptr; }
    if (g_bg_bmp) { DeleteObject(g_bg_bmp); g_bg_bmp = nullptr; }
    if (!g_bg_img) return;
    // 以实际客户区为准：WM_DPICHANGED 的系统建议矩形按比例算出，非客户区
    // 的取整可能让客户区与 sc() 理想值差 1–2 像素，缓存位图必须严格等大。
    RECT client{0, 0, sc(kCliW + g_art_w), sc(kCliH)};
    if (g_panel) GetClientRect(g_panel, &client);
    const int cw = client.right, ch = client.bottom;
    if (cw <= 0 || ch <= 0) return;
    HDC scr = GetDC(nullptr);
    g_bg_dc = CreateCompatibleDC(scr);
    g_bg_bmp = CreateCompatibleBitmap(scr, cw, ch);
    ReleaseDC(nullptr, scr);
    HGDIOBJ old = SelectObject(g_bg_dc, g_bg_bmp);
    Gdiplus::Graphics g(g_bg_dc);
    g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
    // 按高度等比铺满，横向平移让月亮落到立绘条中央（源图月心约 x=390/1915），
    // 控件区不被亮部压字
    const double bg_scale = (double)ch / g_bg_img->GetHeight();
    const int bw = static_cast<int>(g_bg_img->GetWidth() * bg_scale);
    const int moon_x = static_cast<int>(390.0 / g_bg_img->GetWidth() * bw);
    g.DrawImage(g_bg_img, sc(g_art_w) / 2 - moon_x, 0, bw, ch);
    // 控件区压一层半透明深色，浅色文字在任何云纹上都可读
    Gdiplus::SolidBrush dim(Gdiplus::Color(115, 8, 10, 22));
    g.FillRectangle(&dim, sc(g_art_w), 0, cw - sc(g_art_w), ch);
    if (g_art) {  // 立绘居中烘焙进背景
        const int iw = static_cast<int>(g_art->GetWidth()),
                  ih = static_cast<int>(g_art->GetHeight());
        int dw = sc(g_art_w) - sc(8), dh = static_cast<int>((double)dw * ih / iw);
        if (dh > ch) {
            dh = ch;
            dw = static_cast<int>((double)dh * iw / ih);
        }
        g.DrawImage(g_art, (sc(g_art_w) - dw) / 2, (ch - dh) / 2, dw, dh);
    }
    SelectObject(g_bg_dc, old);
}

// 立绘/背景加载：GDI+ 惰性初始化（进程一次）。立绘缺失时 g_art_w=0，
// 窗口退回经典 644 宽紧凑布局，功能不受影响。
void load_panel_art() {
    static ULONG_PTR gdip_token = 0;
    if (!gdip_token) {
        Gdiplus::GdiplusStartupInput input;
        if (Gdiplus::GdiplusStartup(&gdip_token, &input, nullptr) != Gdiplus::Ok) return;
    }
    delete g_art;
    g_art = Gdiplus::Image::FromFile((g_root_w + L"\\media\\qwen-charactor-alpha.png").c_str());
    if (g_art && g_art->GetLastStatus() == Gdiplus::Ok) {
        g_art_w = 210;
    } else {
        delete g_art;
        g_art = nullptr;
        g_art_w = 0;
    }
    delete g_bg_img;
    g_bg_img = Gdiplus::Image::FromFile((g_root_w + L"\\media\\panel-bg.png").c_str());
    if (g_bg_img && g_bg_img->GetLastStatus() != Gdiplus::Ok) {
        delete g_bg_img;
        g_bg_img = nullptr;
    }
}

// owner-draw 按钮：深色圆角卡片。先贴回烘焙背景（圆角外不残留旧像素），
// 再按状态（悬停/按下/选中页签/禁用）绘制。ODS_HOTLIGHT 由主题激活的按钮
// 热追踪提供。
void draw_panel_button(const DRAWITEMSTRUCT* d) {
    const int w = d->rcItem.right - d->rcItem.left,
              hh = d->rcItem.bottom - d->rcItem.top;
    if (g_bg_dc)
        BitBlt(d->hDC, 0, 0, w, hh, g_bg_dc, d->rcItem.left, d->rcItem.top, SRCCOPY);
    else
        FillRect(d->hDC, &d->rcItem, g_panel_bg);  // 无背景图时先铺底色，否则半透明填充会叠在残留白色上
    const bool hot = (d->itemState & ODS_HOTLIGHT) != 0;
    const bool pressed = (d->itemState & ODS_SELECTED) != 0;
    const bool checked = (d->itemState & ODS_CHECKED) != 0;
    const bool disabled = (d->itemState & ODS_DISABLED) != 0;
    Gdiplus::Graphics g(d->hDC);
    g.SetSmoothingMode(Gdiplus::SmoothingModeAntiAlias);
    Gdiplus::Color fill =
        disabled   ? Gdiplus::Color(140, 20, 24, 40)
        : pressed  ? Gdiplus::Color(230, 14, 17, 32)
        : checked  ? Gdiplus::Color(225, 56, 74, 116)   // 当前页签：蓝调强调
        : hot      ? Gdiplus::Color(215, 42, 52, 82)
                   : Gdiplus::Color(205, 26, 32, 54);
    Gdiplus::Color edge = checked ? Gdiplus::Color(255, 130, 160, 220)
                                  : Gdiplus::Color(255, 96, 110, 150);
    Gdiplus::GraphicsPath path;
    const int r = sc(5), x0 = sc(1), y0 = sc(1), x1 = w - sc(2), y1 = hh - sc(2);
    path.AddArc(x0, y0, r * 2, r * 2, 180, 90);
    path.AddArc(x1 - r * 2, y0, r * 2, r * 2, 270, 90);
    path.AddArc(x1 - r * 2, y1 - r * 2, r * 2, r * 2, 0, 90);
    path.AddArc(x0, y1 - r * 2, r * 2, r * 2, 90, 90);
    path.CloseFigure();
    Gdiplus::SolidBrush fb(fill);
    g.FillPath(&fb, &path);
    Gdiplus::Pen pen(edge, static_cast<Gdiplus::REAL>(g_dpi) / 96.0f);
    g.DrawPath(&pen, &path);
    wchar_t txt[128];
    GetWindowTextW(d->hwndItem, txt, 128);
    HFONT hf = reinterpret_cast<HFONT>(SendMessageW(d->hwndItem, WM_GETFONT, 0, 0));
    Gdiplus::Font font(d->hDC, hf);
    Gdiplus::StringFormat sf;
    sf.SetAlignment(Gdiplus::StringAlignmentCenter);
    sf.SetLineAlignment(Gdiplus::StringAlignmentCenter);
    Gdiplus::RectF rc(0, 0, (Gdiplus::REAL)w, (Gdiplus::REAL)hh);
    Gdiplus::SolidBrush tb(disabled ? Gdiplus::Color(255, 112, 120, 142)
                                    : Gdiplus::Color(255, GetRValue(kPanelFg),
                                                     GetGValue(kPanelFg), GetBValue(kPanelFg)));
    g.DrawString(txt, -1, &font, rc, &sf, &tb);
}

LRESULT CALLBACK panel_proc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == WM_DPICHANGED) {
        // 系统建议的新窗口矩形（物理像素）已按新旧 DPI 比例算好，直接采用；
        // 再按记录的 96-DPI 基准重排全部控件，字体/背景缓存一起重建。
        const int dpi = static_cast<int>(HIWORD(wp));
        if (dpi > 0 && dpi != g_dpi) {
            g_dpi = dpi;
            const RECT* want = reinterpret_cast<const RECT*>(lp);
            SetWindowPos(h, nullptr, want->left, want->top, want->right - want->left,
                         want->bottom - want->top, SWP_NOZORDER | SWP_NOACTIVATE);
            bake_panel_bg();  // 背景位图按新客户区尺寸重烘焙（不重读图片）
            HFONT font = panel_font();
            for (const CtlBase& b : g_ctl_base) {
                SetWindowPos(b.h, nullptr, sc(b.r.left), sc(b.r.top),
                             sc(b.r.right - b.r.left), sc(b.r.bottom - b.r.top),
                             SWP_NOZORDER | SWP_NOACTIVATE);
                SendMessageW(b.h, WM_SETFONT, reinterpret_cast<WPARAM>(font), TRUE);
            }
            SendMessageW(h, WM_SETICON, ICON_BIG,
                         reinterpret_cast<LPARAM>(app_icon(sc(32))));
            SendMessageW(h, WM_SETICON, ICON_SMALL,
                         reinterpret_cast<LPARAM>(app_icon(sc(16))));
            RedrawWindow(h, nullptr, nullptr,
                         RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW);
        }
        return 0;
    }
    if (msg == WM_DRAWITEM) {
        const DRAWITEMSTRUCT* d = reinterpret_cast<const DRAWITEMSTRUCT*>(lp);
        if (d->CtlType == ODT_BUTTON && g_panel_bg) {
            draw_panel_button(d);
            return TRUE;
        }
    }
    if (msg == WM_CTLCOLORSTATIC && g_panel_bg) {
        // 只读 Edit（各页检查输出框）：透明底会让 SetWindowText 重绘时旧文字
        // 不被擦除、新旧叠印；改用与输入框一致的深色卡片实心画刷。
        HWND ctl = reinterpret_cast<HWND>(lp);
        wchar_t cls[16];
        if (GetClassNameW(ctl, cls, 16) && wcscmp(cls, L"Edit") == 0 &&
            (GetWindowLongW(ctl, GWL_STYLE) & ES_READONLY) && g_edit_bg) {
            HDC dc = reinterpret_cast<HDC>(wp);
            SetTextColor(dc, kPanelFg);
            SetBkColor(dc, kEditBg);
            return reinterpret_cast<LRESULT>(g_edit_bg);
        }
        // 标签/勾选框/单选钮：浅色文字 + HOLLOW_BRUSH 不填底（实心画刷会填满
        // 整个控件矩形，SetBkMode 只管字符格），夜空背景完整透出
        SetTextColor(reinterpret_cast<HDC>(wp), kPanelFg);
        SetBkMode(reinterpret_cast<HDC>(wp), TRANSPARENT);
        return reinterpret_cast<LRESULT>(GetStockObject(HOLLOW_BRUSH));
    }
    if (msg == WM_CTLCOLOREDIT && g_edit_bg) {
        // 可编辑输入框：深色卡片 + 浅色文字
        HDC dc = reinterpret_cast<HDC>(wp);
        SetTextColor(dc, kPanelFg);
        SetBkColor(dc, kEditBg);
        return reinterpret_cast<LRESULT>(g_edit_bg);
    }
    if (msg == WM_CTLCOLORLISTBOX && g_edit_bg) {
        // 下拉列表（并发路数 ComboBox 展开的列表部分）
        HDC dc = reinterpret_cast<HDC>(wp);
        SetTextColor(dc, kPanelFg);
        SetBkColor(dc, kEditBg);
        return reinterpret_cast<LRESULT>(g_edit_bg);
    }
    if (msg == WM_ERASEBKGND) {
        RECT client;
        GetClientRect(h, &client);
        if (g_bg_dc) {  // 擦除 = 贴回烘焙背景（透明标签下方永远是真实背景像素）
            HDC dc = reinterpret_cast<HDC>(wp);
            BitBlt(dc, 0, 0, client.right, client.bottom, g_bg_dc, 0, 0, SRCCOPY);
            return 1;
        }
        if (g_art_w > 0) {  // 无背景图只有立绘：只擦控件区，立绘条由 WM_PAINT 画
            HDC dc = reinterpret_cast<HDC>(wp);
            RECT r{sc(g_art_w), 0, client.right, client.bottom};
            FillRect(dc, &r, g_panel_bg);
            return 1;
        }
    }
    if (msg == WM_PAINT) {
        PAINTSTRUCT ps;
        HDC dc = BeginPaint(h, &ps);
        if (g_bg_dc) {
            RECT client;
            GetClientRect(h, &client);
            BitBlt(dc, 0, 0, client.right, client.bottom, g_bg_dc, 0, 0, SRCCOPY);
        } else if (g_art && g_art_w > 0) {
            RECT client;
            GetClientRect(h, &client);
            Gdiplus::Graphics g(dc);
            g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
            Gdiplus::SolidBrush bg(Gdiplus::Color(255, GetRValue(kPanelBg), GetGValue(kPanelBg),
                                                  GetBValue(kPanelBg)));
            g.FillRectangle(&bg, 0, 0, sc(g_art_w), client.bottom);
            const int iw = static_cast<int>(g_art->GetWidth()),
                      ih = static_cast<int>(g_art->GetHeight());
            int dw = sc(g_art_w) - sc(8), dh = static_cast<int>((double)dw * ih / iw);
            if (dh > client.bottom) {  // 超高则按高度等比收
                dh = client.bottom;
                dw = static_cast<int>((double)dh * iw / ih);
            }
            g.DrawImage(g_art, (sc(g_art_w) - dw) / 2, (client.bottom - dh) / 2, dw, dh);
        }
        EndPaint(h, &ps);
        return 0;
    }
    if (msg == WM_COMMAND) {
        const UINT id = LOWORD(wp), code = HIWORD(wp);
        // 编辑框输入 / 下拉选择变化 → 即时重检（各页列表独立刷新）
        if (code == EN_CHANGE || code == CBN_SELCHANGE) {
            // 单请求上限的预设下拉：选中即把数值填进编辑框（条目 0 是占位项）。
            // 选完回到占位项，这样再次选中同一个预设也能触发 CBN_SELCHANGE。
            // 模型原生上限 256K（ROPE_ORIGINAL_CTX），超过就必须开 YaRN。
            if (id == kCbMaxctxPreset && code == CBN_SELCHANGE) {
                static const long kTokens[] = {65536, 131072, 196608,
                                               262144, 393216, 524288};
                const int sel = static_cast<int>(
                    SendMessageW(g_pc.maxctx_preset, CB_GETCURSEL, 0, 0));
                if (sel >= 1 && sel <= static_cast<int>(sizeof kTokens / sizeof kTokens[0])) {
                    const long tokens = kTokens[sel - 1];
                    long native_ctx = 262144;
                    pn_int(cfg("ROPE_ORIGINAL_CTX", "262144"), &native_ctx);
                    if (tokens > native_ctx)
                        SendMessageW(g_pc.yarn, BM_SETCHECK, BST_CHECKED, 0);
                    set_edit(g_pc.maxctx, std::to_string(tokens));
                    SendMessageW(g_pc.maxctx_preset, CB_SETCURSEL, 0, 0);
                }
            }
            // 并发总池跟随单请求上限：未手动改过总池框时，上限变了总池框跟着变；
            // 总池框输入非空非 0 → 标记已手动改；输入 0 → 回到跟随并立即回填上限值。
            // g_pool_guard：程序自己 SetWindowText 时不触发这些逻辑。
            if (code == EN_CHANGE && !g_pool_guard) {
                if (id == kEdKvPool) {
                    const std::string s = get_edit(g_pc.kv_pool);
                    if (s == "0") g_pool_touched = false;
                    else if (!s.empty()) g_pool_touched = true;
                }
                if ((id == kEdMaxctx && !g_pool_touched) ||
                    (id == kEdKvPool && get_edit(g_pc.kv_pool) == "0")) {
                    g_pool_guard = true;
                    SetWindowTextW(g_pc.kv_pool, to_w(get_edit(g_pc.maxctx)).c_str());
                    g_pool_guard = false;
                }
            }
            panel_recheck();
            return 0;
        }
        switch (id) {
            case kTab1: case kTab2: case kTab3: case kTab4:
                show_page(static_cast<int>(id - kTab1));
                return 0;
            case kWVer1: case kWVer2: case kWVer3:
                layout_page1(static_cast<int>(id - kWVer1));
                panel_recheck();
                return 0;
            case kBtnSave: {
                std::string error;
                if (!panel_apply_and_save(&error)) {
                    msgbox(std::string(TR("保存配置失败：", "Failed to save config: ")) + error,
                           MB_OK | MB_ICONERROR);
                    return 0;
                }
                // 保存成功即关闭面板（返回 kPanelClose）
                g_panel_result = kPanelClose;
                g_panel_done = true;
                return 0;
            }
            case kBtnCancel:
                g_panel_result = kPanelClose;
                g_panel_done = true;
                return 0;
            case kBtnLang:
                // 切换语言：写回 lang 标记后返回 kPanelRelang，由调用方重开面板
                conf_write_marker_line(kLangMarker, g_lang_en ? "zh" : "en");
                g_panel_result = kPanelRelang;
                g_panel_done = true;
                return 0;
            case kBtnRecheck: panel_recheck(); return 0;
            case kBrModel:
                if (panel_browse(g_pc.row_model.edit)) panel_recheck();
                return 0;
            case kBrNgram:
                if (panel_browse(g_pc.row_ngram.edit)) panel_recheck();
                return 0;
            case kBrOverlay:
                if (panel_browse(g_pc.row_overlay.edit)) panel_recheck();
                return 0;
            case kBrMtp:
                if (panel_browse(g_pc.row_mtp.edit)) panel_recheck();
                return 0;
            case kBrVision:
                if (panel_browse(g_pc.row_vision.edit)) panel_recheck();
                return 0;
            case kBrGguf:
                if (panel_browse(g_pc.row_gguf.edit)) panel_recheck();
                return 0;
            case kBrGgufMtp:
                if (panel_browse(g_pc.row_gguf_mtp.edit)) panel_recheck();
                return 0;
            case kBrGgufVision:
                if (panel_browse(g_pc.row_gguf_vision.edit)) panel_recheck();
                return 0;
            case kBrKvsnapDir:
                if (panel_browse_dir(g_pc.kvsnap_dir)) panel_recheck();
                return 0;
            case kOffOverlay:
                set_optional_row_state(g_pc.row_overlay.off, g_pc.row_overlay.edit,
                                       g_pc.row_overlay.browse);
                panel_recheck();
                return 0;
            case kOffMtp:
                set_optional_row_state(g_pc.row_mtp.off, g_pc.row_mtp.edit, g_pc.row_mtp.browse);
                panel_recheck();
                return 0;
            case kOffVision:
                set_optional_row_state(g_pc.row_vision.off, g_pc.row_vision.edit,
                                       g_pc.row_vision.browse);
                panel_recheck();
                return 0;
            case kOffGgufMtp:
                set_optional_row_state(g_pc.row_gguf_mtp.off, g_pc.row_gguf_mtp.edit,
                                       g_pc.row_gguf_mtp.browse);
                panel_recheck();
                return 0;
            case kOffGgufVision:
                set_optional_row_state(g_pc.row_gguf_vision.off, g_pc.row_gguf_vision.edit,
                                       g_pc.row_gguf_vision.browse);
                panel_recheck();
                return 0;
            case kCkKvPaged: panel_recheck(); return 0;
            case kCkYarn: panel_recheck(); return 0;
            case kCkMoeA4: panel_recheck(); return 0;
            case kCkAffinity: panel_recheck(); return 0;
            default: break;
        }
    }
    if (msg == WM_CLOSE) {
        g_panel_result = kPanelClose;
        g_panel_done = true;
        return 0;
    }
    return DefWindowProcW(h, msg, wp, lp);
}

// 第 1 页文件行：[label][x不使用][EDIT][浏览…]（extra_style 给 label，如 WS_GROUP）
FileRow page1_file_row(HWND parent, int y, const wchar_t* label, int id_edit, int id_browse,
                       int id_off, const std::string& value, bool off,
                       DWORD extra_style = 0) {
    FileRow r;
    r.label = pctl(0, parent, L"Static", SS_LEFTNOWORDWRAP | extra_style, kLblX, y + 3, kLblW,
                   18, label, 0);
    int ex = kEdX, ew = kEdPlainW;
    if (id_off) {
        r.off = pctl(0, parent, L"Button", BS_AUTOCHECKBOX, ex, y, kOffW, 20,
                     TR(L"不使用", L"off"), id_off);
        SendMessageW(r.off, BM_SETCHECK, off ? BST_CHECKED : BST_UNCHECKED, 0);
        ex += kOffW + 8;
        ew = 556 - ex;
    }
    r.edit = pctl(0, parent, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, ex, y, ew, 22,
                  L"", id_edit);
    set_edit(r.edit, value);
    if (id_browse)
        r.browse = pctl(0, parent, L"Button", BS_PUSHBUTTON, kBrX, y - 1, kBrW, 24,
                        TR(L"浏览…", L"Browse"), id_browse);
    if (id_off) set_optional_row_state(r.off, r.edit, r.browse);
    return r;
}

PanelResult panel_run() {
    panel_lang_resolve();  // 每次打开重新解析（语言按钮切换后重开即生效）
    // 服务探测：conf 的 API_PORT 已在监听 → 服务运行中（端口占用/显存检查降级为提示）
    {
        long api_port = 8731;
        if (!pn_int(cfg("API_PORT", "8731"), &api_port)) api_port = 8731;
        g_service_running = port_listening(static_cast<int>(api_port));
    }
    g_panel_done = false;
    g_panel_result = kPanelClose;
    for (auto& v : g_page_ctls) v.clear();
    g_ctl_base.clear();  // 上一轮（语言切换重开）的句柄已随窗口销毁
    HINSTANCE hi = GetModuleHandleW(nullptr);
    static bool registered = false;
    if (!registered) {
        if (!g_panel_bg) g_panel_bg = CreateSolidBrush(kPanelBg);
        if (!g_edit_bg) g_edit_bg = CreateSolidBrush(kEditBg);
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof wc;
        wc.lpfnWndProc = panel_proc;
        wc.hInstance = hi;
        wc.lpszClassName = L"Qwenox_setup_panel";
        wc.hbrBackground = g_panel_bg;
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
        // 窗口图标（任务栏按钮用）：内嵌 ICO，否则任务栏是空白默认图标。
        // 类图标按 96-DPI 基准注册，窗口建好后再按实际 DPI 覆盖。
        static HICON cls_big = app_icon(32);
        static HICON cls_small = app_icon(16);
        wc.hIcon = cls_big;
        wc.hIconSm = cls_small;
        RegisterClassExW(&wc);
        registered = true;
    }
    // 先在光标所在显示器上建隐藏窗口，取得该屏 DPI 后再定尺寸、装背景、建控件。
    // PMv2 下窗口的 DPI 跟随所在显示器；这样副屏打开也不会按主屏 DPI 排版。
    POINT cursor{};
    GetCursorPos(&cursor);
    MONITORINFO mi{};
    mi.cbSize = sizeof mi;
    GetMonitorInfoW(MonitorFromPoint(cursor, MONITOR_DEFAULTTOPRIMARY), &mi);
    HWND h = CreateWindowExW(0, L"Qwenox_setup_panel",
                             TR(L"Qwenox 设置", L"Qwenox Settings"),
                             WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU,
                             mi.rcWork.left, mi.rcWork.top, 100, 100, nullptr, nullptr, hi,
                             nullptr);
    if (!h) {
        msgbox(std::string(TR("无法创建设置窗口，错误码 ", "Cannot create the setup window, error code ")) +
                   std::to_string(GetLastError()),
               MB_OK | MB_ICONERROR);
        return kPanelClose;
    }
    g_panel = h;
    g_dpi = static_cast<int>(windpi::dpi_for_window(h));

    load_panel_art();  // 按 g_dpi 决定背景缓存尺寸；立绘缺失时 g_art_w=0
    RECT rc{0, 0, sc(kCliW + g_art_w), sc(kCliH)};
    const DWORD frame = WS_OVERLAPPED | WS_CAPTION | WS_SYSMENU;
    windpi::adjust_window_rect_ex_for_dpi(&rc, frame, FALSE, 0, static_cast<UINT>(g_dpi));
    const int win_w = rc.right - rc.left, win_h = rc.bottom - rc.top;
    MONITORINFO area{};
    area.cbSize = sizeof area;
    GetMonitorInfoW(MonitorFromWindow(h, MONITOR_DEFAULTTONEAREST), &area);
    const int wx = area.rcWork.left + (area.rcWork.right - area.rcWork.left - win_w) / 2;
    const int wy = area.rcWork.top + (area.rcWork.bottom - area.rcWork.top - win_h) / 2;
    SetWindowPos(h, nullptr, wx, wy, win_w, win_h, SWP_NOZORDER | SWP_NOACTIVATE);
    SendMessageW(h, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(app_icon(sc(32))));
    SendMessageW(h, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(app_icon(sc(16))));
    bake_panel_bg();  // 窗口尺寸已定，按实际客户区烘焙背景缓存
    // 深色标题栏（Win10 1809+/Win11；不支持的平台静默失败保持默认）
    if (g_panel_bg) {
        const BOOL dark = TRUE;
        DwmSetWindowAttribute(h, 20 /*DWMWA_USE_IMMERSIVE_DARK_MODE*/, &dark, sizeof dark);
    }

    const PanelVals init = panel_vals_from_conf();
    g_weights_ver = init.weights_ver;

    // ---- 分页切换（恒可见）：BS_PUSHLIKE 单选按钮，不引入 comctl32
    const wchar_t* tab_names[4] = {TR(L"1 权重", L"1 Weights"),
                                   TR(L"2 推理设置", L"2 Inference"),
                                   TR(L"3 端口与监听", L"3 Ports & listen"),
                                   TR(L"4 显存检查", L"4 VRAM check")};
    for (int i = 0; i < 4; ++i) {
        g_pc.tabs[i] = add_ctrl(h, L"Button",
                                BS_PUSHLIKE | BS_AUTORADIOBUTTON | WS_TABSTOP |
                                    (i == 0 ? WS_GROUP : 0),
                                12 + i * 158, 8, 150, 24, tab_names[i], kTab1 + i);
    }
    SendMessageW(g_pc.tabs[0], BM_SETCHECK, BST_CHECKED, 0);

    // ---- 第 1 页：权重
    plabel(0, h, kLblX, 44, TR(L"权重版本", L"Weights version"));
    const wchar_t* ver_names[3] = {TR(L"V1 版本", L"V1"), TR(L"V2 版本", L"V2"),
                                   TR(L"GGUF 版本", L"GGUF")};
    for (int i = 0; i < 3; ++i) {
        g_pc.wver[i] = pctl(0, h, L"Button",
                            BS_AUTORADIOBUTTON | WS_TABSTOP | (i == 0 ? WS_GROUP : 0),
                            kEdX + i * 92, 44, i == 2 ? 100 : 86, 20, ver_names[i], kWVer1 + i);
    }
    SendMessageW(g_pc.wver[g_weights_ver], BM_SETCHECK, BST_CHECKED, 0);
    int ry = 76;
    g_pc.row_model = page1_file_row(h, ry, TR(L"主模型", L"Main model"), kEdModel, kBrModel, 0,
                                    init.model, false, WS_GROUP);  // WS_GROUP：关闭上面的单选组
    ry += 26;
    g_pc.row_ngram = page1_file_row(h, ry, TR(L"n-gram 表", L"n-gram table"), kEdNgram, kBrNgram, 0,
                                    init.ngram, false);
    ry += 26;
    g_pc.row_overlay = page1_file_row(h, ry, L"overlay", kEdOverlay, kBrOverlay, kOffOverlay,
                                      init.overlay, init.no_overlay);
    ry += 26;
    g_pc.row_mtp = page1_file_row(h, ry, TR(L"MTP 草稿", L"MTP draft"), kEdMtp, kBrMtp, kOffMtp,
                                  init.mtp, init.no_mtp);
    ry += 26;
    g_pc.row_vision = page1_file_row(h, ry, TR(L"视觉塔", L"Vision tower"), kEdVision, kBrVision,
                                     kOffVision, init.vision, init.no_vision);
    ry += 26;
    g_pc.row_gguf = page1_file_row(h, ry, TR(L"GGUF 主分片", L"GGUF main shard"), kEdGguf, kBrGguf,
                                   0, init.gguf, false);
    ry += 26;
    g_pc.row_gguf_mtp = page1_file_row(h, ry, L"GGUF MTP", kEdGgufMtp, kBrGgufMtp, kOffGgufMtp,
                                       init.gguf_mtp, init.no_gguf_mtp);
    ry += 26;
    g_pc.row_gguf_vision = page1_file_row(h, ry, TR(L"GGUF 视觉塔", L"GGUF vision tower"),
                                          kEdGgufVision, kBrGgufVision, kOffGgufVision,
                                          init.gguf_vision, init.no_gguf_vision);
    ry += 26;
    g_pc.row_tok = page1_file_row(h, ry, L"tokenizer", kEdTok, 0, 0, init.tok, false);
    g_pc.tok_hint = pctl(0, h, L"Static", SS_LEFTNOWORDWRAP, kEdX, 210, 434, 18,
                         TR(L"目录内需包含 tokenizer.json", L"Folder must contain tokenizer.json"),
                         0);
    g_pc.ver_note = pctl(0, h, L"Static", SS_LEFTNOWORDWRAP, kEdX, 232, 434, 18, L"", 0);
    // 位置与显隐由 layout_page1 按版本决定（只在 V2 显示）
    g_moe_a4_init = init.moe_a4;
    g_pc.moe_a4 = pctl(0, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, 254, 434, 20,
                       TR(L"prefill 4-bit 加速（快约 9%，会降低精度）",
                          L"4-bit prefill speedup (~9% faster, lowers precision)"),
                       kCkMoeA4);
    SendMessageW(g_pc.moe_a4, BM_SETCHECK, moe_a4_on(init.moe_a4) ? BST_CHECKED : BST_UNCHECKED,
                 0);
    g_pc.out1 = pctl(0, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, 254, 624, kListBottom - 254, L"", kEdOut1);

    // ---- 第 2 页：推理设置（上下文/并发/YaRN/亲和性，行尾无 hint，说明在检查列表顶部）
    int y2 = 46;
    plabel(1, h, kLblX, y2, TR(L"单请求上限 (token)", L"Per-request context (tokens)"));
    g_pc.maxctx = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                       22, L"", kEdMaxctx);
    set_edit(g_pc.maxctx, init.max_context);
    // 快捷选择：只写回上面的编辑框，手输任意值仍然有效
    g_pc.maxctx_preset = pctl(1, h, L"ComboBox", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP,
                              kHintX, y2, 132, 200, L"", kCbMaxctxPreset);
    SendMessageW(g_pc.maxctx_preset, CB_ADDSTRING, 0,
                 reinterpret_cast<LPARAM>(TR(L"快捷选择…", L"Presets…")));
    for (const wchar_t* s : {L"64K (65536)", L"128K (131072)", L"192K (196608)",
                             L"256K (262144)", L"384K (393216)", L"512K (524288)"})
        SendMessageW(g_pc.maxctx_preset, CB_ADDSTRING, 0, reinterpret_cast<LPARAM>(s));
    SendMessageW(g_pc.maxctx_preset, CB_SETCURSEL, 0, 0);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"并发路数", L"Parallel streams"));
    g_pc.parallel = pctl(1, h, L"ComboBox", CBS_DROPDOWNLIST | WS_VSCROLL | WS_TABSTOP, kEdX,
                         y2, 60, 200, L"", kCbParallel);
    for (int i = 1; i <= 8; ++i)
        SendMessageW(g_pc.parallel, CB_ADDSTRING, 0,
                     reinterpret_cast<LPARAM>(std::to_wstring(i).c_str()));
    long par_init = 1;
    pn_int(init.parallel, &par_init);
    SendMessageW(g_pc.parallel, CB_SETCURSEL, static_cast<WPARAM>(par_init - 1), 0);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"并发总池 (token)", L"KV pool (tokens)"));
    g_pc.kv_pool = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                        22, L"", kEdKvPool);
    // conf=0（跟随）时显示单请求上限的值，g_pool_touched=false；
    // conf 有显式值则显示实值并视为已手动设定
    long pool_init = 0;
    pn_int(init.kv_pool, &pool_init);
    g_pool_touched = pool_init != 0;
    g_pool_guard = true;
    set_edit(g_pc.kv_pool, g_pool_touched ? init.kv_pool : init.max_context);
    g_pool_guard = false;
    y2 += 26;
    g_pc.kv_paged = pctl(1, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, y2, 300, 20,
                         TR(L"分页 KV（并发必须开启）", L"Paged KV (required for parallel >1)"),
                         kCkKvPaged);
    SendMessageW(g_pc.kv_paged, BM_SETCHECK, init.kv_paged ? BST_CHECKED : BST_UNCHECKED, 0);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"prefill 分段", L"Prefill chunk"));
    g_pc.prefill = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                        22, L"", kEdPrefill);
    set_edit(g_pc.prefill, init.prefill_chunk);
    plabel(1, h, kHintX, y2, TR(L"0 = 默认 6144（≈5.2 GiB 工作区）",
                                L"0 = default 6144 (≈5.2 GiB workspace)"),
           kHintW);
    y2 += 26;
    {
        // YaRN 勾选框（ROPE_FACTOR 自动推导 = max(2, ceil(上限/原生))，无编辑行）
        double rope_init = 1;
        pn_double(init.rope_factor, &rope_init);
        g_pc.yarn = pctl(1, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, y2, 300, 20,
                         TR(L"YaRN 长上下文扩展", L"YaRN long-context scaling"), kCkYarn);
        SendMessageW(g_pc.yarn, BM_SETCHECK, rope_init > 1 ? BST_CHECKED : BST_UNCHECKED, 0);
    }
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"KV 快照上限 (GiB)", L"KV snapshot cap (GiB)"));
    g_pc.kvsnap = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 120,
                       22, L"", kEdKvsnap);
    set_edit(g_pc.kvsnap, init.kvsnap_max_gb);
    plabel(1, h, kHintX, y2, TR(L"RAM 检查点容量，0 = 关闭", L"RAM checkpoint capacity, 0 = off"),
           kHintW);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"KV 快照目录", L"KV snapshot folder"));
    g_pc.kvsnap_dir = pctl(1, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y2, 354,
                           22, L"", kEdKvsnapDir);
    set_edit(g_pc.kvsnap_dir, init.kvsnap_dir);
    g_pc.kvsnap_dir_br = pctl(1, h, L"Button", BS_PUSHBUTTON, kBrX, y2 - 1, kBrW, 24,
                              TR(L"浏览…", L"Browse"), kBrKvsnapDir);
    y2 += 26;
    plabel(1, h, kLblX, y2, TR(L"处理器亲和性", L"CPU affinity"));
    g_pc.affinity = pctl(1, h, L"Button", BS_AUTOCHECKBOX | WS_TABSTOP, kEdX, y2, 400, 20,
                         TR(L"亲和到 CCD 1（双 CCD 机型）", L"Pin to CCD 1 (dual-CCD systems)"),
                         kCkAffinity);
    SendMessageW(g_pc.affinity, BM_SETCHECK,
                 init.affinity.empty() ? BST_UNCHECKED : BST_CHECKED, 0);
    y2 += 26;
    g_pc.out2 = pctl(1, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, y2 + 10, 624, kListBottom - y2 - 10, L"", kEdOut2);

    // ---- 第 3 页：端口与监听（行尾无 hint，说明在第 3 页检查列表顶部）
    int y3 = 46;
    plabel(2, h, kLblX, y3, TR(L"引擎监听地址", L"Engine bind address"));
    g_pc.ehost = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 140,
                      22, L"", kEdEHost);
    set_edit(g_pc.ehost, init.engine_host);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"引擎端口", L"Engine port"));
    g_pc.eport = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 90, 22,
                      L"", kEdEPort);
    set_edit(g_pc.eport, init.engine_port);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"API 监听地址", L"API bind address"));
    g_pc.ahost = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 140,
                      22, L"", kEdAHost);
    set_edit(g_pc.ahost, init.api_host);
    y3 += 26;
    plabel(2, h, kLblX, y3, TR(L"API 端口", L"API port"));
    g_pc.aport = pctl(2, h, L"Edit", WS_BORDER | ES_AUTOHSCROLL | WS_TABSTOP, kEdX, y3, 90, 22,
                      L"", kEdAPort);
    set_edit(g_pc.aport, init.api_port);
    y3 += 26;
    g_pc.out3 = pctl(2, h, L"Edit",
                     WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                     12, y3 + 8, 624, kListBottom - y3 - 8, L"", kEdOut3);

    // ---- 第 4 页：显存环境检查
    pctl(3, h, L"Button", BS_PUSHBUTTON, 524, 42, 112, 24, TR(L"重新检查", L"Re-check"),
         kBtnRecheck);
    g_pc.report = pctl(3, h, L"Edit",
                       WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                       12, 72, 624, 272, L"", kEdReport);
    plabel(3, h, kLblX, 348, TR(L"其它页的失败汇总", L"Failures from other pages"), 200);
    g_pc.summary = pctl(3, h, L"Edit",
                        WS_BORDER | ES_MULTILINE | ES_READONLY | WS_VSCROLL | ES_AUTOVSCROLL,
                        12, 372, 624, kListBottom - 372, L"", kEdSummary);

    // ---- 底部按钮行（恒可见）
    const int by = 462;
    add_ctrl(h, L"Button", BS_DEFPUSHBUTTON, 297, by, 110, 30, TR(L"保存", L"Save"), kBtnSave);
    add_ctrl(h, L"Button", BS_PUSHBUTTON, 417, by, 90, 30, TR(L"关闭", L"Close"), kBtnCancel);
    // 语言切换：写 lang 标记并返回 kPanelRelang（按钮文字显示目标语言）
    add_ctrl(h, L"Button", BS_PUSHBUTTON, 512, by, 124, 30,
             g_lang_en ? L"切换为中文" : L"Switch to English", kBtnLang);

    show_page(0);
    panel_recheck();
    ShowWindow(h, SW_SHOW);
    UpdateWindow(h);
    SetForegroundWindow(h);

    // 模态循环：不用 PostQuitMessage（托盘打开时 WM_QUIT 会连带杀掉托盘循环）
    MSG m;
    while (!g_panel_done && GetMessageW(&m, nullptr, 0, 0) > 0) {
        if (IsDialogMessageW(h, &m)) continue;
        TranslateMessage(&m);
        DispatchMessageW(&m);
    }
    DestroyWindow(h);
    g_panel = nullptr;
    return g_panel_result;
}

}  // namespace
