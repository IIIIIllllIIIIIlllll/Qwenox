// diagpack.cpp — 见 diagpack.h。
#include "diagpack.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

#include "../crashdiag.h"  // env_report / secret_name（Windows 下带来 winsock2 + windows.h）
#include "../engine_net.h"
#include "../version.h"
#include "miniz.h"

#ifdef _WIN32
#include <winevt.h>
#else
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace diagpack {
namespace {

constexpr size_t kEngineHead = 512 * 1024, kEngineTail = 8 * 1024 * 1024;
constexpr size_t kApiHead = 256 * 1024, kApiTail = 4 * 1024 * 1024;
constexpr size_t kMaxDump = 512ull * 1024 * 1024;  // 单个 .dmp 上限（防意外的完整 dump）
constexpr int kKeepPacks = 10;

int64_t now_unix() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string stamp(int64_t t, bool compact) {
    std::time_t tt = static_cast<std::time_t>(t);
    std::tm lt{};
#ifdef _WIN32
    localtime_s(&lt, &tt);
#else
    localtime_r(&tt, &lt);
#endif
    char b[32];
    std::strftime(b, sizeof b, compact ? "%Y%m%d-%H%M%S" : "%Y-%m-%d %H:%M:%S", &lt);
    return b;
}

int64_t mtime_unix(const std::string& p) {
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA d{};
    if (!GetFileAttributesExA(p.c_str(), GetFileExInfoStandard, &d)) return 0;
    const uint64_t t = (uint64_t(d.ftLastWriteTime.dwHighDateTime) << 32) |
                       d.ftLastWriteTime.dwLowDateTime;
    return int64_t((t - 116444736000000000ULL) / 10000000ULL);
#else
    struct stat st {};
    if (stat(p.c_str(), &st) != 0) return 0;
    return int64_t(st.st_mtime);
#endif
}

uint64_t file_size(const std::string& p) {
    std::error_code ec;
    const auto n = fs::file_size(fs::path(p), ec);
    return ec ? 0 : uint64_t(n);
}

bool read_range(const std::string& p, uint64_t off, uint64_t n, std::string* out) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    f.seekg(std::streamoff(off));
    out->resize(size_t(n));
    f.read(&(*out)[0], std::streamsize(n));
    out->resize(size_t(f.gcount()));
    return true;
}

bool read_all(const std::string& p, std::string* out) {
    return read_range(p, 0, file_size(p), out);
}

// 开头 head + 结尾 tail（中间省略并注明字节数）
std::string head_tail(const std::string& p, size_t head, size_t tail) {
    const uint64_t n = file_size(p);
    std::string out;
    if (n <= head + tail) {
        read_all(p, &out);
        return out;
    }
    std::string a, b;
    read_range(p, 0, head, &a);
    read_range(p, n - tail, tail, &b);
    return a + "\n\n...[diagpack: " + std::to_string(n - head - tail) +
           " bytes skipped]...\n\n" + b;
}

struct Entry {
    std::string path;
    int64_t mtime = 0;
};

// dir 下文件名匹配 pred 的文件，按修改时间从新到旧
template <class Pred>
std::vector<Entry> list_files(const std::string& dir, Pred pred) {
    std::vector<Entry> v;
    std::error_code ec;
    for (fs::directory_iterator it(fs::path(dir), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const std::string name = it->path().filename().string();
        if (!pred(name)) continue;
        Entry e;
        e.path = it->path().string();
        e.mtime = mtime_unix(e.path);
        v.push_back(e);
    }
    std::sort(v.begin(), v.end(), [](const Entry& a, const Entry& b) { return a.mtime > b.mtime; });
    return v;
}

bool starts(const std::string& s, const char* p) { return s.compare(0, strlen(p), p) == 0; }
bool ends(const std::string& s, const char* p) {
    const size_t n = strlen(p);
    return s.size() >= n && s.compare(s.size() - n, n, p) == 0;
}

std::string newest_engine_log(const std::string& root) {
    const std::string logs = root + "/logs";
#ifdef _WIN32
    auto v = list_files(logs, [](const std::string& n) { return starts(n, "engine-") && ends(n, ".log"); });
    if (!v.empty()) return v[0].path;
#endif
    // Linux 启动脚本：logs/<时间>-<pid>/engine.log
    std::vector<Entry> best;
    std::error_code ec;
    for (fs::directory_iterator it(fs::path(logs), ec), end; !ec && it != end; it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        const std::string p = (it->path() / "engine.log").string();
        if (fs::exists(fs::path(p), ec)) best.push_back({p, mtime_unix(p)});
    }
    std::sort(best.begin(), best.end(), [](const Entry& a, const Entry& b) { return a.mtime > b.mtime; });
    if (!best.empty()) return best[0].path;
    auto v2 = list_files(logs, [](const std::string& n) { return starts(n, "engine") && ends(n, ".log"); });
    return v2.empty() ? std::string() : v2[0].path;
}

std::string newest_api_log(const std::string& root, const std::string& engine_log) {
    const std::string logs = root + "/logs";
#ifdef _WIN32
    auto v = list_files(logs, [](const std::string& n) { return starts(n, "api-win-") && ends(n, ".log"); });
    if (!v.empty()) return v[0].path;
#endif
    if (!engine_log.empty()) {  // 同一次运行目录里的 api.log
        const fs::path p = fs::path(engine_log).parent_path() / "api.log";
        std::error_code ec;
        if (fs::exists(p, ec)) return p.string();
    }
    auto v2 = list_files(logs, [](const std::string& n) { return starts(n, "api") && ends(n, ".log"); });
    return v2.empty() ? std::string() : v2[0].path;
}

// 报告里最有用的几行（summary.txt 摘要用）
std::string crash_excerpt(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    std::string line, out;
    int n = 0;
    while (n < 14 && std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        for (const char* k : {"kind:", "code:", "signal:", "message:", "thread:", "fault:",
                              "fault-region:", "pc:", "pc-region:", "time:", "pid:"})
            if (starts(line, k)) {
                out += "    " + line + "\n";
                n++;
                break;
            }
        if (starts(line, "registers:")) break;
    }
    return out;
}

std::string mask_conf(const std::string& text) {
    std::istringstream in(text);
    std::string line, out;
    while (std::getline(in, line)) {
        std::string l = line;
        if (!l.empty() && l.back() == '\r') l.pop_back();
        const size_t a = l.find_first_not_of(" \t");
        const size_t eq = l.find('=');
        if (a != std::string::npos && l[a] != '#' && eq != std::string::npos && eq > a) {
            std::string key = l.substr(a, eq - a);
            while (!key.empty() && (key.back() == ' ' || key.back() == '\t')) key.pop_back();
            if (crashdiag::secret_name(key) && eq + 1 < l.size()) l = l.substr(0, eq + 1) + "***";
        }
        out += l + "\n";
    }
    return out;
}

std::string logs_index(const std::string& root) {
    std::string out;
    for (const char* sub : {"logs", "logs/crash", "logs/diag"}) {
        out += "[" + std::string(sub) + "]\n";
        auto v = list_files(root + "/" + sub, [](const std::string&) { return true; });
        int k = 0;
        for (const auto& e : v) {
            if (k++ >= 200) {
                out += "  ...\n";
                break;
            }
            char b[64];
            snprintf(b, sizeof b, "%14llu  ", static_cast<unsigned long long>(file_size(e.path)));
            out += "  " + stamp(e.mtime, false) + b + fs::path(e.path).filename().string() + "\n";
        }
    }
    return out;
}

// 在线 DIAG 快照：发 "DIAG\n"，读到单独一行 "." 为止（3 s 超时）
std::string live_query(const std::string& host, int port) {
#ifdef _WIN32
    using sk_t = SOCKET;
    const sk_t bad = INVALID_SOCKET;
    auto close_s = [](sk_t s) { closesocket(s); };
#else
    using sk_t = int;
    const sk_t bad = -1;
    auto close_s = [](sk_t s) { close(s); };
#endif
    sockaddr_in a{};
    if (!engine_net::address(engine_net::connect_host(host), port, &a)) return {};
    const sk_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == bad) return {};
#ifdef _WIN32
    DWORD ms = 3000;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof ms);
#else
    timeval tv{3, 0};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
    std::string out;
    if (connect(s, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0 &&
        send(s, "DIAG\n", 5, 0) == 5) {
        char buf[65536];
        while (out.size() < (8u << 20)) {
            const int n = int(recv(s, buf, sizeof buf, 0));
            if (n <= 0) break;
            out.append(buf, size_t(n));
            const size_t end = out.find("\n.\n");
            if (end != std::string::npos) {
                out.resize(end + 1);
                break;
            }
        }
    }
    close_s(s);
    return out;
}

#ifdef _WIN32
// ---- Windows 事件日志 ------------------------------------------------------
std::string level_name(unsigned l) {
    switch (l) {
        case 1: return "Critical";
        case 2: return "Error";
        case 3: return "Warning";
        case 4: return "Info";
        case 0: return "Always";
        default: return "L" + std::to_string(l);
    }
}

std::string filetime_local(ULONGLONG ft) {
    FILETIME f;
    f.dwLowDateTime = DWORD(ft & 0xFFFFFFFF);
    f.dwHighDateTime = DWORD(ft >> 32);
    SYSTEMTIME su{}, sl{};
    FileTimeToSystemTime(&f, &su);
    SystemTimeToTzSpecificLocalTime(nullptr, &su, &sl);
    char b[40];
    snprintf(b, sizeof b, "%04u-%02u-%02u %02u:%02u:%02u", sl.wYear, sl.wMonth, sl.wDay, sl.wHour,
             sl.wMinute, sl.wSecond);
    return b;
}

std::string one_line(std::string s) {
    for (auto& ch : s)
        if (ch == '\r' || ch == '\n' || ch == '\t') ch = ' ';
    std::string out;
    bool sp = false;
    for (char ch : s) {
        if (ch == ' ') {
            if (!sp && !out.empty()) out += ' ';
            sp = true;
        } else {
            out += ch;
            sp = false;
        }
    }
    if (out.size() > 1500) out = out.substr(0, 1500) + " …";
    return out;
}

std::string render_event(EVT_HANDLE ctx, EVT_HANDLE ev, std::map<std::wstring, EVT_HANDLE>& pubs) {
    DWORD used = 0, props = 0;
    EvtRender(ctx, ev, EvtRenderEventValues, 0, nullptr, &used, &props);
    std::vector<BYTE> buf(used + 16);
    if (!EvtRender(ctx, ev, EvtRenderEventValues, DWORD(buf.size()), buf.data(), &used, &props))
        return "(render failed)\n";
    const PEVT_VARIANT v = reinterpret_cast<PEVT_VARIANT>(buf.data());
    const wchar_t* prov = v[EvtSystemProviderName].Type == EvtVarTypeString
                              ? v[EvtSystemProviderName].StringVal
                              : L"?";
    const unsigned id = v[EvtSystemEventID].UInt16Val;
    const unsigned lvl = v[EvtSystemLevel].Type == EvtVarTypeByte ? v[EvtSystemLevel].ByteVal : 0;
    const ULONGLONG ft = v[EvtSystemTimeCreated].FileTimeVal;
    std::string line = filetime_local(ft) + "  " + level_name(lvl) + "  " + crashdiag::w2u(prov) +
                       "  " + std::to_string(id) + "  ";
    // 消息正文：按发布者元数据格式化；不行就退回 EventData 的 XML
    EVT_HANDLE pub = nullptr;
    auto it = pubs.find(prov);
    if (it == pubs.end()) {
        pub = EvtOpenPublisherMetadata(nullptr, prov, nullptr, 0, 0);
        pubs[prov] = pub;
    } else {
        pub = it->second;
    }
    std::string msg;
    if (pub) {
        DWORD need = 0;
        EvtFormatMessage(pub, ev, 0, 0, nullptr, EvtFormatMessageEvent, 0, nullptr, &need);
        if (need) {
            std::wstring w(need + 1, L'\0');
            if (EvtFormatMessage(pub, ev, 0, 0, nullptr, EvtFormatMessageEvent, DWORD(w.size()),
                                 &w[0], &need))
                msg = crashdiag::w2u(w.c_str());
        }
    }
    if (msg.empty()) {
        DWORD need = 0, pc = 0;
        EvtRender(nullptr, ev, EvtRenderEventXml, 0, nullptr, &need, &pc);
        std::wstring w(need / sizeof(wchar_t) + 2, L'\0');
        if (EvtRender(nullptr, ev, EvtRenderEventXml, DWORD(w.size() * sizeof(wchar_t)), &w[0],
                      &need, &pc)) {
            const std::string x = crashdiag::w2u(w.c_str());
            const size_t a = x.find("<EventData>"), b = x.find("</EventData>");
            msg = a != std::string::npos && b != std::string::npos ? x.substr(a, b + 12 - a)
                                                                   : std::string("(no message)");
        }
    }
    return line + one_line(msg) + "\n";
}

std::string query_events(const wchar_t* channel, const std::wstring& xpath, int max, DWORD* err) {
    *err = 0;
    EVT_HANDLE q = EvtQuery(nullptr, channel, xpath.c_str(),
                            EvtQueryChannelPath | EvtQueryReverseDirection);
    if (!q) {
        *err = GetLastError();
        return {};
    }
    EVT_HANDLE ctx = EvtCreateRenderContext(0, nullptr, EvtRenderContextSystem);
    std::map<std::wstring, EVT_HANDLE> pubs;
    std::string out;
    int n = 0;
    EVT_HANDLE ev[16];
    DWORD got = 0;
    while (n < max && EvtNext(q, 16, ev, 3000, 0, &got)) {
        for (DWORD i = 0; i < got; i++) {
            if (n < max) {
                out += render_event(ctx, ev[i], pubs);
                n++;
            }
            EvtClose(ev[i]);
        }
    }
    for (auto& p : pubs)
        if (p.second) EvtClose(p.second);
    if (ctx) EvtClose(ctx);
    EvtClose(q);
    if (!n) out = "  (none)\n";
    return out;
}

std::string win_events(int hours) {
    const std::wstring tw =
        L"TimeCreated[timediff(@SystemTime) <= " + std::to_wstring(int64_t(hours) * 3600000) + L"]";
    const std::wstring sys_prov =
        L"Provider[@Name='Display' or @Name='amdkmdag' or @Name='amdwddmg' or @Name='amdfendr' or "
        L"@Name='Microsoft-Windows-Kernel-Power' or @Name='Microsoft-Windows-Power-Troubleshooter' or "
        L"@Name='Microsoft-Windows-WER-SystemErrorReporting' or @Name='EventLog' or "
        L"@Name='Microsoft-Windows-Kernel-General' or @Name='Microsoft-Windows-Kernel-Boot' or "
        L"@Name='Microsoft-Windows-Resource-Exhaustion-Detector' or @Name='Microsoft-Windows-WHEA-Logger' or "
        L"@Name='volmgr' or @Name='disk' or @Name='Ntfs']";
    std::string out = "Windows event log, last " + std::to_string(hours) +
                      " h, newest first (max 400 per channel).\n"
                      "Key IDs: Display 4101 = GPU driver stopped responding and recovered (TDR); "
                      "Kernel-Power 41 = unexpected reboot, 42/107 = sleep/resume; EventLog 6008 = "
                      "unexpected shutdown; WER-SystemErrorReporting 1001 = bugcheck; "
                      "Resource-Exhaustion-Detector 2004 = low virtual memory; WHEA = hardware errors; "
                      "Application Error 1000 = a program crashed (faulting module + offset); "
                      "Windows Error Reporting 1001 with LiveKernelEvent 141/117 = GPU hang/reset.\n\n";
    DWORD err = 0;
    out += "==== System ====\n";
    std::string s = query_events(
        L"System", L"*[System[(Level=1 or Level=2 or Level=3 or " + sys_prov + L") and " + tw + L"]]",
        400, &err);
    if (err) {  // 部分系统上组合谓词不被接受：拆成两次
        s = "(combined query failed, error " + std::to_string(err) + "; split queries below)\n";
        s += "-- providers --\n" +
             query_events(L"System", L"*[System[" + sys_prov + L" and " + tw + L"]]", 300, &err);
        if (err) s += "  (query failed, error " + std::to_string(err) + ")\n";
        s += "-- errors/warnings --\n" +
             query_events(L"System", L"*[System[(Level=1 or Level=2 or Level=3) and " + tw + L"]]",
                          300, &err);
        if (err) s += "  (query failed, error " + std::to_string(err) + ")\n";
    }
    out += s;
    out += "\n==== Application ====\n";
    s = query_events(L"Application",
                     L"*[System[Provider[@Name='Application Error' or @Name='Windows Error Reporting' "
                     L"or @Name='Application Hang'] and " + tw + L"]]",
                     400, &err);
    if (err) s = "  (query failed, error " + std::to_string(err) + ")\n";
    out += s;
    return out;
}
#else
// ---- Linux 内核日志 --------------------------------------------------------
std::string run_cmd(const std::string& cmd) {
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return "(popen failed)\n";
    char b[4096];
    size_t n;
    while ((n = fread(b, 1, sizeof b, p)) > 0 && out.size() < (4u << 20)) out.append(b, n);
    pclose(p);
    return out;
}
// 崩溃报告里 "  #k  0x… /path/lib.so+0xOFF" 形式的栈帧 → addr2line 函数名与源码行
// （打包时做，不在信号处理里；没有 addr2line 或无调试信息时只得到 ??）
std::string shq(const std::string& s) {
    std::string o = "'";
    for (char ch : s) o += ch == '\'' ? std::string("'\\''") : std::string(1, ch);
    return o + "'";
}
std::string linux_symbolize(const std::string& report) {
    struct Fr {
        std::string label, path, off;
    };
    std::vector<Fr> fr;
    std::istringstream in(report);
    std::string line;
    while (std::getline(in, line)) {
        const bool frame = starts(line, "  #");
        const bool pc = starts(line, "pc: ");
        if (!frame && !pc) continue;
        const size_t a = line.find("0x");
        if (a == std::string::npos) continue;
        const size_t sp = line.find(' ', a);
        const size_t plus = line.rfind("+0x");
        if (sp == std::string::npos || plus == std::string::npos || plus <= sp) continue;
        Fr f;
        f.label = pc ? "pc" : line.substr(2, line.find(' ', 3) - 2);
        f.path = line.substr(sp + 1, plus - sp - 1);
        f.off = line.substr(plus + 1);
        if (!f.path.empty() && f.path[0] == '/') fr.push_back(f);
    }
    if (fr.empty()) return {};
    if (run_cmd("command -v addr2line 2>/dev/null").empty())
        return "(addr2line not found; install binutils to symbolize)\n";
    std::string out = "symbolized with addr2line (function / source line; ?? = no debug info)\n";
    for (const auto& f : fr) {
        std::error_code ec;
        if (!fs::exists(fs::path(f.path), ec)) {
            out += "  " + f.label + "  " + f.path + "+" + f.off + "  (file missing)\n";
            continue;
        }
        std::string r = run_cmd("addr2line -f -C -e " + shq(f.path) + " " + f.off + " 2>&1");
        for (auto& ch : r)
            if (ch == '\n') ch = ' ';
        out += "  " + f.label + "  " + fs::path(f.path).filename().string() + "+" + f.off + "  " + r + "\n";
    }
    return out;
}

std::string linux_kernel(int hours) {
    const std::string pat =
        "amdgpu|kfd|drm|gpu|ttm|iommu|oom|out of memory|killed process|segfault|general protection|"
        "mce|hardware error|watchdog|hung_task|blocked for more than|suspend|resume";
    std::string out = "kernel log, last " + std::to_string(hours) +
                      " h, lines matching: " + pat + "\n\n";
    std::string j = run_cmd("journalctl -k --no-pager -o short-iso --since '-" + std::to_string(hours) +
                            "h' 2>&1 | grep -i -E '" + pat + "' | tail -n 600");
    if (j.empty() || j.find("No journal files") != std::string::npos ||
        j.find("not seeing messages") != std::string::npos)
        j += run_cmd("dmesg -T 2>&1 | grep -i -E '" + pat + "' | tail -n 600");
    return out + (j.empty() ? "(no matching lines, or no permission to read the kernel log)\n" : j);
}
#endif

class Zip {
 public:
    explicit Zip(const std::string& path) : path_(path) {
        ok_ = mz_zip_writer_init_file(&z_, path.c_str(), 0) != 0;
    }
    ~Zip() {
        if (ok_ && !done_) mz_zip_writer_end(&z_);
    }
    bool ok() const { return ok_; }
    void add(const std::string& name, const std::string& data) {
        if (!ok_) return;
        if (!mz_zip_writer_add_mem(&z_, name.c_str(), data.data(), data.size(), MZ_DEFAULT_LEVEL))
            ok_ = false;
        else
            names_.push_back(name + " (" + std::to_string(data.size()) + " bytes)");
    }
    bool finish() {
        if (!ok_) return false;
        const bool a = mz_zip_writer_finalize_archive(&z_) != 0;
        const bool b = mz_zip_writer_end(&z_) != 0;
        done_ = true;
        return a && b;
    }
    const std::vector<std::string>& names() const { return names_; }

 private:
    std::string path_;
    mz_zip_archive z_{};
    bool ok_ = false, done_ = false;
    std::vector<std::string> names_;
};

const char* kReadme =
    "Qwenox diagnostic package / Qwenox 诊断包\n"
    "==========================================\n"
    "\n"
    "请把这个 zip 原样发给开发者。Please send this zip to the developers as is.\n"
    "\n"
    "内容 / Contents\n"
    "  summary.txt      打包原因、退出码含义、崩溃报告摘要 / reason, exit code, crash summary\n"
    "  engine.log       引擎日志（开头 + 结尾）/ engine log (head + tail)\n"
    "  api.log          API 日志（开头 + 结尾）/ API log (head + tail)\n"
    "  crash/           崩溃报告 .txt 与小 minidump .dmp / crash reports and small minidumps\n"
    "  engine-live.*    引擎仍在运行时的在线快照 / live snapshot while the engine was running\n"
    "  system.txt       系统、显卡驱动、电源、TDR 设置 / OS, GPU driver, power, TDR settings\n"
    "  events.txt       Windows 事件日志（驱动复位、睡眠唤醒、程序崩溃…）/ Windows event log\n"
    "  kernel.txt       Linux 内核日志中与 GPU / 内存相关的行 / Linux kernel log (GPU/memory)\n"
    "  service.conf     配置（密钥类的值已打码）/ configuration (secrets masked)\n"
    "  logs-index.txt   logs 目录清单 / listing of the logs folder\n"
    "\n"
    "隐私 / Privacy\n"
    "  不含请求或回复的正文、模型文件、KV 缓存。minidump 只含线程栈与模块列表，栈内存里\n"
    "  可能夹带少量正在处理的文本片段；介意的话删掉 crash/*.dmp 与 engine-live.dmp 再发，\n"
    "  文本报告（.txt）已包含主要信息。\n"
    "  No request/response bodies, model files or KV cache. Minidumps hold thread stacks and\n"
    "  the module list only; stack memory may contain small fragments of text being processed.\n"
    "  Delete crash/*.dmp and engine-live.dmp before sending if that is a concern; the .txt\n"
    "  reports carry the essential information.\n";

}  // namespace

std::string out_dir(const std::string& root) { return root + "/logs/diag"; }

bool valid_name(const std::string& n) {
    if (!starts(n, "qwenox-diag-") || !ends(n, ".zip") || n.size() > 64) return false;
    for (char ch : n)
        if (!(std::isalnum(static_cast<unsigned char>(ch)) || ch == '-' || ch == '.' || ch == '_'))
            return false;
    return true;
}

std::string exit_code_text(unsigned long c) {
    switch (c) {
        case 0: return "normal exit (0)";
        case 1: return "fatal error (1): HIP error / KV pool / startup failure - see the last lines of the engine log and crash/*.txt";
        case 2: return "usage / argument error (2)";
        case 3: return "abort() or std::terminate (3)";
        case 124: return "HIP watchdog timeout (124): GPU copy/synchronization did not return; engine stopped without automatic restart - see crash/*.txt and engine.log";
        case 0xC0000005: return "STATUS_ACCESS_VIOLATION: invalid memory access (see crash/*.txt: fault/pc lines)";
        case 0xC0000409: return "STATUS_STACK_BUFFER_OVERRUN / __fastfail: bypasses in-process handlers; see events.txt (Application Error 1000)";
        case 0xC0000374: return "STATUS_HEAP_CORRUPTION";
        case 0xC00000FD: return "STATUS_STACK_OVERFLOW";
        case 0xC000001D: return "STATUS_ILLEGAL_INSTRUCTION";
        case 0xC0000006: return "STATUS_IN_PAGE_ERROR: a mapped file / page could not be read";
        case 0xC0000017: return "STATUS_NO_MEMORY";
        case 0xC000012D: return "STATUS_COMMITMENT_LIMIT: out of virtual memory (page file)";
        case 0xC000013A: return "STATUS_CONTROL_C_EXIT: Ctrl+C / console closed";
        case 0xE06D7363: return "uncaught C++ exception (MSVC)";
        case 0x80000003: return "STATUS_BREAKPOINT";
        case 0x40010004: return "terminated by a debugger";
        case 134: return "SIGABRT (128+6)";
        case 135: return "SIGBUS (128+7): access to a mapped page with no backing (truncated file / device memory)";
        case 136: return "SIGFPE (128+8)";
        case 137: return "SIGKILL (128+9): killed (OOM killer / watchdog / kill -9)";
        case 139: return "SIGSEGV (128+11): invalid memory access";
        case 143: return "SIGTERM (128+15): asked to stop";
        default: break;
    }
    char b[64];
    snprintf(b, sizeof b, "unknown (0x%08lX)", c);
    return b;
}

std::string build(const Opts& o, std::string* err) {
    try {
        const int64_t now = now_unix();
        const std::string dir = out_dir(o.root);
        std::error_code ec;
        fs::create_directories(fs::path(dir), ec);
        const std::string name = "qwenox-diag-" + stamp(now, true) + ".zip";
        std::string path = dir + "/" + name;
        for (int k = 2; fs::exists(fs::path(path), ec) && k < 100; k++)
            path = dir + "/qwenox-diag-" + stamp(now, true) + "-" + std::to_string(k) + ".zip";
        const std::string tmp = path + ".tmp";
        const int64_t since = o.since_unix ? o.since_unix : now - 72 * 3600;
        // 事件窗口：至少 72 h，覆盖引擎整个运行期但不超过 7 天
        int hours = int((now - since) / 3600) + 1;
        hours = std::max(72, std::min(hours, 168));

        const std::string elog = o.engine_log.empty() ? newest_engine_log(o.root) : o.engine_log;
        const std::string alog = o.api_log.empty() ? newest_api_log(o.root, elog) : o.api_log;

        // 崩溃报告：本次 pid 的全部 + 时间窗内其它的最新 6 个
        std::vector<Entry> crash;
        {
            const std::string pidtag = o.engine_pid ? "-" + std::to_string(o.engine_pid) + "-" : "";
            auto all = list_files(o.root + "/logs/crash", [](const std::string& n) {
                return starts(n, "crash-") && (ends(n, ".txt") || ends(n, ".dmp"));
            });
            int others = 0;
            for (const auto& e : all) {
                const std::string n = fs::path(e.path).filename().string();
                const bool mine = !pidtag.empty() && starts(n, "crash-engine-") &&
                                  n.find(pidtag) != std::string::npos;
                if (mine || (e.mtime >= since - 60 && others++ < 6)) crash.push_back(e);
            }
        }

        Zip z(tmp);
        if (!z.ok()) {
            if (err) *err = "cannot create " + tmp;
            return {};
        }
        z.add("README.txt", kReadme);

        std::string sum;
        sum += "Qwenox diagnostic package\n";
        sum += "created: " + stamp(now, false) + "\n";
        sum += std::string("api version: ") + QWENOX_VERSION_FULL + "\n";
        sum += "reason: " + (o.reason.empty() ? std::string("manual export") : o.reason) + "\n";
        if (o.has_exit) {
            char b[32];
            snprintf(b, sizeof b, "0x%08lX (%lu)", o.exit_code, o.exit_code);
            sum += std::string("engine exit code: ") + b + " = " + exit_code_text(o.exit_code) + "\n";
        }
        if (o.engine_pid) sum += "engine pid: " + std::to_string(o.engine_pid) + "\n";
        sum += "engine log: " + (elog.empty() ? std::string("(not found)") : elog) + "\n";
        sum += "api log: " + (alog.empty() ? std::string("(not found)") : alog) + "\n";
        sum += "window: since " + stamp(since, false) + " (events: last " + std::to_string(hours) + " h)\n";
        sum += "\ncrash reports:\n";
        if (crash.empty()) sum += "  (none in logs/crash for this window)\n";
        for (const auto& e : crash) {
            const std::string n = fs::path(e.path).filename().string();
            sum += "  " + n + "  (" + std::to_string(file_size(e.path)) + " bytes, " +
                   stamp(e.mtime, false) + ")\n";
            if (ends(n, ".txt")) sum += crash_excerpt(e.path);
        }

        if (!elog.empty()) z.add("engine.log", head_tail(elog, kEngineHead, kEngineTail));
        if (!alog.empty() && alog != elog) z.add("api.log", head_tail(alog, kApiHead, kApiTail));
        for (const auto& e : crash) {
            if (file_size(e.path) > kMaxDump) continue;
            std::string data;
            if (!read_all(e.path, &data)) continue;
            const std::string n = fs::path(e.path).filename().string();
            z.add("crash/" + n, data);
#ifndef _WIN32
            if (ends(n, ".txt")) {
                const std::string sym = linux_symbolize(data);
                if (!sym.empty()) z.add("crash/" + n.substr(0, n.size() - 4) + ".sym.txt", sym);
            }
#endif
        }
        if (!o.engine_host.empty() && o.engine_port > 0) {
            const std::string live = live_query(o.engine_host, o.engine_port);
            sum += std::string("\nengine live snapshot: ") +
                   (live.empty() ? "unavailable (engine not answering)" : "engine-live.txt") + "\n";
            if (!live.empty()) z.add("engine-live.txt", live);
        }
        if (!o.live_dump.empty()) {
            std::string data;
            if (file_size(o.live_dump) <= kMaxDump && read_all(o.live_dump, &data) && !data.empty()) {
                z.add("engine-live.dmp", data);
                sum += "engine live minidump: engine-live.dmp\n";
            }
            fs::remove(fs::path(o.live_dump), ec);
        }
        z.add("system.txt", std::string("api process view\n") + crashdiag::env_report());
#ifdef _WIN32
        z.add("events.txt", win_events(hours));
#else
        z.add("kernel.txt", linux_kernel(hours));
#endif
        {
            const std::string cp = o.conf_path.empty() ? o.root + "/service.conf" : o.conf_path;
            std::string text;
            if (read_all(cp, &text) && !text.empty()) z.add("service.conf", mask_conf(text));
        }
        z.add("logs-index.txt", logs_index(o.root));
        sum += "\nfiles:\n";
        for (const auto& n : z.names()) sum += "  " + n + "\n";
        z.add("summary.txt", sum);
        if (!z.finish()) {
            fs::remove(fs::path(tmp), ec);
            if (err) *err = "zip write failed: " + tmp;
            return {};
        }
        fs::rename(fs::path(tmp), fs::path(path), ec);
        if (ec) {
            if (err) *err = "rename failed: " + ec.message();
            return {};
        }
        {  // 绝对路径（托盘在资源管理器里选中它、HTTP 返回给调用方）
            std::error_code ae;
            const fs::path ab = fs::absolute(fs::path(path), ae).lexically_normal();
            if (!ae) path = fs::path(ab).make_preferred().string();
        }
        // 只留最近 kKeepPacks 个
        auto packs = list_files(dir, [](const std::string& n) { return valid_name(n); });
        for (size_t k = kKeepPacks; k < packs.size(); k++) fs::remove(fs::path(packs[k].path), ec);
        return path;
    } catch (const std::exception& e) {
        if (err) *err = std::string("diagpack: ") + e.what();
        return {};
    }
}

}  // namespace diagpack
