// launcher_settings.cpp — launcher_settings：API/引擎之前的网页启动器。
// 双击 start_win.exe（Windows）或运行 ./launcher_settings（Linux）后占用一个端口
// （默认 127.0.0.1:8729，service.conf 的 LAUNCHER_HOST/LAUNCHER_PORT 可改）提供
// 配置页：编辑 service.conf（权重/参数，与 settings_win.exe、API 网页控制台共用
// svcconf 读写层与校验），点启动后才拉起 API + 引擎。
// Windows 上是托盘程序（结构同 qwenox-win 的 tray_win.cpp：主线程托盘消息循环 +
// worker 线程跑 HTTP）：双击托盘图标打开配置页，右键菜单启停/看日志/退出，
// 状态切换弹气泡（service.conf 的 "# start_win: notify=" 可关）。
//
// 拉起语义（与平台现有架构一致，不复制启动逻辑）：
//   Windows：spawn build\qwenox-win.exe（托盘 API），等 API 端口就绪后经
//             POST /admin/engine/start 让它自己拉引擎；停止 = POST stop +
//             结束本进程拉起的 API（其 Job 对象会带走引擎）。已在外部运行的
//             API/引擎会被识别为 external：引擎可停（若非外部启动），API 不动。
//   Linux  ：fork+setsid 跑 bash start_hgn.sh / start_gguf.sh（按
//             "# start_win: weights=" 标记选择），脚本的 EXIT trap 负责清理；
//             停止 = 给脚本进程组发 SIGTERM。launcher 重启后凭
//             logs/.launcher.pid 重新接管仍在运行的服务。
//
// 事件日志：logs/launcher-settings.log（追加）；网页 /api/logs 可看引擎/API 日志尾。
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include "gpu/os_win32.h"  // winsock2 → windows.h 顺序 + wsa_init()
#include <shellapi.h>
#include <shobjidl.h>  // SetCurrentProcessExplicitAppUserModelID
#ifndef NIN_BALLOONCLICK
#define NIN_BALLOONCLICK (WM_USER + 5)  // shellapi.h 在某些 NTDDI 目标下不定义
#endif

#include "launch_win_icon.inc"  // kIconIco：托盘图标（tools/win_icon.py 生成）
#include "win_dpi.h"            // PerMonitorV2
#else
#include <arpa/inet.h>
#include <dirent.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <nlohmann/json.hpp>

#include "api/http.h"
#include "engine_net.h"
#include "service_conf.h"
#include "version.h"
#include "launcher_static_gen.inc"  // 生成物：build 脚本由 src/launcher_static/ 生成

using json = nlohmann::ordered_json;

namespace {

// ---- 小工具 -----------------------------------------------------------------

std::string g_root;       // 项目根（exe 所在目录；启动时 chdir 到这里）
std::string g_conf_path;  // <root>/service.conf

std::string exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    std::string p(buf, n ? static_cast<size_t>(n) : 0);
#else
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    std::string p;
    if (n > 0) p.assign(buf, static_cast<size_t>(n));
#endif
    const size_t slash = p.find_last_of("\\/");
    return slash == std::string::npos ? "." : p.substr(0, slash);
}

bool file_exists(const std::string& path) {
#ifdef _WIN32
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
#endif
}

bool dir_exists(const std::string& path) {
#ifdef _WIN32
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

void ensure_dir(const std::string& path) {
#ifdef _WIN32
    CreateDirectoryA(path.c_str(), nullptr);
#else
    mkdir(path.c_str(), 0755);
#endif
}

std::string now_stamp() {
    time_t t = time(nullptr);
    struct tm tmv;
#ifdef _WIN32
    localtime_s(&tmv, &t);
#else
    localtime_r(&t, &tmv);
#endif
    char buf[32];
    snprintf(buf, sizeof(buf), "%04d%02d%02d-%02d%02d%02d", tmv.tm_year + 1900,
             tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour, tmv.tm_min, tmv.tm_sec);
    return buf;
}

// ---- 事件日志（内存环形 + 落盘） ---------------------------------------------

std::mutex g_log_mtx;
std::deque<std::string> g_events;  // 最近 500 行
std::string g_log_path;            // logs/launcher-settings.log

void log_event(const std::string& msg) {
    const std::string line = "[" + now_stamp() + "] " + msg;
    std::lock_guard<std::mutex> lk(g_log_mtx);
    g_events.push_back(line);
    while (g_events.size() > 500) g_events.pop_front();
    if (!g_log_path.empty()) {
        if (FILE* f = fopen(g_log_path.c_str(), "ab")) {
            fwrite(line.data(), 1, line.size(), f);
            fputc('\n', f);
            fclose(f);
        }
    }
    fprintf(stderr, "%s\n", line.c_str());
}

// ---- 受管键与默认值 -----------------------------------------------------------
// svcconf::kManagedKeys 之外的补充键（API /admin/config 不收的启动参数也在这里）。
const char* kExtraKeys[] = {
    "MODEL_DIR",       "MTP_GAMMA",     "RCKPT_MAX",     "MAX_IMAGES",
    "VIZ",             "REQLOG",        "REQLOG_DIR",    "REQLOG_LEVEL",
    "CONC_PREFILL_CHUNK", "PLE_URING",  "MEMORY_CAP_GB", "MIN_AVAILABLE_GB",
    "START_TIMEOUT",   "STALL_TIMEOUT",
};

bool is_launcher_key(const std::string& key) {
    if (svcconf::is_managed_key(key)) return true;
    for (const char* k : kExtraKeys)
        if (key == k) return true;
    return false;
}

// UI 占位默认（与 service.conf 出厂值一致；键不在 conf 里时按此显示/生效）。
const std::pair<const char*, const char*> kDefaults[] = {
    {"MODEL_DIR", "./models"},
    {"TOKENIZER_DIR", "./models/tokenizer"},
    {"MODEL_FILE", "./models/qwen38-flash-next-v2.hgn"},
    {"NGRAM_FILE", "./models/qwen38-flash-next-ngram.hgn"},
    {"OVERLAY_FILE", ""},
    {"MTP_FILE", "./models/qwen38-flash-next-mtp.hgn"},
    {"VISION_FILE", "./models/qwen38-flash-next-vision.hgn"},
    {"GGUF_FILE", "./models/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"},
    {"GGUF_MTP_FILE", "./models/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf"},
    {"GGUF_VISION_FILE", "./models/mmproj-BF16.gguf"},
    {"ENGINE_HOST", "127.0.0.1"}, {"ENGINE_PORT", "8730"},
    {"API_HOST", "0.0.0.0"}, {"API_PORT", "8731"},
    {"MAX_CONTEXT", "262144"}, {"PARALLEL", "4"},
    {"KV_PAGED", "1"}, {"KV_POOL_TOKENS", "0"},
    {"PREFILL_CHUNK", "0"}, {"CONC_PREFILL_CHUNK", "8192"},
    {"ROPE_FACTOR", "1"}, {"QSA_INDEXER_BUDGET", "0"},
    {"KVSNAP_MAX_GB", "20"}, {"KVSNAP_DIR", ""}, {"RCKPT_MAX", "8"},
    {"MTP_GAMMA", "0"}, {"MOE_A4", "0"}, {"MAX_IMAGES", "32"},
    {"VIZ", "0"}, {"AFFINITY", ""}, {"PLE_URING", "1"},
    {"MEMORY_CAP_GB", "86"}, {"MIN_AVAILABLE_GB", "100"},
    {"START_TIMEOUT", "360"}, {"STALL_TIMEOUT", "60"},
    {"REQLOG", "0"}, {"REQLOG_DIR", "logs/reqlog"}, {"REQLOG_LEVEL", "6"},
};

std::string default_of(const std::string& key) {
    for (const auto& kv : kDefaults)
        if (key == kv.first) return kv.second;
    return "";
}

// ---- service.conf 存取 --------------------------------------------------------

const char* kWeightsMarker = "# start_win: weights=";
const char* kLangMarker = "# start_win: lang=";

std::mutex g_conf_mtx;
svcconf::Conf g_conf;

void reload_conf() {
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    g_conf.clear();
    svcconf::load_conf(g_conf_path, &g_conf);
}

// 有效值（env > conf > 默认）。调用方须持 g_conf_mtx。
std::string resolved(const svcconf::Conf& conf, const std::string& key) {
    return svcconf::cfg_optional(conf, key.c_str(), default_of(key));
}

std::string weights_ver() {
    const std::string marker = svcconf::conf_marker(g_conf_path, kWeightsMarker);
    if (marker == "v1" || marker == "v2" || marker == "gguf") return marker;
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    const std::string model = svcconf::cfg(g_conf, "MODEL_FILE", "");
    const std::string ngram = svcconf::cfg(g_conf, "NGRAM_FILE", "");
    return (ngram.empty() || ngram == model) ? "v1" : "v2";
}

std::string current_lang() {
    const std::string marker = svcconf::conf_marker(g_conf_path, kLangMarker);
    if (marker == "en" || marker == "zh") return marker;
    if (const char* e = getenv("UI_LANG")) return e[0] == 'e' ? "en" : "zh";
    return "zh";
}

// 键是否被环境变量覆盖（此时 conf 改动不生效）。
bool env_overridden(const std::string& key) { return getenv(key.c_str()) != nullptr; }

// ---- TCP/HTTP 探测 ------------------------------------------------------------

sock_t sock_open() {
#ifdef _WIN32
    return socket(AF_INET, SOCK_STREAM, 0);
#else
    return socket(AF_INET, SOCK_STREAM, 0);
#endif
}

void sock_close(sock_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

// 非阻塞 connect + select 超时；成功返回 true。
bool tcp_connect(const std::string& host, int port, int timeout_ms) {
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) return false;
    sock_t s = sock_open();
#ifdef _WIN32
    if (s == INVALID_SOCKET) return false;
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!ok && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    if (s < 0) return false;
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!ok && errno == EINPROGRESS) {
#endif
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(s, &wfds);
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (select(static_cast<int>(s) + 1, nullptr, &wfds, nullptr, &tv) > 0) {
            int soerr = 0;
            socklen_t sl = sizeof(soerr);
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &sl);
            ok = soerr == 0;
        }
    }
    sock_close(s);
    return ok;
}

// 引擎行协议 SNAP 探测：connect → 发 "SNAP\n" → 读一行。空串 = 失败/对端不是引擎。
// SNAP 应答格式：P ts_ms state req pp_toks done total tg_inst tg_avg accept_pct
std::string snap_probe(const std::string& host, int port, int timeout_ms) {
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) return "";
    sock_t s = sock_open();
#ifdef _WIN32
    if (s == INVALID_SOCKET) return "";
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!ok && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    if (s < 0) return "";
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!ok && errno == EINPROGRESS) {
#endif
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(s, &wfds);
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (select(static_cast<int>(s) + 1, nullptr, &wfds, nullptr, &tv) > 0) {
            int soerr = 0;
            socklen_t sl = sizeof(soerr);
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &sl);
            ok = soerr == 0;
        }
    }
    if (!ok) {
        sock_close(s);
        return "";
    }
    // 收超时
#ifdef _WIN32
    u_long zero = 0;
    ioctlsocket(s, FIONBIO, &zero);
    DWORD rcv = static_cast<DWORD>(timeout_ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&rcv),
               sizeof(rcv));
    if (send(s, "SNAP\n", 5, 0) != 5) {
        sock_close(s);
        return "";
    }
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) & ~O_NONBLOCK);
    timeval rcv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
    if (send(s, "SNAP\n", 5, MSG_NOSIGNAL) != 5) {
        sock_close(s);
        return "";
    }
#endif
    std::string line;
    char buf[256];
    while (line.size() < 1024) {
#ifdef _WIN32
        const int n = recv(s, buf, sizeof(buf), 0);
#else
        const ssize_t n = recv(s, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        line.append(buf, static_cast<size_t>(n));
        const size_t nl = line.find('\n');
        if (nl != std::string::npos) {
            line.resize(nl);
            break;
        }
    }
    sock_close(s);
    if (line.compare(0, 2, "P ") != 0) return "";
    return line;
}

// 试探绑定：能 bind 说明端口空闲（不 listen，直接关）。
bool port_free(int port) {
    sock_t s = sock_open();
#ifdef _WIN32
    if (s == INVALID_SOCKET) return true;
#else
    if (s < 0) return true;
#endif
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<unsigned short>(port));
    const bool ok = bind(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    sock_close(s);
    return ok;
}

struct HttpReply {
    int status = 0;
    std::string body;
    std::string err;  // 连接/读取失败原因（status==0 时有效）
};

// 极简阻塞 HTTP 客户端：只用于本机 API 的 admin 接口（Connection: close，读到 EOF）。
// 仅 Windows 用（Linux 走脚本进程 + 端口探测，不经 API admin 接口）。
#ifdef _WIN32
HttpReply http_call(const std::string& method, const std::string& host, int port,
                    const std::string& path, const std::string& body,
                    int timeout_ms) {
    HttpReply r;
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) {
        r.err = "bad host " + host;
        return r;
    }
    sock_t s = sock_open();
#ifdef _WIN32
    if (s == INVALID_SOCKET) {
        r.err = "socket failed";
        return r;
    }
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    bool conn = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!conn && WSAGetLastError() == WSAEWOULDBLOCK) {
#else
    if (s < 0) {
        r.err = "socket failed";
        return r;
    }
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
    bool conn = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    if (!conn && errno == EINPROGRESS) {
#endif
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(s, &wfds);
        timeval tv{timeout_ms / 1000, (timeout_ms % 1000) * 1000};
        if (select(static_cast<int>(s) + 1, nullptr, &wfds, nullptr, &tv) > 0) {
            int soerr = 0;
            socklen_t sl = sizeof(soerr);
            getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &sl);
            conn = soerr == 0;
        }
    }
    if (!conn) {
        r.err = "connect failed";
        sock_close(s);
        return r;
    }
    // 收发改回阻塞 + 超时
#ifdef _WIN32
    u_long zero = 0;
    ioctlsocket(s, FIONBIO, &zero);
    DWORD rcv = timeout_ms > 5000 ? 5000 : static_cast<DWORD>(timeout_ms);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&rcv), sizeof(rcv));
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) & ~O_NONBLOCK);
    timeval rcv{timeout_ms > 5000 ? 5 : timeout_ms / 1000,
                timeout_ms > 5000 ? 0 : (timeout_ms % 1000) * 1000};
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &rcv, sizeof(rcv));
#endif
    std::string req = method + " " + path + " HTTP/1.1\r\nHost: " + host + "\r\n"
                      "Connection: close\r\nContent-Type: application/json\r\n";
    if (const char* k = getenv("QWENOX_API_ADMIN_KEY"))
        if (*k) req += std::string("Authorization: Bearer ") + k + "\r\n";
    req += "Content-Length: " + std::to_string(body.size()) + "\r\n\r\n" + body;
    size_t sent = 0;
    while (sent < req.size()) {
#ifdef _WIN32
        const int n = send(s, req.data() + sent, static_cast<int>(req.size() - sent), 0);
#else
        const ssize_t n = send(s, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
#endif
        if (n <= 0) {
            r.err = "send failed";
            sock_close(s);
            return r;
        }
        sent += static_cast<size_t>(n);
    }
    std::string raw;
    char buf[16384];
    size_t header_end = std::string::npos;
    size_t content_length = std::string::npos;
    for (;;) {
#ifdef _WIN32
        const int n = recv(s, buf, sizeof(buf), 0);
#else
        const ssize_t n = recv(s, buf, sizeof(buf), 0);
#endif
        if (n > 0) {
            raw.append(buf, static_cast<size_t>(n));
            if (header_end == std::string::npos) {
                header_end = raw.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    const std::string head = raw.substr(0, header_end);
                    const size_t sp = head.find(' ');
                    if (sp != std::string::npos)
                        r.status = atoi(head.c_str() + sp + 1);
                    const std::string needle = "\r\nContent-Length: ";
                    size_t p = head.find(needle);
                    if (p == std::string::npos) {
                        const std::string needle2 = "\r\ncontent-length: ";
                        p = head.find(needle2);
                        if (p != std::string::npos)
                            content_length = static_cast<size_t>(
                                atoll(head.c_str() + p + needle2.size()));
                    } else {
                        content_length = static_cast<size_t>(
                            atoll(head.c_str() + p + needle.size()));
                    }
                    header_end += 4;
                }
            }
            if (header_end != std::string::npos && content_length != std::string::npos &&
                raw.size() >= header_end + content_length)
                break;  // 收满
            continue;
        }
        break;  // EOF 或超时：Connection: close 下 EOF 即完整；超时则尽力而为
    }
    sock_close(s);
    if (raw.empty()) {
        r.err = "no response";
        return r;
    }
    if (header_end == std::string::npos) {
        r.err = "bad response";
        return r;
    }
    r.body = raw.substr(header_end);
    return r;
}
#endif  // _WIN32

// ---- 服务状态机 -----------------------------------------------------------------
// state: idle | spawning | waiting_engine | waiting_api | running | stopping |
//        exited | failed
struct Svc {
    std::string state = "idle";
    std::string detail;
    int64_t since = 0;
    // 引擎
    std::string eng_state = "stopped";  // stopped | starting | running | exited
    int eng_pid = 0;
    unsigned long eng_exit = 0;
    bool eng_external = false;  // 非本次服务链路拉起（如 start_win.sh）
    std::string eng_log, eng_detail, eng_diag;
    std::string snap;  // 引擎 SNAP 性能行（monitor 线程刷新，tooltip/悬浮窗用）
    // API
    bool api_up = false;
    bool api_external = false;  // 不是本进程拉起的（launcher 重启后接管也算）
    int api_pid = 0;
#ifdef _WIN32
    HANDLE api_proc = nullptr;
#else
    pid_t child_pid = -1;  // start_*.sh 脚本进程（进程组组长）
#endif
    std::string run_log;  // Linux：脚本日志路径
};
std::mutex g_mtx;
Svc g_svc;
uint64_t g_gen = 0;  // 每次状态机复位自增：让旧监视线程退出

std::string api_connect_host() {
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    return engine_net::connect_host(svcconf::cfg(g_conf, "API_HOST", "0.0.0.0"));
}

int api_port() {
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    long p = 8731;
    svcconf::parse_int(svcconf::cfg(g_conf, "API_PORT", "8731"), &p);
    return static_cast<int>(p);
}

std::string engine_connect_host() {
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    return engine_net::connect_host(svcconf::cfg(g_conf, "ENGINE_HOST", "127.0.0.1"));
}

int engine_port() {
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    long p = 8730;
    svcconf::parse_int(svcconf::cfg(g_conf, "ENGINE_PORT", "8730"), &p);
    return static_cast<int>(p);
}

void set_state_locked(const std::string& s, const std::string& detail) {
    g_svc.state = s;
    g_svc.detail = detail;
    g_svc.since = time(nullptr);
}

void set_state(const std::string& s, const std::string& detail) {
    std::lock_guard<std::mutex> lk(g_mtx);
    set_state_locked(s, detail);
}

#ifdef _WIN32
// 从 API 拉引擎状态并合并进 g_svc；false = API 不可达。
bool poll_engine_from_api() {
    HttpReply r = http_call("GET", api_connect_host(), api_port(),
                            "/admin/engine/status", "", 1500);
    if (r.status != 200) return false;
    json j = json::parse(r.body, nullptr, false);
    if (j.is_discarded()) return false;
    std::lock_guard<std::mutex> lk(g_mtx);
    g_svc.api_up = true;
    g_svc.eng_state = j.value("state", "stopped");
    g_svc.eng_pid = j.value("pid", 0);
    g_svc.eng_exit = j.value("exit_code", 0ul);
    g_svc.eng_external = j.value("external", false);
    g_svc.eng_log = j.value("log", "");
    g_svc.eng_detail = j.value("detail", "");
    g_svc.eng_diag = j.value("diag", "");
    return true;
}
#endif  // _WIN32

#ifdef _WIN32

// 监视线程：spawning →（API 就绪→POST start）→ waiting_engine →（引擎 running）→
// running；进程退出/状态倒退时收尾。state==stopping 时退出（停止流程自己轮询）。
void monitor_windows(uint64_t gen, int64_t spawn_deadline) {
    bool start_posted = false;
    int api_fail = 0;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (gen != g_gen || g_svc.state == "idle") return;
            if (g_svc.state == "stopping") return;  // do_stop 接管
        }
        // 本进程拉起的 API 退出？
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_svc.api_proc &&
                WaitForSingleObject(g_svc.api_proc, 0) == WAIT_OBJECT_0) {
                DWORD code = 0;
                GetExitCodeProcess(g_svc.api_proc, &code);
                CloseHandle(g_svc.api_proc);
                g_svc.api_proc = nullptr;
                g_svc.api_pid = 0;
                g_svc.api_up = false;
                g_svc.eng_state = "stopped";
                set_state_locked("exited", "API 进程退出（exit code " +
                                               std::to_string(code) + "）");
                log_event(g_svc.detail);
                return;
            }
        }
        if (poll_engine_from_api()) {
            api_fail = 0;
            std::string st, ed, diag;
            {
                std::lock_guard<std::mutex> lk(g_mtx);
                st = g_svc.eng_state;
                ed = g_svc.eng_detail;
                diag = g_svc.eng_diag;
            }
            std::unique_lock<std::mutex> lk(g_mtx);
            if (g_svc.state == "spawning" && !start_posted) {
                start_posted = true;
                // API 已就绪：让它按当前 conf 拉引擎
                lk.unlock();
                HttpReply sr = http_call("POST", api_connect_host(), api_port(),
                                         "/admin/engine/start", "{}", 10000);
                lk.lock();
                if (gen != g_gen) return;
                if (sr.status == 200 || sr.status == 409) {
                    set_state_locked("waiting_engine", "");
                    log_event("已请求 API 启动引擎");
                } else {
                    const std::string msg = sr.status
                                                ? "POST /admin/engine/start → " + sr.body
                                                : "POST /admin/engine/start: " + sr.err;
                    set_state_locked("failed", msg);
                    log_event("引擎启动请求失败：" + msg);
                    return;
                }
            } else if (g_svc.state == "waiting_engine") {
                if (st == "running") {
                    set_state_locked("running", "");
                    log_event("引擎就绪");
                } else if (st == "exited") {
                    set_state_locked("failed", "引擎退出：" + ed +
                                                   (diag.empty() ? "" : "；诊断包 " + diag));
                    log_event(g_svc.detail);
                    return;
                }
            } else if (g_svc.state == "running") {
                if (st == "exited") {
                    set_state_locked("exited", "引擎退出（exit code " +
                                                   std::to_string(g_svc.eng_exit) + "）" +
                                                   (diag.empty() ? "" : "；诊断包 " + diag));
                    log_event(g_svc.detail);
                    return;
                }
                if (st == "stopped") {
                    // 被控制台/托盘停掉：回到可再启动状态（API 仍在）
                    set_state_locked("idle", "");
                    log_event("引擎已停止");
                    return;
                }
            }
        } else {
            ++api_fail;
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_svc.state == "spawning" && time(nullptr) > spawn_deadline) {
                set_state_locked("failed", "API 启动超时（90 秒未响应）");
                log_event(g_svc.detail);
                return;
            }
        }
        // 运行中：直连引擎取 SNAP 性能行（托盘 tooltip / 悬浮窗用；引擎侧无日志噪音）
        {
            std::unique_lock<std::mutex> lk(g_mtx);
            if (g_svc.eng_state == "running") {
                lk.unlock();
                std::string snap = snap_probe(engine_connect_host(), engine_port(), 400);
                lk.lock();
                if (gen != g_gen) return;
                g_svc.snap = std::move(snap);
            }
        }
    }
}

// 启动（工作线程）：预检 →（必要时）spawn qwenox-win → 监视线程接力。
void do_start_windows(uint64_t gen) {
    svcconf::Conf conf;
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        g_conf.clear();
        svcconf::load_conf(g_conf_path, &g_conf);
        conf = g_conf;
    }
    std::string err;
    if (!svcconf::validate_conf(conf, &err)) {
        set_state("failed", "配置无效：" + err);
        log_event("启动前校验失败：" + err);
        return;
    }
    // 预检：编译产物与权重文件
    const std::string wv = weights_ver();
    const std::string api_exe = g_root + "\\build\\qwenox-win.exe";
    const std::string eng_exe = g_root + "\\build\\qwenox-engine-win.exe";
    if (!file_exists(api_exe)) {
        set_state("failed", "缺少 build\\qwenox-win.exe（先运行 build_win.bat）");
        return;
    }
    if (!file_exists(eng_exe)) {
        set_state("failed", "缺少 build\\qwenox-engine-win.exe（先运行 build_win.bat）");
        return;
    }
    if (!dir_exists(g_root + "\\build\\rocblas\\library") ||
        !dir_exists(g_root + "\\build\\hipblaslt\\library")) {
        set_state("failed", "缺少 rocBLAS/hipBLASLt kernel db（build\\*\\library）");
        return;
    }
    auto need_file = [&](const char* key, bool optional_overlay) -> bool {
        const std::string p = resolved(conf, key);
        if (p.empty()) return true;  // 显式置空 = 禁用
        if (file_exists(p)) return true;
        if (optional_overlay) {  // overlay 缺失只警告跳过（与启动器语义一致）
            log_event(std::string("警告：找不到 overlay：") + p + "，跳过");
            return true;
        }
        err = std::string(key) + " 不存在：" + p;
        return false;
    };
    bool files_ok = true;
    if (wv == "gguf") {
        files_ok = need_file("GGUF_FILE", false) && need_file("GGUF_MTP_FILE", false) &&
                   need_file("GGUF_VISION_FILE", false);
    } else {
        files_ok = need_file("MODEL_FILE", false) &&
                   (wv == "v1" || need_file("NGRAM_FILE", false)) &&
                   need_file("OVERLAY_FILE", true) && need_file("MTP_FILE", false) &&
                   need_file("VISION_FILE", false);
    }
    // GGUF 权重自带 tokenizer，不查词表目录（API 仍按 conf 的 TOKENIZER_DIR 启动，
    // 缺词表时推理端点 503 是既有行为）
    if (files_ok && wv != "gguf" &&
        !file_exists(resolved(conf, "TOKENIZER_DIR") + "\\tokenizer.json")) {
        err = "tokenizer.json 不存在：" + resolved(conf, "TOKENIZER_DIR");
        files_ok = false;
    }
    if (!files_ok) {
        set_state("failed", "预检失败：" + err);
        log_event(g_svc.detail);
        return;
    }

    // API 已在运行（外部启动/上次残留）？直接复用，不再 spawn。
    if (poll_engine_from_api()) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_svc.api_external = true;
        log_event("检测到已在运行的 API（外部启动），直接接管控制");
    } else {
        long ep = 8731;
        svcconf::parse_int(svcconf::cfg(conf, "API_PORT", "8731"), &ep);
        if (!port_free(static_cast<int>(ep))) {
            set_state("failed", "API 端口 " + std::to_string(ep) +
                                    " 被占用且不是 Qwenox API");
            return;
        }
        STARTUPINFOA si{};
        si.cb = sizeof(si);
        PROCESS_INFORMATION pi{};
        // --no-tray：API 由本 launcher 托管时不带自己的托盘图标（托盘只有 launcher 一个）
        std::string cmd = "\"" + api_exe + "\" --no-tray";
        std::vector<char> cmdline(cmd.begin(), cmd.end());
        cmdline.push_back('\0');
        if (!CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, FALSE, 0, nullptr,
                            g_root.c_str(), &si, &pi)) {
            set_state("failed", "启动 qwenox-win.exe 失败，错误 " +
                                    std::to_string(GetLastError()));
            return;
        }
        CloseHandle(pi.hThread);
        std::lock_guard<std::mutex> lk(g_mtx);
        g_svc.api_proc = pi.hProcess;
        g_svc.api_pid = static_cast<int>(pi.dwProcessId);
        g_svc.api_external = false;
        log_event("已启动 API 进程 qwenox-win.exe（pid " + std::to_string(pi.dwProcessId) +
                  "），等待就绪");
    }
    const int64_t deadline = time(nullptr) + 90;
    std::thread(monitor_windows, gen, deadline).detach();
}

void do_stop_windows() {
    // 引擎外部启动（start_win.sh 等）：API 也管不了，拒绝
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (g_svc.eng_external && g_svc.eng_state == "running") {
            set_state_locked("running", "");
            return;  // 路由层已挡，双保险
        }
    }
    bool ours = false;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        ours = g_svc.api_proc != nullptr;
    }
    // 先停引擎（API 可达时）
    if (poll_engine_from_api()) {
        http_call("POST", api_connect_host(), api_port(), "/admin/engine/stop", "{}",
                  10000);
        const int64_t deadline = time(nullptr) + 20;
        while (time(nullptr) < deadline) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            if (!poll_engine_from_api()) break;
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_svc.eng_state == "stopped" || g_svc.eng_state == "exited") break;
        }
    }
    // 我们拉起的 API 一并结束（其 Job 对象确保引擎不残留）
    if (ours) {
        HANDLE h = nullptr;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            h = g_svc.api_proc;
        }
        if (h) {
            TerminateProcess(h, 0);
            WaitForSingleObject(h, 5000);
            CloseHandle(h);
        }
        std::lock_guard<std::mutex> lk(g_mtx);
        g_svc.api_proc = nullptr;
        g_svc.api_pid = 0;
        g_svc.api_up = false;
    }
    std::lock_guard<std::mutex> lk(g_mtx);
    g_svc.eng_state = "stopped";
    set_state_locked("idle", "");
    log_event("已停止");
}

#else  // ---- Linux ------------------------------------------------------------

std::string pidfile_path() { return "logs/.launcher.pid"; }

void write_pidfile(pid_t pid, const std::string& log) {
    if (FILE* f = fopen(pidfile_path().c_str(), "wb")) {
        fprintf(f, "%d %s\n", static_cast<int>(pid), log.c_str());
        fclose(f);
    }
}

void clear_pidfile() { remove(pidfile_path().c_str()); }

// 子进程 cmdline 是否我们的启动脚本（pidfile 接管校验用）。
bool cmdline_is_launch_script(pid_t pid) {
    const std::string p = "/proc/" + std::to_string(pid) + "/cmdline";
    FILE* f = fopen(p.c_str(), "rb");
    if (!f) return false;
    char buf[512];
    const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    const std::string s(buf, n);
    return s.find("start_hgn.sh") != std::string::npos ||
           s.find("start_gguf.sh") != std::string::npos;
}

void monitor_linux(uint64_t gen) {
    for (;;) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        pid_t pid;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (gen != g_gen || g_svc.state == "idle") return;
            if (g_svc.state == "stopping") return;  // do_stop 接管
            pid = g_svc.child_pid;
        }
        if (pid > 0) {
            int wst = 0;
            const pid_t r = waitpid(pid, &wst, WNOHANG);
            if (r == pid) {
                const int code = WIFEXITED(wst) ? WEXITSTATUS(wst) : 128 + WTERMSIG(wst);
                clear_pidfile();
                std::lock_guard<std::mutex> lk(g_mtx);
                g_svc.child_pid = -1;
                g_svc.api_up = false;
                g_svc.eng_state = "stopped";
                if (g_svc.state == "running" || g_svc.state == "waiting_api") {
                    set_state_locked("exited", "服务脚本退出（exit code " +
                                                   std::to_string(code) + "）");
                } else {
                    set_state_locked("failed", "服务启动失败（脚本退出码 " +
                                                   std::to_string(code) +
                                                   "），查看日志");
                }
                log_event(g_svc.detail);
                return;
            }
        }
        // 引擎探活用 SNAP 安静动词（引擎日志不留 connected/disconnected；裸 TCP
        // 连接会被记成 "client connected and closed" 噪音，一秒一条）。
        const std::string eng_snap =
            snap_probe(engine_connect_host(), engine_port(), 500);
        const bool eng_up = !eng_snap.empty();
        const bool api_up = tcp_connect(api_connect_host(), api_port(), 500);
        std::unique_lock<std::mutex> lk(g_mtx);
        g_svc.api_up = api_up;
        if (eng_up) g_svc.snap = eng_snap;  // 顺带喂给 tooltip / 悬浮窗
        if (g_svc.state == "spawning" || g_svc.state == "waiting_engine") {
            g_svc.eng_state = eng_up ? "running" : "starting";
            if (eng_up) set_state_locked("waiting_api", "引擎已就绪，等待 API");
        } else if (g_svc.state == "waiting_api") {
            g_svc.eng_state = eng_up ? "running" : "starting";
            if (api_up) {
                set_state_locked("running", "");
                log_event("服务已就绪");
            }
        } else if (g_svc.state == "running") {
            g_svc.eng_state = eng_up ? "running" : "starting";
            if (!api_up) set_state_locked("running", "API 端口无响应（等待恢复）");
        }
    }
}

void do_start_linux(uint64_t gen) {
    svcconf::Conf conf;
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        g_conf.clear();
        svcconf::load_conf(g_conf_path, &g_conf);
        conf = g_conf;
    }
    std::string err;
    if (!svcconf::validate_conf(conf, &err)) {
        set_state("failed", "配置无效：" + err);
        log_event("启动前校验失败：" + err);
        return;
    }
    const std::string wv = weights_ver();
    const std::string script = wv == "gguf" ? "start_gguf.sh" : "start_hgn.sh";
    if (!file_exists("build/qwenox-engine") || !file_exists("build/qwenox-api")) {
        set_state("failed", "缺少编译产物 build/qwenox-engine / build/qwenox-api"
                            "（先运行 bash build.sh）");
        return;
    }
    if (!file_exists(script)) {
        set_state("failed", "缺少启动脚本 " + script);
        return;
    }
    auto need_file = [&](const char* key, bool optional_overlay) -> bool {
        const std::string p = resolved(conf, key);
        if (p.empty()) return true;
        if (file_exists(p)) return true;
        if (optional_overlay) {
            log_event(std::string("警告：找不到 overlay：") + p + "，跳过");
            return true;
        }
        err = std::string(key) + " 不存在：" + p;
        return false;
    };
    bool files_ok = true;
    if (wv == "gguf") {
        files_ok = need_file("GGUF_FILE", false) && need_file("GGUF_MTP_FILE", false) &&
                   need_file("GGUF_VISION_FILE", false);
    } else {
        files_ok = need_file("MODEL_FILE", false) &&
                   (wv == "v1" || need_file("NGRAM_FILE", false)) &&
                   need_file("OVERLAY_FILE", true) && need_file("MTP_FILE", false) &&
                   need_file("VISION_FILE", false);
    }
    // GGUF 权重自带 tokenizer，不查词表目录（同上）
    if (files_ok && wv != "gguf" &&
        !file_exists(resolved(conf, "TOKENIZER_DIR") + "/tokenizer.json")) {
        err = "tokenizer.json 不存在：" + resolved(conf, "TOKENIZER_DIR");
        files_ok = false;
    }
    long ep = 8730, ap = 8731;
    svcconf::parse_int(svcconf::cfg(conf, "ENGINE_PORT", "8730"), &ep);
    svcconf::parse_int(svcconf::cfg(conf, "API_PORT", "8731"), &ap);
    if (files_ok && !port_free(static_cast<int>(ep))) {
        err = "引擎端口 " + std::to_string(ep) + " 已被占用";
        files_ok = false;
    }
    if (files_ok && !port_free(static_cast<int>(ap))) {
        err = "API 端口 " + std::to_string(ap) + " 已被占用";
        files_ok = false;
    }
    if (!files_ok) {
        set_state("failed", "预检失败：" + err);
        log_event(g_svc.detail);
        return;
    }

    ensure_dir("logs");
    const std::string log = "logs/launcher-" + now_stamp() + ".log";
    const pid_t pid = fork();
    if (pid < 0) {
        set_state("failed", std::string("fork 失败：") + strerror(errno));
        return;
    }
    if (pid == 0) {
        // 子进程：独立会话/进程组，输出重定向到日志，然后 exec 启动脚本
        setsid();
        const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd >= 0) {
            dup2(fd, STDOUT_FILENO);
            dup2(fd, STDERR_FILENO);
        }
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) dup2(devnull, STDIN_FILENO);
        execlp("bash", "bash", script.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_svc.child_pid = pid;
        g_svc.run_log = log;
        g_svc.eng_state = "starting";
        set_state_locked("spawning", "引擎加载中");
    }
    write_pidfile(pid, log);
    log_event("已启动 " + script + "（pid " + std::to_string(pid) + "），日志 " + log);
    std::thread(monitor_linux, gen).detach();
}

void do_stop_linux() {
    pid_t pid;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        pid = g_svc.child_pid;
    }
    if (pid > 0) {
        kill(-pid, SIGTERM);  // 整个进程组；脚本 trap 负责引擎/API 收尾
        const int64_t deadline = time(nullptr) + 15;
        int wst = 0;
        for (;;) {
            const pid_t r = waitpid(pid, &wst, WNOHANG);
            if (r == pid || (r < 0 && errno == ECHILD)) break;  // ECHILD：已被监视线程回收
            if (time(nullptr) >= deadline) {
                kill(-pid, SIGKILL);
                waitpid(pid, &wst, 0);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        clear_pidfile();
    }
    std::lock_guard<std::mutex> lk(g_mtx);
    g_svc.child_pid = -1;
    g_svc.api_up = false;
    g_svc.eng_state = "stopped";
    set_state_locked("idle", "");
    log_event("已停止");
}

#endif  // _WIN32

bool state_active(const std::string& s) {
    return s == "spawning" || s == "waiting_engine" || s == "waiting_api" ||
           s == "running" || s == "stopping";
}

json status_json() {
    std::lock_guard<std::mutex> lk(g_mtx);
    json j;
    j["state"] = g_svc.state;
    j["detail"] = g_svc.detail;
    j["since"] = g_svc.since;
    json eng;
    eng["state"] = g_svc.eng_state;
    eng["pid"] = g_svc.eng_pid;
    eng["exit_code"] = g_svc.eng_exit;
    eng["external"] = g_svc.eng_external;
    eng["log"] = g_svc.eng_log;
    eng["detail"] = g_svc.eng_detail;
    eng["diag"] = g_svc.eng_diag;
    j["engine"] = eng;
    json api;
    api["up"] = g_svc.api_up;
    api["external"] = g_svc.api_external;
    api["pid"] = g_svc.api_pid;
    j["api"] = api;
    j["run_log"] = g_svc.run_log;
    return j;
}

// ---- 路由 ----------------------------------------------------------------------

void send_json(http::Response* r, const json& j) {
    r->content_type = "application/json; charset=utf-8";
    r->body = j.dump();
    r->set("Cache-Control", "no-store");
}

std::string query_param(const http::Request& q, const std::string& key);

// ---- /api/profiles：方案（保存/切换一组「权重 + 推理设置」配置） -------------------
// 存 data/launcher-profiles.json：{"profiles": {名字: {weights_ver, 键…}}}。
// 权重与推理设置绑定：切换方案经 /api/config 正常落盘 service.conf（含标记与校验）。
// 网络/请求日志（端口、REQLOG）属机器级配置，不进方案。
const char* kProfileKeys[] = {
    // 权重
    "MODEL_FILE", "NGRAM_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE",
    "GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE", "TOKENIZER_DIR",
    // 推理设置
    "MAX_CONTEXT", "PARALLEL", "KV_PAGED", "KV_POOL_TOKENS", "PREFILL_CHUNK",
    "CONC_PREFILL_CHUNK", "QSA_INDEXER_BUDGET", "KVSNAP_MAX_GB", "KVSNAP_DIR",
    "MOE_A4", "MAX_IMAGES", "VIZ", "AFFINITY", "PLE_URING", "MEMORY_CAP_GB",
    "MIN_AVAILABLE_GB", "START_TIMEOUT", "STALL_TIMEOUT",
};

bool is_profile_key(const std::string& key) {
    for (const char* k : kProfileKeys)
        if (key == k) return true;
    return false;
}

std::string profiles_path() { return "data/launcher-profiles.json"; }

json profiles_read() {
    std::string text;
    json j;
    if (svcconf::read_text_file(profiles_path(), &text) && !text.empty()) {
        j = json::parse(text, nullptr, false);
        if (j.is_discarded()) j = json::object();
    }
    if (!j.is_object()) j = json::object();
    if (!j.contains("profiles") || !j["profiles"].is_object()) j["profiles"] = json::object();
    return j;
}

bool profiles_write(const json& j, std::string* err) {
    ensure_dir("data");
    const std::string tmp = profiles_path() + ".tmp";
    {
        FILE* f = fopen(tmp.c_str(), "wb");
        if (!f) {
            if (err) *err = "cannot write " + tmp;
            return false;
        }
        const std::string s = j.dump(2);
        fwrite(s.data(), 1, s.size(), f);
        fclose(f);
    }
#ifdef _WIN32
    // Windows 的 rename 不覆盖已存在目标
    remove(profiles_path().c_str());
#endif
    if (rename(tmp.c_str(), profiles_path().c_str()) != 0) {
        if (err) *err = "cannot rename " + tmp;
        return false;
    }
    return true;
}

void handle_profiles_get(const http::Request&, http::Response* r, http::Stream*) {
    send_json(r, profiles_read());
}

void handle_profiles_post(const http::Request& q, http::Response* r, http::Stream*) {
    const json body = json::parse(q.body, nullptr, false);
    if (body.is_discarded()) http::fail(400, "invalid JSON body");
    const std::string name = body.value("name", "");
    if (name.empty() || name.size() > 64)
        http::fail(400, "name must be 1-64 chars");
    if (!body.contains("data") || !body["data"].is_object())
        http::fail(400, "expected {name, data:{...}}");
    json data = json::object();
    for (auto it = body["data"].begin(); it != body["data"].end(); ++it) {
        if (it.key() == "weights_ver") {
            const std::string wv = it.value().get<std::string>();
            if (wv != "v1" && wv != "v2" && wv != "gguf")
                http::fail(400, "weights_ver must be one of v1, v2, gguf");
            data["weights_ver"] = wv;
            continue;
        }
        if (!is_profile_key(it.key()))
            http::fail(400, "not a weight key: " + it.key());
        if (!it.value().is_string()) http::fail(400, "value must be a string");
        data[it.key()] = it.value().get<std::string>();
    }
    json j = profiles_read();
    j["profiles"][name] = std::move(data);
    std::string perr;
    if (!profiles_write(j, &perr)) http::fail(500, perr);
    log_event("权重方案已保存：" + name);
    send_json(r, profiles_read());
}

void handle_profiles_delete(const http::Request& q, http::Response* r, http::Stream*) {
    const json body = json::parse(q.body, nullptr, false);
    if (body.is_discarded()) http::fail(400, "invalid JSON body");
    const std::string name = body.value("name", "");
    json j = profiles_read();
    j["profiles"].erase(name);
    std::string perr;
    if (!profiles_write(j, &perr)) http::fail(500, perr);
    log_event("权重方案已删除：" + name);
    send_json(r, profiles_read());
}

// ---- /api/files：供配置页浏览选择服务器上的文件 -------------------------------
void handle_files(const http::Request& q, http::Response* r, http::Stream*) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::string rel = query_param(q, "path");
    std::replace(rel.begin(), rel.end(), '\\', '/');
    while (!rel.empty() && rel.back() == '/') rel.pop_back();

#ifdef _WIN32
    // 盘符列表（文件浏览器「计算机」层）
    if (rel == "@drives") {
        json drives = json::array();
        const DWORD mask = GetLogicalDrives();
        for (int i = 0; i < 26; ++i)
            if (mask & (1u << i)) {
                json e;
                e["name"] = std::string(1, static_cast<char>('A' + i)) + ":/";
                drives.push_back(std::move(e));
            }
        json j;
        j["root"] = "";
        j["path"] = "@drives";
        j["dirs"] = std::move(drives);
        j["files"] = json::array();
        send_json(r, j);
        return;
    }
#endif

    fs::path target;
    if (rel.empty() || rel == ".") {
        target = fs::weakly_canonical(fs::path(g_root), ec);  // 空 = 项目根
    } else {
        target = fs::u8path(rel);
        if (!target.is_absolute())
            target = fs::weakly_canonical(fs::path(g_root) / target, ec);
    }
    if (ec) http::fail(400, "bad path: " + ec.message());
    const fs::path canon = fs::weakly_canonical(target, ec);
    if (ec) http::fail(400, "bad path: " + ec.message());
    if (!fs::is_directory(canon, ec) || ec)
        http::fail(400, "path is not a directory or does not exist");
    json dirs = json::array(), files = json::array();
    for (const auto& de : fs::directory_iterator(canon, ec)) {
        const std::string name = de.path().filename().u8string();
        if (name.empty() || name[0] == '.') continue;  // 隐藏条目不进列表
        std::error_code ec2;
        json e;
        e["name"] = name;
        const auto ft = fs::last_write_time(de.path(), ec2);
        if (!ec2) {
            const auto st = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                ft - fs::file_time_type::clock::now() + std::chrono::system_clock::now());
            e["mtime"] = static_cast<long long>(std::chrono::system_clock::to_time_t(st));
        }
        if (de.is_directory(ec2) && !ec2) {
            dirs.push_back(std::move(e));
        } else if (de.is_regular_file(ec2) && !ec2) {
            const auto sz = fs::file_size(de.path(), ec2);
            if (!ec2) e["size"] = static_cast<long long>(sz);
            files.push_back(std::move(e));
        }
    }
    if (ec) http::fail(500, "cannot list directory: " + ec.message(), "server_error",
                       "server_error");
    const auto by_name = [](const json& a, const json& b) {
        return a["name"].get_ref<const std::string&>() <
               b["name"].get_ref<const std::string&>();
    };
    std::sort(dirs.begin(), dirs.end(), by_name);
    std::sort(files.begin(), files.end(), by_name);
    json j;
    j["root"] = g_root;
    // 回显规范化后的绝对路径（前端据此继续导航）
    j["path"] = canon.generic_u8string();
    j["dirs"] = std::move(dirs);
    j["files"] = std::move(files);
    send_json(r, j);
}

// ---- /api/check：批量探测路径存在性（预检区随输入实时刷新用） --------------------
// 与 /api/files 同样的根目录约束：绝对路径、".." 段直接按不存在处理。
void handle_check(const http::Request& q, http::Response* r, http::Stream*) {
    namespace fs = std::filesystem;
    const json body = json::parse(q.body, nullptr, false);
    if (body.is_discarded() || !body.contains("paths") || !body["paths"].is_array())
        http::fail(400, "expected {\"paths\": [...]}");
    std::error_code ec;
    const fs::path root = fs::weakly_canonical(fs::path(g_root), ec);
    if (ec) http::fail(500, ec.message(), "server_error", "server_error");
    json out = json::array();
    for (const auto& p : body["paths"]) {
        if (!p.is_string() || out.size() >= 64) continue;
        std::string rel = p.get<std::string>();
        std::replace(rel.begin(), rel.end(), '\\', '/');
        while (!rel.empty() && rel.front() == '/') rel.erase(rel.begin());
        json e;
        e["path"] = p.get<std::string>();
        fs::path sub = fs::u8path(rel);
        if (sub.is_relative()) sub = root / sub;
        const fs::path canon = fs::weakly_canonical(sub, ec);
        const bool ok = !ec;
        e["exists"] = ok && fs::exists(canon, ec) && !ec;
        e["is_dir"] = ok && fs::is_directory(canon, ec) && !ec;
        out.push_back(std::move(e));
    }
    send_json(r, out);
}

// ---- /media/：面板同款背景与立绘（白名单，防目录穿越） ---------------------------
void handle_media(const http::Request& q, http::Response* r, http::Stream*) {
    const std::string name = q.path.substr(7);  // "/media/" 之后
    if (name != "panel-bg.png" && name != "qwen-charactor-alpha.png") {
        r->status = 404;
        r->body = http::error_json("not found", "not_found", "not_found");
        return;
    }
    std::string data;
    if (!svcconf::read_text_file(g_root + "/media/" + name, &data)) {
        r->status = 404;
        r->body = http::error_json("not found", "not_found", "not_found");
        return;
    }
    r->content_type = "image/png";
    r->body = std::move(data);
    r->set("Cache-Control", "public, max-age=86400");
}

std::string query_param(const http::Request& q, const std::string& key) {
    const std::string& qs = q.query;
    size_t pos = 0;
    while (pos <= qs.size()) {
        const size_t amp = qs.find('&', pos);
        const std::string kv = qs.substr(pos, amp == std::string::npos
                                                 ? std::string::npos : amp - pos);
        const size_t eq = kv.find('=');
        if (eq != std::string::npos && kv.substr(0, eq) == key) {
            const std::string raw = kv.substr(eq + 1);
            std::string out;
            for (size_t i = 0; i < raw.size(); ++i) {
                if (raw[i] == '%' && i + 2 < raw.size() && isxdigit(raw[i + 1]) &&
                    isxdigit(raw[i + 2])) {
                    out += static_cast<char>(std::stoi(raw.substr(i + 1, 2), nullptr, 16));
                    i += 2;
                } else if (raw[i] == '+') {
                    out += ' ';
                } else {
                    out += raw[i];
                }
            }
            return out;
        }
        if (amp == std::string::npos) break;
        pos = amp + 1;
    }
    return "";
}

// 静态页（生成物嵌入；LAUNCHER_WEB_DIR 指向磁盘目录时优先读磁盘，方便前端调试）
void handle_static(const http::Request& q, http::Response* r, http::Stream*) {
    const std::string target = q.path == "/" ? "/" : q.path;
    for (const StaticAsset& a : kStaticAssets) {
        if (target != a.url) continue;
        std::string body;
        if (const char* dir = getenv("LAUNCHER_WEB_DIR")) {
            if (*dir && svcconf::read_text_file(std::string(dir) + "/" + a.file, &body)) {
                r->content_type = a.mime;
                r->body = std::move(body);
                r->set("Cache-Control", "no-store");
                return;
            }
        }
        r->content_type = a.mime;
        r->body.assign(reinterpret_cast<const char*>(a.data), a.len);
        r->set("Cache-Control", "no-store");
        return;
    }
    r->status = 404;
    r->body = http::error_json("not found", "not_found", "not_found");
}

void handle_config_get(const http::Request&, http::Response* r, http::Stream*) {
    json j;
#ifdef _WIN32
    j["platform"] = "windows";
#else
    j["platform"] = "linux";
#endif
    j["conf_path"] = g_conf_path;
    j["lang"] = current_lang();
    j["weights_ver"] = weights_ver();
    json vals = json::object(), defs = json::object(), present = json::object(),
         envov = json::array();
    std::vector<std::string> keys;
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        for (size_t i = 0; i < svcconf::kManagedKeysCount; ++i)
            keys.emplace_back(svcconf::kManagedKeys[i]);
        for (const char* k : kExtraKeys) keys.emplace_back(k);
        for (const auto& key : keys) {
            vals[key] = resolved(g_conf, key);
            defs[key] = default_of(key);
            present[key] = g_conf.count(key) > 0;
            if (env_overridden(key)) envov.push_back(key);
        }
    }
    j["values"] = std::move(vals);
    j["defaults"] = std::move(defs);
    j["present"] = std::move(present);
    j["env_overridden"] = std::move(envov);
    // 权重文件存在性（相对路径相对项目根；启动时已 chdir 到根）
    json files = json::object();
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        const char* file_keys[] = {"MODEL_FILE",  "NGRAM_FILE",      "OVERLAY_FILE",
                                   "MTP_FILE",    "VISION_FILE",     "GGUF_FILE",
                                   "GGUF_MTP_FILE", "GGUF_VISION_FILE", "KVSNAP_DIR"};
        for (const char* key : file_keys) {
            const std::string p = resolved(g_conf, key);
            json f;
            f["path"] = p;
            f["optional"] = svcconf::is_optional_file_key(key);
            if (p.empty()) {
                f["exists"] = false;
            } else if (std::string(key) == "KVSNAP_DIR") {
                f["exists"] = dir_exists(p);
            } else {
                f["exists"] = file_exists(p);
            }
            files[key] = std::move(f);
        }
        const std::string tok = resolved(g_conf, "TOKENIZER_DIR");
        json tj;
        tj["path"] = tok + "/tokenizer.json";
        tj["exists"] = file_exists(tok + "/tokenizer.json") ||
                       file_exists(tok + "\\tokenizer.json");
        files["TOKENIZER_JSON"] = std::move(tj);
    }
    j["files"] = std::move(files);
    // 端口与编译产物预检
    json ports;
    const int ep = engine_port(), ap = api_port();
    ports["engine"] = ep;
    ports["api"] = ap;
    ports["engine_free"] = port_free(ep);
    ports["api_free"] = port_free(ap);
    j["ports"] = std::move(ports);
    json bins;
#ifdef _WIN32
    bins["api"] = file_exists("build\\qwenox-win.exe");
    bins["engine"] = file_exists("build\\qwenox-engine-win.exe");
    bins["kernel_db"] = dir_exists("build\\rocblas\\library") &&
                        dir_exists("build\\hipblaslt\\library");
#else
    bins["api"] = file_exists("build/qwenox-api");
    bins["engine"] = file_exists("build/qwenox-engine");
    bins["scripts"] = file_exists("start_hgn.sh") && file_exists("start_gguf.sh");
#endif
    j["binaries"] = std::move(bins);
    j["service"] = status_json();
    send_json(r, j);
}

// validate_conf 之外的补充语义校验（与 serve_common.sh / engine_sup 的检查对齐）。
bool extra_validate(const svcconf::Conf& conf, std::string* err) {
    auto int_range = [&](const char* key, const char* dflt, long lo, long hi) -> bool {
        const std::string v = svcconf::cfg(conf, key, dflt);
        long n = 0;
        if (!svcconf::parse_int(v, &n) || n < lo || n > hi) {
            *err = std::string(key) + " must be an integer in [" + std::to_string(lo) +
                   ", " + std::to_string(hi) + "] (got \"" + v + "\")";
            return false;
        }
        return true;
    };
    const std::string qsa = svcconf::cfg(conf, "QSA_INDEXER_BUDGET", "0");
    if (qsa != "0" && qsa != "4096") {
        *err = "QSA_INDEXER_BUDGET must be 0 or 4096";
        return false;
    }
    if (!int_range("MTP_GAMMA", "0", 0, 8)) return false;
    if (!int_range("MOE_A4", "0", 0, 3)) return false;
    if (!int_range("MAX_IMAGES", "8", 1, 256)) return false;
    if (!int_range("VIZ", "0", 0, 1)) return false;
    if (!int_range("PLE_URING", "1", 0, 1)) return false;
    if (!int_range("REQLOG", "0", 0, 1)) return false;
    if (!int_range("REQLOG_LEVEL", "6", 0, 9)) return false;
    if (!int_range("RCKPT_MAX", "8", 0, 1024)) return false;
    if (!int_range("CONC_PREFILL_CHUNK", "8192", 0, 1 << 20)) return false;
    if (!int_range("MEMORY_CAP_GB", "86", 1, 2048)) return false;
    if (!int_range("MIN_AVAILABLE_GB", "100", 1, 4096)) return false;
    if (!int_range("START_TIMEOUT", "360", 30, 86400)) return false;
    if (!int_range("STALL_TIMEOUT", "60", 5, 3600)) return false;
    sockaddr_in addr{};
    const std::string eh = svcconf::cfg(conf, "ENGINE_HOST", "127.0.0.1");
    if (!engine_net::address(eh, 1, &addr)) {
        *err = "ENGINE_HOST must be an IPv4 address (got \"" + eh + "\")";
        return false;
    }
    const std::string ah = svcconf::cfg(conf, "API_HOST", "0.0.0.0");
    if (!engine_net::address(ah, 1, &addr)) {
        *err = "API_HOST must be an IPv4 address (got \"" + ah + "\")";
        return false;
    }
    return true;
}

void handle_config_post(const http::Request& q, http::Response* r, http::Stream*) {
    const json body = json::parse(q.body, nullptr, false);
    if (body.is_discarded()) http::fail(400, "invalid JSON body");
    const json& vals_j = body.contains("values") ? body["values"] : body;
    if (!vals_j.is_object()) http::fail(400, "expected {\"values\": {KEY: \"value\"}}");
    std::string wv;
    if (body.contains("weights_ver") && body["weights_ver"].is_string()) {
        wv = body["weights_ver"].get<std::string>();
        if (wv != "v1" && wv != "v2" && wv != "gguf")
            http::fail(400, "weights_ver must be one of v1, v2, gguf");
    }
    std::string lang;
    if (body.contains("lang") && body["lang"].is_string()) {
        lang = body["lang"].get<std::string>();
        if (lang != "zh" && lang != "en") http::fail(400, "lang must be zh or en");
    }
    std::map<std::string, std::string> vals;
    for (auto it = vals_j.begin(); it != vals_j.end(); ++it) {
        if (!is_launcher_key(it.key()))
            http::fail(400, "unknown or unmanaged key: " + it.key());
        if (!it.value().is_string()) http::fail(400, "value must be a string: " + it.key());
        vals[it.key()] = it.value().get<std::string>();
    }
    // 权重键按版本过滤（与 API /admin/config 同语义）
    if (!wv.empty()) {
        static const std::vector<std::string> kWeightKeys = {
            "MODEL_FILE", "NGRAM_FILE",     "OVERLAY_FILE",     "MTP_FILE",
            "VISION_FILE", "GGUF_FILE",     "GGUF_MTP_FILE",    "GGUF_VISION_FILE"};
        static const std::vector<std::string> kVerKeys[] = {
            /* v1 */ {"MODEL_FILE", "OVERLAY_FILE", "MTP_FILE", "VISION_FILE"},
            /* v2 */ {"MODEL_FILE", "NGRAM_FILE", "MTP_FILE", "VISION_FILE"},
            /* gguf */ {"GGUF_FILE", "GGUF_MTP_FILE", "GGUF_VISION_FILE"},
        };
        const auto& allow = kVerKeys[wv == "v2" ? 1 : wv == "gguf" ? 2 : 0];
        for (auto it = vals.begin(); it != vals.end();) {
            const bool is_weight = std::find(kWeightKeys.begin(), kWeightKeys.end(),
                                             it->first) != kWeightKeys.end();
            const bool allowed = std::find(allow.begin(), allow.end(), it->first) !=
                                 allow.end();
            if (is_weight && !allowed)
                it = vals.erase(it);
            else
                ++it;
        }
        if (wv == "v1") vals["NGRAM_FILE"] = "$MODEL_FILE";
        if (wv == "v2") vals["OVERLAY_FILE"] = "";
    }
    std::string err;
    json envov = json::array();
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        // 合并到当前有效值后整表校验（端口互异、YaRN 上限、KV 池上限等）
        svcconf::Conf merged = g_conf;
        for (const auto& kv : vals) merged[kv.first] = kv.second;
        // ROPE_FACTOR 不在网页暴露：按 MAX_CONTEXT 自动推导写回——超过原生
        // 上下文（ROPE_ORIGINAL_CTX，默认 256K）自动开 YaRN，factor =
        // ⌈MAX_CONTEXT / ROPE_ORIGINAL_CTX⌉，否则回落 1。
        long mc = 262144, orig = 262144, pool = 0;
        svcconf::parse_int(svcconf::cfg(merged, "MAX_CONTEXT", "262144"), &mc);
        svcconf::parse_int(svcconf::cfg(merged, "ROPE_ORIGINAL_CTX", "262144"), &orig);
        svcconf::parse_int(svcconf::cfg(merged, "KV_POOL_TOKENS", "0"), &pool);
        if (orig < 1) orig = 1;
        const long factor = mc > orig ? (mc + orig - 1) / orig : 1;
        vals["ROPE_FACTOR"] = std::to_string(factor);
        merged["ROPE_FACTOR"] = vals["ROPE_FACTOR"];
        // KV 页池小于 MAX_CONTEXT 时按 MAX_CONTEXT 算（与引擎语义一致，落盘对齐）
        if (pool > 0 && pool < mc) {
            vals["KV_POOL_TOKENS"] = std::to_string(mc);
            merged["KV_POOL_TOKENS"] = vals["KV_POOL_TOKENS"];
        }
        if (!svcconf::validate_conf(merged, &err)) http::fail(400, err);
        if (!extra_validate(merged, &err)) http::fail(400, err);
        if (!vals.empty() || !wv.empty()) {
            if (!svcconf::save_conf(g_conf_path, vals, &err,
                                    wv.empty() ? nullptr : kWeightsMarker, wv))
                http::fail(500, err);
            g_conf.clear();
            svcconf::load_conf(g_conf_path, &g_conf);
        }
        for (const auto& kv : vals)
            if (env_overridden(kv.first)) envov.push_back(kv.first);
    }
    if (!lang.empty()) svcconf::conf_write_marker(g_conf_path, kLangMarker, lang);
    log_event("配置已保存（" + std::to_string(vals.size()) + " 项）");
    json j;
    j["saved"] = true;
    j["applied"] = "next_start";
    if (!envov.empty()) j["env_overridden"] = std::move(envov);
    j["service"] = status_json();
    send_json(r, j);
}

void handle_status(const http::Request&, http::Response* r, http::Stream*) {
    json j = status_json();
#ifdef _WIN32
    j["platform"] = "windows";
#else
    j["platform"] = "linux";
#endif
    // 控制台地址（API 就绪后前端展示链接）
    std::lock_guard<std::mutex> lk(g_conf_mtx);
    const std::string host = svcconf::cfg(g_conf, "API_HOST", "0.0.0.0");
    j["console_url"] = "http://" + engine_net::connect_host(host) + ":" +
                       svcconf::cfg(g_conf, "API_PORT", "8731") + "/";
    send_json(r, j);
}

// 服务器与托盘全局（在 request_* / 托盘回调 / main 之间共享）
http::Server g_server;
http::Server* g_srv = nullptr;
int g_listen_port = 8729;
#ifdef _WIN32
HWND g_tray_hwnd = nullptr;
constexpr UINT WM_TRAY_QUIT = WM_APP + 2;  // 网页/菜单请求退出（UI 线程收）
#endif

// 启动/停止/退出：HTTP 路由与托盘菜单共用的入口。返回空串 = 已受理。
std::string request_start() {
    uint64_t gen;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (state_active(g_svc.state)) return "service is " + g_svc.state;
        ++g_gen;
        g_svc.detail.clear();
        set_state_locked("spawning", "预检与启动中");
        gen = g_gen;
    }
    log_event("收到启动请求");
#ifdef _WIN32
    std::thread(do_start_windows, gen).detach();
#else
    std::thread(do_start_linux, gen).detach();
#endif
    return "";
}

std::string request_stop() {
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        const bool active = state_active(g_svc.state);
        // 引擎失败/退出但 API 还在（我们拉起的）：允许停止来清理残留
#ifdef _WIN32
        const bool leftover = (g_svc.state == "failed" || g_svc.state == "exited") &&
                              (g_svc.api_proc != nullptr || g_svc.api_up);
#else
        // Linux：脚本退出时其 trap 已回收引擎/API，无残留场景
        const bool leftover = false;
#endif
        if (!active && !leftover) return "service is not running";
        if (g_svc.eng_external && g_svc.eng_state == "running")
            return "引擎由外部启动，请到原启动器停止";
        ++g_gen;  // 监视线程退出，停止流程自己轮询
        set_state_locked("stopping", "");
    }
    log_event("收到停止请求");
#ifdef _WIN32
    std::thread(do_stop_windows).detach();
#else
    std::thread(do_stop_linux).detach();
#endif
    return "";
}

// 退出 launcher：force 时先停掉本进程可管的服务（外部引擎不碰）。
void shutdown_launcher(bool force) {
    std::thread([force] {
        if (force) {
            bool active, eng_ext;
            {
                std::lock_guard<std::mutex> lk(g_mtx);
                active = state_active(g_svc.state);
                eng_ext = g_svc.eng_external;
            }
            if (active && !eng_ext) {
#ifdef _WIN32
                do_stop_windows();
#else
                do_stop_linux();
#endif
            }
        }
        if (g_srv) {
            g_srv->stop();
            // run() 阻塞在 accept：自连一下把它叫醒
            tcp_connect("127.0.0.1", g_listen_port, 500);
        }
#ifdef _WIN32
        if (g_tray_hwnd) PostMessage(g_tray_hwnd, WM_TRAY_QUIT, 0, 0);
#endif
    }).detach();
}

void handle_start(const http::Request&, http::Response* r, http::Stream*) {
    const std::string err = request_start();
    if (!err.empty()) http::fail(409, err);
    send_json(r, status_json());
}

void handle_stop(const http::Request&, http::Response* r, http::Stream*) {
    const std::string err = request_stop();
    if (!err.empty()) http::fail(409, err);
    send_json(r, status_json());
}

// 读文件尾部（最多 max_bytes），按行截到完整行。
std::string tail_file(const std::string& path, size_t max_bytes) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return "";
    fseek(f, 0, SEEK_END);
    const long sz = ftell(f);
    const size_t off = sz > 0 && static_cast<size_t>(sz) > max_bytes
                           ? static_cast<size_t>(sz) - max_bytes : 0;
    fseek(f, static_cast<long>(off), SEEK_SET);
    std::string out;
    char buf[16384];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    fclose(f);
    if (off > 0) {
        const size_t nl = out.find('\n');
        if (nl != std::string::npos) out.erase(0, nl + 1);
    }
    return out;
}

#ifndef _WIN32
// logs/<stamp>-<pid>/<name>：启动脚本的运行目录（$$ == 脚本 pid）。
std::string find_run_log(pid_t pid, const char* name) {
    if (pid <= 0) return "";
    const std::string suffix = "-" + std::to_string(pid);
    DIR* d = opendir("logs");
    if (!d) return "";
    std::string found;
    while (dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() > suffix.size() &&
            n.compare(n.size() - suffix.size(), suffix.size(), suffix) == 0) {
            const std::string p = "logs/" + n + "/" + name;
            if (file_exists(p)) found = p;
        }
    }
    closedir(d);
    return found;
}
#endif

#ifdef _WIN32
// logs 目录里最新的 prefix*.log（API 无托盘模式（--no-tray）落盘的 api-win-*.log 用）
std::string newest_log(const std::string& prefix) {
    namespace fs = std::filesystem;
    std::error_code ec;
    std::string best;
    fs::file_time_type best_t{};
    for (const auto& de : fs::directory_iterator("logs", ec)) {
        const std::string n = de.path().filename().u8string();
        if (n.compare(0, prefix.size(), prefix) != 0) continue;
        if (n.size() < 5 || n.compare(n.size() - 4, 4, ".log") != 0) continue;
        const auto t = fs::last_write_time(de.path(), ec);
        if (ec) continue;
        if (best.empty() || t > best_t) {
            best = de.path().u8string();
            best_t = t;
        }
    }
    return best;
}
#endif  // _WIN32

void handle_logs(const http::Request& q, http::Response* r, http::Stream*) {
    const std::string which = query_param(q, "which");
    std::string path;
    if (which == "launcher") {
        path = g_log_path;
    } else if (which == "engine") {
#ifdef _WIN32
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            path = g_svc.eng_log;
        }
        if (path.empty()) path = "(引擎未启动过，无日志)";
#else
        pid_t pid;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            pid = g_svc.child_pid;
        }
        path = find_run_log(pid, "engine.log");
        if (path.empty()) {
            std::lock_guard<std::mutex> lk(g_mtx);
            path = g_svc.run_log;  // 脚本日志兜底（含错误尾部）
        }
#endif
    } else if (which == "api") {
#ifdef _WIN32
        // --no-tray 模式下 API 的 stdout/stderr 落在 logs\api-win-*.log
        path = newest_log("api-win-");
#else
        pid_t pid;
        {
            std::lock_guard<std::mutex> lk(g_mtx);
            pid = g_svc.child_pid;
        }
        path = find_run_log(pid, "api.log");
#endif
    } else if (which == "script") {
        std::lock_guard<std::mutex> lk(g_mtx);
        path = g_svc.run_log;
    } else {
        http::fail(400, "which must be launcher|engine|api|script");
    }
    json j;
    if (path.empty() || path[0] == '(') {
        j["path"] = "";
        j["text"] = path;
    } else {
        j["path"] = path;
        j["text"] = tail_file(path, 96 * 1024);
    }
    send_json(r, j);
}

void handle_shutdown(const http::Request& q, http::Response* r, http::Stream*) {
    const bool force = query_param(q, "force") == "1";
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        if (state_active(g_svc.state) && !force)
            http::fail(409, "服务运行中；先停止，或带 ?force=1 强制退出");
    }
    json j;
    j["bye"] = true;
    send_json(r, j);
    log_event("launcher 退出");
    shutdown_launcher(force);
}

// ---- main ----------------------------------------------------------------------

// 打开浏览器；false = 没有图形环境（Linux 无 DISPLAY/WAYLAND_DISPLAY）。
bool open_browser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return true;
#else
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) return false;  // 无桌面
    const pid_t pid = fork();
    if (pid == 0) {
        setsid();
        const int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
        }
        execlp("xdg-open", "xdg-open", url.c_str(), static_cast<char*>(nullptr));
        _exit(127);
    }
    return true;
#endif
}

// 无图形环境提示（中英双语）：指向 TUI 配置工具与脚本启动
void print_line(const std::string& s) {
    fprintf(stderr, "%s\n", s.c_str());
    fflush(stderr);
}

void warn_headless(const std::string& url) {
    print_line("");
    print_line("警告：未检测到图形环境，浏览器不会自动打开。");
    print_line("  配置请改用终端设置工具：./qwenox-settings");
    print_line("  启动请直接用脚本：bash start_hgn.sh（hgn 权重）/ bash start_gguf.sh（GGUF）");
    print_line("  本页面仍在监听，可经 SSH 转发访问：ssh -L 8729:127.0.0.1:8729 <本机>，"
               "然后打开 " + url);
    print_line("WARNING: no graphical environment detected; no browser was opened.");
    print_line("  Configure with the terminal tool: ./qwenox-settings");
    print_line("  Start with the scripts: bash start_hgn.sh (hgn) / bash start_gguf.sh (GGUF)");
    print_line("  This page is still listening; reach it via SSH forward: "
               "ssh -L 8729:127.0.0.1:8729 <this-host>, then open " + url);
    print_line("");
}


#ifdef _WIN32
// ---- 托盘（结构移植自 src/api/tray_win.cpp）------------------------------------
// 主线程跑消息循环 + 托盘图标，worker 线程跑 http::Server::run()；菜单启停与
// 网页按钮共用 request_start()/request_stop()。
std::wstring to_w(const std::string& s) {
    if (s.empty()) return {};
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()),
                                      nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), &w[0], n);
    return w;
}

int msgbox(const std::string& text, UINT flags = MB_ICONERROR | MB_OK) {
    return MessageBoxW(nullptr, to_w(text).c_str(), L"Qwenox Launcher",
                       flags | MB_SETFOREGROUND | MB_TOPMOST);
}

// 界面语言与气泡开关（启动时读一次）：lang 标记 > UI_LANG > 系统 UI 语言；
// notify 标记 off/0 = 全部气泡静默。
bool g_lang_en = false;
bool g_notify = true;
#define TR(zh, en) (g_lang_en ? (en) : (zh))

void prefs_resolve() {
    const std::string lang = svcconf::conf_marker(g_conf_path, kLangMarker);
    if (lang == "en") g_lang_en = true;
    else if (lang == "zh") g_lang_en = false;
    else if (const char* e = getenv("UI_LANG")) g_lang_en = _stricmp(e, "en") == 0;
    else g_lang_en = PRIMARYLANGID(GetUserDefaultUILanguage()) != LANG_CHINESE;
    const std::string notify =
        svcconf::conf_marker(g_conf_path, "# start_win: notify=");
    g_notify = notify != "off" && notify != "0";
}

// 从内嵌 ICO 挑不小于 px 的最小条目做 HICON（系统缩放到 px）。
HICON app_icon(int px) {
    const unsigned char* d = kIconIco;
    const size_t total = sizeof(kIconIco);
    auto u16 = [&](size_t o) { return static_cast<unsigned>(d[o] | d[o + 1] << 8); };
    auto u32 = [&](size_t o) { return static_cast<DWORD>(u16(o) | u16(o + 2) << 16); };
    const unsigned n = u16(4);
    int best = -1, best_size = 0;
    for (unsigned i = 0; i < n && 6 + 16 * (i + 1) <= total; ++i) {
        const int s = d[6 + 16 * i] ? d[6 + 16 * i] : 256;
        const bool better = best < 0 || (s >= px ? (best_size < px || s < best_size)
                                                 : (best_size < px && s > best_size));
        if (better) best = i, best_size = s;
    }
    if (best < 0) return nullptr;
    const size_t e = 6 + 16 * static_cast<size_t>(best);
    const DWORD bytes = u32(e + 8), off = u32(e + 12);
    if (off > total || bytes > total - off) return nullptr;
    return CreateIconFromResourceEx(const_cast<PBYTE>(d + off), bytes, TRUE, 0x00030000,
                                    px, px, LR_DEFAULTCOLOR);
}

NOTIFYICONDATAW g_nid{};
UINT g_wm_taskbar_created = 0;
const UINT WM_TRAY = WM_APP + 1;
const UINT WM_DIAG_DONE = WM_APP + 3;
constexpr UINT_PTR kTimerStatus = 1;
std::string g_tray_url;          // 启动器页地址（双击图标打开）
std::string g_tray_state;        // 上次气泡时的状态（跳变检测）
std::string g_balloon_select;    // 气泡点击时在资源管理器选中的文件（诊断包）

std::wstring tray_state_text(const std::string& st) {
    if (st == "running") return TR(L"运行中", L"running");
    if (st == "spawning") return TR(L"启动中…", L"starting…");
    if (st == "waiting_engine") return TR(L"引擎加载中…", L"engine loading…");
    if (st == "waiting_api") return TR(L"等待 API…", L"waiting for API…");
    if (st == "stopping") return TR(L"停止中…", L"stopping…");
    if (st == "exited") return TR(L"已退出", L"exited");
    if (st == "failed") return TR(L"启动失败", L"failed");
    return TR(L"未启动", L"idle");
}

struct TraySnap {
    std::string state, detail;
    bool api_up = false;
    std::string eng_log;
    std::string console_url;
    std::string snap;  // 引擎 SNAP 性能行
};

TraySnap tray_snap() {
    TraySnap s;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        s.state = g_svc.state;
        s.detail = g_svc.detail;
        s.api_up = g_svc.api_up;
        s.eng_log = g_svc.eng_log;
        s.snap = g_svc.snap;
    }
    {
        std::lock_guard<std::mutex> lk(g_conf_mtx);
        s.console_url = "http://" +
                        engine_net::connect_host(svcconf::cfg(g_conf, "API_HOST",
                                                              "0.0.0.0")) +
                        ":" + svcconf::cfg(g_conf, "API_PORT", "8731") + "/";
    }
    return s;
}

void tray_balloon(const std::wstring& title, const std::wstring& text) {
    g_balloon_select.clear();  // 普通气泡；要"点了选中文件"的调用方在之后设置
    if (!g_notify || !g_nid.hWnd) return;
    g_nid.uFlags = NIF_INFO;
    wcsncpy_s(g_nid.szInfoTitle, title.c_str(), _TRUNCATE);
    wcsncpy_s(g_nid.szInfo, text.c_str(), _TRUNCATE);
    g_nid.dwInfoFlags = g_nid.hBalloonIcon ? (NIIF_USER | NIIF_LARGE_ICON) : NIIF_INFO;
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
    g_nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
}
// ---- 性能显示（移植自 tray_win.cpp：SNAP 行 → 人读文本） ------------------------
std::wstring perf_count(uint64_t v) {  // 12K 风格缩写
    wchar_t b[24];
    if (v >= 10000)
        swprintf(b, ARRAYSIZE(b), L"%lluK", (unsigned long long)((v + 500) / 1000));
    else
        swprintf(b, ARRAYSIZE(b), L"%llu", (unsigned long long)v);
    return b;
}

// 空串 = 没取到 / 已过期 / idle。
std::wstring perf_text(const std::string& line) {
    if (line.empty()) return {};
    unsigned long long ts = 0, req = 0, done = 0, total = 0;
    unsigned st = 0;
    double pp = 0, tg_inst = 0, tg_avg = 0, accept = -1;
    if (sscanf(line.c_str(), "P %llu %u %llu %lf %llu %llu %lf %lf %lf", &ts, &st, &req,
               &pp, &done, &total, &tg_inst, &tg_avg, &accept) != 9)
        return {};
    if (st == 0) return {};
    const uint64_t now = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count());
    if (now - ts > 6000) return {};  // 数据过期（引擎被杀兜底：ts 不再更新）
    wchar_t b[96];
    if (st == 1) {  // prefill
        const std::wstring d = perf_count(done), t2 = perf_count(total);
        swprintf(b, ARRAYSIZE(b), L"pp %.0f tok/s · %ls/%ls", pp, d.c_str(), t2.c_str());
    } else if (accept >= 0) {
        swprintf(b, ARRAYSIZE(b), TR(L"tg %.1f tok/s · 接受率 %.1f%%",
                                     L"tg %.1f tok/s · accept %.1f%%"),
                 tg_inst, accept);
    } else {
        swprintf(b, ARRAYSIZE(b), L"tg %.1f tok/s", tg_inst);
    }
    return b;
}

// ---- 悬浮性能监控窗（置顶小窗；托盘菜单开关，左键拖动，右键隐藏） ------------------
HWND g_perf_hwnd = nullptr;
bool g_perf_show = false;

void perf_paint(HWND hwnd) {
    PAINTSTRUCT ps;
    HDC dc = BeginPaint(hwnd, &ps);
    RECT rc;
    GetClientRect(hwnd, &rc);
    HBRUSH bg = CreateSolidBrush(RGB(13, 18, 34));
    FillRect(dc, &rc, bg);
    DeleteObject(bg);
    HPEN pen = CreatePen(PS_SOLID, 1, RGB(120, 140, 190));
    HGDIOBJ oldp = SelectObject(dc, pen);
    HGDIOBJ oldb = SelectObject(dc, GetStockObject(NULL_BRUSH));
    Rectangle(dc, 0, 0, rc.right, rc.bottom);
    SelectObject(dc, oldb);
    SelectObject(dc, oldp);
    DeleteObject(pen);
    SetBkMode(dc, TRANSPARENT);
    std::wstring l1, l2;
    {
        const TraySnap s = tray_snap();
        l1 = std::wstring(L"Qwenox · ") + tray_state_text(s.state);
        std::lock_guard<std::mutex> lk(g_mtx);
        if (g_svc.eng_state == "running" && !g_svc.snap.empty())
            l2 = perf_text(g_svc.snap);
    }
    SetTextColor(dc, RGB(154, 166, 192));
    TextOutW(dc, 10, 8, l1.c_str(), static_cast<int>(l1.size()));
    if (!l2.empty()) {
        SetTextColor(dc, RGB(232, 236, 246));
        TextOutW(dc, 10, 32, l2.c_str(), static_cast<int>(l2.size()));
    }
    EndPaint(hwnd, &ps);
}

LRESULT CALLBACK perf_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    switch (msg) {
    case WM_PAINT:
        perf_paint(hwnd);
        return 0;
    case WM_LBUTTONDOWN:  // 拖动移动
        ReleaseCapture();
        SendMessageW(hwnd, WM_NCLBUTTONDOWN, HTCAPTION, 0);
        return 0;
    case WM_RBUTTONUP:  // 右键隐藏（与菜单勾选同步）
        g_perf_show = false;
        ShowWindow(hwnd, SW_HIDE);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

void perf_toggle() {
    if (!g_perf_hwnd) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = perf_wndproc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"QwenoxLauncherPerf";
        wc.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));  // IDC_ARROW
        RegisterClassExW(&wc);
        RECT wa{};
        SystemParametersInfoW(SPI_GETWORKAREA, 0, &wa, 0);
        const int w = 240, h = 66;
        g_perf_hwnd = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
                                      wc.lpszClassName, L"Qwenox", WS_POPUP,
                                      wa.right - w - 16, wa.bottom - h - 16, w, h, nullptr,
                                      nullptr, wc.hInstance, nullptr);
        if (!g_perf_hwnd) return;
    }
    g_perf_show = !g_perf_show;
    ShowWindow(g_perf_hwnd, g_perf_show ? SW_SHOWNOACTIVATE : SW_HIDE);
    if (g_perf_show) InvalidateRect(g_perf_hwnd, nullptr, TRUE);
}

// ---- 开机自启（HKCU Run；launcher 版带 --no-browser，登录后只出托盘不弹网页） ------
// 与 qwenox-win 托盘的 "Qwenox" 值名区分开，互不覆盖。
bool autostart_get() {
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER,
                      L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run", 0, KEY_READ,
                      &key) != ERROR_SUCCESS)
        return false;
    const LONG r = RegQueryValueExW(key, L"QwenoxLauncher", nullptr, nullptr, nullptr,
                                    nullptr);
    RegCloseKey(key);
    return r == ERROR_SUCCESS;
}

std::string autostart_set(bool on) {  // 空串 = 成功
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER,
                        L"SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run", 0, nullptr,
                        0, KEY_WRITE, nullptr, &key, nullptr) != ERROR_SUCCESS)
        return "RegCreateKeyEx: " + std::to_string(GetLastError());
    LONG r = ERROR_SUCCESS;
    if (on) {
        wchar_t path[MAX_PATH]{};
        const DWORD n = GetModuleFileNameW(nullptr, path, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) {
            r = ERROR_BAD_PATHNAME;
        } else {
            const std::wstring cmd =
                L"\"" + std::wstring(path, n) + L"\" --no-browser";
            r = RegSetValueExW(key, L"QwenoxLauncher", 0, REG_SZ,
                               reinterpret_cast<const BYTE*>(cmd.c_str()),
                               static_cast<DWORD>((cmd.size() + 1) * sizeof(wchar_t)));
        }
    } else {
        r = RegDeleteValueW(key, L"QwenoxLauncher");
        if (r == ERROR_FILE_NOT_FOUND) r = ERROR_SUCCESS;
    }
    RegCloseKey(key);
    return r == ERROR_SUCCESS ? std::string() : "RegSet/DeleteValue: " + std::to_string(r);
}

// ---- 剪贴板 / 资源管理器选中 / 通知注册（移植自 tray_win.cpp） ---------------------
void copy_text(const std::wstring& s) {
    if (!OpenClipboard(g_tray_hwnd)) return;
    EmptyClipboard();
    const size_t bytes = (s.size() + 1) * sizeof(wchar_t);
    HGLOBAL g = GlobalAlloc(GMEM_MOVEABLE, bytes);
    if (g) {
        void* dst = GlobalLock(g);
        if (dst) {
            memcpy(dst, s.c_str(), bytes);
            GlobalUnlock(g);
            if (!SetClipboardData(CF_UNICODETEXT, g)) GlobalFree(g);
        } else {
            GlobalFree(g);
        }
    }
    CloseClipboard();
}

void select_in_explorer(std::string path) {
    for (char& ch : path)
        if (ch == '/') ch = '\\';
    ShellExecuteA(nullptr, "open", "explorer.exe", ("/select,\"" + path + "\"").c_str(),
                  nullptr, SW_SHOWNORMAL);
}

// 通知应用名注册（toast 标题显示 "Qwenox" 而不是 exe 文件名；幂等）。
void register_app_id() {
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER,
                        L"SOFTWARE\\Classes\\AppUserModelIDs\\Qwenox", 0, nullptr, 0,
                        KEY_WRITE, nullptr, &key, nullptr) == ERROR_SUCCESS) {
        const wchar_t* name = L"Qwenox";
        RegSetValueExW(key, L"DisplayName", 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(name),
                       static_cast<DWORD>((wcslen(name) + 1) * sizeof(wchar_t)));
        const std::wstring ico = to_w(g_root + "\\src\\launch_win.ico");
        if (GetFileAttributesW(ico.c_str()) != INVALID_FILE_ATTRIBUTES)
            RegSetValueExW(key, L"IconUri", 0, REG_SZ,
                           reinterpret_cast<const BYTE*>(ico.c_str()),
                           static_cast<DWORD>((ico.size() + 1) * sizeof(wchar_t)));
        RegCloseKey(key);
    }
    SetCurrentProcessExplicitAppUserModelID(L"Qwenox");
}

// ---- 导出诊断包（经 API 的 POST /admin/diag；worker 线程打包） --------------------
std::mutex g_diag_mtx;
bool g_diag_busy = false;
std::string g_diag_path, g_diag_err;

void diag_start() {
    {
        std::lock_guard<std::mutex> lk(g_diag_mtx);
        if (g_diag_busy) return;
        g_diag_busy = true;
    }
    tray_balloon(TR(L"正在生成诊断包…", L"Collecting diagnostics…"),
                 TR(L"几秒后完成，会自动打开所在文件夹。",
                    L"This takes a few seconds; the folder opens when done."));
    std::thread([] {
        HttpReply r = http_call("POST", api_connect_host(), api_port(), "/admin/diag",
                                "{}", 60000);
        std::string p, err;
        if (r.status == 200) {
            const json j = json::parse(r.body, nullptr, false);
            if (!j.is_discarded()) p = j.value("path", "");
        }
        if (p.empty())
            err = r.status ? r.body : r.err;
        {
            std::lock_guard<std::mutex> lk(g_diag_mtx);
            g_diag_path = p;
            g_diag_err = err;
            g_diag_busy = false;
        }
        if (g_tray_hwnd) PostMessageW(g_tray_hwnd, WM_DIAG_DONE, 0, 0);
    }).detach();
}

void diag_done() {
    std::string p, err;
    {
        std::lock_guard<std::mutex> lk(g_diag_mtx);
        p = g_diag_path;
        err = g_diag_err;
    }
    if (p.empty()) {
        msgbox(TR("生成诊断包失败：", "Failed to collect diagnostics: ") + err,
               MB_OK | MB_ICONERROR);
        return;
    }
    select_in_explorer(p);
    tray_balloon(TR(L"诊断包已生成", L"Diagnostics saved"),
                 TR(L"已在资源管理器中选中，把这个 zip 发给开发者即可。",
                    L"Selected in Explorer - send this zip to the developers."));
    g_balloon_select = p;
}


void tray_tip(const TraySnap& s) {
    std::wstring tip = std::wstring(TR(L"Qwenox 启动器 — ", L"Qwenox Launcher — ")) +
                       tray_state_text(s.state);
    if (s.state == "running" && !s.snap.empty()) {
        const std::wstring perf = perf_text(s.snap);
        if (!perf.empty()) tip += L"\n" + perf;
    }
    if (!s.detail.empty() && tip.size() < 90) {
        tip += L"\n";
        tip += to_w(s.detail);
    }
    g_nid.uFlags = NIF_TIP;
    wcsncpy_s(g_nid.szTip, tip.c_str(), _TRUNCATE);
    Shell_NotifyIconW(NIM_MODIFY, &g_nid);
}

void tray_tick() {
    const TraySnap s = tray_snap();
    tray_tip(s);
    if (g_perf_show && g_perf_hwnd) InvalidateRect(g_perf_hwnd, nullptr, TRUE);
    if (s.state == g_tray_state) return;
    g_tray_state = s.state;
    if (s.state == "running")
        tray_balloon(TR(L"服务已就绪", L"Service ready"),
                     TR(L"API 与引擎运行中", L"API and engine are running"));
    else if (s.state == "failed")
        tray_balloon(TR(L"启动失败", L"Start failed"), to_w(s.detail));
    else if (s.state == "exited")
        tray_balloon(TR(L"服务已退出", L"Service exited"), to_w(s.detail));
}

// 用系统默认程序打开：网址 → 浏览器，.log → 记事本兜底
void shell_open(const std::string& target) {
    const INT_PTR r = reinterpret_cast<INT_PTR>(
        ShellExecuteA(nullptr, "open", target.c_str(), nullptr, g_root.c_str(),
                      SW_SHOWNORMAL));
    if (r <= 32 && target.size() > 4 && target.compare(target.size() - 4, 4, ".log") == 0)
        ShellExecuteA(nullptr, "open", "notepad.exe",
                      ("\"" + target + "\"").c_str(), g_root.c_str(), SW_SHOWNORMAL);
}

enum MenuId : UINT {
    kIdOpen = 1, kIdConsole, kIdCopyUrl, kIdPerfMon, kIdAutoStart, kIdNotify,
    kIdStart, kIdStop, kIdLogEngine, kIdLogApi, kIdLogDir, kIdDiag, kIdQuit,
};

void tray_menu() {
    const TraySnap s = tray_snap();
    const bool active = state_active(s.state);
    // 失败/退出但 API 还在：可停（清残留）
    const bool leftover = (s.state == "failed" || s.state == "exited") && s.api_up;
    HMENU m = CreatePopupMenu();
    if (!m) return;
    const std::wstring title =
        L"Qwenox Launcher v" + to_w(QWENOX_VERSION) + L" · " + tray_state_text(s.state);
    AppendMenuW(m, MF_STRING | MF_GRAYED, 0, title.c_str());
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    auto item = [&](UINT id, const wchar_t* text, bool enabled) {
        AppendMenuW(m, MF_STRING | (enabled ? 0 : MF_GRAYED), id, text);
    };
    auto check = [&](UINT id, const wchar_t* text, bool on) {
        AppendMenuW(m, MF_STRING | (on ? MF_CHECKED : 0), id, text);
    };
    item(kIdOpen, TR(L"打开启动器页", L"Open launcher"), true);
    item(kIdConsole, TR(L"打开控制台", L"Open console"), s.api_up);
    item(kIdCopyUrl, TR(L"复制 API 地址", L"Copy API URL"), s.api_up);
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    check(kIdPerfMon, TR(L"性能监控（悬浮窗）", L"Perf monitor (floating)"), g_perf_show);
    check(kIdAutoStart, TR(L"开机自启", L"Launch on startup"), autostart_get());
    check(kIdNotify, TR(L"系统提示", L"Notifications"), g_notify);
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    item(kIdStart, TR(L"启动服务", L"Start service"), !active && !leftover);
    item(kIdStop, TR(L"停止服务", L"Stop service"),
         (active && s.state != "stopping") || leftover);
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    item(kIdLogEngine, TR(L"引擎日志", L"Engine log"), !s.eng_log.empty());
    item(kIdLogApi, TR(L"API 日志", L"API log"),
         !newest_log("api-win-").empty());
    item(kIdLogDir, TR(L"打开日志文件夹", L"Open logs folder"), true);
    item(kIdDiag, TR(L"导出诊断包…", L"Export diagnostics…"), s.api_up);
    AppendMenuW(m, MF_SEPARATOR, 0, nullptr);
    item(kIdQuit, TR(L"退出", L"Quit"), true);
    SetMenuDefaultItem(m, kIdOpen, FALSE);
    POINT pt;
    GetCursorPos(&pt);
    SetForegroundWindow(g_tray_hwnd);  // 让菜单点外部能收起
    TrackPopupMenu(m, TPM_BOTTOMALIGN | TPM_LEFTALIGN | TPM_RIGHTBUTTON, pt.x, pt.y, 0,
                   g_tray_hwnd, nullptr);
    DestroyMenu(m);
}

void tray_command(UINT id) {
    switch (id) {
    case kIdOpen:
        open_browser(g_tray_url);
        break;
    case kIdConsole: {
        const TraySnap s = tray_snap();
        if (s.api_up) open_browser(s.console_url);
        break;
    }
    case kIdCopyUrl: {
        const TraySnap s = tray_snap();
        const std::wstring url = to_w(s.console_url + "v1");
        copy_text(url);
        tray_balloon(TR(L"已复制", L"Copied"), url);
        break;
    }
    case kIdPerfMon:
        perf_toggle();
        break;
    case kIdAutoStart: {
        const bool want = !autostart_get();
        const std::string err = autostart_set(want);
        if (!err.empty())
            msgbox(TR("设置开机自启失败：", "Failed to set autostart: ") + err,
                   MB_OK | MB_ICONERROR);
        else
            tray_balloon(want ? TR(L"已开启开机自启", L"Autostart enabled")
                              : TR(L"已关闭开机自启", L"Autostart disabled"),
                         TR(L"下次登录 Windows 时自动拉起启动器托盘（不弹网页）。",
                            L"The launcher tray will start at next login (no browser)."));
        break;
    }
    case kIdNotify: {
        // 勾选态 ↔ service.conf "# start_win: notify="，即时生效并写回
        const bool want = !g_notify;
        std::string err;
        if (!svcconf::save_conf(g_conf_path, {}, &err, "# start_win: notify=",
                                want ? "on" : "off"))
            msgbox(TR("写入 service.conf 失败：", "Failed to write service.conf: ") + err,
                   MB_OK | MB_ICONERROR);
        else {
            g_notify = want;
            if (want)
                tray_balloon(TR(L"系统提示已开启", L"Notifications enabled"),
                             TR(L"已写回 service.conf。", L"Saved to service.conf."));
        }
        break;
    }
    case kIdStart:
        request_start();
        break;
    case kIdStop:
        request_stop();
        break;
    case kIdLogEngine: {
        const TraySnap s = tray_snap();
        if (!s.eng_log.empty()) shell_open(s.eng_log);
        break;
    }
    case kIdLogApi: {
        const std::string p = newest_log("api-win-");
        if (!p.empty()) shell_open(p);
        break;
    }
    case kIdLogDir:
        shell_open(g_root + "\\logs");
        break;
    case kIdDiag:
        diag_start();
        break;
    case kIdQuit: {
        const TraySnap s = tray_snap();
        if (state_active(s.state) || (s.api_up && (s.state == "failed" || s.state == "exited"))) {
            if (msgbox(TR("服务正在运行。退出启动器会同时停止服务，确定退出？",
                          "The service is running. Quitting the launcher also stops "
                          "it. Quit?"),
                       MB_YESNO | MB_ICONQUESTION) != IDYES)
                return;
            shutdown_launcher(true);
        } else {
            shutdown_launcher(false);
        }
        break;
    }
    }
}

LRESULT CALLBACK tray_wndproc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    if (msg == g_wm_taskbar_created) {
        if (g_nid.hWnd) Shell_NotifyIconW(NIM_ADD, &g_nid);  // 资源管理器重启
        return 0;
    }
    switch (msg) {
    case WM_TRAY:
        if (LOWORD(lp) == WM_LBUTTONDBLCLK) open_browser(g_tray_url);
        else if (LOWORD(lp) == WM_RBUTTONUP) tray_menu();
        else if (LOWORD(lp) == NIN_BALLOONCLICK && !g_balloon_select.empty())
            select_in_explorer(g_balloon_select);
        return 0;
    case WM_TIMER:
        if (wp == kTimerStatus) tray_tick();
        return 0;
    case WM_COMMAND:
        tray_command(LOWORD(wp));
        return 0;
    case WM_DIAG_DONE:
        diag_done();
        return 0;
    case WM_ENDSESSION:
        if (wp) {
            // 注销/关机：本进程拉起的 API 一并结束（其 Job 带走引擎）
            if (g_svc.api_proc) TerminateProcess(g_svc.api_proc, 0);
            Shell_NotifyIconW(NIM_DELETE, &g_nid);
            ExitProcess(0);
        }
        return 0;
    case WM_TRAY_QUIT:
        Shell_NotifyIconW(NIM_DELETE, &g_nid);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wp, lp);
}

bool tray_init() {
    prefs_resolve();
    g_wm_taskbar_created = RegisterWindowMessageW(L"TaskbarCreated");
    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.lpfnWndProc = tray_wndproc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = L"QwenoxLauncherTray";
    if (!RegisterClassExW(&wc) && GetLastError() != ERROR_CLASS_ALREADY_EXISTS)
        return false;
    g_tray_hwnd = CreateWindowExW(0, wc.lpszClassName, L"Qwenox Launcher", 0, 0, 0, 0,
                                  0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
    if (!g_tray_hwnd) return false;
    g_nid.cbSize = sizeof(g_nid);
    g_nid.hWnd = g_tray_hwnd;
    g_nid.uID = 1;
    g_nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP;
    g_nid.uCallbackMessage = WM_TRAY;
    // 托盘/气泡图标按所在屏 DPI 取尺寸（同 tray_win.cpp）
    const UINT dpi = windpi::dpi_for_window(g_tray_hwnd);
    static HICON small_icon = app_icon(windpi::scale(16, dpi));
    static HICON big_icon = app_icon(windpi::scale(32, dpi));  // 通知气泡用
    g_nid.hIcon = small_icon ? small_icon : LoadIconW(nullptr, MAKEINTRESOURCEW(32512));
    g_nid.hBalloonIcon = big_icon;
    wcsncpy_s(g_nid.szTip, L"Qwenox", _TRUNCATE);
    if (!Shell_NotifyIconW(NIM_ADD, &g_nid)) return false;
    g_nid.uVersion = NOTIFYICON_VERSION_4;
    Shell_NotifyIconW(NIM_SETVERSION, &g_nid);
    g_tray_state = "idle";
    SetTimer(g_tray_hwnd, kTimerStatus, 2000, nullptr);
    return true;
}
#endif  // _WIN32

}  // namespace

int main(int argc, char** argv) {
    std::string host;
    long port = 0;
    bool open_ui = true;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--no-browser") {
            open_ui = false;
        } else if (a == "--host" && i + 1 < argc) {
            host = argv[++i];
        } else if (a == "--port" && i + 1 < argc) {
            port = atol(argv[++i]);
        } else if (a == "--console") {
            // Windows：GUI 子系统下调试用；Linux 无效果
#ifdef _WIN32
            AllocConsole();
            freopen("CONOUT$", "w", stdout);
            freopen("CONOUT$", "w", stderr);
#endif
        } else if (a == "-h" || a == "--help") {
            fprintf(stderr,
                    "launcher_settings：Qwenox 网页启动器\n"
                    "用法：launcher_settings [--host H] [--port N] [--no-browser]\n"
                    "默认监听 127.0.0.1:8729（service.conf 的 LAUNCHER_HOST/"
                    "LAUNCHER_PORT 可改）。\n");
            return 0;
        } else {
            fprintf(stderr, "未知参数：%s（--help 查看用法）\n", a.c_str());
            return 2;
        }
    }
    (void)open_ui;

    g_root = exe_dir();
    g_conf_path = g_root + "/service.conf";
#ifdef _WIN32
    SetCurrentDirectoryA(g_root.c_str());
    wsa_init();
    windpi::enable_per_monitor_v2();
#else
    if (chdir(g_root.c_str()) != 0) {
        fprintf(stderr, "无法进入项目目录 %s：%s\n", g_root.c_str(), strerror(errno));
        return 1;
    }
#endif
    reload_conf();

    // 监听地址：命令行 > 环境变量 > service.conf > 默认
    if (host.empty()) {
        if (const char* e = getenv("LAUNCHER_HOST")) {
            host = e;
        } else {
            std::lock_guard<std::mutex> lk(g_conf_mtx);
            host = svcconf::cfg(g_conf, "LAUNCHER_HOST", "127.0.0.1");
        }
    }
    if (port <= 0) {
        if (const char* e = getenv("LAUNCHER_PORT")) {
            port = atol(e);
        } else {
            std::lock_guard<std::mutex> lk(g_conf_mtx);
            long p = 8729;
            svcconf::parse_int(svcconf::cfg(g_conf, "LAUNCHER_PORT", "8729"), &p);
            port = p;
        }
    }
    if (port < 1 || port > 65535) {
        fprintf(stderr, "LAUNCHER_PORT 无效：%ld\n", port);
        return 2;
    }
    g_listen_port = static_cast<int>(port);

    ensure_dir("logs");
    g_log_path = "logs/launcher-settings.log";

#ifdef _WIN32
    // 防多开：已在运行时直接再开一个浏览器页面
    CreateMutexW(nullptr, FALSE, L"Local\\Qwenox-launcher-settings");
    if (GetLastError() == ERROR_ALREADY_EXISTS) {
        open_browser("http://" + (host == "0.0.0.0" ? "127.0.0.1" : host) + ":" +
                     std::to_string(port) + "/");
        return 0;
    }
#endif

    // 启动时接管已在运行的服务（Linux：pidfile；Windows：探测 API）
#ifdef _WIN32
    if (poll_engine_from_api()) {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_svc.api_external = true;  // 不是本进程拉起的，退出时不结束它
        const std::string es = g_svc.eng_state;
        if (es == "running") {
            set_state_locked("running", "接管了已在运行的服务");
        } else if (es == "starting") {
            set_state_locked("waiting_engine", "接管了正在启动的服务");
        }
        if (state_active(g_svc.state)) {
            log_event("检测到已在运行的 API/引擎，接管监控");
            const uint64_t gen = g_gen;
            std::thread(monitor_windows, gen, time(nullptr) + 90).detach();
        }
    }
#else
    if (FILE* f = fopen(pidfile_path().c_str(), "rb")) {
        char buf[1024] = {0};
        const size_t n = fread(buf, 1, sizeof(buf) - 1, f);
        fclose(f);
        long pid = 0;
        char logp[900] = {0};
        if (n > 0 && sscanf(buf, "%ld %899[^\n]", &pid, logp) >= 1 && pid > 0 &&
            kill(static_cast<pid_t>(pid), 0) == 0 &&
            cmdline_is_launch_script(static_cast<pid_t>(pid))) {
            std::lock_guard<std::mutex> lk(g_mtx);
            g_svc.child_pid = static_cast<pid_t>(pid);
            g_svc.run_log = logp;
            const bool eng = tcp_connect(engine_connect_host(), engine_port(), 500);
            const bool api = tcp_connect(api_connect_host(), api_port(), 500);
            g_svc.api_up = api;
            g_svc.eng_state = eng ? "running" : "starting";
            set_state_locked(api ? "running" : (eng ? "waiting_api" : "waiting_engine"),
                             "接管了已在运行的服务");
            log_event("接管已在运行的服务脚本（pid " + std::to_string(pid) + "）");
            std::thread(monitor_linux, g_gen).detach();
        } else {
            clear_pidfile();  // 陈旧 pidfile
        }
    }
#endif

    g_srv = &g_server;
    g_server.on("GET", "/api/config", handle_config_get);
    g_server.on("POST", "/api/config", handle_config_post);
    g_server.on("GET", "/api/status", handle_status);
    g_server.on("POST", "/api/start", handle_start);
    g_server.on("POST", "/api/stop", handle_stop);
    g_server.on("GET", "/api/logs", handle_logs);
    g_server.on("POST", "/api/shutdown", handle_shutdown);
    g_server.on("GET", "/api/files", handle_files);
    g_server.on("POST", "/api/check", handle_check);
    g_server.on("GET", "/api/profiles", handle_profiles_get);
    g_server.on("POST", "/api/profiles", handle_profiles_post);
    g_server.on("POST", "/api/profiles/delete", handle_profiles_delete);
    g_server.on("GET", "/media/panel-bg.png", handle_media);
    g_server.on("GET", "/media/qwen-charactor-alpha.png", handle_media);
    g_server.on("GET", "/", handle_static);
    g_server.set_not_found([](const http::Request& q, http::Response* r, http::Stream*) {
        if (q.path.compare(0, 5, "/api/") == 0) {
            r->status = 404;
            r->body = http::error_json("unknown endpoint", "not_found", "not_found");
            return;
        }
        handle_static(q, r, nullptr);
    });

    const std::string hp = host + ":" + std::to_string(port);
    std::string lerr;
    if (!g_server.listen(hp, &lerr)) {
        const std::string msg = "launcher_settings 监听 " + hp + " 失败：" + lerr +
                                "\n（已在运行？浏览器打开 http://127.0.0.1:" +
                                std::to_string(port) + "/）";
#ifdef _WIN32
        msgbox(msg);
#else
        print_line(msg);
#endif
        return 1;
    }
    const std::string url = "http://" + (host == "0.0.0.0" ? std::string("127.0.0.1")
                                                           : host) +
                            ":" + std::to_string(port) + "/";
    log_event("launcher_settings 就绪：" + url);
    print_line("launcher_settings: " + url);
#ifdef _WIN32
    // 托盘程序：主线程消息循环，HTTP 服务在 worker 线程（同 qwenox-win 的结构）
    register_app_id();  // toast 通知显示 "Qwenox"
    if (!tray_init()) {
        log_event("托盘创建失败，退回无托盘模式");
        if (open_ui) open_browser(url);
        g_server.run();
        return 0;
    }
    g_tray_url = url;
    if (open_ui) open_browser(url);
    std::thread([] { g_server.run(); }).detach();
    MSG msg;
    while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }
    if (g_nid.hWnd) Shell_NotifyIconW(NIM_DELETE, &g_nid);
    return 0;
#else
    if (open_ui && !open_browser(url)) warn_headless(url);
    g_server.run();
    return 0;
#endif
}
