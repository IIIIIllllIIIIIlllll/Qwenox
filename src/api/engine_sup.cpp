// engine_sup.cpp — 见 engine_sup.h。启停语义与 start_win.exe 一致：
// 停止 = 直接 TerminateProcess（与 Linux 脚本 kill 进程组等价），引擎不
// 需要控制协议动词。子进程进 KILL_ON_JOB_CLOSE 的 Job：API 进程无论怎样
// 退出，引擎都会随之结束，不留占着显存的孤儿进程。
#include "engine_sup.h"

#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include "../engine_net.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace enginesup {
namespace {

std::mutex g_mtx;
Status g_st;
#ifdef _WIN32  // 仅 Windows 监管线程/stop() 使用；Linux 下未用会触发 -Werror=unused-variable
bool g_stopping = false;  // stop() 主动停止：监管线程把退出记为 stopped 而非 exited
#endif

// 探测的两段超时（connect、读回复）各 500ms：阻塞 connect 对黑洞地址（防火墙
// 丢包的远程 IP）会挂 ~20s，某些机器环回 RST 被过滤驱动延迟 ~2s；status() 由托盘
// 探测 worker 每 2s 调用、控制台接口也会调，最坏 ~1s 一次。
constexpr int kProbeMs = 500;

#ifdef _WIN32
using sock_t = SOCKET;
constexpr sock_t kBadSock = INVALID_SOCKET;
#else
using sock_t = int;
constexpr sock_t kBadSock = -1;
#endif
#ifdef MSG_NOSIGNAL
constexpr int kNoSigpipe = MSG_NOSIGNAL;  // 对端先关时 send 不触发 SIGPIPE
#else
constexpr int kNoSigpipe = 0;
#endif

void close_sock(sock_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}

void set_nonblocking(sock_t s) {
#ifdef _WIN32
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
#else
    fcntl(s, F_SETFL, fcntl(s, F_GETFL, 0) | O_NONBLOCK);
#endif
}

bool connect_pending() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
#else
    return errno == EINPROGRESS;
#endif
}

bool would_block() {
#ifdef _WIN32
    return WSAGetLastError() == WSAEWOULDBLOCK;
#else
    return errno == EAGAIN || errno == EWOULDBLOCK;
#endif
}

// 等 s 可写（connect 完成）或可读，最多 ms 毫秒。
bool wait_sock(sock_t s, bool for_write, int ms) {
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(s, &fds);
    timeval tv{ms / 1000, (ms % 1000) * 1000};
    return select(static_cast<int>(s) + 1, for_write ? nullptr : &fds,
                  for_write ? &fds : nullptr, nullptr, &tv) > 0;
}

#ifdef _WIN32

HANDLE g_proc = nullptr;
HANDLE g_job = nullptr;
std::string g_host;  // 本次启动的引擎地址（监管线程探测用）
int g_port = 0;

bool file_exists(const std::string& path) {
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

bool dir_exists(const std::string& path) {
    const DWORD a = GetFileAttributesA(path.c_str());
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
}

// 两个路径是否同一文件（卷序列号 + 文件 ID；打不开时退回字符串比较）。
bool same_file(const std::string& a, const std::string& b) {
    if (a == b) return true;
    BY_HANDLE_FILE_INFORMATION ia{}, ib{};
    auto info = [](const std::string& p, BY_HANDLE_FILE_INFORMATION* out) {
        HANDLE h = CreateFileA(p.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) return false;
        const bool ok = GetFileInformationByHandle(h, out) != 0;
        CloseHandle(h);
        return ok;
    };
    if (!info(a, &ia) || !info(b, &ib)) return false;
    return ia.dwVolumeSerialNumber == ib.dwVolumeSerialNumber &&
           ia.nFileIndexHigh == ib.nFileIndexHigh && ia.nFileIndexLow == ib.nFileIndexLow;
}

// 子进程 stdout/stderr → 日志文件（无控制台透传：API 独立运行时引擎输出只落盘）。
DWORD WINAPI tee_thread(LPVOID param) {
    HANDLE pipe_read = static_cast<HANDLE*>(param)[0];
    FILE* log = static_cast<FILE**>(param)[1];
    char buf[65536];
    DWORD n = 0;
    while (ReadFile(pipe_read, buf, sizeof(buf), &n, nullptr) && n > 0)
        if (log) {
            fwrite(buf, 1, n, log);
            fflush(log);
        }
    if (log) fclose(log);
    CloseHandle(pipe_read);
    delete[] static_cast<void**>(param);
    return 0;
}

// 监管线程：starting →（心跳应答）→ running →（进程退出）→ exited/stopped；
// starting 超过 START_TIMEOUT 秒则杀掉并记 exited。
DWORD WINAPI monitor_thread(LPVOID) {
    long timeout = 1800;
    if (const char* e = getenv("START_TIMEOUT")) {
        char* end = nullptr;
        const long n = strtol(e, &end, 10);
        if (end && !*end && n >= 30 && n <= 86400) timeout = n;
    }
    const ULONGLONG deadline = GetTickCount64() + static_cast<ULONGLONG>(timeout) * 1000;
    for (;;) {
        if (WaitForSingleObject(g_proc, 0) == WAIT_OBJECT_0) break;  // 提前退出
        if (probe(g_host, g_port) == Probe::alive) {
            std::lock_guard<std::mutex> lk(g_mtx);
            if (g_st.state == "starting") g_st.state = "running";
            break;
        }
        if (GetTickCount64() > deadline) {
            TerminateProcess(g_proc, 1);
            std::lock_guard<std::mutex> lk(g_mtx);
            g_st.detail = "start timeout (" + std::to_string(timeout) + "s)";
            break;
        }
        Sleep(1000);
    }
    WaitForSingleObject(g_proc, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(g_proc, &code);
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        g_st.exit_code = code;
        g_st.pid = 0;
        // 主动 stop() → stopped；提前退出 / 超时杀掉 / 运行中意外退出 → exited
        g_st.state = g_stopping ? "stopped" : "exited";
        g_stopping = false;
        CloseHandle(g_proc);
        g_proc = nullptr;
    }
    return 0;
}

#endif  // _WIN32

}  // namespace

bool supported() {
#ifdef _WIN32
    return true;
#else
    return false;
#endif
}

Probe probe(const std::string& host, int port, std::string* snap) {
    if (snap) snap->clear();
    const sock_t s = socket(AF_INET, SOCK_STREAM, 0);
    if (s == kBadSock) return Probe::down;
    sockaddr_in addr{};
    if (!engine_net::address(engine_net::connect_host(host), port, &addr)) {
        close_sock(s);
        return Probe::down;
    }
    set_nonblocking(s);
    bool ok = connect(s, reinterpret_cast<sockaddr*>(&addr), sizeof addr) == 0;
    if (!ok && connect_pending() && wait_sock(s, true, kProbeMs)) {
        int soerr = 0;
        socklen_t sl = sizeof soerr;
        getsockopt(s, SOL_SOCKET, SO_ERROR, reinterpret_cast<char*>(&soerr), &sl);
        ok = soerr == 0;
    }
    if (!ok) {
        close_sock(s);
        return Probe::down;
    }
    // 已连上：从这里起任何失败（发不出、超时、对端关闭、回复不对）都算 foreign
    std::string line;
    bool got = false;
    if (send(s, "SNAP\n", 5, kNoSigpipe) == 5) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kProbeMs);
        char buf[256];
        while (line.size() < 1024) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                                  deadline - std::chrono::steady_clock::now()).count();
            if (left <= 0 || !wait_sock(s, false, static_cast<int>(left))) break;
            const int n = static_cast<int>(recv(s, buf, sizeof buf, 0));
            if (n < 0 && would_block()) continue;
            if (n <= 0) break;
            line.append(buf, static_cast<size_t>(n));
            const size_t nl = line.find('\n');
            if (nl != std::string::npos) {
                line.resize(nl);
                got = true;
                break;
            }
        }
    }
    close_sock(s);
    // SNAP 的应答格式：P ts_ms state req pp_toks done total tg_inst tg_avg accept_pct
    if (!got || line.compare(0, 2, "P ") != 0) return Probe::foreign;
    if (snap) *snap = line;
    return Probe::alive;
}

Status status(const std::string& host, int port, std::string* snap) {
    if (snap) snap->clear();
    Status s;
    {
        std::lock_guard<std::mutex> lk(g_mtx);
        s = g_st;  // 只拷贝快照：端口探测在锁外做（connect 可能慢，不能拿着锁）
    }
    if (s.state == "stopped" || s.state == "exited") {
        const Probe p = probe(host, port, snap);
        if (p == Probe::alive) {
            // 引擎在应答但不是本进程拉起的：外部引擎（start_win.exe / 脚本）
            s.state = "running";
            s.external = true;
            s.pid = 0;
        } else if (p == Probe::foreign) {
            s.port_foreign = true;
            if (!s.detail.empty()) s.detail += "; ";
            s.detail += "engine port " + std::to_string(port) +
                        " is in use but does not answer the engine protocol";
        }
    } else if (s.state == "running" && snap) {
        // 本进程拉起的引擎：状态由监管线程判定，这里只顺带取 SNAP（失败留空串）
        probe(host, port, snap);
    }
    return s;
}

bool start(const std::string& root, const svcconf::Conf& conf, std::string* err) {
#ifdef _WIN32
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_st.state == "starting" || (g_st.state == "running" && !g_st.external)) {
        *err = "engine already running";
        return false;
    }
    // 与 launch_win.cpp 相同的取值与默认（env > conf > 内置）
    const std::string model_dir = svcconf::cfg(conf, "MODEL_DIR", "models");
    // weights=gguf（设置页/网页选 GGUF）：引擎直接吃 GGUF 首分片（纯 GGUF 启动，
    // 引擎侧 gguf_base_setup），MTP sidecar 走 QWENOX_GGUF_MTP（空 = 不投机），
    // 视觉塔 GGUF_VISION_FILE（mmproj *.gguf 或 *-vision.hgn 均可）。
    const auto wv = conf.find(kWeightsVerKey);
    const bool gguf = wv != conf.end() && wv->second == "gguf";
    const std::string model_file =
        gguf ? svcconf::cfg(conf, "GGUF_FILE", "")
             : svcconf::cfg(conf, "MODEL_FILE", model_dir + "\\heretic.hgn");
    const std::string ngram_file = gguf ? model_file : svcconf::cfg(conf, "NGRAM_FILE", model_file);
    const std::string mtp_file =
        gguf ? svcconf::cfg_optional(conf, "GGUF_MTP_FILE", "")
             : svcconf::cfg_optional(conf, "MTP_FILE", model_dir + "\\heretic-mtp.hgn");
    const std::string vision_file =
        gguf ? svcconf::cfg_optional(conf, "GGUF_VISION_FILE", "")
             : svcconf::cfg_optional(conf, "VISION_FILE", model_dir + "\\heretic-vision.hgn");
    std::string overlay_file = gguf ? "" : svcconf::cfg_optional(conf, "OVERLAY_FILE", "");
    if (gguf && model_file.empty()) {
        *err = "weights=gguf but GGUF_FILE is not set (first shard *-00001-of-0000N.gguf)";
        return false;
    }
    const std::string engine_host = svcconf::cfg(conf, "ENGINE_HOST", engine_net::kDefaultHost);
    long engine_port = 8730, max_context = 262144, mtp_gamma = 0, kvsnap_max_gb = 20,
         kvsnap_min = 4096, kvsnap_park_min = 4096, park = 1, park_wait_ms = 5000,
         park_max_attempts = 8, kv_reserve_decode = 4096, rckpt_max = 8, kv_paged = 1,
         kv_pool_tokens = 0, parallel = 1, max_images = 8, prefill_chunk = 0,
         rope_original = 262144;
    svcconf::parse_int(svcconf::cfg(conf, "ENGINE_PORT", "8730"), &engine_port);
    svcconf::parse_int(svcconf::cfg(conf, "MAX_CONTEXT", "262144"), &max_context);
    svcconf::parse_int(svcconf::cfg(conf, "MTP_GAMMA", "0"), &mtp_gamma);
    svcconf::parse_int(svcconf::cfg(conf, "KVSNAP_MAX_GB", "20"), &kvsnap_max_gb);
    const std::string kvsnap_min_s = svcconf::cfg(conf, "KVSNAP_MIN", "4096");
    const std::string kvsnap_park_min_s =
        svcconf::cfg(conf, "KVSNAP_PARK_MIN", kvsnap_min_s);
    svcconf::parse_int(kvsnap_min_s, &kvsnap_min);
    svcconf::parse_int(kvsnap_park_min_s, &kvsnap_park_min);
    svcconf::parse_int(svcconf::cfg(conf, "PARK", "1"), &park);
    svcconf::parse_int(svcconf::cfg(conf, "PARK_WAIT_MS", "5000"), &park_wait_ms);
    svcconf::parse_int(svcconf::cfg(conf, "PARK_MAX_ATTEMPTS", "8"), &park_max_attempts);
    svcconf::parse_int(svcconf::cfg(conf, "KV_RESERVE_DECODE", "4096"), &kv_reserve_decode);
    svcconf::parse_int(svcconf::cfg(conf, "RCKPT_MAX", "8"), &rckpt_max);
    svcconf::parse_int(svcconf::cfg(conf, "KV_PAGED", "1"), &kv_paged);
    svcconf::parse_int(svcconf::cfg(conf, "KV_POOL_TOKENS", "0"), &kv_pool_tokens);
    svcconf::parse_int(svcconf::cfg(conf, "PARALLEL", "1"), &parallel);
    svcconf::parse_int(svcconf::cfg(conf, "MAX_IMAGES", "8"), &max_images);
    svcconf::parse_int(svcconf::cfg(conf, "PREFILL_CHUNK", "0"), &prefill_chunk);
    svcconf::parse_int(svcconf::cfg(conf, "ROPE_ORIGINAL_CTX", "262144"), &rope_original);
    const std::string rope_factor_s = svcconf::cfg(conf, "ROPE_FACTOR", "1");
    const std::string rope_fast_s = svcconf::cfg(conf, "ROPE_BETA_FAST", "32");
    const std::string rope_slow_s = svcconf::cfg(conf, "ROPE_BETA_SLOW", "1");
    const std::string rope_attn_s = svcconf::cfg(conf, "ROPE_ATTN_SCALE", "0");

    // 启动前文件检查（同 launch_win.cpp 启动检查；overlay 缺失只警告并跳过）
    const std::string exe = root + "\\build\\qwenox-engine-win.exe";
    if (!file_exists(exe)) {
        *err = "missing " + exe;
        return false;
    }
    if (!file_exists(model_file)) {
        *err = "model not found: " + model_file;
        return false;
    }
    if (!file_exists(ngram_file)) {
        *err = "n-gram table not found: " + ngram_file;
        return false;
    }
    if (!mtp_file.empty() && !file_exists(mtp_file)) {
        *err = "MTP weights not found: " + mtp_file;
        return false;
    }
    if (!vision_file.empty() && !file_exists(vision_file)) {
        *err = "vision tower not found: " + vision_file;
        return false;
    }
    if (!overlay_file.empty() && !file_exists(overlay_file)) overlay_file.clear();
    if (!dir_exists(root + "\\build\\rocblas\\library") ||
        !dir_exists(root + "\\build\\hipblaslt\\library")) {
        *err = "missing rocBLAS/hipBLASLt kernel db (build\\*\\library)";
        return false;
    }

    // 生产选项与 launch_win.cpp / Linux start.sh 一致：设在自身进程，子进程继承
    const char* flags[] = {
        "QWENOX_QSA_KV_BF16", "QWENOX_QSA_WMMA", "QWENOX_QSA_WMMA_BTV",
        "QWENOX_MOE_LT", "QWENOX_MOE_LT_BF16", "QWENOX_GR_BF16",
        "QWENOX_GDN_STREAM", "QWENOX_GDN_WAVE", "QWENOX_NOWARMUP",
        "QWENOX_GEMM_WMMA", "QWENOX_GDN_FUSED",
        "QWENOX_INDEX_FUSED2", "QWENOX_PP_MOE_OUT", "QWENOX_INDEX_STREAM_SELECT",
    };
    for (const char* f : flags) SetEnvironmentVariableA(f, "1");
    SetEnvironmentVariableA("QWENOX_KVSNAP", kvsnap_max_gb ? "1" : "0");
    SetEnvironmentVariableA("QWENOX_KVSNAP_MAX_GB", std::to_string(kvsnap_max_gb).c_str());
    SetEnvironmentVariableA("QWENOX_KVSNAP_MIN", std::to_string(kvsnap_min).c_str());
    SetEnvironmentVariableA("QWENOX_KVSNAP_PARK_MIN", std::to_string(kvsnap_park_min).c_str());
    SetEnvironmentVariableA("QWENOX_PARK", park ? "1" : "0");
    SetEnvironmentVariableA("QWENOX_PARK_WAIT_MS", std::to_string(park_wait_ms).c_str());
    SetEnvironmentVariableA("QWENOX_PARK_MAX_ATTEMPTS", std::to_string(park_max_attempts).c_str());
    SetEnvironmentVariableA("QWENOX_KV_RESERVE_DECODE", std::to_string(kv_reserve_decode).c_str());
    // KVSNAP_DIR（见 service.conf）：缓存放在它下面的 qwenox-kvsnap 子目录；
    // 留空时不动 QWENOX_KVSNAP_DIR（引擎默认 data\kvsnap，外部指定照常生效）
    {
        std::string kd = svcconf::cfg(conf, "KVSNAP_DIR", "");
        while (!kd.empty() && (kd.back() == '\\' || kd.back() == '/')) kd.pop_back();
        if (!kd.empty())
            SetEnvironmentVariableA("QWENOX_KVSNAP_DIR", (kd + "\\qwenox-kvsnap").c_str());
    }
    SetEnvironmentVariableA("QWENOX_RCKPT_MAX", std::to_string(rckpt_max).c_str());
    // 0 = 引擎按模式自选：不设（并删掉外部残留）
    SetEnvironmentVariableA("QWENOX_SPEC_GAMMA",
                            mtp_gamma ? std::to_string(mtp_gamma).c_str() : nullptr);
    SetEnvironmentVariableA("QWENOX_KV_PAGED", kv_paged ? "1" : nullptr);
    SetEnvironmentVariableA("QWENOX_KV_POOL_TOKENS",
                            kv_paged && kv_pool_tokens
                                ? std::to_string(kv_pool_tokens).c_str()
                                : nullptr);
    SetEnvironmentVariableA("QWENOX_PARALLEL", std::to_string(parallel).c_str());
    // VIZ=1（见 service.conf）：运行时可视化埋点，GET /viz 取数据
    SetEnvironmentVariableA("QWENOX_VIZ",
                            svcconf::cfg(conf, "VIZ", "0") == "1" ? "1" : nullptr);
    // MOE_A4=1–3（见 service.conf）：prefill MoE 4-bit 激活，其它值 = 关
    {
        const std::string a4 = svcconf::cfg(conf, "MOE_A4", "0");
        const bool on = a4 == "1" || a4 == "2" || a4 == "3";
        SetEnvironmentVariableA("QWENOX_MOE_A4", on ? a4.c_str() : nullptr);
    }
    // MAX_IMAGES 同时被本进程（API 的 vision::max_images）读取：
    // SetEnvironmentVariableA 只改 Win32 环境块，本进程 CRT 的 getenv 看不到，
    // 必须用 _putenv（CRT 缓存与 Win32 块一起更新，子进程照常继承）。
    _putenv((std::string("QWENOX_API_MAX_IMAGES=") + std::to_string(max_images)).c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_FACTOR", rope_factor_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_ORIGINAL_CTX", std::to_string(rope_original).c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_BETA_FAST", rope_fast_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_BETA_SLOW", rope_slow_s.c_str());
    SetEnvironmentVariableA("QWENOX_ROPE_ATTN_SCALE", rope_attn_s.c_str());
    if (prefill_chunk)
        SetEnvironmentVariableA("QWENOX_PREFILL_CHUNK", std::to_string(prefill_chunk).c_str());
    // GGUF：MTP sidecar 不是位置参数而是 QWENOX_GGUF_MTP。conf 里没有
    // GGUF_MTP_FILE 键 = 不设（引擎在分片目录自动找唯一的 mtp-*.gguf）；键显式
    // 置空 = "none"（引擎不投机；Windows 环境变量无法可靠地传空串）。
    const bool mtp_key = getenv("GGUF_MTP_FILE") || conf.count("GGUF_MTP_FILE");
    SetEnvironmentVariableA("QWENOX_GGUF_MTP",
                            !gguf || !mtp_key ? nullptr
                            : mtp_file.empty() ? "none"
                                               : mtp_file.c_str());
    for (const char* k : {"QWENOX_GGUF", "QWENOX_GGUF_DENSE", "QWENOX_GGUF_PLE", "QWENOX_GGUF_EXPERTS"})
        SetEnvironmentVariableA(k, nullptr);

    std::vector<std::string> args = {model_file};
    if (!overlay_file.empty()) args.push_back(overlay_file);
    if (!same_file(ngram_file, model_file)) args.push_back(ngram_file);
    if (!gguf && !mtp_file.empty()) args.push_back(mtp_file);
    args.insert(args.end(),
                {"--serve", "--host", engine_host, "--port", std::to_string(engine_port),
                 "--maxctx", std::to_string(max_context)});
    if (!vision_file.empty())
        args.insert(args.end(), {"--vision-tower", vision_file});

    CreateDirectoryA((root + "\\logs").c_str(), nullptr);
    SYSTEMTIME st;
    GetLocalTime(&st);
    char stamp[32];
    snprintf(stamp, sizeof(stamp), "%04d%02d%02d-%02d%02d%02d", st.wYear, st.wMonth,
             st.wDay, st.wHour, st.wMinute, st.wSecond);
    const std::string log_path = root + "\\logs\\engine-api-" + stamp + ".log";

    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE pipe_read = nullptr, pipe_write = nullptr;
    if (!CreatePipe(&pipe_read, &pipe_write, &sa, 1u << 20)) {
        *err = "CreatePipe failed";
        return false;
    }
    SetHandleInformation(pipe_read, HANDLE_FLAG_INHERIT, 0);
    HANDLE in = CreateFileA("NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                            OPEN_EXISTING, 0, nullptr);

    std::string cmd = "\"" + exe + "\"";
    for (const std::string& a : args) cmd += " \"" + a + "\"";
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdOutput = pipe_write;
    si.hStdError = pipe_write;
    si.hStdInput = in == INVALID_HANDLE_VALUE ? nullptr : in;
    PROCESS_INFORMATION pi{};
    std::vector<char> cmdline(cmd.begin(), cmd.end());
    cmdline.push_back('\0');
    const BOOL ok = CreateProcessA(nullptr, cmdline.data(), nullptr, nullptr, TRUE,
                                   CREATE_NO_WINDOW, nullptr, root.c_str(), &si, &pi);
    const DWORD create_err = GetLastError();
    if (in != INVALID_HANDLE_VALUE) CloseHandle(in);
    CloseHandle(pipe_write);
    if (!ok) {
        CloseHandle(pipe_read);
        *err = "CreateProcess failed (" + exe + "), error " + std::to_string(create_err);
        return false;
    }
    // API 进程退出（含被任务管理器结束）时引擎随之结束
    if (!g_job) {
        g_job = CreateJobObjectA(nullptr, nullptr);
        if (g_job) {
            JOBOBJECT_EXTENDED_LIMIT_INFORMATION li{};
            li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if (!SetInformationJobObject(g_job, JobObjectExtendedLimitInformation, &li,
                                         sizeof(li))) {
                CloseHandle(g_job);
                g_job = nullptr;
            }
        }
    }
    if (g_job) AssignProcessToJobObject(g_job, pi.hProcess);  // 失败也不影响运行
    CloseHandle(pi.hThread);

    FILE* log = fopen(log_path.c_str(), "wb");
    if (!log)
        fprintf(stderr, "qwenox-api: warning: cannot write engine log %s\n", log_path.c_str());
    void** tee_args = new void*[2]{pipe_read, log};
    HANDLE t = CreateThread(nullptr, 0, tee_thread, tee_args, 0, nullptr);
    if (t) CloseHandle(t);

    g_proc = pi.hProcess;
    g_host = engine_host;
    g_port = static_cast<int>(engine_port);
    g_st = Status{};
    g_st.state = "starting";
    g_st.pid = static_cast<int>(pi.dwProcessId);
    g_st.log = log_path;
    g_stopping = false;
    HANDLE m = CreateThread(nullptr, 0, monitor_thread, nullptr, 0, nullptr);
    if (m) CloseHandle(m);
    fprintf(stderr, "qwenox-api: engine starting (pid %lu), log: %s\n",
            static_cast<unsigned long>(pi.dwProcessId), log_path.c_str());
    return true;
#else
    (void)root;
    (void)conf;
    *err = "engine start/stop is only supported on Windows";
    return false;
#endif
}

bool stop(std::string* err) {
#ifdef _WIN32
    std::lock_guard<std::mutex> lk(g_mtx);
    // 外部引擎（端口在听但非本进程拉起）由调用方先经 status() 判定并拒绝
    if (!g_proc) {
        *err = "engine not running";
        return false;
    }
    g_stopping = true;
    TerminateProcess(g_proc, 1);  // 锁内终止：避免与监管线程的 CloseHandle 竞态
    return true;                  // 状态由监管线程翻转为 stopped（前端轮询可见）
#else
    *err = "engine start/stop is only supported on Windows";
    return false;
#endif
}

}  // namespace enginesup
