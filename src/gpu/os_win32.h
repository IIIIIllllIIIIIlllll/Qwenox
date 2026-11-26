// os_win32.h — Windows 兼容层：为 qwenox.cpp / hgn.h 提供 POSIX 等价物。
// 仅 _WIN32 下包含。设计原则：qwenox.cpp 主体代码基本不动，差异全部收敛到这里。
#pragma once
#ifndef _WIN32
#error "os_win32.h 仅供 Windows 构建包含"
#endif

#ifndef NOMINMAX
#define NOMINMAX  // 屏蔽 windows.h 的 min/max 宏（std::min/size_t> 会被污染）
#endif
#include <winsock2.h>  // 必须先于 windows.h
#include <ws2tcpip.h>
#include <windows.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <direct.h>
#include <fcntl.h>
#include <io.h>
#include <string>
#include <sys/stat.h>
#include <time.h>

#pragma comment(lib, "ws2_32")

// ---------- madvise（纯提示，Windows 降级为 no-op；快路径走 IOCP 批量读） ----------
#ifndef MADV_WILLNEED
#define MADV_WILLNEED 0
#endif
static inline int madvise(void*, size_t, int) { return 0; }
// 批量异步预取映射冷页（单次系统调用并发下发 16 行 PLE NVMe 缺页，消除 QD=1 串行硬缺页）
static inline void os_prefetch_ple(const uint8_t* const* addrs, size_t n, size_t rb) {
  struct Win32VmRange {
    void* VirtualAddress;
    SIZE_T NumberOfBytes;
  };
  using PfnPrefetchVm = BOOL(WINAPI*)(HANDLE, ULONG_PTR, Win32VmRange*, ULONG);
  static const PfnPrefetchVm pfn = []() -> PfnPrefetchVm {
    const char* e = getenv("QWENOX_WIN_PLE_PREFETCH");
    if (e && !strcmp(e, "0")) return nullptr;
    HMODULE k32 = GetModuleHandleA("kernel32.dll");
    return k32 ? (PfnPrefetchVm)GetProcAddress(k32, "PrefetchVirtualMemory") : nullptr;
  }();
  if (!pfn || !n || !rb) return;
  // 热页免预取：探测首行首字节，页在 working set（或 standby 软缺页）时
  // 访问极快（<<100us）——此时 PrefetchVirtualMemory 对常驻页只是空走页表，
  // 实测反而耗 ~0.3-0.4ms/tok（OPT_ple_pf0 热页快 1.5% 的实证），直接跳过。
  // 首行硬缺页（Phison ~0.3-1ms）才为其余行并发预取：与旧串行路径同价的
  // 一次缺页，换掉余下 15 行的串行硬缺页。
  LARGE_INTEGER pt0, pt1, fr;
  QueryPerformanceCounter(&pt0);
  (void)*(const volatile uint8_t*)addrs[0];
  QueryPerformanceCounter(&pt1);
  QueryPerformanceFrequency(&fr);
  if ((double)(pt1.QuadPart - pt0.QuadPart) * 1e6 / (double)fr.QuadPart < 100.0)
    return;
  Win32VmRange vr[64];
  ULONG_PTR cnt = 0;
  for (size_t i = 0; i < n && cnt < 64; i++) {
    uintptr_t p0 = (uintptr_t)addrs[i] & ~(uintptr_t)4095;
    uintptr_t p1 = ((uintptr_t)addrs[i] + rb - 1) & ~(uintptr_t)4095;
    SIZE_T bytes = (SIZE_T)(p1 - p0 + 4096);
    bool dup = false;
    for (ULONG_PTR j = 0; j < cnt; j++) {
      if ((uintptr_t)vr[j].VirtualAddress == p0 && vr[j].NumberOfBytes >= bytes) {
        dup = true;
        break;
      }
    }
    if (!dup) vr[cnt++] = {(void*)p0, bytes};
  }
  if (cnt) pfn(GetCurrentProcess(), cnt, vr, 0);
}
#define isatty _isatty
#define mkdir(p, m) _mkdir(p)

// ---------- 杂项 ----------
#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif
typedef SSIZE_T ssize_t;
typedef intptr_t sockfd_t;  // SOCKET 在 Win64 是 64 位，不能截断成 int

static inline size_t strnlen_(const char* s, size_t n) {
  size_t i = 0;
  while (i < n && s[i]) i++;
  return i;
}
#define strnlen strnlen_

// clock_gettime(CLOCK_REALTIME) / localtime_r
#ifndef CLOCK_REALTIME
#define CLOCK_REALTIME 0
#endif
static inline int clock_gettime(int, timespec* ts) {
  FILETIME ft;
  GetSystemTimePreciseAsFileTime(&ft);
  uint64_t t = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
  t -= 116444736000000000ull;  // 1601 -> 1970
  ts->tv_sec = (time_t)(t / 10000000ull);
  ts->tv_nsec = (long)((t % 10000000ull) * 100);
  return 0;
}
static inline struct tm* localtime_r(const time_t* t, struct tm* out) {
  return localtime_s(out, t) == 0 ? out : nullptr;
}

// ---------- Winsock ----------
static inline bool wsa_init() {
  WSADATA w;
  return WSAStartup(MAKEWORD(2, 2), &w) == 0;
}
static inline sockfd_t sock_tcp() {
  SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  return s == INVALID_SOCKET ? (sockfd_t)-1 : (sockfd_t)s;
}
static inline int sock_close(sockfd_t fd) { return closesocket((SOCKET)fd); }
static inline int sock_shutdown(sockfd_t fd) { return shutdown((SOCKET)fd, SD_BOTH); }
// flags 仅支持 0 与 MSG_DONTWAIT（后者通过临时切换非阻塞模式实现）
static inline ssize_t sock_recv(sockfd_t fd, void* buf, size_t n, bool dontwait) {
  SOCKET s = (SOCKET)fd;
  if (dontwait) {
    u_long nb = 1;
    ioctlsocket(s, FIONBIO, &nb);
    int r = recv(s, (char*)buf, (int)n, 0);
    nb = 0;
    ioctlsocket(s, FIONBIO, &nb);
    if (r == SOCKET_ERROR) return WSAGetLastError() == WSAEWOULDBLOCK ? 0 : -1;
    return r == 0 ? 0 : (ssize_t)r;  // 对端关闭按"此刻无数据"处理,下次阻塞读给出 EOF
  }
  int r = recv(s, (char*)buf, (int)n, 0);
  return r == SOCKET_ERROR ? -1 : (ssize_t)r;
}
static inline ssize_t sock_send(sockfd_t fd, const void* buf, size_t n) {
  int r = send((SOCKET)fd, (const char*)buf, (int)n, 0);
  return r == SOCKET_ERROR ? -1 : (ssize_t)r;
}
static inline int sock_errno() { return WSAGetLastError(); }
static inline const char* sock_strerror(int e) {
  static thread_local char b[96];
  snprintf(b, sizeof b, "WSA error %d", e);
  return b;
}

// ---------- 文件 ----------
// 二进制模式（MSVC 默认文本模式会改写 \n，必须显式 _O_BINARY）
static inline int os_open_rd(const char* path) {
  return _open(path, _O_RDONLY | _O_BINARY);
}
static inline int os_open_wr(const char* path) {  // create+truncate
  return _open(path, _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY, 0644);
}
static inline int os_close(int fd) { return _close(fd); }
static inline ssize_t os_write(int fd, const void* p, size_t n) {
  // _write 单次限 4 GiB（int 返回值），调用方本就有循环
  return _write(fd, p, (unsigned int)n);
}
static inline int64_t os_fsize(int fd) { return _filelengthi64(fd); }
static inline ssize_t os_read(int fd, void* p, size_t n) {
  return _read(fd, p, (unsigned int)n);  // 同 _write：单次 < 4 GiB，调用方循环
}
static inline int os_sync(int fd) { return _commit(fd); }
static inline void os_drop_cache(int) {}  // Windows 无 posix_fadvise 等价物，跳过

// 只读映射整个文件（mmap PROT_READ 等价物），返回 nullptr 失败
static inline void* os_map_ro(const char* path, size_t len) {
  HANDLE fh = CreateFileA(path, GENERIC_READ,
                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                          nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (fh == INVALID_HANDLE_VALUE) return nullptr;
  HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mh) {
    CloseHandle(fh);
    return nullptr;
  }
  void* p = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, len);
  CloseHandle(mh);
  CloseHandle(fh);
  return p;
}
static inline void os_unmap_ro(void* p, size_t) { UnmapViewOfFile(p); }

// 目录在网络位置上（UNC \\server\share 或映射的网络盘符）：kvsnap 用多个写线程
// （网络盘上每个文件的打开/落盘/改名往返占大头）
static inline bool os_path_is_remote(const char* path) {
  char full[MAX_PATH];
  const DWORD n = GetFullPathNameA(path, MAX_PATH, full, nullptr);
  if (!n || n >= MAX_PATH) return false;
  const auto sep = [](char c) { return c == '\\' || c == '/'; };
  if (sep(full[0]) && sep(full[1])) return true;  // UNC
  if (full[0] && full[1] == ':') {
    const char root[4] = {full[0], ':', '\\', 0};
    return GetDriveTypeA(root) == DRIVE_REMOTE;
  }
  return false;
}

// 直通（绕过系统文件缓存）整文件读写，供 kvsnap 用；语义与 00_platform.inc 的
// POSIX 版一致：缓冲按 OS_DIO_ALIGN 对齐且容得下 n 向上取整的长度；写按补齐长度
// 写入再把文件截回 n，读按补齐长度请求、到 EOF 为止。返回 1 成功、0 IO 错误、
// -1 此处不支持直通 IO（调用方退回普通 IO）。
static const size_t OS_DIO_ALIGN = 4096;
static inline size_t os_dio_round(size_t n) { return (n + OS_DIO_ALIGN - 1) & ~(OS_DIO_ALIGN - 1); }
static inline int os_write_direct(const char* path, const char* p, size_t n) {
  HANDLE h = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
  if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_INVALID_PARAMETER ? -1 : 0;
  const size_t cap = os_dio_round(n);
  size_t off = 0;
  int rc = 1;
  while (off < cap) {
    const DWORD want = (DWORD)(cap - off < ((size_t)1 << 30) ? cap - off : (size_t)1 << 30);
    DWORD got = 0;
    if (!WriteFile(h, p + off, want, &got, nullptr) || !got) {
      rc = off == 0 && GetLastError() == ERROR_INVALID_PARAMETER ? -1 : 0;
      break;
    }
    off += got;
  }
  if (rc == 1) {
    FILE_END_OF_FILE_INFO eof;  // 文件大小可设为任意值，只有 IO 要对齐
    eof.EndOfFile.QuadPart = (LONGLONG)n;
    if (!SetFileInformationByHandle(h, FileEndOfFileInfo, &eof, sizeof eof) ||
        !FlushFileBuffers(h))
      rc = 0;
  }
  if (!CloseHandle(h)) rc = 0;
  return rc;
}
// 读恰好 n 字节的整个文件（大小不符返回 0）
static inline int os_read_direct(const char* path, char* p, size_t n) {
  HANDLE h = CreateFileA(path, GENERIC_READ,
                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                         OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING |
                             FILE_FLAG_SEQUENTIAL_SCAN,
                         nullptr);
  if (h == INVALID_HANDLE_VALUE) return GetLastError() == ERROR_INVALID_PARAMETER ? -1 : 0;
  LARGE_INTEGER sz;
  int rc = GetFileSizeEx(h, &sz) && (uint64_t)sz.QuadPart == n ? 1 : 0;
  const size_t cap = os_dio_round(n);
  size_t off = 0;
  while (rc == 1 && off < n) {
    const DWORD want = (DWORD)(cap - off < ((size_t)1 << 30) ? cap - off : (size_t)1 << 30);
    DWORD got = 0;
    if (!ReadFile(h, p + off, want, &got, nullptr)) {
      rc = off == 0 && GetLastError() == ERROR_INVALID_PARAMETER ? -1 : 0;
      break;
    }
    if (!got) {
      rc = 0;
      break;
    }
    off += got;
    if (off < n && (off & (OS_DIO_ALIGN - 1))) rc = 0;  // 文件中段短读
  }
  CloseHandle(h);
  return rc;
}

static inline std::string os_last_error(const char* what) {
  DWORD e = GetLastError();
  char buf[512];
  DWORD n = FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM, nullptr, e, 0, buf,
                           sizeof buf, nullptr);
  std::string s = what;
  s += ": ";
  s += n ? buf : "unknown error";
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
  return s;
}
