#ifndef _WIN32
// POSIX side of the socket shim (Windows side lives in os_win32.h): lets the
// serve loop below use one spelling on both platforms.
typedef int sockfd_t;
static inline bool wsa_init() { return true; }
static inline sockfd_t sock_tcp() { return socket(AF_INET, SOCK_STREAM, 0); }
static inline int sock_close(sockfd_t fd) { return close(fd); }
static inline int sock_shutdown(sockfd_t fd) { return shutdown(fd, SHUT_RDWR); }
static inline ssize_t sock_recv(sockfd_t fd, void* buf, size_t n, bool dontwait) {
  return recv(fd, buf, n, dontwait ? MSG_DONTWAIT : 0);
}
static inline ssize_t sock_send(sockfd_t fd, const void* buf, size_t n) {
  return send(fd, buf, n, MSG_NOSIGNAL);
}
static inline int sock_errno() { return errno; }
static inline const char* sock_strerror(int e) { return strerror(e); }

// POSIX side of the os_* file shims (Windows side lives in os_win32.h).
static inline int os_open_rd(const char* path) { return open(path, O_RDONLY); }
static inline int os_open_wr(const char* path) {
  return open(path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
}
static inline int os_close(int fd) { return close(fd); }
static inline ssize_t os_write(int fd, const void* p, size_t n) {
  return write(fd, p, n);
}
static inline ssize_t os_read(int fd, void* p, size_t n) { return read(fd, p, n); }
static inline int os_sync(int fd) { return fdatasync(fd); }
// written/read data will not be needed again soon: keep the page cache (which
// competes with device memory on a unified-memory APU) from holding it
static inline void os_drop_cache(int fd) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); }
static inline int64_t os_fsize(int fd) {
  struct stat sb;
  return fstat(fd, &sb) ? -1 : (int64_t)sb.st_size;
}
static inline void* os_map_ro(const char* path, size_t len) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return nullptr;
  void* p = mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : p;
}
static inline void os_unmap_ro(void* p, size_t len) { munmap(p, len); }

// A directory on a network file system (NFS/SMB/CIFS/9P/Ceph/FUSE): kvsnap
// writes it with several threads (per-file round trips dominate there).
static inline bool os_path_is_remote(const char* path) {
  struct statfs sf;
  if (statfs(path, &sf)) return false;
  switch ((unsigned long)sf.f_type) {
    case 0x6969UL:      // NFS
    case 0x517BUL:      // SMB
    case 0xFF534D42UL:  // CIFS
    case 0xFE534D42UL:  // SMB2
    case 0x01021997UL:  // 9P
    case 0x00C36400UL:  // Ceph
    case 0x65735546UL:  // FUSE (sshfs, rclone, ...)
      return true;
    default:
      return false;
  }
}

// Direct (page-cache bypassing) whole-file IO, kvsnap. The buffer must be
// OS_DIO_ALIGN-aligned with room for n rounded up to OS_DIO_ALIGN: writes
// send the padded length and cut the file back to n, reads ask for it and
// stop at EOF. Returns 1 ok, 0 IO error, -1 direct IO not supported here
// (the caller falls back to buffered IO).
static const size_t OS_DIO_ALIGN = 4096;
static inline size_t os_dio_round(size_t n) { return (n + OS_DIO_ALIGN - 1) & ~(OS_DIO_ALIGN - 1); }
static inline int os_write_direct(const char* path, const char* p, size_t n) {
  int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC | O_DIRECT | O_CLOEXEC, 0644);
  if (fd < 0) return errno == EINVAL ? -1 : 0;
  const size_t cap = os_dio_round(n);
  size_t off = 0;
  int rc = 1;
  while (off < cap) {
    const size_t want = cap - off < ((size_t)1 << 30) ? cap - off : (size_t)1 << 30;
    const ssize_t w = pwrite(fd, p + off, want, (off_t)off);
    if (w <= 0) {
      rc = w < 0 && errno == EINVAL && off == 0 ? -1 : 0;
      break;
    }
    off += (size_t)w;
  }
  if (rc == 1 && (ftruncate(fd, (off_t)n) || fdatasync(fd))) rc = 0;
  if (close(fd)) rc = 0;
  return rc;
}
// Whole file of exactly n bytes (0 if its size differs).
static inline int os_read_direct(const char* path, char* p, size_t n) {
  int fd = open(path, O_RDONLY | O_DIRECT | O_CLOEXEC);
  if (fd < 0) return errno == EINVAL ? -1 : 0;
  struct stat sb;
  int rc = !fstat(fd, &sb) && (uint64_t)sb.st_size == n ? 1 : 0;
  const size_t cap = os_dio_round(n);
  size_t off = 0;
  while (rc == 1 && off < n) {
    const size_t want = cap - off < ((size_t)1 << 30) ? cap - off : (size_t)1 << 30;
    const ssize_t r = pread(fd, p + off, want, (off_t)off);
    if (r <= 0) {
      rc = r < 0 && errno == EINVAL && off == 0 ? -1 : 0;
      break;
    }
    off += (size_t)r;
    if (off < n && (off & (OS_DIO_ALIGN - 1))) rc = 0;  // short read mid-file
  }
  close(fd);
  return rc;
}
#endif

static inline void sock_rcvtimeo(sockfd_t fd, int sec) {
#ifdef _WIN32
  DWORD ms = (DWORD)(sec * 1000);
  setsockopt((SOCKET)fd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&ms, sizeof ms);
#else
  timeval tv{sec, 0};
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
#endif
}
static inline void sock_sndtimeo(sockfd_t fd, int sec) {
#ifdef _WIN32
  DWORD ms = (DWORD)(sec * 1000);
  setsockopt((SOCKET)fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&ms, sizeof ms);
#else
  timeval tv{sec, 0};
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#endif
}

using hgn::Checkpoint;
using hgn::Tensor;

// CK：每次调用前把调用点（文件:行 + 表达式，编译期字面量）记进
// crashdiag 的 HIP 调用点环（崩溃报告里的 "last HIP calls"，开销 ~几十 ns）；
// 失败时 stderr 打同一行原因，再写一份 kind=fatal 的崩溃报告后 exit(1)。
#define CK_STR2_(x) #x
#define CK_STR_(x) CK_STR2_(x)
#define CK(x)                                                          \
  do {                                                                 \
    crashdiag::site(__FILE__ ":" CK_STR_(__LINE__) " " #x);            \
    hipError_t e_ = (x);                                               \
    if (e_ != hipSuccess)                                              \
      crashdiag::fatal("HIP error %s at %s:%d", hipGetErrorString(e_), \
                       __FILE__, __LINE__);                            \
  } while (0)

static int g_hip_device = 0;
// HIP 的当前设备按宿主线程记录；每个会调用 HIP 的工作线程都要显式绑定。
static inline void hip_use_selected_device() { CK(hipSetDevice(g_hip_device)); }

// 崩溃链路自测（分发后验证报告/诊断包是否生成）：
//   QWENOX_CRASH_TEST=<kind>   kind 见 crashdiag::self_test_kinds()，另有
//                              hipfatal（走一次 CK 失败路径）
//   QWENOX_CRASH_TEST_AT=start|ready|gen   在哪个时刻崩（默认 start：装好
//                              处理器、打印环境后立刻崩，不加载模型）
// 也可直接运行 qwenox-engine --crash-test <kind>（同 start）。
[[maybe_unused]] static void crash_test_now(const char* kind) {
  crashdiag::rearm();  // HIP 初始化时 amd_comgr 的 LLVM 会抢走顶层异常过滤器
  if (!strcmp(kind, "hipfatal")) {
    CK(hipSetDevice(9999));
    return;
  }
  if (!crashdiag::self_test(kind))
    fprintf(stderr, "crash test: unknown kind '%s' (known: %s hipfatal)\n", kind,
            crashdiag::self_test_kinds());
}
[[maybe_unused]] static void crash_test_point(const char* at) {
  static const char* kind = getenv("QWENOX_CRASH_TEST");
  if (!kind || !*kind) return;
  const char* want = getenv("QWENOX_CRASH_TEST_AT");
  if (!want || !*want) want = "start";
  if (strcmp(want, at) == 0) crash_test_now(kind);
}

// "[2026-09-18 11:25:03.123]" — wall-clock prefix for human-facing logs.
// thread_local so concurrent loader/decode threads never share the buffer.
// One call per fprintf: two calls in one expression would alias.
static const char* log_ts() {
  static thread_local char b[32];
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  struct tm tmv;
  localtime_r(&ts.tv_sec, &tmv);
  snprintf(b, sizeof b, "[%04d-%02d-%02d %02d:%02d:%02d.%03d]",
           tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday, tmv.tm_hour,
           tmv.tm_min, tmv.tm_sec, (int)(ts.tv_nsec / 1000000));
  return b;
}

struct NgramMod {
  using token_t=int32_t; static constexpr token_t EMPTY=-1;
  int n=8,n_min=8,n_max=64; size_t learned=0; std::vector<token_t> table;
  NgramMod(){ init(0); }
  // nn>0：联合起草的短窗表（绕开 QWENOX_NGRAM_MATCH 的 [4,64] 钳制；
  // QWENOX_JOINT_N 可调，默认 3 = halogen PLD 的 N）。n_min=1：命中即放行，
  // 链长由调用方的 want 封顶。
  explicit NgramMod(int nn){ init(nn); }
  void init(int nn){
    if(nn>0){ n=nn; n_min=1; }
    const char* em = getenv(nn>0?"QWENOX_JOINT_N":"QWENOX_NGRAM_MATCH");
    if(em) n=std::max(nn>0?2:4,std::min(64,atoi(em)));
    if(nn==0){
      if(const char* e=getenv("QWENOX_NGRAM_MIN")) n_min=std::max(1,std::min(64,atoi(e)));
      if(const char* e=getenv("QWENOX_NGRAM_MAX")) n_max=std::max(1,std::min(64,atoi(e)));
    }
    table.resize((size_t)4*1024*1024,EMPTY);
  }
  // 槽位条目默认构造后是主表窗口；slots_init 用它对齐联合表的窗口。
  // 仅在表还空着时有效（学习后换窗口会让既有条目失址）。
  void set_n(int nn){ if(learned==0) n=nn; }
  size_t slot(const std::vector<int64_t>&h,size_t off)const{uint64_t z=0;for(int i=0;i<n;i++) z=z*6364136223846793005ULL+(uint32_t)h[off+i];return(size_t)(z%table.size());}
  void reset(){std::fill(table.begin(),table.end(),EMPTY);learned=0;}
  void observe(const std::vector<int64_t>&h){if(h.size()<=(size_t)n)return;size_t lim=h.size()-n;if(learned>lim)learned=lim;for(size_t i=learned;i<lim;i++)table[slot(h,i)]=(token_t)h[i+n];learned=lim;}
  std::vector<int> draft(const std::vector<int64_t>&h,int pending,int want)const{std::vector<int>o;if(h.size()+1<(size_t)n)return o;std::vector<int64_t>c((size_t)n);for(int i=0;i<n-1;i++)c[i]=h[h.size()-(size_t)(n-1)+i];c[n-1]=pending;for(int i=0;i<std::min(want,n_max);i++){token_t t=table[slot(c,0)];if(t==EMPTY){if(i<n_min)o.clear();break;}o.push_back((int)t);for(int j=0;j<n-1;j++)c[j]=c[j+1];c[n-1]=(int)t;}return o;}
};

