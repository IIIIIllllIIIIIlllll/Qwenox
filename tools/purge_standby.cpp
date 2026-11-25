// purge_standby (Windows): 清空系统 Standby 页缓存，效果等同 RAMMap 的
// "Empty -> Empty Standby List" / EmptyStandbyList.exe。
// 用途：prefill 冷态测试前强制 PLE 页全部落到磁盘（页缓存对 prefill 跑分的
// 影响 ±6~8%，见 dev-docs/RESULT-WIN-PREFILL-FIX-20261125.md）。
// 需要 SeProfileSingleProcessPrivilege（管理员账户默认持有；若进程的
// 过滤令牌里没有会打印失败原因）。
//   cl /O2 purge_standby.cpp        或        clang++ -O2 purge_standby.cpp
#define NOMINMAX
#include <windows.h>
#include <stdio.h>

typedef LONG NTSTATUS;
typedef NTSTATUS(NTAPI* NtSetSystemInformation_t)(ULONG, PVOID, ULONG);

bool purge_standby_list() {
  HMODULE ntdll = GetModuleHandleA("ntdll.dll");
  if (!ntdll) return false;
  auto NtSetSystemInformation =
      (NtSetSystemInformation_t)GetProcAddress(ntdll, "NtSetSystemInformation");
  if (!NtSetSystemInformation) return false;
  HANDLE token = nullptr;
  if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                       &token)) {
    LUID luid;
    if (LookupPrivilegeValueA(nullptr, "SeProfileSingleProcessPrivilege", &luid)) {
      TOKEN_PRIVILEGES tp{};
      tp.PrivilegeCount = 1;
      tp.Privileges[0].Luid = luid;
      tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
      AdjustTokenPrivileges(token, FALSE, &tp, sizeof(tp), nullptr, nullptr);
    }
    CloseHandle(token);
  }
  // SystemMemoryListInformation = 0x50, MemoryPurgeStandbyList = 4
  ULONG cmd = 4;
  NTSTATUS st = NtSetSystemInformation(0x50, &cmd, sizeof(cmd));
  return st >= 0;
}

#ifndef QWENOX_PURGE_NO_MAIN
int main() {
  if (purge_standby_list()) {
    printf("standby list purged\n");
    return 0;
  }
  fprintf(stderr, "purge failed (need SeProfileSingleProcessPrivilege / admin)\n");
  return 1;
}
#endif
