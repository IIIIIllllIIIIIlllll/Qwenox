// reqstat_read_test — recorder + reader self-test (no tokenizer/server
// needed; safe to run anywhere). Also used by tools/reqstat_api_verify.sh:
//
//   reqstat_read_test                 run the full assertion suite (PASS/FAIL)
//   reqstat_read_test --fixtures DIR  only write the 200-record fixture chain
//                                     (+1 junk record) into DIR and exit
//
// Fixture shape (deterministic): 200 records, i = 0..199.
//   ts: i < 100 -> 2026-09-28 12:00Z + i min ; else 2026-09-29 12:00Z + (i-100) min
//   req_seq = i+1, n_prompt = 1000+i, n_cached = i%7, n_gen = 50
//   drafter: i%5==0 -> serial (proposed=0); else chain (proposed=40, rounds=10,
//            commit=25 -> acceptance (25-10)/40 = 0.375; of it n-gram
//            ng_rounds=2, ng_proposed=16, ng_accepted=3)
//   finish = 1+(i%3)   prefill 2s, decode 5s, ttft 500ms (i%10==0 -> 0)
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "reqstat.h"
#include "reqstat_read.h"

namespace {

int failures = 0;

void check(bool ok, const char* what) {
  if (ok)
    printf("ok   %s\n", what);
  else {
    printf("FAIL %s\n", what);
    failures++;
  }
}

int64_t days_from_civil(int64_t y, unsigned m, unsigned d) {
  y -= m <= 2;
  const int64_t era = (y >= 0 ? y : y - 399) / 400;
  const unsigned yoe = (unsigned)(y - era * 400);
  const unsigned doy = (153 * (m > 2 ? m - 3 : m + 9) + 2) / 5 + d - 1;
  const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
  return era * 146097 + (int64_t)doe - 719468;
}

const uint64_t D1 = (uint64_t)days_from_civil(2026, 9, 28) * 86400000 +
                    12ull * 3600000;  // 2026-09-28 12:00Z
const uint64_t D2 = D1 + 86400000ull;  // 2026-09-29 12:00Z

reqstat::Entry fixture_entry(int i) {
  reqstat::Entry e;
  e.ts_ms = i < 100 ? D1 + (uint64_t)i * 60000 : D2 + (uint64_t)(i - 100) * 60000;
  e.req_seq = (uint64_t)i + 1;
  e.n_prompt = 1000 + i;
  e.n_cached = i % 7;
  e.n_gen = 50;
  if (i % 5 == 0) {  // serial
    e.proposed = e.commit = e.rounds = 0;
    e.flags = 0 << 4;
  } else {  // chain
    e.proposed = 40;
    e.commit = 25;
    e.rounds = 10;
    e.ng_rounds = 2;
    e.ng_proposed = 16;
    e.ng_accepted = 3;
    e.flags = 4u << 4;
  }
  e.prefill_us = 2000000;
  e.decode_us = 5000000;
  e.ttft_us = i % 10 == 0 ? 0 : 500000;
  e.flags |= (uint32_t)(1 + i % 3);
  return e;
}

void write_fixtures() {
  for (int i = 0; i < 200; i++) reqstat::record(fixture_entry(i));
}

void append_junk(const std::string& dir) {
  FILE* f = fopen((dir + "/reqstat.bin").c_str(), "ab");
  uint8_t junk[80];  // one whole record slot (version 2)
  memset(junk, 0xA5, sizeof junk);
  fwrite(junk, 1, sizeof junk, f);
  fclose(f);
}

// A version 1 reqstat.bin (64-byte records, no n-gram split) as the
// pre-11-10 recorder wrote it: n chain records, seq 5.
uint16_t crc16(const uint8_t* p, size_t n) {
  uint16_t c = 0xFFFF;
  for (size_t i = 0; i < n; i++) {
    c ^= (uint16_t)p[i] << 8;
    for (int b = 0; b < 8; b++) c = (c & 0x8000) ? (c << 1) ^ 0x1021 : c << 1;
  }
  return c;
}
void write_v1_file(const std::string& path, int n) {
  FILE* f = fopen(path.c_str(), "wb");
  uint8_t h[256] = {};
  const uint32_t magic = 0x47525153, recsz = 64, seq = 5;
  const uint16_t ver = 1, hsz = 256;
  const uint64_t count = (uint64_t)n;
  memcpy(h + 0, &magic, 4);
  memcpy(h + 4, &ver, 2);
  memcpy(h + 6, &hsz, 2);
  memcpy(h + 8, &recsz, 4);
  memcpy(h + 12, &seq, 4);
  memcpy(h + 16, &count, 8);
  const uint64_t rot = 256ull << 20;
  memcpy(h + 48, &rot, 8);
  fwrite(h, 1, sizeof h, f);
  for (int i = 0; i < n; i++) {
    uint8_t r[64] = {};
    const uint64_t ts = D1 - 3600000ull + (uint64_t)i * 1000, rs = 9000 + (uint64_t)i;
    const uint32_t v[10] = {500, 0, 20, 40, 25, 10, 1000000, 2000000, 0, (4u << 4) | 1u};
    memcpy(r + 0, &ts, 8);
    memcpy(r + 8, &rs, 8);
    memcpy(r + 16, v, sizeof v);
    const uint16_t c = crc16(r, 56);
    memcpy(r + 56, &c, 2);
    fwrite(r, 1, sizeof r, f);
  }
  fclose(f);
}


void set_env(const char* k, const std::string& v) {
#ifdef _WIN32
  _putenv_s(k, v.c_str());
#else
  setenv(k, v.c_str(), 1);
#endif
}

struct Agg {
  uint64_t n = 0, tp = 0, tc = 0, tg = 0, dec_us = 0, ttft_n = 0;
  uint64_t chain_n = 0, chain_prop = 0, chain_acc = 0;
  uint64_t split_n = 0, ng_rounds = 0, ng_prop = 0, ng_acc = 0;
};

Agg aggregate(uint64_t t0, uint64_t t1, reqstat::ScanInfo* info) {
  Agg a;
  std::string err;
  reqstat::scan(
      t0, t1,
      [&](const reqstat::QEntry& e) {
        a.n++;
        a.tp += e.n_prompt;
        a.tc += e.n_cached;
        a.tg += e.n_gen;
        a.dec_us += e.decode_us;
        if (e.ttft_us) a.ttft_n++;
        if (e.proposed) {
          a.chain_n++;
          a.chain_prop += e.proposed;
          a.chain_acc += e.commit - e.rounds;
          if (e.split) {
            a.split_n++;
            a.ng_rounds += e.ng_rounds;
            a.ng_prop += e.ng_proposed;
            a.ng_acc += e.ng_accepted;
          }
        }
        return true;
      },
      info, &err);
  return a;
}

}  // namespace

// --v1-upgrade DIR (own process: the recorder binds one directory per process)
int v1_upgrade(const std::string& dir) {
  namespace fs = std::filesystem;
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  write_v1_file(dir + "/reqstat.bin", 10);
  set_env("QWENOX_REQSTAT_DIR", dir);
  set_env("QWENOX_REQSTAT_MAX_MB", "256");
  for (int i = 0; i < 3; i++) reqstat::record(fixture_entry(1 + i));  // chain records
  check(fs::exists(dir + "/reqstat-000005.bin", ec), "v1 file kept as rotated reqstat-000005.bin");
  bool v2_live = false;
  if (FILE* f = fopen((dir + "/reqstat.bin").c_str(), "rb")) {
    uint8_t h[16] = {};
    if (fread(h, 1, sizeof h, f) == sizeof h) {
      uint16_t ver;
      uint32_t recsz, seq;
      memcpy(&ver, h + 4, 2);
      memcpy(&recsz, h + 8, 4);
      memcpy(&seq, h + 12, 4);
      v2_live = ver == 2 && recsz == 80 && seq == 6;
    }
    fclose(f);
  }
  check(v2_live, "fresh live file is version 2, seq 6");
  std::vector<reqstat::QEntry> out;
  reqstat::ScanInfo info;
  std::string err;
  const bool ok = reqstat::tail(100, &out, &info, &err);
  check(ok && out.size() == 13 && info.files_total == 2, "chain = 10 v1 + 3 v2 records");
  bool order = out.size() == 13;
  for (size_t i = 0; order && i < out.size(); i++)
    order = i < 10 ? out[i].req_seq == 9000 + i && !out[i].split && out[i].proposed == 40
                   : out[i].req_seq == i - 8 && out[i].split && out[i].ng_proposed == 16;
  check(order, "v1 records read (no split), then v2 records (split)");
  Agg a = aggregate(0, ~(uint64_t)0, &info);
  check(a.n == 13 && a.chain_prop == 13 * 40 && a.split_n == 3 && info.bad_crc == 0,
        "scan spans both record versions");
  fs::remove_all(dir, ec);
  return failures;
}

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  if (argc == 3 && !strcmp(argv[1], "--v1-upgrade")) return v1_upgrade(argv[2]) ? 1 : 0;
  if (argc == 3 && !strcmp(argv[1], "--fixtures")) {
    std::error_code ec;
    fs::create_directories(argv[2], ec);
    set_env("QWENOX_REQSTAT_DIR", argv[2]);
    set_env("QWENOX_REQSTAT_MAX_MB", "0.01");
    write_fixtures();
    append_junk(argv[2]);
    printf("fixtures written to %s\n", argv[2]);
    return 0;
  }

  const std::string dir =
      (fs::temp_directory_path() / "reqstat_read_test").string();
  std::error_code ec;
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  set_env("QWENOX_REQSTAT_DIR", dir);
  set_env("QWENOX_REQSTAT_MAX_MB", "0.01");  // 10 KiB -> rotates mid-run

  // Empty directory: valid scan, zero records.
  {
    reqstat::ScanInfo info;
    Agg a = aggregate(0, ~(uint64_t)0, &info);
    check(a.n == 0 && info.files_total == 0, "empty dir scans clean");
  }

  write_fixtures();

  // Rotation: 10 KiB cap -> the chain is reqstat-000001.bin + live reqstat.bin.
  uint64_t rotated_records = 0, live_records = 0;
  {
    reqstat::ScanInfo info;
    Agg a = aggregate(0, ~(uint64_t)0, &info);
    check(a.n == 200 && info.files_total == 2, "rotation: 200 records, 2 files");
    check(a.tp == 219900, "sum prompt tokens");
    check(a.tc == 594, "sum cached tokens");
    check(a.tg == 10000 && a.dec_us == 1000ull * 1000000,
          "sum gen tokens / decode time");
    check(a.ttft_n == 180, "ttft present on 180 records");
    check(a.chain_n == 160 && a.chain_prop == 6400 && a.chain_acc == 2400,
          "chain acceptance sums (0.375)");
    check(a.split_n == 160 && a.ng_rounds == 320 && a.ng_prop == 2560 && a.ng_acc == 480,
          "n-gram share round-trips (v2 records)");
    check(info.bad_crc == 0, "no bad crc yet");
  }

  // Per-record flush: the live file on disk already holds every record we
  // wrote (no atexit, no explicit flush) — file size is the truth.
  {
    const uint64_t sz = fs::file_size(dir + "/reqstat.bin", ec);
    live_records = sz >= 256 ? (sz - 256) / 80 : 0;
    const uint64_t rsz = fs::file_size(dir + "/reqstat-000001.bin", ec);
    rotated_records = rsz >= 256 ? (rsz - 256) / 80 : 0;
    check(rotated_records + live_records == 200,
          "records hit disk immediately (no buffering)");
    check(rotated_records == 127, "rotated file holds 127 records (10 KiB cap)");
  }

  // Time window: day 2 only.
  {
    reqstat::ScanInfo info;
    Agg a = aggregate(D2, D2 + 86399999ull, &info);
    check(a.n == 100, "window scan finds day-2 records only");
  }

  // tail: last N in chain order.
  {
    std::vector<reqstat::QEntry> out;
    reqstat::ScanInfo info;
    std::string err;
    check(reqstat::tail(50, &out, &info, &err) && out.size() == 50,
          "tail(50) returns 50");
    check(!out.empty() && out.front().req_seq == 151 && out.back().req_seq == 200,
          "tail(50) is the last 50 in chain order");
    bool increasing = true;
    for (size_t i = 1; i < out.size(); i++)
      if (out[i].req_seq != out[i - 1].req_seq + 1) increasing = false;
    check(increasing, "tail order is contiguous");
    check(reqstat::tail(500, &out, &info, &err) && out.size() == 200,
          "tail(500) caps at history size");
    check(reqstat::tail(41, &out, &info, &err) && out.size() == 41 &&
              info.files_scanned == 1,
          "tail within the live file reads no rotated file");
  }

  // CRC: a junk record is skipped and counted.
  {
    append_junk(dir);
    reqstat::ScanInfo info;
    Agg a = aggregate(0, ~(uint64_t)0, &info);
    check(a.n == 200 && info.bad_crc == 1, "junk record skipped, bad_crc=1");
  }

  // Missing directory: not an error, zero records.
  {
    set_env("QWENOX_REQSTAT_DIR", dir + "/no-such");
    reqstat::ScanInfo info;
    Agg a = aggregate(0, ~(uint64_t)0, &info);
    std::vector<reqstat::QEntry> out;
    std::string err;
    const bool ok = reqstat::tail(10, &out, &info, &err);
    check(a.n == 0 && ok && out.empty(), "missing dir scans clean");
    set_env("QWENOX_REQSTAT_DIR", dir);
  }

  fs::remove_all(dir, ec);

  // Version 1 -> 2 upgrade, in a child process (fresh recorder).
  {
    const std::string vdir = (fs::temp_directory_path() / "reqstat_read_test_v1").string();
    const std::string cmd = "\"" + std::string(argv[0]) + "\" --v1-upgrade \"" + vdir + "\"";
    check(std::system(cmd.c_str()) == 0, "v1 upgrade (child process)");
  }
  if (failures) {
    printf("FAIL (%d checks failed)\n", failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
