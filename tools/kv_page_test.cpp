#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

static constexpr int KV_PAGE = 256;

#include "kv_page_alloc.inc"

static void require(bool passed, const char* name) {
  if (!passed) throw std::runtime_error(name);
}

static void require_unchanged(const KvPagePool& pool, const KvSeqTable& seq,
                              const std::vector<int>& table,
                              const std::vector<uint16_t>& refs, int mapped, int free_pages,
                              const char* name) {
  require(seq.tab == table && pool.ref == refs && seq.mapped == mapped &&
              pool.nfree() == free_pages,
          name);
}

int main() {
  KvPagePool pool;
  KvSeqTable seq;
  pool.init(6, 1);
  seq.init(6, pool.guard());
  require(seq.prepare_need(pool, 0, 0) == 0, "empty range");
  require(seq.prepare_need(pool, 0, 5 * KV_PAGE) == 5, "unmapped tail");
  require(seq.prepare_need(pool, 0, 6 * KV_PAGE + 1) == 7,
          "beyond-table demand is reported before mutation");

  require(seq.reserve(pool, 3 * KV_PAGE), "initial reserve");
  const std::vector<int> initial_table = seq.tab;
  const std::vector<uint16_t> initial_refs = pool.ref;
  require(seq.prepare_need(pool, 3 * KV_PAGE, 5 * KV_PAGE) == 2, "tail after reserve");
  require_unchanged(pool, seq, initial_table, initial_refs, 3, 3,
                    "tail preflight mutated state");

  const std::vector<int> pinned = seq.pin(pool, 2 * KV_PAGE);
  const std::vector<uint16_t> shared_refs = pool.ref;
  require(seq.prepare_need(pool, 0, 3 * KV_PAGE) == 2, "shared prefix COW");
  require(seq.prepare_need(pool, KV_PAGE, 3 * KV_PAGE) == 1, "partial shared prefix COW");
  require(seq.prepare_need(pool, KV_PAGE, 5 * KV_PAGE) == 3, "shared prefix plus tail");
  require_unchanged(pool, seq, initial_table, shared_refs, 3, 3,
                    "shared preflight mutated state");

  KvSeqTable::unpin(pool, pinned);
  require(seq.prepare_need(pool, KV_PAGE, 5 * KV_PAGE) == 2,
          "evicted checkpoint removes COW demand");

  pool.init(3, 1);
  seq.init(6, pool.guard());
  require(seq.reserve(pool, 3 * KV_PAGE), "small pool reserve");
  const std::vector<int> full_table = seq.tab;
  const std::vector<uint16_t> full_refs = pool.ref;
  require(pool.nfree() < seq.prepare_need(pool, 3 * KV_PAGE, 5 * KV_PAGE),
          "full pool must fail preflight");
  require_unchanged(pool, seq, full_table, full_refs, 3, 0,
                    "failed preflight mutated state");

  std::puts("PASS KV prepare preflight / COW accounting / failure atomicity");
  return 0;
}
