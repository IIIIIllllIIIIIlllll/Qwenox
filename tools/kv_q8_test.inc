// Q8 KV storage and attention regressions; included by ktest.cu.
template <class T>
static std::vector<T> q8_get(const T* p, size_t n) {
  std::vector<T> v(n);
  TCK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

static void test_q8_prefill_expand() {
  // Reverse pages, cross a page boundary, and leave a partial final block.
  // Poison every future Q8 row and guard the outputs against overrun.
  for (int sparse = 0; sparse < 2; sparse++) {
    const int base = sparse ? g_index_first : KV_PAGE - 3;
    const int P = 10, N = base + P, padded = (N + 3) & ~3;
    const int nr = ((padded + KV_PAGE - 1) / KV_PAGE) * KV_PAGE;
    std::vector<float> k((size_t)nr * 512), v(k.size()), q((size_t)P * 6144);
    for (size_t i = 0; i < k.size(); i++) {
      float scale = (i / 32) % 3 == 0 ? 0.f : ((i / 32) % 3 == 1 ? 1e-7f : 4.f);
      k[i] = frand() * scale; v[i] = frand() * scale * 3.f;
    }
    for (auto& x : q) x = frand();
    std::vector<int> tab(nr / KV_PAGE), sel((size_t)P * g_index_budget);
    for (size_t i = 0; i < tab.size(); i++) tab[i] = (int)tab.size() - 1 - (int)i;
    for (int r = 0; r < P; r++) for (int i = 0; i < g_index_budget; i++)
      // Production union consumes sorted, unique complete-block selections.
      sel[(size_t)r * g_index_budget + i] = (base + r + 1) / 4 - g_index_budget + i;
    auto* dk = dup(k); auto* dv = dup(v); auto* dq = dup(q);
    auto* dt = dup(tab); auto* ds = dup(sel);
    std::vector<KvQ8Block> empty((size_t)nr * 16);
    auto* kq = dup(empty); auto* vq = dup(empty);
    k_store_kv_q8<<<nr * 4, 128>>>(dk, dv, kq, vq, nr, 0, dt);
    auto hk = q8_get(kq, empty.size()), hv = q8_get(vq, empty.size());
    const uint16_t sentinel = 0x7fc1;
    std::vector<uint16_t> kb((size_t)(padded + 4) * 512, sentinel), vt(kb.size(), sentinel);
    for (int t = 0; t < padded; t++) for (int d = 0; d < 512; d++) {
      const size_t src = (size_t)(tab[t / KV_PAGE] * KV_PAGE + t % KV_PAGE) * 16 + d / 32;
      kb[(size_t)t * 512 + d] = t < N ? bf16_bits(__half2float(hk[src].scale) * hk[src].q[d % 32]) : 0;
      vt[((size_t)(t / 4) * 512 + d) * 4 + t % 4] =
          t < N ? bf16_bits(__half2float(hv[src].scale) * hv[src].q[d % 32]) : 0;
    }
    auto* ek = dup(kb); auto* ev = dup(vt);
    std::vector<uint16_t> poison(kb.size(), sentinel);
    auto* sk = dup(poison); auto* sv = dup(poison);
    k_q8_prefill_expand<<<((N + 3) / 4 * 16 + 3) / 4, 128>>>(kq, vq, sk, sv, N, dt);
    auto gotk = q8_get(sk, kb.size()), gotv = q8_get(sv, vt.size());
    check(sparse ? "q8_expand_sparse_paged_tail" : "q8_expand_dense_paged_tail",
          memcmp(kb.data(), gotk.data(), kb.size() * 2) ||
          memcmp(vt.data(), gotv.data(), vt.size() * 2), 0);
    float *a = dalloc(q.size()), *ref = dalloc(q.size());
    if (sparse) {
      const int ga = (base + 3) & ~3, ge = (base + P) & ~3, ng = (ge - ga) / 4;
      auto* ub = dup(std::vector<uint16_t>((size_t)ng * QSA_UCAP));
      auto* um = dup(std::vector<uint16_t>((size_t)ng * QSA_UCAP));
      auto* uc = dup(std::vector<int>(ng));
      k_qsa_union_merge<<<ng, 128>>>(ds, ga, ub, um, uc, QSA_UCAP, base, P);
      k_qsa_wmma_u<<<dim3(ng, 2), 256>>>(dq, sk, sv, a, 256, 2, 24,
          ub, um, uc, QSA_UCAP, ga, base, nullptr, P);
      k_qsa_wmma_u<<<dim3(ng, 2), 256>>>(dq, ek, ev, ref, 256, 2, 24,
          ub, um, uc, QSA_UCAP, ga, base, nullptr, P);
      for (auto edge : {std::pair<int,int>{base, ga - base}, {ge, base + P - ge}}) {
        if (!edge.second) continue;
        auto* es = ds + (size_t)(edge.first - base) * g_index_budget;
        k_qsa_wmma<true><<<dim3(edge.second, 2), 256>>>(dq, sk, (const uint16_t*)nullptr, sv, a, P, 256, 2, 24, es, edge.first, base);
        k_qsa_wmma<true><<<dim3(edge.second, 2), 256>>>(dq, ek, (const uint16_t*)nullptr, ev, ref, P, 256, 2, 24, es, edge.first, base);
      }
      TCK(hipFree(ub)); TCK(hipFree(um)); TCK(hipFree(uc));
    } else {
      k_qsa_wmma<true, false, true><<<dim3(P, 2), 256>>>(dq, sk, (const uint16_t*)nullptr, sv, a, P, 256, 2, 24, nullptr, base, base);
      k_qsa_wmma<true, false, true><<<dim3(P, 2), 256>>>(dq, ek, (const uint16_t*)nullptr, ev, ref, P, 256, 2, 24, nullptr, base, base);
    }
    auto ah = dget(a, q.size()), rh = dget(ref, q.size());
    double bad = 0;
    for (size_t i = 0; i < ah.size(); i++)
      bad += !std::isfinite(ah[i]) || ah[i] != rh[i];
    check(sparse ? "q8_expand_union_ragged_reference" : "q8_expand_dense_btv_reference", bad, 0);
    for (void* p : {(void*)dk, (void*)dv, (void*)dq, (void*)dt, (void*)ds,
                    (void*)kq, (void*)vq, (void*)ek, (void*)ev, (void*)sk,
                    (void*)sv, (void*)a, (void*)ref}) TCK(hipFree(p));
  }
}

static void test_kv_q8() {
  const auto saved_rng = rng;
  test_q8_prefill_expand();
  rng = saved_rng;  // keep all existing deterministic regression inputs
  // Reverse physical pages and write across a page seam. Zero, subnormal,
  // normal and outlier groups exercise independent scales for K and V.
  const int P = KV_PAGE + 7, base = KV_PAGE - 3, rows = 3 * KV_PAGE;
  std::vector<float> k((size_t)P * 512), v(k.size());
  for (size_t i = 0; i < k.size(); i++) {
    float s = (i / 32) % 4 == 0 ? 0.f : ((i / 32) % 4 == 1 ? 1e-7f : 3.f);
    k[i] = frand() * s;
    v[i] = frand() * s * 7.f;
  }
  std::vector<int> tab = {2, 0, 1};
  int* pt = dup(tab);
  float *dk = dup(k), *dv = dup(v);
  std::vector<KvQ8Block> zero((size_t)rows * 16);
  KvQ8Block *kc = dup(zero), *vc = dup(zero), *ks = dup(zero), *vs = dup(zero);
  k_store_kv_q8<<<P * 4, 128>>>(dk, dv, kc, vc, P, base, pt);
  int* dp = dup(std::vector<int>{base});
  for (int t = 0; t < P; t++) {
    int pos = base + t;
    TCK(hipMemcpy(dp, &pos, sizeof pos, hipMemcpyHostToDevice));
    k_store_kv_q8<<<4, 128>>>(dk + (size_t)t * 512, dv + (size_t)t * 512,
                            ks, vs, 1, 0, pt, dp);
  }
  auto hk = q8_get(kc, zero.size()), hv = q8_get(vc, zero.size());
  auto sk = q8_get(ks, zero.size()), sv = q8_get(vs, zero.size());
  check("q8_batch_vs_step", memcmp(hk.data(), sk.data(), hk.size() * sizeof(KvQ8Block)) ||
        memcmp(hv.data(), sv.data(), hv.size() * sizeof(KvQ8Block)), 0);
  double bad = 0;
  std::vector<KvQ8Block> ek = zero, ev = zero;
  auto reference = [&](const std::vector<float>& x, std::vector<KvQ8Block>& expected) {
    for (int t = 0; t < P; t++) for (int g = 0; g < 16; g++) {
      float mx = 0;
      for (int j = 0; j < 32; j++) mx = std::max(mx, fabsf(x[(size_t)t * 512 + g * 32 + j]));
      int pos = base + t, row = tab[pos / KV_PAGE] * KV_PAGE + pos % KV_PAGE;
      auto& b = expected[(size_t)row * 16 + g];
      b.scale = __float2half_rn(mx == 0 ? 0.f : std::max(mx / 127.f, 0x1p-24f));
      float s = __half2float(b.scale);
      for (int j = 0; j < 32; j++) {
        float a = x[(size_t)t * 512 + g * 32 + j];
        b.q[j] = s == 0 ? 0 : (int8_t)std::max(-127.f, std::min(127.f, nearbyintf(a / s)));
      }
    }
  };
  reference(k, ek); reference(v, ev);
  for (size_t i = 0; i < hk.size(); i++)
    bad += memcmp(&hk[i], &ek[i], sizeof(KvQ8Block)) != 0 ||
           memcmp(&hv[i], &ev[i], sizeof(KvQ8Block)) != 0;
  check("q8_paged_cpu_reference", bad, 0);
  for (void* p : {(void*)pt, (void*)dk, (void*)dv, (void*)kc, (void*)vc,
                  (void*)ks, (void*)vs, (void*)dp}) TCK(hipFree(p));

  // Dense and sparse attention, contiguous and reversed-page layouts.
  // Compare to explicit dequantization, so quantization error cannot hide a
  // pointer/scale bug. Sparse case includes the first sparse query and tail.
  for (int sparse = 0; sparse < 2; sparse++) {
    const int b = sparse ? g_index_first : KV_PAGE - 2, N = b + 4;
    const int nr = ((N + KV_PAGE - 1) / KV_PAGE) * KV_PAGE;
    std::vector<float> kh((size_t)nr * 512), vh(kh.size()), q(4 * 6144);
    for (auto& x : kh) x = frand();
    for (auto& x : vh) x = frand();
    for (auto& x : q) x = frand();
    auto* kf = dup(kh); auto* vf = dup(vh); auto* dq = dup(q);
    std::vector<KvQ8Block> empty((size_t)nr * 16);
    auto* kq = dup(empty); auto* vq = dup(empty);
    k_store_kv_q8<<<nr * 4, 128>>>(kf, vf, kq, vq, nr, 0);
    auto kr = q8_get(kq, empty.size()), vr = q8_get(vq, empty.size());
    std::vector<uint16_t> kb(kh.size()), vb(vh.size());
    for (size_t i = 0; i < kh.size(); i++) {
      kh[i] = __half2float(kr[i / 32].scale) * kr[i / 32].q[i % 32];
      vh[i] = __half2float(vr[i / 32].scale) * vr[i / 32].q[i % 32];
      kb[i] = bf16_bits(kh[i]); vb[i] = bf16_bits(vh[i]);
    }
    TCK(hipMemcpy(kf, kh.data(), kh.size() * 4, hipMemcpyHostToDevice));
    TCK(hipMemcpy(vf, vh.data(), vh.size() * 4, hipMemcpyHostToDevice));
    auto* kbf = dup(kb); auto* vbf = dup(vb);
    std::vector<int> tabs(nr / KV_PAGE), sel(4 * g_index_budget);
    for (size_t i = 0; i < tabs.size(); i++) tabs[i] = (int)tabs.size() - 1 - (int)i;
    for (int r = 0; r < 4; r++) for (int i = 0; i < g_index_budget; i++)
      sel[(size_t)r * g_index_budget + i] = (i * 13 + r) % ((b + r + 1) / 4);
    auto* dt = dup(tabs); auto* ds = dup(sel);
    auto kp = empty, vp = empty;
    for (int t = 0; t < nr; t++) {
      int pr = tabs[t / KV_PAGE] * KV_PAGE + t % KV_PAGE;
      memcpy(kp.data() + (size_t)pr * 16, kr.data() + (size_t)t * 16, KV_Q8_ROW_BYTES);
      memcpy(vp.data() + (size_t)pr * 16, vr.data() + (size_t)t * 16, KV_Q8_ROW_BYTES);
    }
    auto* pk = dup(kp); auto* pv = dup(vp);
    float *a = dalloc(q.size()), *ref = dalloc(q.size()), *paged = dalloc(q.size());
    int* pos = dup(std::vector<int>{N - 1});
    float *pm = dalloc(2 * QSA_DEC_NSPLIT * 12 * 2),
          *pa = dalloc(2 * QSA_DEC_NSPLIT * 12 * 256);
    k_qsa_flash<true, false, KvQ8Block, true><<<dim3(QSA_DEC_NSPLIT, 2), 128>>>(
        dq, pk, pv, a, 1, 256, 2, 24, sparse ? ds : nullptr, 0, pos, pa, pm, 0, dt);
    k_qsa_flash_combine<<<dim3(12, 2), 64>>>(pm, pa, a, 12, 256, pos);
    k_qsa_flash<true, false, float, true><<<dim3(QSA_DEC_NSPLIT, 2), 128>>>(
        dq, kf, vf, ref, 1, 256, 2, 24, sparse ? ds : nullptr, 0, pos, pa, pm);
    k_qsa_flash_combine<<<dim3(12, 2), 64>>>(pm, pa, ref, 12, 256, pos);
    auto ah = dget(a, 6144), rh = dget(ref, 6144);
    double err = 0;
    for (size_t i = 0; i < ah.size(); i++) err = std::max(err, fabs((double)ah[i] - rh[i]));
    check(sparse ? "q8_sparse_decode" : "q8_dense_decode", err, 2e-6);
    if (sparse) {
      k_qsa_wmma<false><<<dim3(4, 2), 256>>>(dq, kq, vq, nullptr, a, 4, 256, 2, 24, ds, b, b);
      k_qsa_wmma<false, true><<<dim3(4, 2), 256>>>(dq, pk, pv, nullptr, paged, 4, 256, 2, 24, ds, b, b, dt);
      k_qsa_wmma<false><<<dim3(4, 2), 256>>>(dq, kbf, vbf, nullptr, ref, 4, 256, 2, 24, ds, b, b);
    } else {
      k_qsa_wmma<false, false, true><<<dim3(4, 2), 256>>>(dq, kq, vq, nullptr, a, 4, 256, 2, 24, nullptr, b, b);
      k_qsa_wmma<false, true, true><<<dim3(4, 2), 256>>>(dq, pk, pv, nullptr, paged, 4, 256, 2, 24, nullptr, b, b, dt);
      k_qsa_wmma<false, false, true><<<dim3(4, 2), 256>>>(dq, kbf, vbf, nullptr, ref, 4, 256, 2, 24, nullptr, b, b);
    }
    ah = dget(a, q.size()); rh = dget(ref, q.size()); auto ph = dget(paged, q.size());
    err = 0;
    for (size_t i = 0; i < ah.size(); i++) err = std::max(err, fabs((double)ah[i] - rh[i]));
    check(sparse ? "q8_sparse_wmma_reference" : "q8_dense_wmma_reference", err, 2e-5);
    check(sparse ? "q8_sparse_wmma_paged" : "q8_dense_wmma_paged",
          memcmp(ah.data(), ph.data(), ah.size() * 4) != 0, 0);
    // Changing future V must not perturb accepted rows of a verify batch.
    TCK(hipMemset(vq + (size_t)(b + 3) * 16, 0, KV_Q8_ROW_BYTES));
    if (sparse)
      k_qsa_wmma<false><<<dim3(3, 2), 256>>>(dq, kq, vq, nullptr, a, 3, 256, 2, 24, ds, b, b);
    else
      k_qsa_wmma<false, false, true><<<dim3(3, 2), 256>>>(dq, kq, vq, nullptr, a, 3, 256, 2, 24, nullptr, b, b);
    auto first3 = dget(a, 3 * 6144);
    check(sparse ? "q8_sparse_future_mask" : "q8_dense_future_mask",
          memcmp(ah.data(), first3.data(), first3.size() * 4) != 0, 0);
    for (void* p : {(void*)kf, (void*)vf, (void*)dq, (void*)kq, (void*)vq,
                    (void*)kbf, (void*)vbf, (void*)dt, (void*)ds, (void*)pk,
                    (void*)pv, (void*)a, (void*)ref, (void*)paged, (void*)pos,
                    (void*)pm, (void*)pa}) TCK(hipFree(p));
  }
}
