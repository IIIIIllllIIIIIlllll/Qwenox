// Native FP16 KV regressions; included by ktest.cu after the Q8 helpers.
static void test_kv_fp16() {
  const auto saved_rng = rng;
  const int P = KV_PAGE + 7, base = KV_PAGE - 3, rows = 3 * KV_PAGE;
  std::vector<float> k((size_t)P * 512), v(k.size());
  for (auto& x : k) x = frand() * 7.f;
  for (auto& x : v) x = frand() * 31.f;
  // Known IEEE binary16 encodings: signed zero, subnormals, ties-to-even,
  // finer-than-BF16 mantissas, max finite and overflow.
  const float cases[] = {0.f, -0.f, 0x1p-24f, -0x1p-24f, 0x1p-25f,
                         0x1.8p-24f, 1.f + 0x1p-10f, 1.f + 0x1p-11f,
                         1.f + 0x1.8p-10f, 65504.f, 65520.f, -65520.f};
  const uint16_t bits[] = {0x0000, 0x8000, 0x0001, 0x8001, 0x0000,
                           0x0002, 0x3c01, 0x3c00, 0x3c02, 0x7bff, 0x7c00, 0xfc00};
  for (size_t i = 0; i < sizeof bits / sizeof *bits; i++) k[i] = v[i] = cases[i];
  std::vector<int> tab = {2, 0, 1};
  auto* pt = dup(tab);
  auto* dk = dup(k); auto* dv = dup(v);
  std::vector<__half> empty((size_t)rows * 512, __float2half_rn(-3.f));
  auto* kc = dup(empty); auto* vc = dup(empty);
  auto* ks = dup(empty); auto* vs = dup(empty);
  k_store_kv_fp16<<<P * 2, 256>>>(dk, dv, kc, vc, P, base, pt);
  auto* dp = dup(std::vector<int>{base});
  for (int t = 0; t < P; t++) {
    int pos = base + t;
    TCK(hipMemcpy(dp, &pos, sizeof pos, hipMemcpyHostToDevice));
    k_store_kv_fp16<<<2, 256>>>(dk + (size_t)t * 512, dv + (size_t)t * 512,
                              ks, vs, 1, 0, pt, dp);
  }
  auto hk = q8_get(kc, empty.size()), hv = q8_get(vc, empty.size());
  auto sk = q8_get(ks, empty.size()), sv = q8_get(vs, empty.size());
  check("fp16_batch_vs_step", memcmp(hk.data(), sk.data(), hk.size() * 2) ||
        memcmp(hv.data(), sv.data(), hv.size() * 2), 0);
  auto ek = empty, ev = empty;
  for (int t = 0; t < P; t++) {
    int pos = base + t, pr = tab[pos / KV_PAGE] * KV_PAGE + pos % KV_PAGE;
    for (int j = 0; j < 512; j++) {
      ek[(size_t)pr * 512 + j] = __float2half_rn(k[(size_t)t * 512 + j]);
      ev[(size_t)pr * 512 + j] = __float2half_rn(v[(size_t)t * 512 + j]);
    }
  }
  check("fp16_paged_cpu_reference", memcmp(hk.data(), ek.data(), hk.size() * 2) ||
        memcmp(hv.data(), ev.data(), hv.size() * 2), 0);
  const size_t first = (size_t)(tab[base / KV_PAGE] * KV_PAGE + base % KV_PAGE) * 512;
  check("fp16_ieee_rne_bits", memcmp(hk.data() + first, bits, sizeof bits) ||
        memcmp(hv.data() + first, bits, sizeof bits), 0);

  // Capture once, then replay with two different device bases across a seam.
  hipStream_t stream;
  hipGraph_t graph;
  hipGraphExec_t exec;
  TCK(hipStreamCreate(&stream));
  TCK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
  k_store_kv_fp16<<<6, 256, 0, stream>>>(dk, dv, ks, vs, 3, 0, pt, dp);
  TCK(hipStreamEndCapture(stream, &graph));
  TCK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  for (int b : {KV_PAGE - 1, 2 * KV_PAGE - 1}) {
    TCK(hipMemcpy(ks, empty.data(), empty.size() * 2, hipMemcpyHostToDevice));
    TCK(hipMemcpy(vs, empty.data(), empty.size() * 2, hipMemcpyHostToDevice));
    TCK(hipMemcpy(dp, &b, sizeof b, hipMemcpyHostToDevice));
    TCK(hipGraphLaunch(exec, stream));
    TCK(hipStreamSynchronize(stream));
    ek = empty; ev = empty;
    for (int t = 0; t < 3; t++) {
      const int pr = tab[(b + t) / KV_PAGE] * KV_PAGE + (b + t) % KV_PAGE;
      for (int j = 0; j < 512; j++) {
        ek[(size_t)pr * 512 + j] = __float2half_rn(k[(size_t)t * 512 + j]);
        ev[(size_t)pr * 512 + j] = __float2half_rn(v[(size_t)t * 512 + j]);
      }
    }
    sk = q8_get(ks, empty.size()); sv = q8_get(vs, empty.size());
    check("fp16_graph_replay_base", memcmp(sk.data(), ek.data(), sk.size() * 2) ||
          memcmp(sv.data(), ev.data(), sv.size() * 2), 0);
  }
  TCK(hipGraphExecDestroy(exec)); TCK(hipGraphDestroy(graph)); TCK(hipStreamDestroy(stream));
  for (void* p : {(void*)pt, (void*)dk, (void*)dv, (void*)kc, (void*)vc,
                  (void*)ks, (void*)vs, (void*)dp}) TCK(hipFree(p));

  for (int sparse = 0; sparse < 2; sparse++) {
    const int b = sparse ? g_index_first : KV_PAGE - 2, N = b + 4;
    const int nr = ((N + KV_PAGE - 1) / KV_PAGE) * KV_PAGE;
    std::vector<float> kh((size_t)nr * 512), vh(kh.size()), q(4 * 6144);
    for (auto& x : kh) x = __half2float(__float2half_rn(frand()));
    for (auto& x : vh) x = __half2float(__float2half_rn(frand()));
    for (auto& x : q) x = frand();
    std::vector<__half> h(kh.size()), vv(vh.size());
    for (size_t i = 0; i < h.size(); i++) {
      h[i] = __float2half_rn(kh[i]); vv[i] = __float2half_rn(vh[i]);
    }
    std::vector<int> tabs(nr / KV_PAGE), sel(4 * g_index_budget);
    for (size_t i = 0; i < tabs.size(); i++) tabs[i] = (int)tabs.size() - 1 - (int)i;
    for (int r = 0; r < 4; r++) for (int i = 0; i < g_index_budget; i++)
      sel[(size_t)r * g_index_budget + i] = (i * 13 + r) % ((b + r + 1) / 4);
    auto hp = h, vp = vv;
    for (int t = 0; t < nr; t++) {
      int pr = tabs[t / KV_PAGE] * KV_PAGE + t % KV_PAGE;
      memcpy(hp.data() + (size_t)pr * 512, h.data() + (size_t)t * 512, 1024);
      memcpy(vp.data() + (size_t)pr * 512, vv.data() + (size_t)t * 512, 1024);
    }
    auto* dt = dup(tabs); auto* ds = dup(sel); auto* dq = dup(q);
    auto* kf = dup(kh); auto* vf = dup(vh);
    auto* kn = dup(h); auto* vn = dup(vv); auto* pk = dup(hp); auto* pv = dup(vp);
    float *a = dalloc(q.size()), *ref = dalloc(q.size()), *paged = dalloc(q.size());
    int* pos = dup(std::vector<int>{N - 1});
    float *pm = dalloc(2 * QSA_DEC_NSPLIT * 12 * 2),
          *pa = dalloc(2 * QSA_DEC_NSPLIT * 12 * 256);
    k_qsa_flash<true, false, __half, true><<<dim3(QSA_DEC_NSPLIT, 2), 128>>>(
        dq, pk, pv, a, 1, 256, 2, 24, sparse ? ds : nullptr, 0, pos, pa, pm, 0, dt);
    k_qsa_flash_combine<<<dim3(12, 2), 64>>>(pm, pa, a, 12, 256, pos);
    k_qsa_flash<true, false, float, true><<<dim3(QSA_DEC_NSPLIT, 2), 128>>>(
        dq, kf, vf, ref, 1, 256, 2, 24, sparse ? ds : nullptr, 0, pos, pa, pm);
    k_qsa_flash_combine<<<dim3(12, 2), 64>>>(pm, pa, ref, 12, 256, pos);
    auto ah = dget(a, 6144), rh = dget(ref, 6144);
    double err = 0;
    for (size_t i = 0; i < ah.size(); i++) {
      if (!std::isfinite(ah[i])) err = INFINITY;
      else err = std::max(err, fabs((double)ah[i] - rh[i]));
    }
    check(sparse ? "fp16_sparse_decode" : "fp16_dense_decode", err, 2e-6);
    if (sparse) {
      k_qsa_wmma<false><<<dim3(4, 2), 256>>>(dq, kn, vn, nullptr, a, 4, 256, 2, 24, ds, b, b);
      k_qsa_wmma<false, true><<<dim3(4, 2), 256>>>(dq, pk, pv, nullptr, paged, 4, 256, 2, 24, ds, b, b, dt);
    } else {
      k_qsa_wmma<false, false, true><<<dim3(4, 2), 256>>>(dq, kn, vn, nullptr, a, 4, 256, 2, 24, nullptr, b, b);
      k_qsa_wmma<false, true, true><<<dim3(4, 2), 256>>>(dq, pk, pv, nullptr, paged, 4, 256, 2, 24, nullptr, b, b, dt);
    }
    ah = dget(a, q.size()); auto ph = dget(paged, q.size());
    check(sparse ? "fp16_sparse_wmma_paged" : "fp16_dense_wmma_paged",
          memcmp(ah.data(), ph.data(), ah.size() * 4), 0);
    // Independent attention reference: FP16 Q/K/V, exact softmax and PV.
    // WMMA rounds P to FP16 in each online-softmax tile, so allow that error.
    err = 0;
    for (int r = 0; r < 4; r++) for (int head = 0; head < 24; head++) {
      const int visible = b + r + 1;
      const int nt = sparse ? g_index_tokens + visible % 4 : visible;
      auto source = [&](int j) {
        return !sparse ? j : j < g_index_tokens ?
            4 * sel[(size_t)r * g_index_budget + j / 4] + j % 4 :
            (visible / 4) * 4 + j - g_index_tokens;
      };
      std::vector<double> scores(nt);
      double mx = -INFINITY, sum = 0;
      for (int j = 0; j < nt; j++) {
        double dot = 0;
        const size_t off = (size_t)source(j) * 512 + (head / 12) * 256;
        for (int d = 0; d < 256; d++)
          dot += __half2float(__float2half_rn(q[((size_t)r * 24 + head) * 256 + d])) *
                 (double)kh[off + d];
        scores[j] = dot / 16.; mx = std::max(mx, scores[j]);
      }
      for (auto& score : scores) { score = exp(score - mx); sum += score; }
      for (int d = 0; d < 256; d++) {
        double out = 0;
        for (int j = 0; j < nt; j++)
          out += scores[j] * vh[(size_t)source(j) * 512 + (head / 12) * 256 + d];
        const float got = ah[((size_t)r * 24 + head) * 256 + d];
        if (!std::isfinite(got)) err = INFINITY;
        else err = std::max(err, fabs(got - out / sum));
      }
    }
    check(sparse ? "fp16_sparse_wmma_cpu" : "fp16_dense_wmma_cpu", err, 8e-5);
    // Future values (including NaNs) cannot affect accepted verify rows.
    std::vector<__half> poison(512, __float2half_rn(NAN));
    TCK(hipMemcpy(vn + (size_t)(b + 3) * 512, poison.data(), 1024, hipMemcpyHostToDevice));
    if (sparse)
      k_qsa_wmma<false><<<dim3(3, 2), 256>>>(dq, kn, vn, nullptr, a, 3, 256, 2, 24, ds, b, b);
    else
      k_qsa_wmma<false, false, true><<<dim3(3, 2), 256>>>(dq, kn, vn, nullptr, a, 3, 256, 2, 24, nullptr, b, b);
    auto first3 = dget(a, 3 * 6144);
    check(sparse ? "fp16_sparse_future_mask" : "fp16_dense_future_mask",
          memcmp(ah.data(), first3.data(), first3.size() * 4), 0);
    for (void* p : {(void*)dt, (void*)ds, (void*)dq, (void*)kf, (void*)vf,
                    (void*)kn, (void*)vn, (void*)pk, (void*)pv, (void*)a,
                    (void*)ref, (void*)paged, (void*)pos, (void*)pm, (void*)pa}) TCK(hipFree(p));
  }
  rng = saved_rng;
}
