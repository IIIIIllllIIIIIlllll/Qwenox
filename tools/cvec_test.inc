// 控制向量投影内核回归（k_cvec_apply；included by ktest.cu）。
// 覆盖 add/project × fp32/bf16 四种实例：与 host 公式逐元素比对，并检查
// project 后残差与单位方向的正交性（|cos| 门限）。
static float cvec_bf16_round(float f) {
  const uint16_t b = bf16_bits(f);
  const uint32_t u = (uint32_t)b << 16;
  float r;
  memcpy(&r, &u, 4);
  return r;
}

template <class T>
static std::vector<T> cvec_get(const T* p, size_t n) {
  std::vector<T> v(n);
  TCK(hipMemcpy(v.data(), p, n * sizeof(T), hipMemcpyDeviceToHost));
  return v;
}

static void test_cvec_apply() {
  const int n = 2560, br = 4, P = 3;
  const int row_stride = br * n;
  for (int mode = 0; mode < 2; mode++) {      // 0 = add, 1 = project
    const bool proj = mode == 1;
    for (int is_bf16 = 0; is_bf16 < 2; is_bf16++) {
      std::vector<float> v(n), res((size_t)P * row_stride);
      double nrm = 0;
      for (auto& x : v) { x = frand(); nrm += (double)x * x; }
      nrm = sqrt(nrm);
      for (auto& x : v) x = (float)(x / nrm);  // 单位方向（加载器 project 后同形态）
      for (auto& x : res) x = frand() * 4.f;
      if (is_bf16) for (auto& x : res) x = cvec_bf16_round(x);  // 输入先落 bf16
      // project 的正交性检查要求 s=1（s<1 会按 (1-s) 保留残差分量，cos 不为 0）。
      const float s = 1.0f;

      // host 参考
      std::vector<double> ref((size_t)P * row_stride);
      for (int r = 0; r < P; r++)
        for (int b = 0; b < br; b++) {
          const float* h = &res[(size_t)r * row_stride + (size_t)b * n];
          double dot = 0;
          for (int j = 0; j < n; j++) dot += (double)h[j] * v[j];
          const double c = proj ? -s * dot : s;
          double* o = &ref[(size_t)r * row_stride + (size_t)b * n];
          for (int j = 0; j < n; j++) {
            float y = (float)(h[j] + c * v[j]);
            if (is_bf16) y = cvec_bf16_round(y);
            o[j] = y;
          }
        }

      float* dv = nullptr;
      TCK(hipMalloc((void**)&dv, n * sizeof(float)));
      TCK(hipMemcpy(dv, v.data(), n * sizeof(float), hipMemcpyHostToDevice));
      std::vector<float> got((size_t)P * row_stride);
      if (is_bf16) {
        std::vector<uint16_t> hb((size_t)P * row_stride);
        for (size_t i = 0; i < hb.size(); i++) hb[i] = bf16_bits(res[i]);
        uint16_t* db = dup(hb);
        if (proj)
          k_cvec_apply<true, true><<<dim3(P, br), 256>>>(db, dv, s, row_stride, n);
        else
          k_cvec_apply<true, false><<<dim3(P, br), 256>>>(db, dv, s, row_stride, n);
        auto hk = cvec_get(db, hb.size());
        for (size_t i = 0; i < got.size(); i++) {
          const uint32_t u = (uint32_t)hk[i] << 16;
          memcpy(&got[i], &u, 4);
        }
        TCK(hipFree(db));
      } else {
        float* df = dup(res);
        if (proj)
          k_cvec_apply<false, true><<<dim3(P, br), 256>>>(df, dv, s, row_stride, n);
        else
          k_cvec_apply<false, false><<<dim3(P, br), 256>>>(df, dv, s, row_stride, n);
        got = dget(df, got.size());
        TCK(hipFree(df));
      }
      TCK(hipFree(dv));

      // 逐元素误差（bf16 的参考也按 bf16 舍入；dot 求和顺序不同，留 1 ulp 余量）
      double maxrel = 0;
      for (size_t i = 0; i < got.size(); i++) {
        const double e = fabs((double)got[i] - ref[i]);
        const double q = is_bf16 ? 1.0 / 128.0 : 1e-6;  // bf16 相对分辨率
        maxrel = std::max(maxrel, e / (q * std::max(1.0, fabs(ref[i])) * 4.0));
      }
      char name[64];
      snprintf(name, sizeof name, "cvec_%s_%s", proj ? "project" : "add",
               is_bf16 ? "bf16" : "f32");
      check(name, maxrel, 1.0);

      if (proj) {  // 投影后应与方向正交
        double maxcos = 0;
        for (int r = 0; r < P; r++)
          for (int b = 0; b < br; b++) {
            const float* h = &got[(size_t)r * row_stride + (size_t)b * n];
            double d = 0, q = 0;
            for (int j = 0; j < n; j++) { d += (double)h[j] * v[j]; q += (double)h[j] * h[j]; }
            maxcos = std::max(maxcos, q > 0 ? fabs(d / sqrt(q)) : 0.0);
          }
        snprintf(name, sizeof name, "cvec_cos_%s", is_bf16 ? "bf16" : "f32");
        check(name, maxcos, is_bf16 ? 2e-2 : 1e-5);
      }
    }
  }
}
