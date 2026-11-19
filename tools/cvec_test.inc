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

template <class T> static std::vector<T> cvec_get(const T* p, size_t n);

// Exercise the production fused write+norm paths, including the small verify
// batches and a partial T=4 block. Compare residuals AND the following norm
// with an independent host formula; testing k_cvec_apply alone misses these.
static void test_cvec_fused() {
  const int n = 2560, br = 4;
  for (int mode = 1; mode < 3; ++mode) {  // project, disabled; add uses standalone apply
    const bool project = mode == 1;
    for (int variant = 0; variant < 6; ++variant) {
      const bool bf16 = variant < 4;
      double max_r = 0, max_norm = 0;
      for (int P : {1, 3, 4, 5, 9}) {
        const size_t count = (size_t)P * br * n;
        std::vector<float> input(count), y((size_t)P * n), w((size_t)br * n), v(n);
        for (float& x : input) x = bf16 ? cvec_bf16_round(frand() * 2.f) : frand() * 2.f;
        for (float& x : y) x = frand() * .25f;
        for (float& x : w) x = frand() * .125f;
        double norm_v = 0;
        for (float& x : v) { x = frand() * .2f; norm_v += (double)x * x; }
        if (project) for (float& x : v) x = (float)(x / sqrt(norm_v));
        const float scale = project ? 1.375f : .1f;
        std::vector<float> ref(count), ref_norm(count);
        for (int row = 0; row < P * br; ++row) {
          const size_t base = (size_t)row * n;
          double dot = 0;
          for (int j = 0; j < n; ++j) {
            float x = input[base+j] + y[(size_t)(row/br)*n+j];
            ref[base+j] = bf16 ? cvec_bf16_round(x) : x;
            dot += (double)ref[base+j] * v[j];
          }
          double ss = 0;
          for (int j = 0; j < n; ++j) {
            float x = ref[base+j];
            if (mode != 2) x = (float)(x + (project ? -scale * dot : scale) * v[j]);
            ref[base+j] = bf16 ? cvec_bf16_round(x) : x;
            ss += (double)ref[base+j] * ref[base+j];
          }
          const double inv = 1. / sqrt(ss/n + 1e-6f);
          for (int j = 0; j < n; ++j) {
            const float x = (float)(ref[base+j] * inv * (1.f+w[(size_t)(row%br)*n+j]));
            ref_norm[base+j] = bf16 ? cvec_bf16_round(x) : x;
          }
        }
        float* dv = dup(v), *dy = dup(y), *dw = dup(w);
        float* dw4 = dup(std::vector<float>((size_t)P*br, 0.f));  // inject == 1 exactly
        float* dout = dup(std::vector<float>((size_t)P*br*br, 0.f));
        uint16_t* dwi = dup(std::vector<uint16_t>((size_t)br*br*n, 0));
        uint16_t* dr16 = dup(std::vector<uint16_t>(count, 0));
        uint16_t* dh16 = dup(std::vector<uint16_t>(count, 0));
        float* dr = dup(input), *dh = dup(std::vector<float>(count, 0.f));
        const float* dir = mode == 2 ? nullptr : dv;
        if (bf16) {
          std::vector<uint16_t> bits(count);
          for (size_t i = 0; i < count; ++i) bits[i] = bf16_bits(input[i]);
          TCK(hipMemcpy(dr16, bits.data(), count*2, hipMemcpyHostToDevice));
          if (variant == 0)
            k_gr_scatter_norm_b_hc_bf16<<<P*br, 256>>>(dw4, dr16, dy, dw, dh16, n, br, P, 1e-6f, dir, scale);
          else if (variant == 1)
            k_gr_scatter_norm_b_hc_bf16_f4<<<P, 256>>>(dw4, dr16, dy, dw, dh16, n, br, P, 1e-6f, dir, scale);
          else if (variant == 2)
            k_gr_scatter_norm_inj_bf16<1><<<P*br, 256>>>(dw4, nullptr, dr16, dy, dw, dh16, dwi, dout, n, P, 1e-6f, nullptr, dir, scale);
          else
            k_gr_scatter_norm_inj_bf16<4><<<br*((P+3)/4), 256>>>(dw4, nullptr, dr16, dy, dw, dh16, dwi, dout, n, P, 1e-6f, nullptr, dir, scale);
        } else if (variant == 4)
          k_gr_scatter_norm_b<<<P*br, 1024>>>(dw4, dr, dy, dw, dh, n, br, P, 1e-6f, nullptr, dir, scale);
        else
          k_gr_scatter_norm_b<true><<<P*br, 1024>>>(dw4, dr, dy, dw, dh, n, br, P, 1e-6f, dh16, dir, scale);
        TCK(hipGetLastError());
        auto got = dget(dr, count), got_norm = dget(dh, count);
        if (bf16) {
          auto r = cvec_get(dr16, count), h = cvec_get(dh16, count);
          for (size_t i = 0; i < count; ++i) {
            uint32_t a = (uint32_t)r[i] << 16, b = (uint32_t)h[i] << 16;
            memcpy(&got[i], &a, 4); memcpy(&got_norm[i], &b, 4);
          }
        }
        const double tol = bf16 ? 4./128 : 4e-6;
        for (size_t i = 0; i < count; ++i) {
          if (!std::isfinite(got[i]) || !std::isfinite(got_norm[i])) {
            max_r = max_norm = HUGE_VAL;
            continue;
          }
          max_r = std::max(max_r, fabs((double)got[i]-ref[i])/(tol*std::max(1., fabs((double)ref[i]))));
          max_norm = std::max(max_norm, fabs((double)got_norm[i]-ref_norm[i])/(tol*std::max(1., fabs((double)ref_norm[i]))));
        }
        TCK(hipFree(dv)); TCK(hipFree(dy)); TCK(hipFree(dw)); TCK(hipFree(dw4));
        TCK(hipFree(dout)); TCK(hipFree(dwi)); TCK(hipFree(dr16)); TCK(hipFree(dh16));
        TCK(hipFree(dr)); TCK(hipFree(dh));
      }
      char label[64];
      const char* mode_name = mode == 2 ? "off" : project ? "project" : "add";
      snprintf(label, sizeof label, "cvec_fused_%s_%d", mode_name, variant);
      check(label, std::max(max_r,max_norm), 1.);
    }
  }
}

struct CvecTestEnv {
  std::vector<std::pair<std::string, std::string>> values;
  std::vector<bool> present;
  CvecTestEnv() {
    for (const char* key : {"QWENOX_CVEC", "QWENOX_CVEC_MODE", "QWENOX_CVEC_SCALE",
                            "QWENOX_CVEC_LAYERS", "QWENOX_CVEC_DIR", "QWENOX_CVEC_PROBE"}) {
      const char* v = getenv(key);
      values.emplace_back(key, v ? v : ""); present.push_back(v != nullptr);
    }
  }
  static void set(const char* key, const char* value) {
#ifdef _WIN32
    _putenv_s(key, value ? value : "");
#else
    if (value) setenv(key, value, 1); else unsetenv(key);
#endif
  }
  ~CvecTestEnv() {
    for (size_t i = 0; i < values.size(); ++i)
      set(values[i].first.c_str(), present[i] ? values[i].second.c_str() : nullptr);
  }
};

// Load real GGUFs so normalization and effective scale are exercised together
// with the fingerprint, rather than populating the Cvec fields by hand.
static void test_cvec_fingerprint() {
  CvecTestEnv env;
  const auto dir = std::filesystem::temp_directory_path() /
      ("qwenox-cvec-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(dir);
  const auto a_path = dir / "unit.gguf", b_path = dir / "double.gguf";
  auto write_gguf = [&](const std::filesystem::path& path, float magnitude) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write((const char*)&v, 4); };
    auto u64 = [&](uint64_t v) { f.write((const char*)&v, 8); };
    auto str = [&](const char* v) { u64(strlen(v)); f.write(v, strlen(v)); };
    u32(0x46554747); u32(3); u64(1); u64(1);
    str("general.architecture"); u32(8); str("controlvector");
    str("direction.4"); u32(1); u64(g_cfg.d); u32(gguf::F32); u64(0);
    while ((uint64_t)f.tellp() % 32) f.put(0);
    std::vector<float> raw(g_cfg.d, 0.f); raw[0] = magnitude;
    f.write((const char*)raw.data(), raw.size()*sizeof(float));
    if (!f) throw std::runtime_error("cvec test: cannot write GGUF fixture");
  };
  write_gguf(a_path, 1.f); write_gguf(b_path, 2.f);
  CvecTestEnv::set("QWENOX_CVEC_MODE", "project");
  CvecTestEnv::set("QWENOX_CVEC_SCALE", "1");
  CvecTestEnv::set("QWENOX_CVEC_LAYERS", "4 4");
  CvecTestEnv::set("QWENOX_CVEC_DIR", "per-layer");
  CvecTestEnv::set("QWENOX_CVEC_PROBE", nullptr);
  CvecTestEnv::set("QWENOX_CVEC", a_path.string().c_str());
  Cvec a; a.load();
  CvecTestEnv::set("QWENOX_CVEC", b_path.string().c_str());
  Cvec b; b.load();
  check("cvec_fingerprint_norm", a.fingerprint() == b.fingerprint() ? 1. : 0., 0.);
  check("cvec_effective_scale", fabs(a.s(4)-1.f) + fabs(b.s(4)-2.f), 0.);
  Cvec disabled;
  check("cvec_fingerprint_off", disabled.fingerprint() == 0 ? 0. : 1., 0.);
  a.release(); b.release();
  std::filesystem::remove(a_path); std::filesystem::remove(b_path); std::filesystem::remove(dir);
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
