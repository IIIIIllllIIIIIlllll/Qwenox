#!/usr/bin/env python3
"""用 Windows 宿主编译器对照真实 Sampler，并检查异常输入拒绝契约。

须从 Windows Git Bash 调用 Windows Python，使用 --cxx 指定 TheRock 10.1
的 clang++.exe。本测试不启动 GPU，也不代替真实服务的异常恢复验收。
"""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


SOURCE_PATH = "src/gpu/parts/51_host_cfg.inc"


def extract_sampler(source):
    """只抽取真实结构体，不复制或改写任何采样逻辑。"""
    start = source.index("struct HostSampler {")
    end = source.index("// Default drafter", start)
    block = source[start:end].strip()
    if not block.endswith("};"):
        raise ValueError("HostSampler 抽取边界异常")
    return block


HARNESS = r'''
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

@@SAMPLERS@@

static void require(bool ok, const std::string& label) {
  if (!ok) throw std::runtime_error("检查失败：" + label);
}

static uint32_t bits(float value) {
  uint32_t result;
  static_assert(sizeof(result) == sizeof(value), "浮点位宽不符");
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static void same_vector(const std::vector<float>& a,
                        const std::vector<float>& b, const std::string& label) {
  require(a.size() == b.size(), label + " 长度");
  for (size_t i = 0; i < a.size(); ++i)
    require(bits(a[i]) == bits(b[i]), label + " 浮点位模式，位置=" + std::to_string(i));
}

struct Config {
  float temp, top_p, min_p;
  int top_k;
  float presence, frequency;
};

template<class S>
static S make_sampler(const Config& c, uint64_t seed) {
  S s{};
  s.temp = c.temp;
  s.top_p = c.top_p;
  s.min_p = c.min_p;
  s.top_k = c.top_k;
  s.presence = c.presence;
  s.frequency = c.frequency;
  s.rng = seed;
  s.bias = {{0, 0.3f}, {2, -0.7f}, {8, 0.15f}, {-1, 0.4f}, {9999, 1.0f}};
  s.remember(2);
  s.remember(2);
  s.remember(8);
  return s;
}

static void same_state(const HostSampler& a, const BaselineSampler& b,
                       const std::string& label) {
  require(a.rng == b.rng, label + " RNG");
  require(a.hist == b.hist, label + " remember 历史");
  require(a.cand == b.cand, label + " 候选顺序");
  same_vector(a.work, b.work, label + " 工作概率向量");
}

static void same_draw(int a, int b, float la, float lb,
                      const HostSampler& s, const BaselineSampler& base,
                      const std::string& label) {
  require(a == b, label + " token");
  require(bits(la) == bits(lb), label + " logprob 浮点位模式");
  require(std::isfinite(la), label + " 合法 logprob");
  same_state(s, base, label);
}

static void distribution(const std::vector<float>& pr, const std::string& label) {
  double total = 0;
  for (float p : pr) {
    require(std::isfinite(p) && p >= 0, label + " 概率有限非负");
    total += p;
  }
  require(std::isfinite(total) && std::abs(total - 1.0) < 1e-5,
          label + " 概率归一化");
}

static void valid_cases() {
  const Config configs[] = {
    {1.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f},
    {0.7f, 1.0f, 0.0f, 7, 0.3f, 0.12f},
    {1.3f, 0.83f, 0.0f, 0, -0.2f, 0.08f},
    {0.9f, 1.0f, 0.12f, 0, 0.4f, 0.1f},
    {0.65f, 0.91f, 0.04f, 11, 0.25f, 0.07f}
  };
  const uint64_t seeds[] = {0, 1, 0x123456789abcdef0ULL, 0xffffffffffffffffULL};
  std::mt19937 gen(0x51c0ffeeU);
  std::vector<std::vector<float>> rows;
  for (int r = 0; r < 12; ++r) {
    std::vector<float> row(37);
    for (float& v : row) v = (static_cast<int>(gen() % 20001) - 10000) / 731.0f;
    rows.push_back(row);
  }
  rows.push_back({std::numeric_limits<float>::max(), -std::numeric_limits<float>::max(),
                  0.0f, 1e30f, -1e30f, 2.0f, -2.0f, 17.0f, -17.0f});
  rows.push_back({-INFINITY, 0.5f, -INFINITY, -4.0f, 3.0f,
                  -INFINITY, 1.0f, -12.0f, 2.0f});
  rows.push_back(std::vector<float>(19, 0.0f));
  size_t dense_count = 0, sparse_count = 0, residual_count = 0, sparse_residual_count = 0;
  for (const Config& c : configs) for (uint64_t seed : seeds) {
    auto s = make_sampler<HostSampler>(c, seed);
    auto base = make_sampler<BaselineSampler>(c, seed);
    for (size_t ri = 0; ri < rows.size(); ++ri) {
      const std::string label = "配置/种子/输入=" + std::to_string(&c - configs) + "/" +
                                std::to_string(seed) + "/" + std::to_string(ri);
      for (int draw = 0; draw < 16; ++draw) {
        auto a = rows[ri], b = a;
        float la = 0, lb = 0;
        int ta = s.sample(a, &la), tb = base.sample(b, &lb);
        same_vector(a, b, label + " dense 概率");
        distribution(a, label + " dense");
        same_draw(ta, tb, la, lb, s, base, label + " dense sample");
        ++dense_count;
      }
      std::vector<int> ids(rows[ri].size());
      for (size_t j = 0; j < ids.size(); ++j) ids[j] = static_cast<int>(j * 3 + 1);
      HostSampler::SDist da;
      BaselineSampler::SDist db;
      s.prepare_sparse(ids.data(), rows[ri].data(), static_cast<int>(ids.size()), da);
      base.prepare_sparse(ids.data(), rows[ri].data(), static_cast<int>(ids.size()), db);
      require(da.ids == db.ids, label + " sparse 支持集");
      same_vector(da.pr, db.pr, label + " sparse 概率");
      distribution(da.pr, label + " sparse");
      same_state(s, base, label + " prepare_sparse");
      for (int draw = 0; draw < 32; ++draw) {
        float la = 0, lb = 0;
        const bool remember = draw % 2 == 0;
        int ta = s.draw_sparse(da, &la, remember), tb = base.draw_sparse(db, &lb, remember);
        same_draw(ta, tb, la, lb, s, base, label + " draw_sparse");
        ++sparse_count;
      }
      // 正残差和 p=q 回退均多次抽样；概率直接作为测试输入，不重写采样算法。
      const std::vector<float> p = {0.125f, 0.375f, 0.0f, 0.5f};
      for (bool equal : {false, true}) {
        const std::vector<float> q = equal ? p : std::vector<float>{0.5f, 0.25f, 0.125f, 0.125f};
        HostSampler::SDist pa{{9, 2, 17, 5}, p}, qa{{9, 2, 17, 5}, q};
        BaselineSampler::SDist pb{pa.ids, p}, qb{qa.ids, q};
        for (int draw = 0; draw < 32; ++draw) {
          float la = 0, lb = 0;
          int ta = s.sample_residual(p, q, &la), tb = base.sample_residual(p, q, &lb);
          same_draw(ta, tb, la, lb, s, base, label + (equal ? " dense p=q" : " dense 正残差"));
          s.remember(ta); base.remember(tb);
          same_state(s, base, label + " dense residual remember");
          ++residual_count;
          ta = s.sample_residual_sparse(pa, qa, &la);
          tb = base.sample_residual_sparse(pb, qb, &lb);
          same_draw(ta, tb, la, lb, s, base, label + (equal ? " sparse p=q" : " sparse 正残差"));
          s.remember(ta); base.remember(tb);
          same_state(s, base, label + " sparse residual remember");
          ++sparse_residual_count;
        }
        same_vector(pa.pr, pb.pr, label + " residual target 概率");
        same_vector(qa.pr, qb.pr, label + " residual draft 概率");
      }
      same_state(s, base, label + " 最终状态");
    }
  }
  std::cout << "PASS 合法对照：dense=" << dense_count << " sparse=" << sparse_count
            << " dense残差=" << residual_count << " sparse残差=" << sparse_residual_count
            << "；token/概率位模式/logprob位模式/RNG/历史一致\n";
}

template<class F>
static void rejects(const std::string& label, F invoke) {
  auto s = make_sampler<HostSampler>({1.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f}, 0xabc123ULL);
  const uint64_t rng = s.rng;
  bool rejected = false;
  try { invoke(s); }
  catch (const std::runtime_error&) { rejected = true; }
  catch (const std::exception& e) {
    throw std::runtime_error(label + " 抛出非 runtime_error：" + e.what());
  }
  require(rejected, label + " 必须抛出 std::runtime_error");
  require(s.rng == rng, label + " 拒绝前不得消耗 RNG");
  std::cout << "PASS 拒绝且 RNG 不变：" << label << "\n";
}

static void invalid_cases() {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  const std::vector<std::pair<std::string, std::vector<float>>> raw = {
    {"空 raw logits", {}}, {"raw NaN", {0.0f, nan}},
    {"raw +Inf", {0.0f, inf}}, {"raw 全 -Inf", {-inf, -inf}}
  };
  for (const auto& row : raw) {
    rejects("prepare " + row.first, [&](HostSampler& s) {
      auto v = row.second; s.prepare(v);
    });
    rejects("sample " + row.first, [&](HostSampler& s) {
      auto v = row.second; float lp; s.sample(v, &lp);
    });
    rejects("prepare_sparse " + row.first, [&](HostSampler& s) {
      HostSampler::SDist d;
      // k=0 使用有效哨兵地址，避免测试自身构造空指针算术。
      const int ids[] = {1, 2}; const float sentinel = 0;
      s.prepare_sparse(ids, row.second.empty() ? &sentinel : row.second.data(),
                       static_cast<int>(row.second.size()), d);
    });
  }
  rejects("draw_prepared 空候选", [](HostSampler& s) {
    float lp; s.cand.clear(); s.draw_prepared({1.0f}, &lp, true);
  });
  rejects("draw_sparse 空支持集", [](HostSampler& s) {
    float lp; s.draw_sparse({}, &lp, true);
  });
  const std::vector<std::pair<std::string, std::vector<float>>> masses = {
    {"全零", {0.0f, 0.0f}}, {"NaN", {nan, nan}},
    {"Inf", {inf, inf}}, {"有限质量混入 NaN", {0.5f, nan}},
    {"有限质量混入 Inf", {0.5f, inf}}, {"负概率", {0.5f, -0.1f}}
  };
  for (const auto& row : masses) {
    rejects("draw_prepared 概率" + row.first, [&](HostSampler& s) {
      float lp; s.cand = {0, 1}; s.draw_prepared(row.second, &lp, true);
    });
    rejects("draw_sparse 概率" + row.first, [&](HostSampler& s) {
      float lp; HostSampler::SDist d{{1, 2}, row.second}; s.draw_sparse(d, &lp, true);
    });
  }
  rejects("dense residual 空 target", [](HostSampler& s) {
    float lp; s.sample_residual({}, {}, &lp);
  });
  rejects("sparse residual 空 target SDist", [](HostSampler& s) {
    float lp; s.sample_residual_sparse({}, {}, &lp);
  });
  // 全零 target 必须拒绝；与合法 p=q 的零残差回退是两种不同情况。
  for (const auto& row : masses) {
    rejects("dense residual target " + row.first, [&](HostSampler& s) {
      float lp; s.sample_residual(row.second, {0.0f, 0.0f}, &lp);
    });
    rejects("sparse residual target " + row.first, [&](HostSampler& s) {
      float lp; HostSampler::SDist p{{1, 2}, row.second}, q{{1, 2}, {0.0f, 0.0f}};
      s.sample_residual_sparse(p, q, &lp);
    });
    if (row.first != "全零") {
      rejects("dense residual draft " + row.first, [&](HostSampler& s) {
        float lp; s.sample_residual({0.5f, 0.5f}, row.second, &lp);
      });
      rejects("sparse residual draft " + row.first, [&](HostSampler& s) {
        float lp; HostSampler::SDist p{{1, 2}, {0.5f, 0.5f}}, q{{1, 2}, row.second};
        s.sample_residual_sparse(p, q, &lp);
      });
    }
  }
}

int main(int argc, char** argv) {
#ifdef _WIN32
  // 仅抑制 Windows 崩溃对话框；复现模式不捕获基线 UB，也不计为 PASS。
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX);
  SetConsoleOutputCP(CP_UTF8);
#endif
  if (argc == 2 && std::string(argv[1]) == "empty-residual") {
    std::cerr << "仅复现基线 empty-residual 未定义行为；退出状态不代表测试通过\n";
    auto base = make_sampler<BaselineSampler>({1.0f, 1.0f, 0.0f, 0, 0.0f, 0.0f}, 1);
    float lp = 0;
    const int token = base.sample_residual_sparse({}, {}, &lp);
    std::cerr << "基线调用返回 token=" << token << "；未定义行为仍不能视为 PASS\n";
    return 3;
  }
  try {
    valid_cases();
    invalid_cases();
    std::cout << "PASS Sampler 回归完成；不包含真实服务异常恢复验收\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "失败：" << e.what() << "\n";
    return 1;
  }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cxx", required=True, help="TheRock 10.1 clang++.exe 的 Windows 路径")
    parser.add_argument("--baseline-ref", default="153d15c", help="对照基线 Git 引用")
    parser.add_argument("--case", choices=["regression", "empty-residual"], default="regression",
                        help="empty-residual 仅单独复现基线 UB，不属于正常测试")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("禁止 Linux 编译；请通过 Windows Git Bash 运行 Windows Python")
    compiler = Path(args.cxx).resolve()
    if not compiler.is_file() or compiler.name.lower() != "clang++.exe":
        parser.error("--cxx 必须指向 TheRock 10.1 的 clang++.exe 文件")
    root = Path(__file__).resolve().parent.parent
    build = root / "build"
    if not build.is_dir():
        parser.error("root/build 必须已经存在；脚本不会创建其他持久目录")
    current = extract_sampler((root / SOURCE_PATH).read_text(encoding="utf-8"))
    baseline_source = subprocess.run(
        ["git", "-C", str(root), "show", f"{args.baseline_ref}:{SOURCE_PATH}"],
        check=True, capture_output=True, encoding="utf-8", timeout=30,
    ).stdout
    baseline = re.sub(r"\bHostSampler\b", "BaselineSampler", extract_sampler(baseline_source))
    program = HARNESS.replace("@@SAMPLERS@@", current + "\n\n" + baseline)
    with tempfile.TemporaryDirectory(prefix="sampler-regression-", dir=build) as directory:
        work = Path(directory).resolve()
        work.relative_to(build.resolve())
        source = work / "sampler_regression_test.cpp"
        binary = work / "sampler_regression_test.exe"
        source.write_text(program, encoding="utf-8", newline="\n")
        command = [str(compiler), "-std=c++17", "-O3", "-Wall", "-Wextra", "-Werror",
                   "-D_CRT_SECURE_NO_WARNINGS", "-finput-charset=UTF-8",
                   "-fexec-charset=UTF-8", str(source), "-o", str(binary)]
        print(f"编译真实 Sampler：当前源码对照 {args.baseline_ref}", flush=True)
        subprocess.run(command, check=True, timeout=120)
        invocation = [str(binary)]
        if args.case == "empty-residual":
            print("注意：本模式仅复现基线 UB，非零退出或崩溃均不属于 PASS。", flush=True)
            invocation.append(args.case)
        result = subprocess.run(invocation, capture_output=True, encoding="utf-8",
                                errors="replace", timeout=120)
        print(result.stdout, end="", flush=True)
        print(result.stderr, end="", file=sys.stderr, flush=True)
        if args.case == "empty-residual":
            print(f"基线复现进程退出码：{result.returncode}；临时文件自动清理", flush=True)
            return 3
        return result.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"测试未完成：{exc}", file=sys.stderr)
        raise SystemExit(1)
