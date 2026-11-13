// gguf_kv.h — 极简 GGUF v2/v3 元数据读取（只读 KV 表，不碰张量数据）。
// 供 API 的 GGUF 内嵌词表加载用（tokenizer.ggml.*）；引擎侧完整的张量视图
// loader 在 src/gguf.h（mmap 全部落盘、依赖 os_win32 兼容层），这里刻意不引用：
// 只用 stdio 读头部几 MB，无平台依赖、无 mmap。
//
// 格式（llama.cpp 约定）：magic "GGUF" + u32 version + u64 tensor_count +
// u64 kv_count，随后 kv_count 个键值对（键 = u64 长度 + 字节），值类型：
//   0 u8 / 1 i8 / 2 u16 / 3 i16 / 4 u32 / 5 i32 / 6 f32 / 7 bool /
//   8 string(u64 len + bytes) / 9 array(u32 元素类型 + u64 个数 + 元素) /
//   10 u64 / 11 i64 / 12 f64
// 分片 GGUF 的完整 KV 表只在第 1 个分片，读它即可。
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace gguf_kv {

struct Value {
    uint32_t type = 0;     // GGUF 值类型（数组时为元素类型）
    bool is_array = false;
    int64_t i = 0;                      // 标量整型（bool/uint 一并放宽到这里）
    double f = 0.0;                     // 标量浮点
    std::string s;                      // 字符串
    std::vector<int64_t> ai;            // 整型数组
    std::vector<std::string> as;        // 字符串数组
};

inline bool load(const std::string& path, std::unordered_map<std::string, Value>* out,
                 std::string* err) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) {
        if (err) *err = "cannot open " + path;
        return false;
    }
    auto fail = [&](const std::string& m) {
        fclose(fp);
        if (err) *err = path + ": " + m;
        return false;
    };
    auto rd = [&](void* p, size_t n) { return fread(p, 1, n, fp) == n; };
    auto rd_u32 = [&](uint32_t* v) { return rd(v, 4); };
    auto rd_u64 = [&](uint64_t* v) { return rd(v, 8); };
    auto rd_str = [&](std::string* s) {
        uint64_t n = 0;
        if (!rd_u64(&n) || n > (1ull << 30)) return false;
        s->resize(static_cast<size_t>(n));
        return n == 0 || rd(&(*s)[0], static_cast<size_t>(n));
    };
    // 整型元素按字节宽读进来放宽到 int64
    auto rd_ints = [&](uint32_t etype, uint64_t n, std::vector<int64_t>* dst) {
        static const uint8_t width[14] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8, 0};
        if (etype > 12 || !width[etype] || etype == 6 || etype == 12) return false;
        dst->resize(static_cast<size_t>(n));
        for (uint64_t k = 0; k < n; ++k) {
            uint8_t b[8] = {0};
            if (!rd(b, width[etype])) return false;
            switch (etype) {
            case 0: (*dst)[k] = b[0]; break;
            case 1: (*dst)[k] = static_cast<int8_t>(b[0]); break;
            case 7: (*dst)[k] = b[0] ? 1 : 0; break;
            case 2: case 4: case 10: {
                uint64_t v = 0;
                memcpy(&v, b, width[etype]);
                (*dst)[k] = static_cast<int64_t>(v);
                break;
            }
            default: {
                int64_t v = 0;
                memcpy(&v, b, width[etype]);
                (*dst)[k] = v;  // 3/5/11：小端符号扩展由定宽整型语义保证
                if (etype == 3 && width[etype] == 2) {
                    int16_t s16 = 0;
                    memcpy(&s16, b, 2);
                    (*dst)[k] = s16;
                } else if (etype == 5) {
                    int32_t s32 = 0;
                    memcpy(&s32, b, 4);
                    (*dst)[k] = s32;
                }
                break;
            }
            }
        }
        return true;
    };
    auto skip = [&](uint32_t type, uint64_t n) {
        static const uint8_t width[13] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
        if (type > 12 || !width[type]) return false;
        return fseek(fp, static_cast<long>(width[type] * n), SEEK_CUR) == 0;
    };

    char magic[4];
    uint32_t version = 0;
    uint64_t tensor_count = 0, kv_count = 0;
    if (!rd(magic, 4) || memcmp(magic, "GGUF", 4) != 0) return fail("not a GGUF file");
    if (!rd_u32(&version) || version < 2 || version > 3)
        return fail("unsupported GGUF version " + std::to_string(version));
    if (!rd_u64(&tensor_count) || !rd_u64(&kv_count)) return fail("bad header");
    if (kv_count > 100000) return fail("kv_count looks corrupt");

    for (uint64_t e = 0; e < kv_count; ++e) {
        std::string key;
        uint32_t vtype = 0;
        if (!rd_str(&key) || !rd_u32(&vtype)) return fail("bad kv entry");
        Value v;
        v.type = vtype;
        if (vtype == 8) {
            if (!rd_str(&v.s)) return fail("bad string value");
        } else if (vtype == 9) {
            uint32_t etype = 0;
            uint64_t n = 0;
            if (!rd_u32(&etype) || !rd_u64(&n)) return fail("bad array header");
            v.is_array = true;
            v.type = etype;
            if (etype == 8) {
                v.as.resize(static_cast<size_t>(n));
                for (uint64_t k = 0; k < n; ++k)
                    if (!rd_str(&v.as[static_cast<size_t>(k)]))
                        return fail("bad string array");
            } else if (etype != 6 && etype != 12 && etype <= 12) {
                if (!rd_ints(etype, n, &v.ai)) return fail("bad int array");
            } else {
                if (!skip(etype, n)) return fail("bad float array");
            }
        } else if (vtype <= 12) {
            std::vector<int64_t> one;
            if (vtype == 6 || vtype == 12) {
                uint8_t b[8] = {0};
                if (!rd(b, vtype == 6 ? 4 : 8)) return fail("bad float value");
                if (vtype == 6) {
                    float f32;
                    memcpy(&f32, b, 4);
                    v.f = f32;
                } else {
                    memcpy(&v.f, b, 8);
                }
            } else {
                if (!rd_ints(vtype, 1, &one)) return fail("bad int value");
                v.i = one[0];
            }
        } else {
            return fail("unknown value type " + std::to_string(vtype));
        }
        (*out)[std::move(key)] = std::move(v);
    }
    fclose(fp);
    return true;
}

}  // namespace gguf_kv
